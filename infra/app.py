"""AWS infrastructure for Sentinel (CDK).

    cdk deploy            # builds the image, creates everything, prints the URL

One Lambda (container image + Lambda Web Adapter, response streaming) behind a public Function URL.
State in DynamoDB, secrets in Secrets Manager, metrics via CloudWatch EMF, alarms on errors/latency.
"""
import os

import aws_cdk as cdk
from aws_cdk import (
    Duration,
    RemovalPolicy,
)
from aws_cdk import (
    aws_cloudwatch as cw,
)
from aws_cdk import (
    aws_dynamodb as ddb,
)
from aws_cdk import (
    aws_ecr_assets as ecr_assets,
)
from aws_cdk import (
    aws_events as events,
)
from aws_cdk import (
    aws_events_targets as targets,
)
from aws_cdk import (
    aws_lambda as lambda_,
)
from aws_cdk import (
    aws_logs as logs,
)
from aws_cdk import (
    aws_secretsmanager as sm,
)
from constructs import Construct

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class SentinelStack(cdk.Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs: object) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Values filled in after deploy (scripts/put_secrets.sh); the code reads them at cold start
        secret = sm.Secret(self, "AppSecret", secret_name="sentinel/app", description="Sentinel API keys")

        checkpoints = ddb.TableV2(
            self, "Checkpoints", partition_key=ddb.Attribute(name="PK", type=ddb.AttributeType.STRING),
            sort_key=ddb.Attribute(name="SK", type=ddb.AttributeType.STRING),
            time_to_live_attribute="ttl", removal_policy=RemovalPolicy.DESTROY)
        applications = ddb.TableV2(
            self, "Applications",
            partition_key=ddb.Attribute(name="application_id", type=ddb.AttributeType.STRING),
            global_secondary_indexes=[
                ddb.GlobalSecondaryIndexPropsV2(
                    index_name="by_status", partition_key=ddb.Attribute(name="status", type=ddb.AttributeType.STRING),
                    sort_key=ddb.Attribute(name="created_at", type=ddb.AttributeType.STRING)),
                ddb.GlobalSecondaryIndexPropsV2(
                    index_name="recent", partition_key=ddb.Attribute(name="kind", type=ddb.AttributeType.STRING),
                    sort_key=ddb.Attribute(name="created_at", type=ddb.AttributeType.STRING)),
            ],
            point_in_time_recovery_specification=ddb.PointInTimeRecoverySpecification(
                point_in_time_recovery_enabled=True),  # audit records
            removal_policy=RemovalPolicy.RETAIN)
        locks = ddb.TableV2(
            self, "Locks", partition_key=ddb.Attribute(name="lock_id", type=ddb.AttributeType.STRING),
            time_to_live_attribute="expires_at", removal_policy=RemovalPolicy.DESTROY)

        log_group = logs.LogGroup(self, "Logs", retention=logs.RetentionDays.ONE_MONTH,
                                  removal_policy=RemovalPolicy.DESTROY)

        fn = lambda_.DockerImageFunction(
            self, "Api",
            code=lambda_.DockerImageCode.from_image_asset(
                ROOT, platform=ecr_assets.Platform.LINUX_ARM64,
                exclude=["infra", "cdk.out", ".venv", ".git", "_scratch", "data", "logs"]),
            architecture=lambda_.Architecture.ARM_64,
            memory_size=3008,  # Lambda allocates CPU in proportion to memory: ~2 vCPUs, faster cold starts
            timeout=Duration.seconds(120),
            log_group=log_group,
            environment={
                "ENVIRONMENT": "aws",
                # If start-up passes Lambda's 10 s init limit, finish it in the first request instead of
                # being killed and restarted
                "AWS_LWA_ASYNC_INIT": "true",
                "SECRETS_ARN": secret.secret_arn,
                "CHECKPOINTER": "dynamodb",
                "CHECKPOINT_TABLE": checkpoints.table_name,
                "APPLICATIONS_TABLE": applications.table_name,
                "LOCKS_TABLE": locks.table_name,
                "METRICS_BACKEND": "cloudwatch",
                "TELEGRAM_MODE": "webhook",
                "LANGFUSE_PUBLIC_TRACES": "true",
                "FLUSH_TRACES_EACH_TURN": "true",  # the process freezes between invocations
                "RATE_LIMIT_PER_MINUTE": "15",
                "TURN_TIMEOUT_S": "60",
            },
        )
        secret.grant_read(fn)
        for table in (checkpoints, applications, locks):
            table.grant_read_write_data(fn)
        fn.add_to_role_policy(cdk.aws_iam.PolicyStatement(actions=["cloudwatch:GetMetricData"], resources=["*"]))

        # Keep one instance warm so a reviewer's first click doesn't pay a cold start. The adapter
        # delivers non-HTTP events to POST /events.
        events.Rule(self, "KeepWarm", schedule=events.Schedule.rate(Duration.minutes(5)),
                    targets=[targets.LambdaFunction(fn)])

        url = fn.add_function_url(auth_type=lambda_.FunctionUrlAuthType.NONE,
                                  invoke_mode=lambda_.InvokeMode.RESPONSE_STREAM)

        # Alarms: visible in the CloudWatch console; add an SNS topic to page someone in a real deployment
        turn_errors = cw.Metric(namespace="Sentinel", metric_name="Turns", dimensions_map={"Outcome": "error"},
                                statistic="Sum", period=Duration.minutes(5))
        p95 = cw.Metric(namespace="Sentinel", metric_name="TurnLatency", dimensions_map={"Outcome": "ok"},
                        statistic="p95", period=Duration.minutes(5))
        llm_errors = cw.Metric(namespace="Sentinel", metric_name="LLMErrors", statistic="Sum",
                               period=Duration.minutes(5))
        cw.Alarm(self, "TurnErrors", metric=turn_errors, threshold=3, evaluation_periods=1,
                 alarm_description="Chat turns failing", treat_missing_data=cw.TreatMissingData.NOT_BREACHING)
        cw.Alarm(self, "SlowTurns", metric=p95, threshold=15_000, evaluation_periods=2,
                 alarm_description="p95 turn latency above 15s", treat_missing_data=cw.TreatMissingData.NOT_BREACHING)
        cw.Alarm(self, "LLMFailures", metric=llm_errors, threshold=5, evaluation_periods=1,
                 alarm_description="LLM provider errors after retries",
                 treat_missing_data=cw.TreatMissingData.NOT_BREACHING)
        cw.Alarm(self, "Throttles", metric=fn.metric_throttles(period=Duration.minutes(5)), threshold=1,
                 evaluation_periods=1, alarm_description="Lambda concurrency limit reached",
                 treat_missing_data=cw.TreatMissingData.NOT_BREACHING)

        dashboard = cw.Dashboard(self, "Dashboard", dashboard_name="Sentinel")
        dashboard.add_widgets(
            cw.GraphWidget(title="Turns by outcome", left=[
                cw.Metric(namespace="Sentinel", metric_name="Turns", dimensions_map={"Outcome": o}, statistic="Sum",
                          period=Duration.minutes(5), label=o) for o in ("ok", "error", "timeout", "rate_limited")]),
            cw.GraphWidget(title="Turn latency (ms)", left=[
                cw.Metric(namespace="Sentinel", metric_name="TurnLatency", dimensions_map={"Outcome": "ok"},
                          statistic=s, period=Duration.minutes(5), label=s) for s in ("p50", "p95", "p99")]),
            cw.GraphWidget(title="LLM tokens and errors", left=[
                cw.Metric(namespace="Sentinel", metric_name="LLMTokens", statistic="Sum", period=Duration.minutes(5))],
                right=[llm_errors]),
            cw.GraphWidget(title="Decisions", left=[
                cw.Metric(namespace="Sentinel", metric_name="Decisions", dimensions_map={"Decision": d},
                          statistic="Sum", period=Duration.hours(1), label=d)
                for d in ("approved", "declined", "referred", "underwriter_approved", "underwriter_declined")]),
            cw.GraphWidget(title="Lambda", left=[fn.metric_invocations(), fn.metric_errors(), fn.metric_throttles()],
                           right=[fn.metric_duration(statistic="p95")]),
        )

        cdk.CfnOutput(self, "Url", value=url.url)
        cdk.CfnOutput(self, "SecretArn", value=secret.secret_arn)
        cdk.CfnOutput(self, "FunctionName", value=fn.function_name)


app = cdk.App()
SentinelStack(app, "Sentinel", env=cdk.Environment(account=os.getenv("CDK_DEFAULT_ACCOUNT"),
                                                   region=os.getenv("CDK_DEFAULT_REGION", "us-east-1")))
app.synth()
