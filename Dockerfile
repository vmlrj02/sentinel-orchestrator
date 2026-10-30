# One image for local Docker and AWS Lambda. On Lambda, the Lambda Web Adapter extension forwards
# invocations to uvicorn and streams responses back (SSE works unchanged); elsewhere it is inert.
FROM public.ecr.aws/docker/library/python:3.12-slim-bookworm

COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.9.1 /lambda-adapter /opt/extensions/lambda-adapter
COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /bin/uv

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PORT=8080 \
    AWS_LWA_INVOKE_MODE=response_stream \
    AWS_LWA_READINESS_CHECK_PATH=/healthz

WORKDIR /app
# Dependencies first so code changes don't reinstall them
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY sentinel ./sentinel
RUN useradd --create-home app
USER app

EXPOSE 8080
CMD ["/app/.venv/bin/uvicorn", "sentinel.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8080"]
