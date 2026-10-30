"""Things an applicant can say at any point, handled in code with no LLM call."""
import re

_COMMANDS = [
    ("start_over", r"(start (over|again)|restart|reset|cancel( (my )?application)?|new application)"),
    ("help", r"(help|\?|what can you (do|help with)|how does this work)"),
    ("summary", r"(what (do you have|have i (told|given) you)( so far)?|show (me )?(my )?(details|application|"
                r"answers)|summary|my details)"),
    ("human", r"((talk|speak|chat) (to|with) (a |an |some)?(person|human|someone|agent|advisor|officer|staff)"
              r"|human|agent|real person|customer (service|support))"),
    ("skip", r"(skip( (this|it|for now|this one))?|later|ask me later)"),
    ("apply", r"(apply for a loan|i('d| would) like to apply|start (an |my )?application)"),
    ("change", r"(change|update|edit|fix|correct)(?: (?:my|the))? ?(?P<what>.*)"),
]


def match_command(text: str) -> tuple[str, str] | None:
    """The whole message must be the command ("change my income", not a sentence that mentions it)."""
    cleaned = re.sub(r"\s+", " ", text.strip().lower())
    cleaned = cleaned if cleaned == "?" else cleaned.rstrip(".!?")
    cleaned = re.sub(r"^(please|pls|can i|could i|i want to|i'd like to)\s+", "", cleaned)
    for name, pattern in _COMMANDS:
        if match := re.fullmatch(rf"{pattern}( please)?", cleaned):
            what = (match.groupdict().get("what") or "").removesuffix(" please")
            if name == "change" and re.search(r"\d", what):
                return None  # "change my income to 40k" carries a value: the interviewer extracts it
            return name, what
    return None
