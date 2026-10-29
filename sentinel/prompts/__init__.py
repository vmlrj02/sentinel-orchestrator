"""Prompts live in versioned files (name.vN.txt); the highest version is used and logged on every trace."""
import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path

PROMPT_DIR = Path(__file__).parent


@dataclass(frozen=True)
class Prompt:
    name: str
    version: str
    template: str

    def render(self, **values: str) -> str:
        return self.template.format(**values)


@cache
def load_prompt(name: str) -> Prompt:
    versions = {
        int(m.group(1)): path
        for path in PROMPT_DIR.glob(f"{name}.v*.txt")
        if (m := re.fullmatch(rf"{name}\.v(\d+)\.txt", path.name))
    }
    if not versions:
        raise FileNotFoundError(f"No prompt files for {name!r} in {PROMPT_DIR}")
    latest = max(versions)
    return Prompt(name, f"v{latest}", versions[latest].read_text())


def prompt_versions() -> dict[str, str]:
    return {f"prompt_{name}": load_prompt(name).version for name in ("interviewer", "advisor")}
