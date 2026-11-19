
import re
from typing import Final

HEADER_FOOTER_RE: Final[re.Pattern[str]] = re.compile(
    r"(?im)^\s*(?:Page\s*\d+|[\-\*]{3,}|"
    r"Supreme Court of India|IN THE SUPREME COURT OF INDIA)\s*$"
)

MULTIPLE_SPACES_RE: Final[re.Pattern[str]] = re.compile(r"[^\S\n]+")
BLANK_LINES_RE: Final[re.Pattern[str]] = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """Normalize legal text while preserving paragraph boundaries."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\t", " ")

    cleaned_lines = []

    for line in normalized.split("\n"):
        line = MULTIPLE_SPACES_RE.sub(" ", line).strip()

        if not line:
            if cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")
            continue

        if HEADER_FOOTER_RE.fullmatch(line):
            continue

        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines)
    cleaned = BLANK_LINES_RE.sub("\n\n", cleaned)

    return cleaned.strip()
