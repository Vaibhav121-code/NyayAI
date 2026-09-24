import re
from typing import Final

WHITESPACE_RE: Final[re.Pattern[str]] = re.compile(r"\s+")
BLANK_LINE_RE: Final[re.Pattern[str]] = re.compile(r"\n{3,}")
HEADER_FOOTER_RE: Final[re.Pattern[str]] = re.compile(
    r"(?im)^\s*(?:Page\s*\d+|[\-\*]{3,}|Supreme Court of India|IN THE SUPREME COURT OF INDIA)\s*$"
)


def clean_text(text: str) -> str:
    """Apply conservative text normalization to legal judgment text."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\t", " ")
    normalized = WHITESPACE_RE.sub(" ", normalized)
    normalized = normalized.replace(" \n", "\n").replace("\n ", "\n")
    normalized = normalized.replace("\n{", "{").replace("}\n", "}")
    normalized = BLANK_LINE_RE.sub("\n\n", normalized)

    lines = [line.strip() for line in normalized.split("\n")]
    cleaned_lines: list[str] = []
    for line in lines:
        if not line:
            if cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")
            continue
        if HEADER_FOOTER_RE.fullmatch(line):
            continue
        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r" *\n *", "\n", cleaned)
    cleaned = re.sub(r"\n +", "\n", cleaned)
    cleaned = re.sub(r" +", " ", cleaned)
    return cleaned.strip()
