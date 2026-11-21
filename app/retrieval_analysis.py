from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable
from typing import Any

_IGNORED_TERMS = {
    "about", "after", "also", "and", "any", "are", "been", "before", "being",
    "between", "both", "but", "can", "case", "cases", "could", "did", "does",
    "each", "for", "from", "had", "has", "have", "how", "into", "its", "may",
    "more", "most", "not", "our", "out", "over", "same", "should", "some",
    "such", "than", "that", "the", "their", "them", "there", "these", "they",
    "this", "those", "through", "under", "was", "were", "what", "when",
    "where", "which", "while", "who", "with", "would", "your",
    "must",
}

_REVIEW_QUESTIONS = [
    "Which material facts in your summary are supported by evidence, disputed, or still unverified?",
    "What legal test and holding does the full judgment apply, and does your case have the same procedural context?",
    "Which facts distinguish your case from each retrieved passage, and where are those facts shown in the record?",
    "What contrary evidence or alternative account should counsel address?",
]


def _term_counts(text: str) -> Counter[str]:
    terms = re.findall(r"[a-z][a-z0-9]{2,}|[0-9]{2,}", text.casefold())
    return Counter(term for term in terms if term not in _IGNORED_TERMS)


def _ranked_terms(
    counts: Counter[str],
    predicate: Callable[[str], bool],
) -> list[str]:
    return [
        term
        for term, _count in sorted(
            ((term, count) for term, count in counts.items() if predicate(term)),
            key=lambda item: (-item[1], item[0]),
        )[:8]
    ]


def build_retrieval_analysis(
    case_summary: str,
    sources: list[dict[str, Any]],
) -> dict[str, Any]:
    summary_terms = _term_counts(case_summary)
    comparisons = []
    for source in sources:
        passage_terms = _term_counts(str(source.get("excerpt", "")))
        comparisons.append({
            "source_id": source["source_id"],
            "shared_terms": _ranked_terms(
                summary_terms,
                lambda term: term in passage_terms,
            ),
            "summary_only_terms": _ranked_terms(
                summary_terms,
                lambda term: term not in passage_terms,
            ),
            "passage_only_terms": _ranked_terms(
                passage_terms,
                lambda term: term not in summary_terms,
            ),
        })

    return {
        "case_summary": case_summary.strip(),
        "comparisons": comparisons,
        "review_questions": _REVIEW_QUESTIONS.copy(),
    }
