"""
Guardrails Module
Provides input validation and output quality checks for the RAG pipeline.

Input Guardrails:
  - Off-topic detection (non-financial queries)
  - Prompt injection detection
  - Query length / quality check

Output Guardrails:
  - Hallucination flag (answer contains facts not in context)
  - Low-confidence block (auto-refuse if confidence too low)
  - Refusal passthrough (model said "I don't know" → honor it)
"""

import re
from dataclasses import dataclass
from typing import Optional


# ── Schemas ───────────────────────────────────────────────────────────────────

@dataclass
class GuardrailCheck:
    passed: bool
    triggered: bool
    guardrail_type: Optional[str]   # "off_topic" | "injection" | "low_confidence" | "hallucination" | "empty_query"
    message: str                    # User-facing explanation
    severity: str                   # "block" | "warn" | "pass"


PASS = GuardrailCheck(passed=True, triggered=False, guardrail_type=None, message="", severity="pass")


# ── Input Guardrails ──────────────────────────────────────────────────────────

# Keywords that strongly indicate a financial/business query
FINANCIAL_KEYWORDS = {
    "revenue", "sales", "income", "profit", "loss", "earnings", "ebitda",
    "margin", "cash", "debt", "equity", "assets", "liabilities", "balance",
    "fiscal", "quarter", "annual", "growth", "dividend", "eps", "share",
    "stock", "market", "product", "segment", "risk", "business", "strategy",
    "outlook", "guidance", "acquisition", "investment", "capital", "rate",
    "cost", "expense", "operating", "gross", "net", "total", "billion",
    "million", "trillion", "percent", "ceo", "executive", "report", "10-k",
    "filing", "sec", "compare", "summarize", "extract", "financial", "company",
    "apple", "tesla", "microsoft", "amazon", "alphabet", "google", "nvidia",
    "forecast", "performance", "employee", "headcount", "r&d", "research",
    "cloud", "subscription", "customer", "competition", "competitor",
}

# Phrases that strongly suggest off-topic (non-financial) queries
OFF_TOPIC_PATTERNS = [
    r"\b(recipe|cook|food|restaurant|meal|diet|nutrition)\b",
    r"\b(movie|film|music|song|album|artist|celebrity|actor)\b",
    r"\b(weather|temperature|forecast|rain|snow|climate change)\b",
    r"\b(sports|football|basketball|soccer|nfl|nba|baseball)\b",
    r"\b(politics|election|vote|president|congress|senator)\b",
    r"\b(write me a poem|tell me a joke|play a game)\b",
    r"\b(relationship|love|date|marriage|divorce)\b",
    r"\b(medical|doctor|prescription|symptom|disease|hospital)\b",
]

# Prompt injection patterns
INJECTION_PATTERNS = [
    r"ignore (previous|all|your) instructions",
    r"disregard (the|your) (system|context|prompt)",
    r"you are now",
    r"new personality",
    r"act as (if you are|a|an)",
    r"forget (everything|all|your training)",
    r"jailbreak",
    r"DAN\b",
]


def check_input(query: str, strict_mode: bool = False) -> GuardrailCheck:
    """
    Validate a user query before processing.

    Args:
        query: Raw user input.
        strict_mode: If True, requires financial keywords (blocks vague queries).

    Returns:
        GuardrailCheck — passes through most queries, blocks clear violations.
    """
    query_lower = query.lower().strip()

    # Empty / too short
    if len(query_lower) < 5:
        return GuardrailCheck(
            passed=False, triggered=True,
            guardrail_type="empty_query",
            message="Query is too short. Please ask a specific question about the financial documents.",
            severity="block",
        )

    # Too long (likely injection or abuse)
    if len(query) > 2000:
        return GuardrailCheck(
            passed=False, triggered=True,
            guardrail_type="injection",
            message="Query exceeds maximum length. Please keep questions concise.",
            severity="block",
        )

    # Prompt injection
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, query_lower, re.IGNORECASE):
            return GuardrailCheck(
                passed=False, triggered=True,
                guardrail_type="injection",
                message="Query contains disallowed patterns. Please ask financial questions directly.",
                severity="block",
            )

    # Off-topic check
    has_off_topic = any(re.search(p, query_lower) for p in OFF_TOPIC_PATTERNS)
    has_financial = any(kw in query_lower for kw in FINANCIAL_KEYWORDS)

    if has_off_topic and not has_financial:
        return GuardrailCheck(
            passed=False, triggered=True,
            guardrail_type="off_topic",
            message=(
                "This appears to be a non-financial question. "
                "This tool is specialized for SEC filings analysis — try asking about "
                "revenue, risks, strategy, or financial metrics."
            ),
            severity="block",
        )

    # Strict mode: require at least some financial relevance
    if strict_mode and not has_financial:
        return GuardrailCheck(
            passed=True, triggered=True,
            guardrail_type="off_topic",
            message=(
                "⚠️ This query may not be about financial documents. "
                "Results may have low relevance. Consider asking about specific metrics, "
                "risks, or business topics from the SEC filings."
            ),
            severity="warn",
        )

    return PASS


# ── Output Guardrails ─────────────────────────────────────────────────────────

# Phrases the model uses when it genuinely lacks context — these are GOOD
HONEST_REFUSAL_PHRASES = [
    "the provided documents don't contain",
    "not mentioned in the",
    "cannot find this information",
    "not available in the context",
    "the documents do not",
    "no information about",
    "i don't have",
    "i cannot determine",
]

# Phrases that might indicate hallucination or over-confidence
SPECULATION_PHRASES = [
    r"\bprobably\b",
    r"\blikely\b",
    r"\bmight be\b",
    r"\bi believe\b",
    r"\bi think\b",
    r"\bassume\b",
    r"\bspeculate\b",
    r"\bmy understanding\b",
    r"\bgenerally speaking\b",
    r"\bin general\b",
    r"\btypically\b",
]

# Numbers in the answer that should ideally appear in context too
NUMBER_PATTERN = r"\$[\d,.]+[BMKbmk]?|\b\d+\.?\d*\s?(?:billion|million|trillion|percent|%)\b"


def check_output(
    answer: str,
    context: str,
    confidence_score: float,
    min_confidence: float = 0.30,
    warn_confidence: float = 0.50,
) -> GuardrailCheck:
    """
    Validate a generated answer before returning to user.

    Args:
        answer: LLM-generated answer.
        context: Retrieved context that was provided to the LLM.
        confidence_score: Overall confidence from ConfidenceScorer.
        min_confidence: Below this → block with refusal message.
        warn_confidence: Below this → pass but attach warning.

    Returns:
        GuardrailCheck result.
    """
    answer_lower = answer.lower()

    # Honest refusal — always pass through
    if any(phrase in answer_lower for phrase in HONEST_REFUSAL_PHRASES):
        return GuardrailCheck(
            passed=True, triggered=False,
            guardrail_type=None,
            message="",
            severity="pass",
        )

    # Very low confidence — block
    if confidence_score < min_confidence:
        return GuardrailCheck(
            passed=False, triggered=True,
            guardrail_type="low_confidence",
            message=(
                f"⚠️ Response blocked: confidence score {confidence_score:.2f} is below the minimum "
                f"threshold ({min_confidence}). The retrieved documents may not contain sufficient "
                "information to answer this question reliably. Try rephrasing or checking if the "
                "relevant documents are indexed."
            ),
            severity="block",
        )

    # Hallucination signal: numbers in answer not in context
    answer_nums = set(re.findall(NUMBER_PATTERN, answer, re.IGNORECASE))
    context_nums = set(re.findall(NUMBER_PATTERN, context, re.IGNORECASE))
    if answer_nums:
        uncovered = answer_nums - context_nums
        uncovered_rate = len(uncovered) / len(answer_nums)
        if uncovered_rate > 0.5 and len(uncovered) >= 2:
            return GuardrailCheck(
                passed=True, triggered=True,
                guardrail_type="hallucination",
                message=(
                    f"⚠️ Fact-check warning: {len(uncovered)} number(s) in this answer "
                    f"({', '.join(list(uncovered)[:3])}) were not found in the retrieved context. "
                    "Please verify these figures against the source documents."
                ),
                severity="warn",
            )

    # Speculation language warning
    speculation_count = sum(
        1 for p in SPECULATION_PHRASES
        if re.search(p, answer_lower)
    )
    if speculation_count >= 2:
        return GuardrailCheck(
            passed=True, triggered=True,
            guardrail_type="speculation",
            message=(
                "⚠️ This answer contains speculative language. "
                "Some claims may be inferences rather than direct facts from the documents."
            ),
            severity="warn",
        )

    # Low confidence warning (but not blocked)
    if confidence_score < warn_confidence:
        return GuardrailCheck(
            passed=True, triggered=True,
            guardrail_type="low_confidence",
            message=(
                f"⚠️ Moderate confidence ({confidence_score:.2f}). "
                "The retrieved chunks may not fully cover this topic. "
                "Consider verifying key figures in the original filings."
            ),
            severity="warn",
        )

    return PASS


class Guardrails:
    """Convenience wrapper combining input and output guardrails."""

    def __init__(
        self,
        strict_input: bool = False,
        min_confidence: float = 0.12,
        warn_confidence: float = 0.28,
    ):
        self.strict_input = strict_input
        self.min_confidence = min_confidence
        self.warn_confidence = warn_confidence

    def check_input(self, query: str) -> GuardrailCheck:
        return check_input(query, strict_mode=self.strict_input)

    def check_output(
        self, answer: str, context: str, confidence_score: float
    ) -> GuardrailCheck:
        return check_output(
            answer, context, confidence_score,
            min_confidence=self.min_confidence,
            warn_confidence=self.warn_confidence,
        )
