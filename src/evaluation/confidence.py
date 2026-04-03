"""
Confidence Scoring
Computes multi-dimensional confidence scores for every RAG response:

  1. Retrieval Confidence  — how well the top-k chunks match the query
  2. Coverage Score        — how many of top-k chunks are high-quality
  3. Groundedness Score    — how much the answer is grounded in the context
  4. Overall Confidence    — weighted combination of all signals

Each score is in [0.0, 1.0]. Signals are fully explainable.
"""

import re
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ConfidenceResult:
    # Component scores
    retrieval_confidence: float      # Avg cosine similarity of top-k chunks
    coverage_score: float            # % chunks above quality threshold
    groundedness_score: float        # Lexical + semantic answer-context overlap
    consistency_score: float         # Agreement across chunks on topic
    overall_confidence: float        # Weighted composite

    # Per-chunk detail
    chunk_scores: list[dict] = field(default_factory=list)

    # Human-readable verdict
    verdict: str = ""
    explanation: str = ""

    def to_dict(self) -> dict:
        return {
            "retrieval_confidence": round(self.retrieval_confidence, 3),
            "coverage_score": round(self.coverage_score, 3),
            "groundedness_score": round(self.groundedness_score, 3),
            "consistency_score": round(self.consistency_score, 3),
            "overall_confidence": round(self.overall_confidence, 3),
            "verdict": self.verdict,
            "explanation": self.explanation,
            "chunk_scores": self.chunk_scores,
        }


# ── Lexical Groundedness ──────────────────────────────────────────────────────

def _tokenize(text: str) -> set[str]:
    """Simple unigram tokenizer, lowercased, stopwords removed."""
    STOP = {
        "the","a","an","is","are","was","were","be","been","being",
        "have","has","had","do","does","did","will","would","could","should",
        "may","might","shall","can","need","dare","ought","of","in","on",
        "at","to","for","by","with","from","into","through","and","or","but",
        "not","no","so","yet","both","either","neither","each","every","all",
        "any","few","more","most","other","some","such","than","too","very",
        "just","also","only","then","that","this","these","those","it","its",
    }
    tokens = re.findall(r"\b[a-z]+\b", text.lower())
    return {t for t in tokens if t not in STOP and len(t) > 2}


def _f1_overlap(answer: str, context: str) -> float:
    """Token F1 between answer tokens and context tokens."""
    a_toks = _tokenize(answer)
    c_toks = _tokenize(context)
    if not a_toks or not c_toks:
        return 0.0
    shared = a_toks & c_toks
    if not shared:
        return 0.0
    precision = len(shared) / len(a_toks)
    recall = len(shared) / len(c_toks)
    return 2 * precision * recall / (precision + recall)


def _sentence_overlap(answer: str, context: str, min_len: int = 5) -> float:
    """
    Fraction of answer sentences that have significant lexical overlap
    with at least one sentence in the context.
    """
    a_sents = [s.strip() for s in re.split(r"[.!?]", answer) if len(s.strip()) > min_len]
    c_sents = [s.strip() for s in re.split(r"[.!?]", context) if len(s.strip()) > min_len]
    if not a_sents:
        return 0.0
    grounded_count = 0
    for a_s in a_sents:
        if any(_f1_overlap(a_s, c_s) > 0.25 for c_s in c_sents):
            grounded_count += 1
    return grounded_count / len(a_sents)


def _number_coverage(answer: str, context: str) -> float:
    """
    What fraction of numbers/percentages mentioned in the answer
    also appear in the retrieved context? (financial accuracy signal)
    """
    num_pattern = r"\$[\d,.]+[BMKbmk]?|\b\d+\.?\d*%|\b\d{4}\b|\b\d+\.?\d*\s?(?:billion|million|trillion)\b"
    a_nums = set(re.findall(num_pattern, answer, re.IGNORECASE))
    c_nums = set(re.findall(num_pattern, context, re.IGNORECASE))
    if not a_nums:
        return 1.0   # No numbers to verify — neutral
    covered = a_nums & c_nums
    return len(covered) / len(a_nums)


# ── Main Scorer ───────────────────────────────────────────────────────────────

class ConfidenceScorer:
    """
    Scores the confidence and groundedness of a RAG response.

    Usage:
        scorer = ConfidenceScorer()
        result = scorer.score(query, answer, retrieved_results)
    """

    # Weights for overall confidence
    WEIGHTS = {
        "retrieval": 0.30,
        "coverage": 0.20,
        "groundedness": 0.35,
        "consistency": 0.15,
    }

    # Retrieval score threshold to count as "high quality"
    HIGH_QUALITY_THRESHOLD = 0.55

    def score(
        self,
        query: str,
        answer: str,
        retrieved_results: list,   # list of SearchResult objects
        query_vec: Optional[np.ndarray] = None,
    ) -> ConfidenceResult:
        """
        Compute full confidence breakdown.

        Args:
            query: User query string.
            answer: Generated answer text.
            retrieved_results: List of SearchResult objects.
            query_vec: Optional query embedding for semantic scoring.

        Returns:
            ConfidenceResult with all scores and explanations.
        """
        if not retrieved_results:
            return ConfidenceResult(
                retrieval_confidence=0.0,
                coverage_score=0.0,
                groundedness_score=0.0,
                consistency_score=0.0,
                overall_confidence=0.0,
                verdict="❌ No Evidence",
                explanation="No chunks were retrieved to support this answer.",
            )

        raw_scores = [r.score for r in retrieved_results]

        # ── 1. Retrieval Confidence ───────────────────────────────────────────
        # Normalize cosine similarity (inner product for L2-normalized vecs)
        # Typical range for good matches: 0.7–1.0, bad: 0.3–0.5
        retrieval_conf = float(np.mean(raw_scores))
        # Normalize to [0,1]: 0.3 → 0.0, 1.0 → 1.0
        retrieval_norm = max(0.0, min(1.0, (retrieval_conf - 0.3) / 0.7))

        # ── 2. Coverage Score ─────────────────────────────────────────────────
        # What fraction of retrieved chunks are above quality threshold?
        high_quality = sum(1 for s in raw_scores if s >= self.HIGH_QUALITY_THRESHOLD)
        coverage = high_quality / len(raw_scores)

        # ── 3. Groundedness Score ─────────────────────────────────────────────
        # How much of the answer is grounded in retrieved context?
        full_context = " ".join(r.text for r in retrieved_results)

        f1 = _f1_overlap(answer, full_context)
        sent_overlap = _sentence_overlap(answer, full_context)
        num_coverage = _number_coverage(answer, full_context)

        # "I don't know" or refusal → high groundedness (model is being honest)
        if any(phrase in answer.lower() for phrase in [
            "don't contain", "not available", "not mentioned",
            "cannot find", "no information"
        ]):
            groundedness = 0.85  # Honest refusal is grounded behavior
        else:
            groundedness = 0.45 * sent_overlap + 0.30 * f1 + 0.25 * num_coverage

        # ── 4. Consistency Score ──────────────────────────────────────────────
        # Do retrieved chunks agree on topic? Measure pairwise overlap.
        if len(retrieved_results) >= 2:
            pair_overlaps = []
            for i in range(len(retrieved_results)):
                for j in range(i + 1, min(i + 3, len(retrieved_results))):
                    pair_overlaps.append(
                        _f1_overlap(retrieved_results[i].text, retrieved_results[j].text)
                    )
            consistency = float(np.mean(pair_overlaps)) if pair_overlaps else 0.5
            # Some overlap is good (same topic), too much = redundant chunks
            # Optimal: 0.2–0.5 overlap. Normalize accordingly.
            consistency = 1.0 - abs(consistency - 0.3) / 0.3
            consistency = max(0.0, min(1.0, consistency))
        else:
            consistency = 0.5  # Neutral with single chunk

        # ── Overall ───────────────────────────────────────────────────────────
        overall = (
            self.WEIGHTS["retrieval"] * retrieval_norm
            + self.WEIGHTS["coverage"] * coverage
            + self.WEIGHTS["groundedness"] * groundedness
            + self.WEIGHTS["consistency"] * consistency
        )
        overall = max(0.0, min(1.0, overall))

        # ── Per-chunk scores ──────────────────────────────────────────────────
        chunk_scores = []
        for r in retrieved_results:
            chunk_f1 = _f1_overlap(answer, r.text)
            chunk_scores.append({
                "chunk_id": r.chunk_id,
                "company": r.company,
                "section": r.section,
                "retrieval_score": round(r.score, 4),
                "answer_overlap": round(chunk_f1, 3),
                "contributed": chunk_f1 > 0.15,
            })

        # ── Verdict ───────────────────────────────────────────────────────────
        verdict, explanation = self._interpret(overall, retrieval_norm, groundedness, raw_scores)

        return ConfidenceResult(
            retrieval_confidence=round(retrieval_norm, 3),
            coverage_score=round(coverage, 3),
            groundedness_score=round(groundedness, 3),
            consistency_score=round(consistency, 3),
            overall_confidence=round(overall, 3),
            chunk_scores=chunk_scores,
            verdict=verdict,
            explanation=explanation,
        )

    def _interpret(
        self,
        overall: float,
        retrieval: float,
        groundedness: float,
        raw_scores: list[float],
    ) -> tuple[str, str]:
        """Generate human-readable verdict and explanation."""
        top_score = max(raw_scores) if raw_scores else 0.0

        if overall >= 0.78:
            verdict = "✅ High Confidence"
            explanation = (
                f"Strong retrieval match (top score: {top_score:.2f}) with high answer-context overlap. "
                "The answer is well-grounded in the source documents."
            )
        elif overall >= 0.55:
            verdict = "⚠️ Medium Confidence"
            if groundedness < 0.45:
                explanation = (
                    f"Retrieval quality is adequate (top: {top_score:.2f}), but the answer may "
                    "contain inferences beyond what's explicitly stated in the source documents."
                )
            elif retrieval < 0.50:
                explanation = (
                    "The answer is well-grounded in retrieved text, but retrieval relevance "
                    "is moderate — related documents may not be perfectly indexed."
                )
            else:
                explanation = (
                    f"Moderate confidence. Top retrieval score: {top_score:.2f}. "
                    "Verify key figures against the original filings."
                )
        elif overall >= 0.35:
            verdict = "🔶 Low Confidence"
            explanation = (
                f"Weak retrieval match (top: {top_score:.2f}) or low answer-context overlap. "
                "The answer may include hallucinations. Recommend verifying against source documents."
            )
        else:
            verdict = "❌ Very Low Confidence"
            explanation = (
                "Very poor retrieval quality or minimal grounding in source text. "
                "This answer should not be relied upon without manual verification."
            )
        return verdict, explanation
