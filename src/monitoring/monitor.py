import os
import json
import time
import uuid
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from typing import Optional
from collections import defaultdict


@dataclass
class QueryEvent:
    event_id: str
    timestamp: str
    query: str
    mode: str                      
    doc_id: Optional[str]
    latency_ms: float
    cached: bool
    num_chunks_retrieved: int
    top_retrieval_score: float
    mean_retrieval_score: float
    answer_length: int
    confidence_score: float        
    groundedness_score: float     
    guardrail_triggered: bool
    guardrail_type: Optional[str]  #"off_topic" | "low_confidence" | "hallucination"
    model: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SessionStats:
    total_queries: int = 0
    cache_hits: int = 0
    guardrail_triggers: int = 0
    total_latency_ms: float = 0.0
    mode_counts: dict = field(default_factory=lambda: defaultdict(int))
    confidence_scores: list = field(default_factory=list)
    groundedness_scores: list = field(default_factory=list)
    retrieval_scores: list = field(default_factory=list)


class RAGMonitor:
   
    def __init__(self, log_dir: str = "data/monitoring"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / "query_log.jsonl"
        self._session = SessionStats()


    def log(self, event: QueryEvent):
        """Append event to log file and update in-memory session stats."""
        with open(self.log_path, "a") as f:
            f.write(json.dumps(event.to_dict()) + "\n")

        s = self._session
        s.total_queries += 1
        s.total_latency_ms += event.latency_ms
        s.mode_counts[event.mode] += 1
        if event.cached:
            s.cache_hits += 1
        if event.guardrail_triggered:
            s.guardrail_triggers += 1
        s.confidence_scores.append(event.confidence_score)
        s.groundedness_scores.append(event.groundedness_score)
        if event.top_retrieval_score > 0:
            s.retrieval_scores.append(event.top_retrieval_score)

    def make_event(
        self,
        query: str,
        mode: str,
        latency_ms: float,
        cached: bool,
        retrieval_scores: list[float],
        answer_length: int,
        confidence_score: float,
        groundedness_score: float,
        guardrail_triggered: bool,
        guardrail_type: Optional[str],
        doc_id: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
    ) -> QueryEvent:
        return QueryEvent(
            event_id=str(uuid.uuid4())[:8],
            timestamp=datetime.utcnow().isoformat(),
            query=query,
            mode=mode,
            doc_id=doc_id,
            latency_ms=latency_ms,
            cached=cached,
            num_chunks_retrieved=len(retrieval_scores),
            top_retrieval_score=max(retrieval_scores) if retrieval_scores else 0.0,
            mean_retrieval_score=sum(retrieval_scores) / len(retrieval_scores) if retrieval_scores else 0.0,
            answer_length=answer_length,
            confidence_score=confidence_score,
            groundedness_score=groundedness_score,
            guardrail_triggered=guardrail_triggered,
            guardrail_type=guardrail_type,
            model=model,
        )

    def load_history(self, last_n: int = 200) -> list[dict]:
    
        if not self.log_path.exists():
            return []
        events = []
        with open(self.log_path) as f:
            for line in f:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return events[-last_n:]

    def session_metrics(self) -> dict:
        
        s = self._session
        n = max(s.total_queries, 1)
        return {
            "total_queries": s.total_queries,
            "cache_hit_rate": round(s.cache_hits / n, 3),
            "guardrail_rate": round(s.guardrail_triggers / n, 3),
            "avg_latency_ms": round(s.total_latency_ms / n, 1),
            "avg_confidence": round(sum(s.confidence_scores) / max(len(s.confidence_scores), 1), 3),
            "avg_groundedness": round(sum(s.groundedness_scores) / max(len(s.groundedness_scores), 1), 3),
            "avg_retrieval_score": round(sum(s.retrieval_scores) / max(len(s.retrieval_scores), 1), 3),
            "mode_distribution": dict(s.mode_counts),
        }

    def historical_metrics(self, hours: int = 24) -> dict:
        
        events = self.load_history(last_n=1000)
        if not events:
            return {}

        cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
        recent = [e for e in events if e.get("timestamp", "") >= cutoff]
        if not recent:
            recent = events  # fallback to all

        n = len(recent)
        latencies = [e["latency_ms"] for e in recent]
        confidences = [e["confidence_score"] for e in recent]
        groundedness = [e["groundedness_score"] for e in recent]
        retrieval = [e["top_retrieval_score"] for e in recent if e["top_retrieval_score"] > 0]

        mode_counts = defaultdict(int)
        for e in recent:
            mode_counts[e.get("mode", "unknown")] += 1

        return {
            "period_hours": hours,
            "total_queries": n,
            "cache_hit_rate": round(sum(1 for e in recent if e["cached"]) / n, 3),
            "guardrail_rate": round(sum(1 for e in recent if e["guardrail_triggered"]) / n, 3),
            "avg_latency_ms": round(sum(latencies) / n, 1),
            "p95_latency_ms": round(sorted(latencies)[int(0.95 * len(latencies))], 1) if latencies else 0,
            "avg_confidence": round(sum(confidences) / n, 3),
            "avg_groundedness": round(sum(groundedness) / n, 3),
            "avg_top_retrieval_score": round(sum(retrieval) / max(len(retrieval), 1), 3),
            "mode_distribution": dict(mode_counts),
            "total_guardrail_triggers": sum(1 for e in recent if e["guardrail_triggered"]),
        }

    def query_trends(self, bucket_hours: int = 1, num_buckets: int = 24) -> list[dict]:
        
        events = self.load_history(last_n=2000)
        buckets = []
        now = datetime.utcnow()
        for i in range(num_buckets, -1, -1):
            bucket_start = now - timedelta(hours=(i + 1) * bucket_hours)
            bucket_end = now - timedelta(hours=i * bucket_hours)
            count = sum(
                1 for e in events
                if bucket_start.isoformat() <= e.get("timestamp", "") < bucket_end.isoformat()
            )
            avg_conf = 0.0
            bucket_events = [
                e for e in events
                if bucket_start.isoformat() <= e.get("timestamp", "") < bucket_end.isoformat()
            ]
            if bucket_events:
                avg_conf = sum(e["confidence_score"] for e in bucket_events) / len(bucket_events)
            buckets.append({
                "label": bucket_end.strftime("%H:%M"),
                "count": count,
                "avg_confidence": round(avg_conf, 3),
            })
        return buckets

    def top_queries(self, n: int = 10) -> list[dict]:
       
        events = self.load_history(last_n=500)
        counts = defaultdict(int)
        for e in events:
            counts[e.get("query", "")] += 1
        sorted_q = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        return [{"query": q, "count": c} for q, c in sorted_q[:n]]


_monitor: Optional[RAGMonitor] = None

def get_monitor(log_dir: str = "data/monitoring") -> RAGMonitor:
    global _monitor
    if _monitor is None:
        _monitor = RAGMonitor(log_dir=log_dir)
    return _monitor
