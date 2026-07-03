"""Deterministic lexical search (spec section 11.3).

Pure-Python BM25-style scoring on record search_text. No external embedding API.
Result order is deterministic for fixed query and corpus.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable

from .models import LegalRecord, ValidActionWorld
from .render import snippet


_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_PATTERN.findall(text or "")]


class LexicalIndex:
    def __init__(self, records: Iterable[LegalRecord]):
        self.records: list[LegalRecord] = list(records)
        self._docs: list[list[str]] = [_tokenize(r.search_text) for r in self.records]
        self._df: Counter[str] = Counter()
        for tokens in self._docs:
            for token in set(tokens):
                self._df[token] += 1
        self._avgdl = (
            sum(len(doc) for doc in self._docs) / len(self._docs) if self._docs else 0
        )
        self._k1 = 1.5
        self._b = 0.75

    def _idf(self, term: str) -> float:
        n = len(self._docs)
        df = self._df.get(term, 0)
        if df == 0:
            return 0.0
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(
        self,
        query: str,
        *,
        entity_id: str | None = None,
        record_type: str | None = None,
        max_results: int = 5,
    ) -> list[dict[str, object]]:
        if not query.strip():
            raise ValueError("empty query")
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores: list[tuple[float, int]] = []
        for idx, doc in enumerate(self._docs):
            record = self.records[idx]
            if entity_id is not None and record.entity_id != entity_id:
                continue
            if record_type is not None and record.record_type.value != record_type:
                continue
            doc_len = len(doc) if doc else 0
            tf_map: Counter[str] = Counter(doc)
            score = 0.0
            for token in tokens:
                if token not in tf_map:
                    continue
                tf = tf_map[token]
                idf = self._idf(token)
                denom = tf + self._k1 * (
                    1 - self._b + self._b * doc_len / max(self._avgdl, 1)
                )
                score += idf * (tf * (self._k1 + 1)) / max(denom, 1e-9)
            if score > 0:
                scores.append((score, idx))
        scores.sort(key=lambda x: (-x[0], x[1]))
        out: list[dict[str, object]] = []
        for _, idx in scores[:max_results]:
            record = self.records[idx]
            matched = next(
                (
                    snippet(section.text, query)
                    for section in record.sections
                    if any(token in section.text.lower() for token in tokens)
                ),
                snippet(record.search_text, query),
            )
            out.append(
                {
                    "record_id": record.record_id,
                    "title": record.title,
                    "record_type": record.record_type.value,
                    "entity_id": record.entity_id,
                    "effective_date": record.effective_date.isoformat(),
                    "status": record.status,
                    "snippet": matched,
                }
            )
        return out


def build_index(world: ValidActionWorld) -> LexicalIndex:
    return LexicalIndex(world.records)


__all__ = ["LexicalIndex", "build_index"]
