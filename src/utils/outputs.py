# when definind a new span extraction/generation type, define an enum and add a field to ExtractionResults
from __future__ import annotations

import dataclasses
import enum
from typing import List, Union, Optional


class SpanType(enum.Enum):
    GREEDY = 'greedy'
    BEAM = 'beam_search'
    ML_SPAN = 'ml_span'
    ML_SPAN_LENGTH_NORMALIZED = 'ml_span_norm'
    LABEL = 'label'

@dataclasses.dataclass
class PredictionWithoutMetadata:
    decoded: str
    nll: float = 0.0


@dataclasses.dataclass
class PredictionWithMetadata:
    tokens: List[str]
    tokens_ids: List[int]
    tokens_nlls: List[float]
    decoded: str
    nll: float

@dataclasses.dataclass
class AggregatedPredictionsOfType:
    top_k: Union[List[PredictionWithoutMetadata],List[PredictionWithMetadata]]  # not using optional to avoid casing on None, empty is "not calculated"
    tpe: SpanType
    k: int  # used in merging with other instance to know how many items to keep in top_k

    def merge(self, other: AggregatedPredictionsOfType) -> AggregatedPredictionsOfType:
        assert self.tpe == other.tpe, f"merging two results of different types: ({self.tpe},{other.tpe}."
        k = max(self.k, other.k)
        unique_ks = list(dict([(s.decoded, s) for s in (self.top_k + other.top_k)]).values())
        top_k = sorted(unique_ks, key=lambda x: x.nll, reverse=False)[:k]
        return AggregatedPredictionsOfType(top_k=top_k, tpe=self.tpe, k=k)

    @property
    def top(self):
        return self.top_k[0]


@dataclasses.dataclass
class ExtractionResults:
    greedy: Optional[AggregatedPredictionsOfType]
    beam: Optional[AggregatedPredictionsOfType]
    ml_span: Optional[AggregatedPredictionsOfType]
    ml_span_norm: Optional[AggregatedPredictionsOfType]
    label: Optional[AggregatedPredictionsOfType]
    input: str
    context_tokens: List[int]

    def _merge_field(self, other, field_name) -> Optional[AggregatedPredictionsOfType]:
        self_field: Optional[AggregatedPredictionsOfType] = getattr(self, field_name)
        other_field: Optional[AggregatedPredictionsOfType] = getattr(other, field_name)
        assert not (self_field is None and other_field is not None)
        assert not (self_field is not None and other_field is None)
        if self_field is None and other_field is None:
            return None
        else:
            return self_field.merge(other_field)

    def merge(self, other: ExtractionResults):
        fields = list(map(lambda x: x.name, dataclasses.fields(self)))
        d = dict((f, self._merge_field(other, f)) for f in fields)
        return ExtractionResults(**d)

    @classmethod
    def empty(cls):
        return ExtractionResults(
        greedy=None,
        beam=None,
        ml_span=None,
        ml_span_norm=None,
        label=None,
        input=None,
        context_tokens=None
    )
