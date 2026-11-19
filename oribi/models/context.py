"""Canonical result returned by :func:`oribi.inspect`."""

from dataclasses import dataclass
from textwrap import shorten
from typing import Any

from .evidence import CardinalityContext, ColumnRelationshipContext, GrainContext, KeyCandidate, Recommendation, RelationshipContext, SemanticContext, Warning
from .profile import DatasetProfile, StatisticsContext


@dataclass(frozen=True)
class ColumnContext:
    name: str
    dtype: str
    logical_type: str
    statistics: StatisticsContext
    semantic: SemanticContext
    recommendations: tuple[Recommendation, ...] = ()
    warnings: tuple[Warning, ...] = ()
    cardinality: CardinalityContext | None = None
    relationships: tuple[ColumnRelationshipContext, ...] = ()


@dataclass(frozen=True)
class DatasetContext:
    profile: DatasetProfile
    columns: tuple[ColumnContext, ...]
    candidate_keys: tuple[KeyCandidate, ...] = ()
    grain: GrainContext | None = None
    relationships: tuple[RelationshipContext, ...] = ()
    warnings: tuple[Warning, ...] = ()
    schema_version: str = "2"

    def to_dict(self, *, include_values: bool = False) -> dict[str, Any]:
        from ..serialization.dict import to_dict
        return to_dict(self, include_values=include_values)

    def to_json(self, *, indent: int | None = 2, include_values: bool = False) -> str:
        from ..serialization.dict import to_json
        return to_json(self, indent=indent, include_values=include_values)

    def to_markdown(self, *, include_values: bool = False) -> str:
        """Render a report with evidence for inferences and warnings."""
        from ..serialization.markdown import to_markdown
        return to_markdown(self, include_values=include_values)

    def to_context(self, *, max_chars: int = 8000, include_values: bool = False) -> str:
        """Render compact context with warning and aggregation evidence when space allows."""
        from ..serialization.context import to_context
        return to_context(self, max_chars=max_chars, include_values=include_values)

    def summary(self) -> str:
        """Summarize the profile, likely grain, and up to three analytical warnings."""
        rows = self.profile.row_count
        cols = self.profile.column_count
        parts = [f"Dataset: {rows:,} rows × {cols:,} columns."]
        if self.profile.duplicate_row_count:
            parts.append(f"{self.profile.duplicate_row_count:,} duplicate rows observed.")
        if self.grain and self.grain.status == "estimated" and self.grain.label:
            parts.append(f"Likely grain: {self.grain.label} (confidence {self.grain.confidence:.2f}).")
        elif self.grain:
            parts.append("Likely grain could not be determined confidently.")
        if self.warnings:
            from ..serialization.context import prioritize_warnings

            parts.append(f"{len(self.warnings)} analytical warning(s) detected.")
            findings = (
                f"[{warning.severity}] {shorten(warning.message, width=140, placeholder='…')}"
                for warning in prioritize_warnings(self.warnings)[:3]
            )
            parts.append("Key findings: " + " · ".join(findings))
        return " ".join(parts)
