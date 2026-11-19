"""Observed per-value cardinality summaries."""

import pandas as pd

from ..models.evidence import CardinalityContext, Evidence


def _constancy_evidence(
    observed: pd.Series, observed_count: int, dominant_count: int | None, missing_count: int,
) -> Evidence | None:
    if dominant_count is None or observed_count < 2:
        return None
    share = dominant_count / observed_count
    if share < .95:
        return None
    if observed.dtype == object and pd.api.types.infer_dtype(observed, skipna=True) in {"mixed", "mixed-integer"}:
        # Object equality can group unlike values, such as True and 1.
        return None
    if dominant_count == observed_count:
        code = "constant_column"
        statement = f"Constant across {observed_count:,} non-missing rows"
    else:
        code = "near_constant_column"
        statement = f"Near-constant ({dominant_count:,} of {observed_count:,} non-missing rows share one value)"
    return Evidence(code, statement, {
        "observed_count": observed_count,
        "dominant_count": dominant_count,
        "dominant_share": share,
        "missing_count": missing_count,
    })


def profile_cardinality(series: pd.Series, row_count: int) -> CardinalityContext:
    """Describe observed frequencies without assigning business meaning.

    Missing values are excluded from the frequency distribution; row_count
    still includes them so callers can see the coverage of the column.
    """
    observed = series.dropna()
    exact_values = True
    try:
        counts = observed.value_counts(dropna=True, sort=False)
    except (TypeError, ValueError):
        # Match AnalysisSession's fallback for unhashable object values.
        counts = observed.astype(str).value_counts(dropna=True, sort=False)
        exact_values = False
    counts = counts[counts > 0]
    observed_count = int(counts.sum())
    unique_count = int(len(counts))
    if not unique_count:
        ratio = average = None
        minimum = maximum = None
    else:
        ratio = unique_count / observed_count
        average = observed_count / unique_count
        minimum = int(counts.min())
        maximum = int(counts.max())
    missing_count = row_count - observed_count
    return CardinalityContext(
        row_count=row_count,
        observed_count=observed_count,
        unique_count=unique_count,
        uniqueness_ratio=ratio,
        average_records_per_value=average,
        minimum_records_per_value=minimum,
        maximum_records_per_value=maximum,
        missing_count=missing_count,
        missing_rate=missing_count / row_count if row_count else 0.0,
        constancy=_constancy_evidence(observed, observed_count, maximum, missing_count) if exact_values else None,
    )
