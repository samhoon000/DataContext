"""Structured warnings and aggregation recommendations."""

from ..models.evidence import Evidence, Recommendation, Warning


def reason_column(name: str, logical_type: str, semantic: object, stats: object, row_count: int) -> tuple[tuple[Recommendation, ...], tuple[Warning, ...]]:
    recommendations: list[Recommendation] = []
    warnings: list[Warning] = []
    if stats.missing_rate >= .2:
        warnings.append(Warning("high_missingness", "medium", "High missingness", f"{name} is missing {stats.missing_rate:.1%} of its values.", (Evidence("missing_rate", f"{stats.missing_count} of values are missing", {"missing_count": stats.missing_count, "missing_rate": stats.missing_rate}),), "Analyses using this column may exclude many rows.", "Check whether missing values have a meaningful interpretation."))
    if semantic.role == "identifier":
        recommendations.append(Recommendation("COUNT DISTINCT", "recommended", "Distinct counting is more appropriate for counting entities than summing identifier values.", semantic.evidence))
        if stats.distinct_count < row_count - stats.missing_count:
            warnings.append(Warning("non_unique_identifier", "medium", "Identifier-like column is not unique", f"{name} has {stats.distinct_count} distinct values across rows and may identify repeated entities.", semantic.evidence, "Row counts may exceed entity counts.", "Use a distinct count when counting these entities."))
    if semantic.role == "measure":
        for op, disposition in (("SUM", "recommended"), ("MEAN", "recommended"), ("MEDIAN", "useful"), ("MIN", "valid"), ("MAX", "valid")):
            recommendations.append(Recommendation(op, disposition, "The column is numeric and inferred as a likely measure.", semantic.evidence))
        if stats.skewness is not None and abs(stats.skewness) >= 2:
            warnings.append(Warning("high_skew", "low", "Highly skewed measure", f"{name} has skewness {stats.skewness:.2f}.", (Evidence("skewness", "Observed distribution is strongly skewed", {"skewness": stats.skewness}),), "The mean may be sensitive to a small number of large values.", "Consider reporting the median alongside the mean."))
        if stats.negative_count:
            warnings.append(Warning("negative_measure_values", "low", f"Negative values in {name}", f"{stats.negative_count} negative values were observed in likely measure {name}.", (Evidence("negative_count", f"{stats.negative_count} values are below zero", {"negative_count": stats.negative_count}),), "Negative values may represent refunds, reversals, or data issues.", "Check domain meaning before filtering negative values."))
    if logical_type == "categorical" and stats.distinct_count > 100:
        warnings.append(Warning("high_cardinality_category", "low", "High-cardinality category", f"{name} has {stats.distinct_count} distinct categories.", (), "Grouping by this column may produce many small groups."))
    if logical_type == "text" and semantic.role not in {"identifier", "temporal"} and stats.distinct_count > 100:
        # Identifiers and dates are often expected to have many distinct values.
        observed = row_count - stats.missing_count
        ratio = stats.distinct_count / observed if observed else 0.0
        warnings.append(Warning("high_cardinality_text", "low", "High-cardinality text", f"{name} has {stats.distinct_count} distinct text values across {observed} non-missing rows.", (Evidence("distinct_count", f"{stats.distinct_count} distinct values ({ratio:.1%} of non-missing rows)", {"distinct_count": stats.distinct_count, "non_missing_count": observed, "uniqueness_ratio": ratio}),), "Grouping or aggregating by this column may produce many small groups.", "Use it as a label or free text rather than as a grouping dimension, or bucket it before grouping."))
    return tuple(recommendations), tuple(warnings)


def dataset_warnings(duplicate_count: int | None, row_count: int) -> tuple[Warning, ...]:
    if duplicate_count:
        return (Warning("duplicate_rows", "medium", "Duplicate rows detected", f"{duplicate_count} fully duplicated rows were observed.", (Evidence("duplicate_count", f"{duplicate_count} duplicate rows across {row_count} total rows", {"duplicate_row_count": duplicate_count, "row_count": row_count}),), "Duplicates can inflate aggregates.", "Confirm whether repeated rows are expected before aggregation."),)
    return ()
