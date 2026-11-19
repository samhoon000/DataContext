"""Composable rules for likely column roles."""

import re

from ..models.evidence import Evidence, SemanticContext


_MIN_ROLE_SCORE = 0.55
_AMBIGUITY_MARGIN = 0.12

def infer_semantic(
    name: str,
    logical_type: str,
    stats: object,
    row_count: int,
    *,
    is_float: bool = False,
    date_evidence: Evidence | None = None,
) -> SemanticContext:
    votes: dict[str, float] = {}
    evidence: list[Evidence] = []
    lname = name.lower()
    distinct = stats.distinct_count
    rate = stats.distinct_rate
    tokens = set(re.split(r"[^a-z0-9]+", lname))
    explicit_identifier_name = "id" in tokens or lname.endswith("_key")
    code_name = lname.endswith("_code")
    # A suffix alone cannot establish identity when only a few codes recur.
    categorical_code_shape = row_count >= 20 and distinct <= 3 and rate <= 0.15
    if explicit_identifier_name and distinct > 1:
        votes["identifier"] = .48 if categorical_code_shape else .78
        evidence.append(Evidence("name_identifier_hint", "Column name strongly suggests an identifier"))
        if categorical_code_shape:
            evidence.append(Evidence("few_repeated_codes", "A few values repeat across many rows; entity identity is unestablished", {"distinct_count": distinct, "row_count": row_count}))
        evidence.append(Evidence(
            "identifier_value_variation",
            f"Observed {distinct} distinct non-missing values",
            {"distinct_count": distinct, "row_count": row_count},
        ))
    elif code_name and distinct > 1:
        votes["identifier"] = .48 if categorical_code_shape else .75
        evidence.append(Evidence("name_identifier_hint", "Column name suggests an identifier"))
        evidence.append(Evidence(
            "identifier_value_variation",
            f"Observed {distinct} distinct non-missing values",
            {"distinct_count": distinct, "row_count": row_count},
        ))
    if row_count and rate >= .98 and distinct > 1 and logical_type != "duration":
        # Uniqueness is supporting evidence, not proof of business identity.
        # Elapsed durations are skipped: unique durations do not identify entities.
        # It is deliberately weaker for numeric columns, which commonly hold
        # continuous measures that happen to be unique in small samples.
        unique_vote = .28 if logical_type == "numeric" else .62
        votes["identifier"] = max(votes.get("identifier", 0), unique_vote)
        evidence.append(Evidence("mostly_unique", f"{distinct} distinct non-null values across {row_count} rows", {"distinct_count": distinct, "row_count": row_count, "distinct_rate": rate}))
    if logical_type == "numeric":
        votes["measure"] = .48
        evidence.append(Evidence("numeric_type", "Column has a numeric dtype"))
        if is_float:
            votes["measure"] = max(votes["measure"], .56)
            evidence.append(Evidence("floating_point_type", "Floating-point values are consistent with a quantitative measure"))
        elif row_count and rate < .98 and distinct > min(20, max(3, row_count // 10)):
            votes["measure"] = max(votes["measure"], .56)
            evidence.append(Evidence("repeated_numeric_measure_values", "Numeric values repeat across rows without having low categorical cardinality"))
    elif logical_type == "temporal":
        votes["temporal"] = .86
        evidence.append(Evidence("datetime_dtype", "Column has a datetime dtype"))
    elif date_evidence is not None:
        votes["temporal"] = .84
        evidence.append(date_evidence)
    elif logical_type == "boolean":
        votes["boolean"] = .9
        evidence.append(Evidence("boolean_dtype", "Column has a boolean dtype"))
    elif logical_type == "categorical":
        votes["categorical"] = .78
        evidence.append(Evidence("categorical_dtype", "Column has a categorical dtype"))
    elif logical_type == "text":
        if rate > .5 and distinct > 20:
            votes["text"] = .62
            evidence.append(Evidence("high_cardinality_text", "Text values have relatively high cardinality"))
        else:
            votes["dimension"] = .62
            evidence.append(Evidence("text_dimension", "Text values are reusable across rows"))
    if logical_type == "numeric" and not is_float and (
        distinct <= min(20, max(3, row_count // 10))
        or (row_count >= 20 and rate <= .1)
    ):
        votes["dimension"] = max(votes.get("dimension", 0), .72 if row_count >= 20 and rate <= .1 else .64)
        evidence.append(Evidence("low_cardinality_numeric", "Numeric values have low cardinality"))
    if not votes:
        return SemanticContext("unknown", 0.0, tuple(evidence))
    role, score = max(votes.items(), key=lambda item: (item[1], item[0]))
    ranked_votes = sorted(votes.items(), key=lambda item: (-item[1], item[0]))
    if score < _MIN_ROLE_SCORE:
        return SemanticContext("unknown", round(score, 2), tuple(evidence))
    if len(ranked_votes) > 1 and score - ranked_votes[1][1] < _AMBIGUITY_MARGIN:
        return SemanticContext("unknown", round(min(score, .5), 2), tuple(evidence))
    return SemanticContext(role, round(min(score, .98), 2), tuple(evidence))
