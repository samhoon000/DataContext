from datetime import date, datetime, timezone

import pandas as pd
import pytest

from oribi import inspect


def test_roles_have_evidence(sales_frame):
    context = inspect(sales_frame)
    roles = {c.name: c.semantic for c in context.columns}
    assert roles["order_id"].role == "identifier"
    assert roles["revenue"].role == "measure"
    assert roles["order_date"].role == "temporal"
    assert roles["order_id"].evidence


def test_numeric_low_cardinality_can_be_dimension():
    context = inspect(__import__("pandas").DataFrame({"rating": [1, 2, 1, 3, 2]}))
    assert context.columns[0].semantic.role == "dimension"


def test_unique_numeric_business_measure_stays_a_measure():
    import pandas as pd

    context = inspect(pd.DataFrame({"revenue": [10.0, 20.0, 35.0, 80.0]}))
    column = context.columns[0]
    assert column.semantic.role == "measure"
    assert {r.operation for r in column.recommendations} >= {"SUM", "MEAN", "MEDIAN"}


def test_numeric_identifier_name_outweighs_numeric_measure_type():
    import pandas as pd

    context = inspect(pd.DataFrame({"customer_id": [101, 102, 103, 104]}))
    assert context.columns[0].semantic.role == "identifier"


def test_reusable_code_is_eligible_for_relationship_detection():
    import pandas as pd

    frame = pd.DataFrame({"store_code": ["a", "a", "b", "b"], "store_name": ["East", "East", "West", "West"]})
    context = inspect(frame)
    assert context.columns[0].semantic.role == "identifier"
    assert any(r.source_column == "store_code" and r.target_column == "store_name" for r in context.relationships)


def test_ambiguous_unique_numeric_column_remains_unknown():
    import pandas as pd

    context = inspect(pd.DataFrame({"score": [11, 17, 23, 41]}))
    assert context.columns[0].semantic.role == "unknown"


def test_short_numeric_measure_name_is_enough_for_measure_inference():
    import pandas as pd

    context = inspect(pd.DataFrame({"amount": [2.0, 4.5, 8.0]}))
    assert context.columns[0].semantic.role == "measure"


@pytest.mark.parametrize("dtype", ["object", "string"])
def test_iso_dates_and_timestamps_infer_a_temporal_role(dtype):
    frame = pd.DataFrame({
        "day": pd.Series(["2026-01-01", "2026-01-02", None, "2026-01-02"], dtype=dtype),
        "logged": pd.Series(["2026-11-01T01:30:00-04:00", "2026-11-01T06:30:00Z", None], dtype=dtype),
    })
    original = frame.copy(deep=True)
    columns = inspect(frame).columns
    assert [column.semantic.role for column in columns] == ["temporal", "temporal"]
    assert all(column.logical_type == "text" for column in columns)
    day_shape = next(e for e in columns[0].semantic.evidence if e.code == "iso_temporal_values")
    assert day_shape.details == {"observed_count": 3, "distinct_count": 2}
    assert not any(w.code == "potential_temporal_gap" for column in columns for w in column.warnings)
    pd.testing.assert_frame_equal(frame, original)


def test_python_dates_in_an_object_column_infer_a_temporal_role():
    values = pd.Series([date(2026, 1, 1), datetime(2026, 1, 2, tzinfo=timezone.utc)], dtype=object)
    column = inspect(pd.DataFrame({"observed": values})).columns[0]
    assert column.semantic.role == "temporal"
    assert any(e.code == "temporal_object_values" for e in column.semantic.evidence)


@pytest.mark.parametrize("values", [
    ["20260101", "20260102"],
    ["01/02/2026", "01/03/2026"],
    ["2026-01-01", "date pending"],
    ["2026-01-01", "2026-02-30"],
    ["2026-01-01", "2026-02-01T12:00:00+00:75"],
    ["2026-01-01", "2026-01-01"],
    [date(2026, 1, 1), "date pending"],
    ["2026-01-01"] * 101 + ["not a date"],
])
def test_ambiguous_or_mixed_values_are_not_inferred_as_temporal(values):
    column = inspect(pd.DataFrame({"value": pd.Series(values, dtype=object)})).columns[0]
    assert column.semantic.role != "temporal"
    assert not any(e.code in {"iso_temporal_values", "temporal_object_values"} for e in column.semantic.evidence)


def test_identifier_name_can_leave_date_like_strings_ambiguous():
    column = inspect(pd.DataFrame({"batch_id": ["2026-01-01", "2026-01-02"]})).columns[0]
    assert column.semantic.role == "unknown"
    assert {e.code for e in column.semantic.evidence} >= {"name_identifier_hint", "iso_temporal_values"}


def test_distinct_iso_dates_are_not_reported_as_high_cardinality_text():
    values = pd.date_range("2026-01-01", periods=120, freq="D").strftime("%Y-%m-%d")
    column = inspect(pd.DataFrame({"day": values})).columns[0]
    assert column.semantic.role == "temporal"
    assert not any(w.code == "high_cardinality_text" for w in column.warnings)
