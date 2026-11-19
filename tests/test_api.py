import math
from dataclasses import replace

import pandas as pd
import pytest

from oribi import ProfileOptions, StatisticsContext, inspect
from oribi.exceptions import InvalidInputError
from oribi.models.evidence import Warning


def test_statistics_context_preserves_positional_date_range():
    date_range = "2026-01-01 to 2026-01-05"
    statistics = StatisticsContext(
        0, 0.0, 2, 1.0, None, None, None, None, None, {}, None, None, None, (), date_range
    )
    assert statistics.date_range == date_range
    assert statistics.top_frequencies == ()


def test_inspect_returns_context_and_is_deterministic(sales_frame):
    a = inspect(sales_frame)
    b = inspect(sales_frame)
    assert a.to_dict() == b.to_dict()
    assert a.profile.row_count == 4
    assert a.profile.column_count == 5
    assert "Dataset:" in a.summary()


def test_summary_shows_a_warning_and_the_likely_grain():
    frame = pd.DataFrame({
        "order_id": [1, 2, 3, 4, 5, 6],
        "amount": [10, None, None, None, None, 20],
    })
    summary = inspect(frame).summary()
    assert "Likely grain: order_id" in summary
    assert "1 analytical warning(s) detected." in summary
    assert "[medium] amount is missing 66.7% of its values." in summary


def test_summary_without_warnings_keeps_the_profile_and_grain():
    context = inspect(pd.DataFrame({"order_id": [1, 2, 3, 4], "amount": [10, 20, 30, 40]}))
    assert not context.warnings
    summary = context.summary()
    assert "Dataset: 4 rows × 2 columns." in summary
    assert "Likely grain: order_id" in summary
    assert "Key findings" not in summary


def test_summary_prioritizes_distinct_warnings_and_limits_added_text(sales_frame):
    low = Warning("low", "low", "Low", "Low-priority detail.")
    high = Warning("high", "high", "High", "High-priority detail.")
    critical = Warning("critical", "critical", "Critical", "Critical detail.")
    wordy = Warning("medium", "medium", "Medium", "Medium detail. " + "word " * 1000)
    context = replace(inspect(sales_frame), warnings=(low, high, critical, high, wordy))

    summary = context.summary()

    assert summary.index("[critical]") < summary.index("[high]") < summary.index("[medium]")
    assert summary.count("High-priority detail") == 1
    assert "Medium detail" in summary
    assert "Low-priority detail" not in summary
    assert len(summary) < 450


def test_options_control_quantiles(sales_frame):
    context = inspect(sales_frame, options=ProfileOptions(include_quantiles=False))
    assert context.columns[2].statistics.quantiles == {}


def test_max_exact_rows_includes_quantiles_at_limit_and_omits_them_above():
    def quantiles(row_count):
        frame = pd.DataFrame({"value": [float(index) for index in range(row_count)]})
        return inspect(frame, options=ProfileOptions(max_exact_rows=3)).columns[0].statistics.quantiles

    assert quantiles(2)
    assert quantiles(3)
    assert quantiles(4) == {}


def test_rejects_invalid_inputs():
    with pytest.raises(InvalidInputError):
        inspect([1, 2, 3])
    frame = pd.DataFrame([[1, 2]], columns=["x", "x"])
    with pytest.raises(InvalidInputError):
        inspect(frame)


@pytest.mark.parametrize("name, value", [
    ("top_k", 1.5), ("top_k", True), ("top_k", "2"),
    ("max_exact_rows", 1.5), ("max_key_columns", False),
    ("max_relationships", 2.5), ("max_relationship_columns", "4"),
    ("max_gap_examples", True), ("max_gap_warnings", 1.5),
    ("top_k", 0), ("max_exact_rows", 0), ("max_key_columns", 1),
    ("max_relationships", -1), ("max_relationship_columns", -1),
    ("max_gap_examples", -1), ("max_gap_warnings", 0),
])
def test_invalid_integer_options_raise_at_configuration_time(name, value):
    with pytest.raises(ValueError, match=name):
        ProfileOptions(**{name: value})


def test_valid_option_boundaries_and_defaults():
    options = ProfileOptions(top_k=1, max_exact_rows=1, max_key_columns=2,
                             max_relationships=0, max_relationship_columns=0,
                             max_gap_examples=0, max_gap_warnings=1)
    assert options.top_k == 1
    assert ProfileOptions().max_gap_warnings == 10
    assert ProfileOptions(top_k=pd.Series([2]).iloc[0]).top_k == 2


@pytest.mark.parametrize("value", [1, "false", None])
def test_include_quantiles_requires_boolean(value):
    with pytest.raises(ValueError, match="include_quantiles"):
        ProfileOptions(include_quantiles=value)


@pytest.mark.parametrize("values", [[1 + 2j, 3 + 4j], [1 + 2j, None]])
def test_complex_column_raises_public_input_error(values):
    frame = pd.DataFrame({"signal": pd.Series(values, dtype="complex128")})
    with pytest.raises(InvalidInputError, match="signal"):
        inspect(frame)


def test_valid_numeric_column_still_profiles():
    column = inspect(pd.DataFrame({"signal": [1.0, 2.0, 3.0]})).columns[0]
    assert column.statistics.quantiles["0.5"] == 2.0
@pytest.mark.parametrize("infinite", [float("inf"), float("-inf")])
def test_numeric_quantiles_with_infinity_are_finite(infinite):
    frame = pd.DataFrame({"value": [1.0, 2.0, infinite, 4.0]})
    stats = inspect(frame).columns[0].statistics
    assert all(q is None or (pd.notna(q) and math.isfinite(q)) for q in stats.quantiles.values())


@pytest.mark.parametrize("invalid_options", [False, 0, {}])
def test_inspect_rejects_falsy_invalid_options(invalid_options):
    frame = pd.DataFrame({"value": [1, 2, 3]})
    with pytest.raises(InvalidInputError, match="ProfileOptions"):
        inspect(frame, options=invalid_options)


def test_inspect_none_and_omitted_options_match_defaults():
    frame = pd.DataFrame({"value": [1, 2, 3]})
    assert inspect(frame).to_dict() == inspect(frame, options=None).to_dict()
