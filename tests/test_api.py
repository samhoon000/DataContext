import math

import pandas as pd
import pytest

from oribi import ProfileOptions, inspect
from oribi.exceptions import InvalidInputError


def test_inspect_returns_context_and_is_deterministic(sales_frame):
    a = inspect(sales_frame)
    b = inspect(sales_frame)
    assert a.to_dict() == b.to_dict()
    assert a.profile.row_count == 4
    assert a.profile.column_count == 5
    assert "Dataset:" in a.summary()


def test_options_control_quantiles(sales_frame):
    context = inspect(sales_frame, options=ProfileOptions(include_quantiles=False))
    assert context.columns[2].statistics.quantiles == {}


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
