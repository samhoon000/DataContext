import json

import pytest

from oribi import inspect
from oribi.serialization.context import _format_aggregations, _format_range_value


def test_serialization_outputs(sales_frame):
    context = inspect(sales_frame)
    assert json.loads(context.to_json()) == context.to_dict()
    assert "## Columns" in context.to_markdown()
    compact = context.to_context(max_chars=500)
    assert len(compact) <= 500
    assert "DATASET" in compact


def test_context_renderers_use_scannable_hierarchy_and_compact_columns(sales_frame):
    context = inspect(sales_frame)
    compact = context.to_context()
    assert compact.index("DATASET\n") < compact.index("GRAIN") < compact.index("COLUMNS")
    assert "Missing " in compact
    assert "Unique " in compact
    assert "Aggregations:" in compact
    assert "→" in context.to_context(include_values=True)
    assert "Evidence:" not in compact
    assert "confidence" in compact
    assert "DATASET SUMMARY" not in compact
    assert "- order_date: temporal dimension" in compact
    assert " · " in compact
    assert "Missing 25.0%" not in compact

    markdown = context.to_markdown()
    assert markdown.index("## Dataset summary") < markdown.index("## Grain") < markdown.index("## Columns")
    assert "Recommended aggregations:" in markdown
    assert "<details>" in markdown
    assert "<summary>Inference evidence</summary>" in markdown


def test_compact_context_formats_column_warnings_concisely():
    import pandas as pd

    context = inspect(pd.DataFrame({"region": ["East", "West", "East", None]}))
    rendered = context.to_context()
    assert "⚠ region · 25% missing" in rendered
    assert "[medium]" not in rendered


def test_aggregation_formatter_preserves_names_as_complete_values():
    assert _format_aggregations(["COUNT DISTINCT"]) == "COUNT DISTINCT"
    assert _format_aggregations(["SUM", "MEAN"]) == "SUM · MEAN"
    assert _format_aggregations([]) == "—"
    assert _format_aggregations(["SUM, MEAN", "COUNT DISTINCT"]) == "SUM, MEAN · COUNT DISTINCT"
    assert _format_aggregations("COUNT DISTINCT") == "COUNT DISTINCT"


def test_range_formatter_removes_unnecessary_time_and_decimal_parts():
    import pandas as pd

    assert _format_range_value(pd.Timestamp("2026-01-01 00:00:00")) == "2026-01-01"
    assert _format_range_value(pd.Timestamp("2026-01-01 12:30:00")) == "2026-01-01 12:30:00"
    assert _format_range_value(85.0) == "85"
    assert _format_range_value(85.5) == "85.5"


def test_compact_context_formats_date_and_whole_number_ranges():
    import pandas as pd

    context = inspect(pd.DataFrame({
        "event_date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
        "amount": [85.0, 210.0],
    }))
    rendered = context.to_context(include_values=True)
    assert "Range 2026-01-01 → 2026-01-02" in rendered
    assert "Range 85 → 210" in rendered
    safe_rendered = context.to_context()
    assert "Range 2026-01-01" not in safe_rendered
    assert "Range 85" not in safe_rendered


def test_result_mappings_cannot_be_mutated_through_the_snapshot(sales_frame):
    context = inspect(sales_frame)
    revenue = next(c for c in context.columns if c.name == "revenue")
    try:
        revenue.statistics.quantiles["0.5"] = -100
    except TypeError:
        pass
    else:
        raise AssertionError("quantiles should be read-only")
    assert context.to_dict()["columns"][2]["statistics"]["quantiles"]["0.5"] == 6.25


@pytest.mark.parametrize(
    "new_value,mutate",
    [
        pytest.param(lambda: ["a"], lambda value: value.append("later"), id="list"),
        pytest.param(lambda: {"label": "a"}, lambda value: value.update(label="later"), id="dictionary"),
        pytest.param(
            lambda: {"labels": ["a"]},
            lambda value: value["labels"].append("later"),
            id="nested-list",
        ),
    ],
)
def test_inspect_snapshots_mutable_top_values(new_value, mutate):
    import pandas as pd

    frame = pd.DataFrame({"value": [new_value(), new_value()], "region": ["East", "East"]})
    context = inspect(frame)
    expected = new_value()
    statistics = context.columns[0].statistics
    assert statistics.top_values == ((expected, 2),)
    before = context.to_dict(include_values=True)
    assert before["columns"][0]["statistics"]["top_values"] == [[expected, 2]]
    json_before = context.to_json(include_values=True)
    markdown_before = context.to_markdown(include_values=True)
    compact_before = context.to_context(include_values=True)

    mutate(frame.at[0, "value"])

    assert statistics.top_values == ((expected, 2),)
    assert context.columns[1].statistics.top_values == (("East", 2),)
    assert context.to_dict(include_values=True) == before
    assert context.to_json(include_values=True) == json_before
    assert context.to_markdown(include_values=True) == markdown_before
    assert context.to_context(include_values=True) == compact_before


def test_inspect_preserves_cycles_and_shared_items_in_a_snapshot():
    import pandas as pd

    shared = ["a"]
    value = {"first": shared, "second": shared}
    value["cycle"] = value
    context = inspect(pd.DataFrame({"value": [value]}))
    snapshot, count = context.columns[0].statistics.top_values[0]

    assert count == 1
    assert snapshot is not value
    assert snapshot["cycle"] is snapshot
    assert snapshot["first"] is snapshot["second"]
    assert snapshot["first"] is not shared


def test_context_limit_keeps_an_explicit_marker_for_long_labels():
    import pandas as pd

    context = inspect(pd.DataFrame({"x" * 1_000: [1, 2, 3]}))
    rendered = context.to_context(max_chars=100)
    assert len(rendered) <= 100
    assert "[Context truncated.]" in rendered


def test_markdown_prints_each_warning_message_once():
    import pandas as pd

    frame = pd.DataFrame({"amount": [1.0, -2.0, 3.0, 3.0]}).iloc[[0, 1, 2, 3, 3]]
    context = inspect(frame)
    markdown = context.to_markdown()
    column_warning = next(w for w in context.warnings if w.code == "negative_measure_values")
    dataset_warning = next(w for w in context.warnings if w.code == "duplicate_rows")
    assert markdown.count(column_warning.message) == 1
    assert markdown.count("negative values were observed") == 1
    assert markdown.count(dataset_warning.message) == 1
    column_section = markdown[markdown.index("### amount"):markdown.index("## Analytical warnings")]
    assert f"Warning [{column_warning.severity}]: {column_warning.title}" in column_section
    assert column_warning.message not in column_section
    warnings_section = markdown[markdown.index("## Analytical warnings"):]
    assert column_warning.message in warnings_section
    assert column_warning.implication in warnings_section
    assert dataset_warning.title in warnings_section


def test_markdown_warning_change_leaves_structured_output_alone():
    import pandas as pd

    context = inspect(pd.DataFrame({"amount": [1.0, -2.0, 3.0]}))
    column = next(c for c in context.columns if c.name == "amount")
    assert [w.code for w in column.warnings] == ["negative_measure_values"]
    assert "negative_measure_values" in [w.code for w in context.warnings]
    exported = context.to_dict()
    assert any(w["code"] == "negative_measure_values" for w in exported["warnings"])
    assert json.loads(context.to_json()) == exported
