# Build figures from saved descriptive rows; never execute experiments in a plot function.
"""Small optional Plotly views that remain independent of notebook execution state."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def paired_effect_figure(rows: Sequence[dict[str, Any]]) -> Any:
    """Plot complete paired differences; incomplete comparisons remain in the table.

    No error bars are invented from correlated repetitions. Importing this module
    does not require Plotly; only calling this optional figure function does.
    """
    # Import the optional display library only when a saved-result figure is requested.
    import plotly.graph_objects as go

    # Keep incomplete evidence out of numeric coordinates without deleting its table rows.
    complete = [row for row in rows if row["status"] == "complete"]
    # Label individual paired repetitions rather than implying independence across tokens.
    labels = [f"{row['pair_id']} / {row['replicate']}" for row in complete]
    # Plot only the predeclared changed-minus-baseline likelihood margin.
    figure = go.Figure(go.Bar(x=labels, y=[row["delta"] for row in complete], marker_color="#0f766e"))
    # Use a neutral, readable analysis style without a custom frontend framework.
    figure.update_layout(template="plotly_white", title="Paired score changes — descriptive, not a causal claim", xaxis_title="Pair / replicate", yaxis_title="Second − first target log-probability margin (nats)")
    # Mark the no-change reference point explicitly.
    figure.add_hline(y=0, line_color="#64748b", line_dash="dash")
    # Return a display object without showing it, saving files, or mutating notebook state.
    return figure
