"""Create dependency-free SVG charts for Day 10 forecasts and attention weights."""

from __future__ import annotations

from pathlib import Path


def _polyline(values, x0, y0, width, height, low, high):
    span = high - low or 1.0
    denominator = max(len(values) - 1, 1)
    points = []
    for index, value in enumerate(values):
        x = x0 + index * width / denominator
        y = y0 + height - (float(value) - low) * height / span
        points.append(f"{x:.2f},{y:.2f}")
    return " ".join(points)


def write_attention_heatmap(weights, output_path: str | Path) -> None:
    """Write a 14-by-30 SVG heatmap, with rows representing forecast days."""
    rows = len(weights)
    columns = len(weights[0]) if rows else 0
    if rows != 14 or columns != 30:
        raise ValueError(f"Expected attention weights shaped (14, 30), got ({rows}, {columns}).")
    cell = 18
    left, top = 72, 42
    width, height = left + columns * cell + 24, top + rows * cell + 56
    rectangles = []
    labels = []
    for row_index, row in enumerate(weights):
        labels.append(
            f'<text x="8" y="{top + row_index * cell + 13}" font-size="11">Day {row_index + 1}</text>'
        )
        for column_index, weight in enumerate(row):
            intensity = max(0, min(255, round(255 * (1.0 - float(weight) * 30))))
            color = f"#{intensity:02x}{intensity:02x}ff"
            rectangles.append(
                f'<rect x="{left + column_index * cell}" y="{top + row_index * cell}" '
                f'width="{cell}" height="{cell}" fill="{color}" stroke="white" stroke-width="1"/>'
            )
    for column in range(0, columns, 5):
        labels.append(
            f'<text x="{left + column * cell}" y="{top + rows * cell + 18}" font-size="10">Day {column + 1}</text>'
        )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        '<rect width="100%" height="100%" fill="white"/>'
        '<text x="250" y="22" font-size="15">Historical Attention by Forecast Day</text>'
        + "".join(rectangles)
        + "".join(labels)
        + f'<text x="{left + 145}" y="{height - 8}" font-size="12">Historical day (30-day input)</text>'
        + "</svg>\n"
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")


def write_forecast_chart(
    actual,
    predicted,
    output_path: str | Path,
    title: str,
    actual_label: str = "Actual",
    predicted_label: str = "Predicted",
) -> None:
    """Write an actual-versus-predicted or forecast SVG chart for a 14-day horizon."""
    if len(actual) != 14 or len(predicted) != 14:
        raise ValueError("Forecast charts require exactly 14 actual and predicted values.")
    low = min(min(actual), min(predicted))
    high = max(max(actual), max(predicted))
    left, top, plot_width, plot_height = 66, 42, 640, 300
    actual_line = _polyline(actual, left, top, plot_width, plot_height, low, high)
    predicted_line = _polyline(predicted, left, top, plot_width, plot_height, low, high)
    ticks = "".join(
        f'<text x="{left + index * plot_width / 13:.1f}" y="{top + plot_height + 20}" '
        f'font-size="10">Day {index + 1}</text>'
        for index in range(0, 14, 2)
    )
    legend = (
        '<line x1="520" y1="25" x2="545" y2="25" stroke="#2563eb" stroke-width="3"/>'
        f'<text x="550" y="29" font-size="11">{actual_label}</text>'
        '<line x1="610" y1="25" x2="635" y2="25" stroke="#dc2626" stroke-width="3"/>'
        f'<text x="640" y="29" font-size="11">{predicted_label}</text>'
    )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="740" height="400">'
        '<rect width="100%" height="100%" fill="white"/>'
        f'<text x="66" y="22" font-size="15">{title}</text>{legend}'
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#111"/>'
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" '
        f'y2="{top + plot_height}" stroke="#111"/>'
        f'<polyline points="{actual_line}" fill="none" stroke="#2563eb" stroke-width="2.5"/>'
        f'<polyline points="{predicted_line}" fill="none" stroke="#dc2626" stroke-width="2.5"/>'
        f'{ticks}<text x="8" y="190" transform="rotate(-90 8 190)" font-size="12">Sales</text>'
        '</svg>\n'
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")


def write_training_loss(history: dict, output_path: str | Path) -> None:
    """Write an SVG line chart for training and validation loss."""
    training = history.get("loss", [])
    validation = history.get("val_loss", [])
    if not training or not validation:
        raise ValueError("Training history must include loss and val_loss.")
    low = min(min(training), min(validation))
    high = max(max(training), max(validation))
    width, height = 720, 390
    chart = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        '<rect width="100%" height="100%" fill="white"/>'
        '<text x="60" y="24" font-size="16">Training and Validation Loss</text>'
        f'<polyline points="{_polyline(training, 60, 45, 600, 280, low, high)}" '
        'fill="none" stroke="#2563eb" stroke-width="2.5"/>'
        f'<polyline points="{_polyline(validation, 60, 45, 600, 280, low, high)}" '
        'fill="none" stroke="#dc2626" stroke-width="2.5"/>'
        '<text x="530" y="360" font-size="11" fill="#2563eb">Training</text>'
        '<text x="600" y="360" font-size="11" fill="#dc2626">Validation</text>'
        '<text x="250" y="380" font-size="12">Epoch</text>'
        '</svg>\n'
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(chart, encoding="utf-8")
