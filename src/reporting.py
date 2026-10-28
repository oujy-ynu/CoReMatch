"""Formatting for training and evaluation metrics."""

from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP


def truncate_mse(value):
    """Scale raw MSE by 1000 and truncate to three decimal places."""
    scaled = Decimal(str(value)) * Decimal(1000)
    return format(scaled.quantize(Decimal("0.001"), rounding=ROUND_DOWN), ".3f")


def round_metric(value):
    """Round half up; normalize insignificant binary error at exact half points."""
    number = Decimal(str(value)).quantize(Decimal("0.000000000001"), rounding=ROUND_HALF_UP)
    return format(number.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP), ".3f")


def display_metrics(metrics):
    return {"mse_x1000": truncate_mse(metrics["mse"]),
            **{key: round_metric(metrics[key]) for key in ("rho", "tau", "p10", "p20")}}


def format_metrics(metrics):
    values = display_metrics(metrics)
    return " ".join(f"{key}={value}" for key, value in values.items())
