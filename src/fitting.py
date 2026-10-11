"""Оценка физических параметров при известных начальных условиях."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from models import (
    solve_membrane,
    solve_heat,
    reconstruct,
)


@dataclass
class FitResult:
    parameters: dict
    prediction: np.ndarray
    rmse: float
    optimization: object


def metrics(observed, predicted):
    """Вычислить RMSE и MAE без неявного изменения формы массивов."""
    observed = np.asarray(observed)
    predicted = np.asarray(predicted)

    if (
        observed.shape != predicted.shape
        or observed.size == 0
        or not np.isfinite(observed).all()
        or not np.isfinite(predicted).all()
    ):
        raise ValueError(
            "Нужны конечные непустые массивы одинаковой формы"
        )

    error = predicted - observed

    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
    }


def _fit(predict, names, observed, x0, bounds):
    observed = np.asarray(observed, dtype=float)

    if observed.ndim == 1:
        observed = observed[:, None]

    # Проверка формы до оптимизации исключает broadcasting,
    # который мог бы незаметно исказить целевую функцию.
    metrics(observed, predict(x0))

    result = least_squares(
        lambda p: (predict(p) - observed).ravel(),
        x0,
        bounds=bounds,
        method="trf",
        x_scale="jac",
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
        max_nfev=300,
    )

    if not result.success:
        raise RuntimeError(result.message)

    prediction = predict(result.x)

    return FitResult(
        parameters=dict(
            zip(names, map(float, result.x))
        ),
        prediction=prediction,
        rmse=metrics(observed, prediction)["rmse"],
        optimization=result,
    )


def fit_membrane(
    times,
    observed,
    points,
    modes,
    initial,
    velocity=None,
    *,
    x0=(0.8, 0.05),
    bounds=((0.2, 0.0), (2.0, 1.0)),
):
    """
    Подобрать вектор (c, gamma).

    observed: прогиб в точках points, форма (N, P).
    Это должны быть измерения прогиба, не звукового давления.

    Начальные модальные амплитуды и скорости считаются известными.
    Границы по умолчанию относятся к безразмерному эксперименту.
    """
    def predict(p):
        a, _ = solve_membrane(
            times,
            modes,
            initial,
            velocity,
            c=p[0],
            gamma=p[1],
        )

        return reconstruct(a, modes, points)

    return _fit(
        predict,
        ("c", "gamma"),
        observed,
        x0,
        bounds,
    )


def fit_heat(
    times,
    observed,
    points,
    modes,
    initial,
    *,
    x0=(0.03, 0.02),
    bounds=((0.005, 0.0), (0.2, 0.5)),
):
    """
    Подобрать вектор (alpha, beta).

    observed: превышение температуры над фоном, форма (N, P).
    Начальные модальные амплитуды считаются известными.

    Для разделения alpha и beta нужны наблюдаемые моды
    с различными собственными значениями.
    """
    def predict(p):
        b = solve_heat(
            times,
            modes,
            initial,
            alpha=p[0],
            beta=p[1],
        )

        return reconstruct(b, modes, points)

    return _fit(
        predict,
        ("alpha", "beta"),
        observed,
        x0,
        bounds,
    )