# physics.py
"""Пространственный оператор, численная эволюция и идентификация."""

import numpy as np
from scipy import sparse
from scipy.optimize import lsq_linear
from scipy.sparse.linalg import eigsh, factorized


def time_step(t):
    differences = np.diff(np.asarray(t, dtype=float))
    if (
        len(differences) == 0
        or np.any(differences <= 0)
        or not np.allclose(differences, differences[0], rtol=1e-4)
    ):
        raise ValueError("Нужна равномерная возрастающая временная сетка.")
    return float(differences[0])


def laplacian(grid):
    """Пятиточечный лапласиан.

    Для круга: нулевые значения вне mask, ступенчатая граница.
    Для прямоугольника: периодические границы.
    """
    ny, nx = grid.mask.shape
    numbering = np.full((ny, nx), -1, dtype=int)
    numbering[grid.mask] = np.arange(grid.mask.sum())
    rows, cols, values = [], [], []

    for j, i in np.argwhere(grid.mask):
        row = numbering[j, i]
        rows.append(row)
        cols.append(row)
        values.append(-2 / grid.dx**2 - 2 / grid.dy**2)

        for dj, di, weight in (
            (0, -1, 1 / grid.dx**2),
            (0, 1, 1 / grid.dx**2),
            (-1, 0, 1 / grid.dy**2),
            (1, 0, 1 / grid.dy**2),
        ):
            jj, ii = j + dj, i + di
            if grid.periodic:
                jj, ii = jj % ny, ii % nx
            if 0 <= jj < ny and 0 <= ii < nx and grid.mask[jj, ii]:
                rows.append(row)
                cols.append(numbering[jj, ii])
                values.append(weight)

    n = grid.mask.sum()
    return sparse.csr_matrix((values, (rows, cols)), shape=(n, n))


def evolve(operator, initial, t, d, kind, loss=0.0):
    """Эволюция поля по известному или обученному оператору.

    heat: initial[P], схема Кранка–Николсон.
    wave: initial[2,P], центральная схема с затуханием.
    """
    if kind not in {"heat", "wave"} or d <= 0 or loss < 0:
        raise ValueError("Некорректные физические параметры.")

    dt = time_step(t)
    operator = sparse.csc_matrix(operator)
    n = operator.shape[0]
    eye = sparse.eye(n, format="csc")
    out = np.empty((len(t), n), dtype=float)

    if kind == "heat":
        if np.asarray(initial).shape != (n,):
            raise ValueError("Для heat нужен один начальный кадр.")
        generator = d * operator - loss * eye
        solve = factorized(eye - 0.5 * dt * generator)
        right = eye + 0.5 * dt * generator
        out[0] = initial
        for k in range(len(t) - 1):
            out[k + 1] = solve(right @ out[k])
    else:
        if len(t) < 3 or np.asarray(initial).shape != (2, n):
            raise ValueError("Для wave нужны два начальных кадра.")

        # Условие устойчивости для отрицательно полуопределённого L.
        if n <= 3:
            smallest = np.linalg.eigvalsh(operator.toarray())[0]
        else:
            smallest = eigsh(
                operator, k=1, which="SA", return_eigenvectors=False
            )[0]
        if dt**2 * d * max(0.0, -smallest) >= 4:
            raise ValueError("Нарушено условие устойчивости волновой схемы.")

        out[:2] = initial
        damping = loss * dt / 2
        for k in range(1, len(t) - 1):
            out[k + 1] = (
                2 * out[k]
                - (1 - damping) * out[k - 1]
                + dt**2 * d * (operator @ out[k])
            ) / (1 + damping)

    if not np.isfinite(out).all():
        raise FloatingPointError("Численное решение разошлось.")
    return out


def regression_terms(u, t, kind):
    """Дискретные производные и состояния для регрессии PDE."""
    dt = time_step(t)
    if kind == "heat":
        state = 0.5 * (u[:, 1:] + u[:, :-1])
        derivative = (u[:, 1:] - u[:, :-1]) / dt
        damping_state = state
    elif kind == "wave":
        state = u[:, 1:-1]
        derivative = (u[:, 2:] - 2*u[:, 1:-1] + u[:, :-2]) / dt**2
        damping_state = (u[:, 2:] - u[:, :-2]) / (2*dt)
    else:
        raise ValueError("Неизвестная PDE.")
    return state, derivative, damping_state


def identify(u, t, operator, kind, fit_loss=False):
    """Подобрать d=alpha/c**2 и, при необходимости, beta/gamma.

    Передавать одну траекторию или несколько с общими параметрами.
    Для оценки качества использовать другие, поздние кадры.
    """
    u = np.asarray(u)
    if u.ndim == 2:
        u = u[None]

    state, target, damping = regression_terms(u, t, kind)
    spatial = (operator @ state.reshape(-1, state.shape[-1]).T).T
    columns = [spatial.ravel()]
    if fit_loss:
        columns.append(-damping.ravel())

    design = np.column_stack(columns)
    result = lsq_linear(design, target.ravel(), bounds=(0, np.inf))
    if not result.success:
        raise RuntimeError(result.message)

    return {
        "d": float(result.x[0]),
        "loss": float(result.x[1]) if fit_loss else 0.0,
        "relative_residual": float(
            np.linalg.norm(design @ result.x - target.ravel())
            / max(np.linalg.norm(target), 1e-12)
        ),
    }


def audit(fields, operator, tolerance=None):
    """Проверка PDE по заявленным параметрам, без их подгонки.

    Это диагностика: на грубой сетке и у границ есть ошибка
    дискретизации. Порог нужно обосновать сеточным исследованием.
    """
    state, derivative, _ = regression_terms(
        fields.u, fields.t, fields.kind
    )
    spatial = (operator @ state.reshape(-1, state.shape[-1]).T).T
    spatial = spatial.reshape(state.shape)
    expected = fields.d[:, None, None] * spatial

    numerator = np.linalg.norm(
        (derivative - expected).reshape(len(state), -1), axis=1
    )
    denominator = np.linalg.norm(
        derivative.reshape(len(state), -1), axis=1
    )
    errors = numerator / np.maximum(denominator, 1e-12)

    if tolerance is not None and np.any(errors > tolerance):
        raise ValueError(
            "Высокая невязка PDE: проверьте метки, время, координаты "
            "и погрешность пространственной дискретизации."
        )
    return errors


def metrics(reference, prediction, start=0):
    """Не включать переданные модели начальные кадры в основную оценку."""
    reference = np.asarray(reference)[start:]
    prediction = np.asarray(prediction)[start:]
    difference = prediction - reference

    return {
        "relative_l2": float(
            np.linalg.norm(difference)
            / max(np.linalg.norm(reference), 1e-12)
        ),
        "mae": float(np.mean(np.abs(difference))),
        "relative_l2_by_time": (
            np.linalg.norm(difference, axis=-1)
            / np.maximum(np.linalg.norm(reference, axis=-1), 1e-12)
        ),
    }