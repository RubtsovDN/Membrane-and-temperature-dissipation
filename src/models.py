"""Свободные колебания и охлаждение прямоугольника с нулевыми границами."""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class RectModes:
    """Синусоидальные моды прямоугольной области."""

    modes: tuple = ((1, 1), (2, 1))
    lx: float = 1.0
    ly: float = 1.0

    def __post_init__(self):
        if (
            not np.isfinite([self.lx, self.ly]).all()
            or min(self.lx, self.ly) <= 0
        ):
            raise ValueError("Размеры области должны быть положительными")

        modes = tuple(tuple(pair) for pair in self.modes)

        if not modes or any(
            len(pair) != 2
            or any(
                not isinstance(n, (int, np.integer)) or n < 1
                for n in pair
            )
            for pair in modes
        ):
            raise ValueError(
                "Моды задаются парами положительных целых чисел"
            )

        if len(set(modes)) != len(modes):
            raise ValueError("Моды не должны повторяться")

        object.__setattr__(self, "modes", modes)

    @property
    def eigenvalues(self):
        """Собственные значения оператора -Delta."""
        m, n = np.asarray(self.modes).T

        return np.pi**2 * (
            (m / self.lx)**2 + (n / self.ly)**2
        )

    def basis(self, points):
        """
        Вычислить пространственные моды в заданных точках.

        points: массив формы (P, 2), координаты (x, y).
        Возвращает матрицу формы (P, K).
        """
        points = np.atleast_2d(
            np.asarray(points, dtype=float)
        )

        if (
            points.ndim != 2
            or points.shape[1] != 2
            or not np.isfinite(points).all()
        ):
            raise ValueError(
                "Ожидаются конечные координаты формы (P, 2)"
            )

        if (
            np.any(points < 0)
            or np.any(points > [self.lx, self.ly])
        ):
            raise ValueError("Точка вне прямоугольника")

        m, n = np.asarray(self.modes).T

        return (
            np.sin(np.pi * points[:, 0, None] * m / self.lx)
            * np.sin(np.pi * points[:, 1, None] * n / self.ly)
        )


def _vector(value, size):
    value = np.asarray(value, dtype=float)

    if (
        value.shape != (size,)
        or not np.isfinite(value).all()
    ):
        raise ValueError(
            f"Ожидается конечный вектор длины {size}"
        )

    return value


def _parameters(transport, loss):
    if (
        not np.isfinite([transport, loss]).all()
        or transport <= 0
        or loss < 0
    ):
        raise ValueError(
            "Коэффициент переноса должен быть > 0, потерь >= 0"
        )


def _integrate(rhs, times, initial):
    times = np.asarray(times, dtype=float)

    if (
        times.ndim != 1
        or times.size == 0
        or not np.isfinite(times).all()
        or np.any(np.diff(times) <= 0)
    ):
        raise ValueError(
            "Время должно быть конечным и строго возрастающим"
        )

    # Начальные условия относятся к times[0].
    if times.size == 1:
        return initial[None, :].copy()

    solution = solve_ivp(
        rhs,
        (times[0], times[-1]),
        initial,
        t_eval=times,
        method="DOP853",
        rtol=1e-9,
        atol=1e-11,
    )

    if not solution.success:
        raise RuntimeError(solution.message)

    return solution.y.T


def solve_membrane(
    times,
    modes,
    initial,
    velocity=None,
    *,
    c=1.0,
    gamma=0.15,
):
    """
    Решить u_tt + 2*gamma*u_t = c**2*Delta(u).

    initial: начальные модальные амплитуды.
    velocity: начальные производные модальных амплитуд.
    Возвращает (a, a_t), каждый массив формы (N, K).

    Базовая модель получается при gamma=0.
    """
    _parameters(c, gamma)

    k = len(modes.modes)
    initial = _vector(initial, k)

    velocity = (
        np.zeros(k)
        if velocity is None
        else _vector(velocity, k)
    )

    eigenvalues = modes.eigenvalues

    def rhs(t, state):
        a = state[:k]
        v = state[k:]

        acceleration = (
            -2 * gamma * v
            - c**2 * eigenvalues * a
        )

        return np.concatenate((v, acceleration))

    state = _integrate(
        rhs,
        times,
        np.concatenate((initial, velocity)),
    )

    return state[:, :k], state[:, k:]


def solve_heat(
    times,
    modes,
    initial,
    *,
    alpha=0.05,
    beta=0.1,
):
    """
    Решить theta_t = alpha*Delta(theta) - beta*theta.

    theta — превышение температуры над окружающей средой.
    Возвращает модальные амплитуды формы (N, K).

    Базовая модель получается при beta=0.
    """
    _parameters(alpha, beta)

    initial = _vector(initial, len(modes.modes))
    rates = alpha * modes.eigenvalues + beta

    return _integrate(
        lambda t, b: -rates * b,
        times,
        initial,
    )


def reconstruct(amplitudes, modes, points):
    """
    Восстановить поле по модальным амплитудам.

    amplitudes: массив формы (N, K).
    points: координаты формы (P, 2).
    Результат: значения поля формы (N, P).
    """
    amplitudes = np.asarray(amplitudes, dtype=float)

    if (
        amplitudes.ndim != 2
        or amplitudes.shape[1] != len(modes.modes)
        or not np.isfinite(amplitudes).all()
    ):
        raise ValueError(
            "Неверная матрица модальных амплитуд"
        )

    return amplitudes @ modes.basis(points).T