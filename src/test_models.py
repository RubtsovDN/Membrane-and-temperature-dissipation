import unittest

import numpy as np
from numpy.testing import assert_allclose

from models import (
    RectModes,
    solve_membrane,
    solve_heat,
    reconstruct,
)
from fitting import fit_membrane, fit_heat


class ModelTests(unittest.TestCase):
    def test_membrane_against_analytic_solution(self):
        modes = RectModes(((1, 1),))
        t = np.linspace(2, 5, 151)
        c, gamma = 1.0, 0.15

        a, v = solve_membrane(
            t,
            modes,
            [1.0],
            c=c,
            gamma=gamma,
        )

        omega = np.sqrt(
            c**2 * modes.eigenvalues[0] - gamma**2
        )

        # Проверяем также ненулевой начальный момент времени.
        dt = t - t[0]

        exact = np.exp(-gamma * dt) * (
            np.cos(omega * dt)
            + gamma / omega * np.sin(omega * dt)
        )

        assert_allclose(a[:, 0], exact, atol=1e-8)
        assert_allclose(v[0], [0.0], atol=1e-12)

    def test_heat_against_analytic_solution(self):
        modes = RectModes()
        t = np.linspace(0, 3, 80)
        initial = np.array([1.0, 0.3])

        actual = solve_heat(
            t,
            modes,
            initial,
            alpha=0.05,
            beta=0.1,
        )

        exact = initial * np.exp(
            -t[:, None]
            * (0.05 * modes.eigenvalues + 0.1)
        )

        assert_allclose(actual, exact, atol=1e-9)

        # При отсутствии источников норма поля убывает.
        norm_squared = np.sum(actual**2, axis=1)
        self.assertTrue(
            np.all(np.diff(norm_squared) <= 1e-10)
        )

    def test_boundaries_and_energy(self):
        modes = RectModes()
        t = np.linspace(0, 3, 200)

        a, v = solve_membrane(
            t,
            modes,
            [1.0, 0.3],
        )

        boundary = [
            [0, 0.3],
            [1, 0.6],
            [0.4, 0],
            [0.6, 1],
        ]

        assert_allclose(
            reconstruct(a, modes, boundary),
            0,
            atol=1e-14,
        )

        # Здесь c=1. Общий множитель нормировки мод
        # не влияет на монотонность энергии.
        energy = 0.5 * np.sum(
            v**2 + modes.eigenvalues * a**2,
            axis=1,
        )

        self.assertTrue(
            np.all(np.diff(energy) <= 1e-8)
        )

    def check_parameter_recovery(self, kind):
        modes = RectModes()
        t = np.linspace(0, 2, 100)
        initial = np.array([1.0, 0.3])

        points = [
            [0.25, 0.5],
            [0.65, 0.5],
        ]

        # Эталон генерируется аналитически,
        # а оптимизация использует численный солвер.
        if kind == "heat":
            truth = np.array([0.05, 0.1])

            a = initial * np.exp(
                -t[:, None]
                * (
                    truth[0] * modes.eigenvalues
                    + truth[1]
                )
            )

            observed = reconstruct(a, modes, points)

            fit = fit_heat(
                t,
                observed,
                points,
                modes,
                initial,
            )

        else:
            truth = np.array([1.0, 0.15])

            omega = np.sqrt(
                truth[0]**2 * modes.eigenvalues
                - truth[1]**2
            )

            a = (
                initial
                * np.exp(-truth[1] * t[:, None])
                * (
                    np.cos(t[:, None] * omega)
                    + truth[1] / omega
                    * np.sin(t[:, None] * omega)
                )
            )

            observed = reconstruct(a, modes, points)

            fit = fit_membrane(
                t,
                observed,
                points,
                modes,
                initial,
            )

        assert_allclose(
            list(fit.parameters.values()),
            truth,
            rtol=1e-4,
            atol=1e-6,
        )

        self.assertLess(fit.rmse, 1e-7)

    def test_heat_parameter_recovery(self):
        self.check_parameter_recovery("heat")

    def test_membrane_parameter_recovery(self):
        self.check_parameter_recovery("membrane")

    def test_invalid_time_and_parameters(self):
        with self.assertRaises(ValueError):
            solve_heat(
                [0, 0],
                RectModes(),
                [1, 0.3],
            )

        with self.assertRaises(ValueError):
            solve_membrane(
                [0, 1],
                RectModes(),
                [1, 0.3],
                gamma=-1,
            )