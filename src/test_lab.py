# test_lab.py
import networkx as nx
import numpy as np

from data import Fields, Grid, split_indices
from equations import graph_representations
from model import PODOperator
from physics import evolve, identify, laplacian, metrics


def periodic_problem(n=20):
    axis = np.arange(n) / n
    grid = Grid(
        axis, axis.copy(), np.ones((n, n), dtype=bool), True
    )
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    mode = np.sin(2*np.pi*xx) * np.sin(2*np.pi*yy)
    operator = laplacian(grid)

    # Точное собственное значение дискретного лапласиана.
    eigenvalue = -8 * n**2 * np.sin(np.pi / n)**2
    return grid, operator, mode.ravel(), eigenvalue


def test_split_has_no_leakage():
    train, valid, test = map(set, split_indices(100))
    assert not train & valid
    assert not train & test
    assert not valid & test
    assert len(train | valid | test) == 100


def test_graphs_are_dags():
    for kind in ("heat", "wave"):
        result = graph_representations(kind)
        assert nx.is_directed_acyclic_graph(result["graph"])
        assert result["adjacency"].sum() == result["graph"].number_of_edges()


def test_periodic_laplacian():
    grid, operator, mode, eigenvalue = periodic_problem()
    np.testing.assert_allclose(
        operator @ np.ones(grid.mask.sum()), 0, atol=1e-10
    )
    np.testing.assert_allclose(
        operator @ mode, eigenvalue * mode, atol=1e-10
    )


def test_heat_against_analytic_mode():
    _, operator, mode, eigenvalue = periodic_problem()
    t = np.linspace(0, 0.2, 101)
    alpha = 0.02

    predicted = evolve(operator, mode, t, alpha, "heat")
    exact = np.exp(alpha * eigenvalue * t[:, None]) * mode

    assert metrics(exact, predicted, start=1)["relative_l2"] < 1e-5


def test_wave_against_analytic_mode():
    _, operator, mode, eigenvalue = periodic_problem()
    t = np.linspace(0, 0.1, 101)
    c = 1.0
    omega = c * np.sqrt(-eigenvalue)
    exact = np.cos(omega * t[:, None]) * mode

    predicted = evolve(operator, exact[:2], t, c**2, "wave")
    assert metrics(exact, predicted, start=2)["relative_l2"] < 1e-4


def test_parameter_identification():
    _, operator, mode, _ = periodic_problem()
    t = np.linspace(0, 0.2, 101)
    alpha, beta = 0.02, 0.3

    # Вторая мода позволяет разделить alpha и beta.
    n = 20
    x, y = np.meshgrid(np.arange(n)/n, np.arange(n)/n)
    initial = mode + 0.3 * (
        np.sin(4*np.pi*x) * np.sin(2*np.pi*y)
    ).ravel()

    u = evolve(operator, initial, t, alpha, "heat", beta)
    estimate = identify(u, t, operator, "heat", fit_loss=True)

    np.testing.assert_allclose(estimate["d"], alpha, rtol=1e-5)
    np.testing.assert_allclose(estimate["loss"], beta, rtol=1e-5)


def test_learned_heat_operator_on_new_initial_field():
    grid, operator, mode, _ = periodic_problem()
    t = np.linspace(0, 0.2, 101)
    n = len(grid.x)
    x, y = np.meshgrid(grid.x, grid.y)
    second = (
        np.sin(4*np.pi*x) * np.sin(2*np.pi*y)
    ).ravel()

    coefficients = np.array([0.01, 0.015, 0.02])
    initial_fields = [
        mode + 0.2 * second,
        0.3 * mode + second,
        mode - 0.7 * second,
    ]
    trajectories = np.stack([
        evolve(operator, initial, t, alpha, "heat")
        for initial, alpha in zip(initial_fields, coefficients)
    ])
    train = Fields(trajectories, t, coefficients, grid, "heat")
    model = PODOperator("heat", rank=2, ridge=1e-10).fit(train)

    # Новые коэффициент и комбинация пространственных мод.
    initial = 0.7 * mode + 0.4 * second
    alpha = 0.017
    reference = evolve(operator, initial, t, alpha, "heat")
    predicted = model.predict(initial, t, alpha)

    assert metrics(reference, predicted, start=1)["relative_l2"] < 1e-4