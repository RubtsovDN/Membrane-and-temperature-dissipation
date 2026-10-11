import unittest
import networkx as nx
import numpy as np
import sympy as sp
from equation_graphs import equation_graph, graph_data, evaluate_graph


class TestEquationGraphs(unittest.TestCase):
    def test_matrix_and_dictionary(self):
        for kind, size in [('membrane', 10), ('heat', 7)]:
            g = equation_graph(kind)
            data, Z = graph_data(g)
            order = data['order']
            self.assertTrue(nx.is_directed_acyclic_graph(g))
            self.assertEqual(Z.shape, (size, size))
            self.assertTrue(np.array_equal(Z, np.tril(Z, -1)))
            for i, target in enumerate(order):
                for j, source in enumerate(order):
                    self.assertEqual(Z[i, j], int(g.has_edge(source, target)))
                self.assertEqual(set(g.predecessors(target)),
                                 set(data['nodes'][target]['args']))
            self.assertEqual(data['nodes']['rhs']['args'][-1], 'loss')
            if kind == 'membrane':
                self.assertEqual(data['nodes']['c2']['shape'], 'scalar')

    def test_membrane_polynomial(self):
        x, y, t, c, gamma = sp.symbols('x y t c gamma')
        u = x**2 + 3*y**2 + t*x
        actual = evaluate_graph(equation_graph('membrane'),
                                dict(u=u, c=c, gamma=gamma), (x, y), t)
        self.assertEqual(sp.expand(actual - (8*c**2 - 2*gamma*x)), 0)
        self.assertEqual(sp.simplify(actual.subs(gamma, 0) - 8*c**2), 0)

    def test_heat_polynomial_in_1d_and_2d(self):
        x, y, t, alpha, beta = sp.symbols('x y t alpha beta')
        theta = x**2 + 3*y**2 + t
        for coords, lap in [((x,), 2), ((x, y), 8)]:
            actual = evaluate_graph(equation_graph('heat'),
                                    dict(theta=theta, alpha=alpha, beta=beta),
                                    coords, t)
            self.assertEqual(sp.expand(actual - (lap*alpha-beta*theta)), 0)
            self.assertEqual(sp.simplify(actual.subs(beta, 0)-lap*alpha), 0)

    def test_mode_matches_solver_equations(self):
        x, y, t, c, gamma, alpha, beta = sp.symbols('x y t c gamma alpha beta')
        a = sp.Function('a')(t)
        phi = sp.sin(sp.pi*x)*sp.sin(sp.pi*y)
        m = evaluate_graph(equation_graph('membrane'),
                           dict(u=a*phi, c=c, gamma=gamma), (x, y), t)
        h = evaluate_graph(equation_graph('heat'),
                           dict(theta=a*phi, alpha=alpha, beta=beta), (x, y), t)
        self.assertEqual(sp.simplify(m/phi + 2*sp.pi**2*c**2*a
                                    + 2*gamma*sp.diff(a, t)), 0)
        self.assertEqual(sp.simplify(h/phi + (2*sp.pi**2*alpha+beta)*a), 0)


if __name__ == '__main__':
    unittest.main()
