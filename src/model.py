# model.py
"""POD + обучение пространственного оператора методом ridge-регрессии."""

from dataclasses import dataclass

import numpy as np
from sklearn.utils.extmath import randomized_svd

from physics import evolve, regression_terms


@dataclass
class PODOperator:
    kind: str
    rank: int = 24
    ridge: float = 1e-6
    seed: int = 42

    def fit(self, fields, max_snapshots=3000):
        if fields.kind != self.kind:
            raise ValueError("Тип данных не соответствует модели.")
        if self.rank < 1 or self.ridge <= 0:
            raise ValueError("rank и ridge должны быть положительными.")

        snapshots = fields.u.reshape(-1, fields.u.shape[-1])
        rng = np.random.default_rng(self.seed)

        if len(snapshots) > max_snapshots:
            indices = rng.choice(
                len(snapshots), max_snapshots, replace=False
            )
            snapshots = snapshots[indices]

        rank = min(self.rank, *snapshots.shape)
        _, singular, vt = randomized_svd(
            snapshots,
            n_components=rank,
            n_iter=5,
            random_state=self.seed,
        )

        keep = singular > max(singular[0] * 1e-10, 1e-14)
        if not np.any(keep):
            raise ValueError("Обучающие поля практически нулевые.")

        self.basis_ = vt[keep].T
        z = fields.u @ self.basis_

        state, derivative, _ = regression_terms(
            z, fields.t, self.kind
        )
        target = derivative / fields.d[:, None, None]

        x = state.reshape(-1, state.shape[-1])
        y = target.reshape(-1, target.shape[-1])

        gram = x.T @ x
        regularization = self.ridge * max(
            np.trace(gram) / len(gram), 1e-12
        )

        coefficients = np.linalg.solve(
            gram + regularization * np.eye(len(gram)),
            x.T @ y,
        )
        raw = coefficients.T

        # Физическая регуляризация пространственного оператора.
        symmetric = 0.5 * (raw + raw.T)
        eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
        eigenvalues = np.minimum(eigenvalues, 0.0)

        self.operator_ = (
            eigenvectors * eigenvalues
        ) @ eigenvectors.T

        projection = (
            snapshots @ self.basis_
        ) @ self.basis_.T

        self.projection_error_ = float(
            np.linalg.norm(snapshots - projection)
            / max(np.linalg.norm(snapshots), 1e-12)
        )
        return self

    def predict(self, initial, t, d, loss=0.0):
        if not hasattr(self, "basis_"):
            raise RuntimeError("Сначала вызовите fit().")

        latent_initial = np.asarray(initial) @ self.basis_
        latent = evolve(
            self.operator_,
            latent_initial,
            t,
            d,
            self.kind,
            loss,
        )
        return latent @ self.basis_.T

    def save(self, path):
        np.savez_compressed(
            path,
            kind=self.kind,
            rank=self.rank,
            ridge=self.ridge,
            seed=self.seed,
            basis=self.basis_,
            operator=self.operator_,
            projection_error=self.projection_error_,
        )

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            model = cls(
                kind=str(data["kind"].item()),
                rank=int(data["rank"].item()),
                ridge=float(data["ridge"].item()),
                seed=int(data["seed"].item()),
            )
            model.basis_ = data["basis"].copy()
            model.operator_ = data["operator"].copy()
            model.projection_error_ = float(
                data["projection_error"].item()
            )
        return model