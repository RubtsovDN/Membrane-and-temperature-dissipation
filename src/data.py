# data.py
"""HTTP-данные и единый формат пространственно-временных полей."""

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import shutil
import zipfile

import fsspec
import h5py
import numpy as np
import requests
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay


WAVE_URL = (
    "https://zenodo.org/records/15397980/files/"
    "Wave-C-Sines.nc?download=1"
)
MAE_URL = (
    "https://zenodo.org/records/13355846/files/"
    "mae_pdes_data.zip?download=1"
)


def fingerprint(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url, path):
    """Потоковая загрузка; недокачанный файл не становится итоговым."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")

    with requests.get(url, stream=True, timeout=(20, 120)) as response:
        response.raise_for_status()
        with temporary.open("wb") as stream:
            for block in response.iter_content(2**20):
                stream.write(block)

    temporary.replace(path)
    record = {"url": url, "sha256": fingerprint(path)}
    path.with_suffix(path.suffix + ".source.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    return path


def zip_members(url=MAE_URL):
    """Получить имена HDF5 через HTTP Range, не распаковывая архив."""
    with fsspec.open(url, "rb", block_size=2**20) as stream:
        with zipfile.ZipFile(stream) as archive:
            return [
                name for name in archive.namelist()
                if name.lower().endswith((".h5", ".hdf5"))
            ]


def download_zip_member(member, path, url=MAE_URL):
    """Скачать один явно выбранный файл из удалённого ZIP."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")

    with fsspec.open(url, "rb", block_size=2**20) as stream:
        with zipfile.ZipFile(stream) as archive:
            with archive.open(member) as source, temporary.open("wb") as out:
                shutil.copyfileobj(source, out, length=2**20)

    temporary.replace(path)
    record = {
        "url": url,
        "archive_member": member,
        "sha256": fingerprint(path),
    }
    path.with_suffix(path.suffix + ".source.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    return path


def h5_inventory(path):
    """Показать реальные пути и формы массивов перед чтением."""
    result = {}
    with h5py.File(path, "r") as file:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                result[name] = tuple(obj.shape)
        file.visititems(visit)
    return result


def _array(node):
    """В NetCDF переменная может лежать внутри одноимённой группы."""
    if isinstance(node, h5py.Dataset):
        return node

    arrays = []
    node.visititems(
        lambda _, obj: arrays.append(obj)
        if isinstance(obj, h5py.Dataset) and obj.ndim == 4 else None
    )
    if len(arrays) != 1:
        raise ValueError(
            "Укажите точный путь к массиву из h5_inventory()."
        )
    return arrays[0]


@dataclass
class Grid:
    x: np.ndarray
    y: np.ndarray
    mask: np.ndarray
    periodic: bool = False

    @property
    def dx(self):
        return float(self.x[1] - self.x[0])

    @property
    def dy(self):
        return float(self.y[1] - self.y[0])

    @property
    def points(self):
        xx, yy = np.meshgrid(self.x, self.y, indexing="xy")
        return np.column_stack([xx[self.mask], yy[self.mask]])

    def image(self, values):
        """Активные узлы -> двумерное поле для графика."""
        out = np.full(self.mask.shape, np.nan)
        out[self.mask] = values
        return out


@dataclass
class Fields:
    # u: [траектория, время, активный пространственный узел]
    u: np.ndarray
    t: np.ndarray
    d: np.ndarray       # alpha для heat; c**2 для wave
    grid: Grid
    kind: str

    def __post_init__(self):
        self.u = np.asarray(self.u, dtype=float)
        self.t = np.asarray(self.t, dtype=float)
        self.d = np.asarray(self.d, dtype=float)

        if self.kind not in {"heat", "wave"}:
            raise ValueError("kind должен быть heat или wave.")
        if self.u.ndim != 3:
            raise ValueError("Ожидается u[N, T, P].")
        n, nt, p = self.u.shape
        if nt != len(self.t) or p != self.grid.mask.sum():
            raise ValueError("Несогласованные размеры полей и сетки.")
        if self.d.shape != (n,) or np.any(self.d <= 0):
            raise ValueError("Нужен положительный коэффициент на траекторию.")
        if not np.isfinite(self.u).all():
            raise ValueError("В данных есть NaN/Inf.")
        if nt < 3 or np.any(np.diff(self.t) <= 0):
            raise ValueError("Нужно минимум три возрастающих момента времени.")

    def subset(self, indices):
        return Fields(
            self.u[indices], self.t.copy(), self.d[indices],
            self.grid, self.kind
        )


def split_indices(n, seed=42):
    """Разделяем траектории, а не отдельные кадры."""
    if n < 10:
        raise ValueError("Для разбиения нужно минимум 10 траекторий.")
    indices = np.random.default_rng(seed).permutation(n)
    a, b = int(0.7 * n), int(0.85 * n)
    return indices[:a], indices[a:b], indices[b:]


def load_wave(path, count=128, resolution=40, u_key="u", x_key="x"):
    """RIGNO Wave-C-Sines -> регулярная сетка внутри круга.

    Временная ось взята из описания статьи:
    21 сохранённый кадр на [0, 0.1].
    """
    with h5py.File(path, "r") as file:
        values = np.asarray(_array(file[u_key])[:count])
        coords = _array(file[x_key])
        xy = np.asarray(coords[0, 0])

        if values.ndim != 4 or values.shape[-1] != 1:
            raise ValueError("Ожидается одно скалярное поле [N,T,P,1].")
        if values.shape[1] != 21:
            raise ValueError("Временная сетка отличается от описания источника.")
        if coords.shape[0] > 1:
            for i in range(len(values)):
                if not np.allclose(coords[i, 0], xy):
                    raise ValueError("Этот адаптер требует общей сетки.")

    axis = np.linspace(-0.5, 1.5, resolution)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    targets = np.column_stack([xx.ravel(), yy.ravel()])
    triangulation = Delaunay(xy)

    inside_circle = (xx - 0.5)**2 + (yy - 0.5)**2 < 1.0
    inside_mesh = (
        triangulation.find_simplex(targets).reshape(xx.shape) >= 0
    )
    grid = Grid(axis, axis.copy(), inside_circle & inside_mesh)
    projected = []

    for trajectory in values[..., 0]:
        interpolator = LinearNDInterpolator(
            triangulation, trajectory.T
        )
        projected.append(interpolator(grid.points).T)

    return Fields(
        np.stack(projected),
        np.linspace(0.0, 0.1, 21),
        np.full(len(projected), 4.0),  # c=2 -> c**2=4
        grid,
        "wave",
    )


def load_heat(path, group="train", count=128):
    """MAE-PDEs: выбрать строки с метаданными чистой 2D Heat.

    После чтения обязательно вызвать audit() из physics.py:
    метаданные источника сами по себе не гарантируют корректную PDE.
    """
    with h5py.File(path, "r") as file:
        data = file[group]
        nu = np.asarray(data["nu"])
        selected = nu > 0

        for name in ("ax", "ay", "cx", "cy"):
            selected &= np.isclose(np.asarray(data[name]), 0.0)

        indices = np.flatnonzero(selected)[:count]
        if not len(indices):
            raise ValueError("Не найдены строки с метками чистой Heat.")

        u = np.asarray(data["u"][indices])
        coords = np.asarray(data["x"])
        t = np.asarray(data["t"])

    if u.ndim != 4 or coords.shape != (2, *u.shape[2:]):
        raise ValueError("Ожидаются u[N,T,H,W] и x[2,H,W].")

    # Не предполагаем порядок осей meshgrid в исходном файле.
    flat_x, flat_y = coords[0].ravel(), coords[1].ravel()
    x, y = np.unique(flat_x), np.unique(flat_y)
    if len(x) * len(y) != len(flat_x):
        raise ValueError("Нужна полная прямоугольная сетка.")

    order = np.lexsort((flat_x, flat_y))
    grid = Grid(x, y, np.ones((len(y), len(x)), dtype=bool), True)

    for axis in (x, y):
        if not np.allclose(np.diff(axis), axis[1] - axis[0], rtol=1e-4):
            raise ValueError("Сетка неравномерная.")
        if not np.isclose(len(axis) * (axis[1] - axis[0]), 2.0, rtol=1e-3):
            raise ValueError(
                "Проверьте период и наличие дублированной конечной точки."
            )

    u = u.reshape(len(indices), len(t), -1)[..., order]
    return Fields(u, t, nu[indices], grid, "heat")