"""HTTP-загрузка и чтение временных рядов из CSV и ZIP."""

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from zipfile import ZipFile

import csv
import hashlib
import io
import os
import tempfile

import numpy as np


SOURCES = {
    "membrane": {
        "doi": "10.5281/zenodo.22344228",
        "url": (
            "https://zenodo.org/records/22344228/files/"
            "jtcam-data-13030.zip?download=1"
        ),
    },
    "heat": {
        "doi": "10.17632/sbcf36976g.1",
        "page": (
            "https://data.mendeley.com/datasets/"
            "sbcf36976g/1"
        ),
        # Прямой URL файла берётся через Download.
        # Адрес страницы набора не является адресом CSV.
    },
}


def sha256(path):
    """Рассчитать контрольную сумму файла без загрузки целиком в память."""
    digest = hashlib.sha256()

    with open(path, "rb") as stream:
        for block in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def download(
    url,
    destination,
    *,
    expected_sha256=None,
    force=False,
):
    """
    Скачать файл по HTTP(S), используя локальный кэш.

    Файл заменяется только после успешной загрузки и проверки.
    Без expected_sha256 существующий кэш считается доверенным.
    force=True позволяет повторно скачать файл.
    """
    if urlparse(url).scheme not in {"http", "https"}:
        raise ValueError(
            "Разрешены только HTTP(S)-адреса"
        )

    destination = Path(destination)
    expected = (
        expected_sha256.lower()
        if expected_sha256
        else None
    )

    if destination.exists() and not force:
        if expected and sha256(destination) != expected:
            raise ValueError(
                "Контрольная сумма кэша не совпадает"
            )
        return destination

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = None

    try:
        request = Request(
            url,
            headers={"User-Agent": "PhysicsLab/1.0"},
        )

        with urlopen(request, timeout=60) as response:
            if response.status != 200:
                raise OSError(f"HTTP {response.status}")

            # Временный файл находится на том же диске:
            # последующая замена destination будет атомарной.
            with tempfile.NamedTemporaryFile(
                dir=destination.parent,
                delete=False,
            ) as out:
                temporary = Path(out.name)

                for block in iter(
                    lambda: response.read(1024 * 1024),
                    b"",
                ):
                    out.write(block)

        if temporary.stat().st_size == 0:
            raise ValueError("Получен пустой файл")

        if expected and sha256(temporary) != expected:
            raise ValueError(
                "Контрольная сумма загрузки не совпадает"
            )

        os.replace(temporary, destination)
        return destination

    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@dataclass
class TimeSeries:
    time: np.ndarray
    values: np.ndarray
    columns: tuple


def read_timeseries_csv(
    path,
    *,
    time_column,
    value_columns,
    delimiter=",",
    decimal=".",
    time_scale=1.0,
    zip_member=None,
):
    """
    Прочитать CSV с заголовком.

    time_column: название столбца времени.
    value_columns: список столбцов измерений.
    time_scale: множитель времени; для мс -> с равен 0.001.
    zip_member: имя CSV внутри ZIP, если читается архив.

    Возвращает:
        time: (N,)
        values: (N, число выбранных столбцов)

    Пропуски и повторяющиеся моменты времени не исправляются
    автоматически, а приводят к явной ошибке.
    """
    columns = tuple(value_columns)

    if (
        not columns
        or len(set(columns)) != len(columns)
        or time_column in columns
        or not np.isfinite(time_scale)
        or time_scale <= 0
    ):
        raise ValueError(
            "Некорректные столбцы или масштаб времени"
        )

    if zip_member is None:
        text = Path(path).read_text(
            encoding="utf-8-sig"
        )
    else:
        # Архив не распаковывается на диск.
        with ZipFile(path) as archive:
            text = archive.read(zip_member).decode(
                "utf-8-sig"
            )

    reader = csv.DictReader(
        io.StringIO(text),
        delimiter=delimiter,
    )

    requested = (time_column,) + columns

    if not set(requested).issubset(
        reader.fieldnames or []
    ):
        raise ValueError(
            f"Отсутствуют столбцы: {requested}"
        )

    rows = []

    for row in reader:
        try:
            rows.append([
                float(row[name].replace(decimal, "."))
                for name in requested
            ])
        except (
            ValueError,
            TypeError,
            AttributeError,
        ) as error:
            raise ValueError(
                "Нечисловое значение в строке "
                f"{reader.line_num}"
            ) from error

    data = np.asarray(rows, dtype=float)

    if data.size == 0 or not np.isfinite(data).all():
        raise ValueError(
            "Пустые данные, NaN или бесконечность"
        )

    time = data[:, 0] * time_scale

    if (
        not np.isfinite(time).all()
        or np.any(np.diff(time) <= 0)
    ):
        raise ValueError(
            "Время должно строго возрастать без повторений"
        )

    return TimeSeries(
        time=time,
        values=data[:, 1:],
        columns=columns,
    )