import hashlib
import io

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np

import data_io


class Response(io.BytesIO):
    """Поддельный HTTP-ответ: тесты не требуют сети."""

    status = 200


class DataTests(TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_download_and_cache(self):
        payload = b"time,value\n0,1\n1,2\n"
        target = self.root / "data.csv"

        checksum = hashlib.sha256(
            payload
        ).hexdigest()

        with patch.object(
            data_io,
            "urlopen",
            return_value=Response(payload),
        ) as request:
            for _ in range(2):
                data_io.download(
                    "https://example.org/data.csv",
                    target,
                    expected_sha256=checksum,
                )

        self.assertEqual(
            target.read_bytes(),
            payload,
        )

        # Второе обращение должно использовать кэш.
        self.assertEqual(request.call_count, 1)

    def test_failed_download_preserves_existing_file(self):
        target = self.root / "data.csv"
        target.write_bytes(b"original")

        with patch.object(
            data_io,
            "urlopen",
            return_value=Response(b"wrong"),
        ):
            with self.assertRaisesRegex(
                ValueError,
                "Контрольная сумма",
            ):
                data_io.download(
                    "https://example.org/data.csv",
                    target,
                    expected_sha256="0" * 64,
                    force=True,
                )

        # Неудачная загрузка не должна портить исходный файл.
        self.assertEqual(
            target.read_bytes(),
            b"original",
        )

        # Временный файл должен быть удалён.
        self.assertEqual(
            list(self.root.iterdir()),
            [target],
        )

    def test_csv_inside_zip(self):
        path = self.root / "data.zip"

        with ZipFile(path, "w") as archive:
            archive.writestr(
                "trial.csv",
                "ms;T1;T2\n"
                "0;20,0;21,0\n"
                "1000;19,5;20,5\n",
            )

        data = data_io.read_timeseries_csv(
            path,
            zip_member="trial.csv",
            time_column="ms",
            value_columns=["T1", "T2"],
            delimiter=";",
            decimal=",",
            time_scale=0.001,
        )

        np.testing.assert_allclose(
            data.time,
            [0, 1],
        )

        np.testing.assert_allclose(
            data.values,
            [[20, 21], [19.5, 20.5]],
        )

    def test_invalid_csv(self):
        cases = [
            "t,T\n0,1\n0,2\n",   # Повтор времени.
            "t,T\n0,nan\n",      # Пропуск измерения.
            "t,other\n0,1\n",    # Нет нужного столбца.
            "t,T\n",            # Нет данных.
        ]

        for content in cases:
            with self.subTest(content=content):
                path = self.root / "bad.csv"
                path.write_text(content)

                with self.assertRaises(ValueError):
                    data_io.read_timeseries_csv(
                        path,
                        time_column="t",
                        value_columns=["T"],
                    )