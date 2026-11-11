"""Helpers shared by the dataset loaders."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kwan.data import registry


def dataset_attrs(key: str, subset: str | None = None, **extra: Any) -> dict[str, Any]:
    """What a loader records in `df.attrs["dataset"]`: the registry key, licence and commercial flag."""
    out: dict[str, Any] = {
        "key": registry.dataset_key(key),
        "licence": registry.licence(key),
        "commercial_params_ok": registry.commercial_ok(key),
    }
    if subset is not None:
        out["subset"] = subset
    out.update(extra)
    return out


def need(path: Path, hint: str = "run `make data`") -> Path:
    """Return `path`, or raise FileNotFoundError with what to do about it."""
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing ({hint})")
    return path


def read_csv_chunks(
    path: str | Path, columns: dict[str, Any], block_bytes: int = 32 << 20, multiline: bool = False
) -> Any:
    """Read some columns of a large CSV in chunks with pyarrow and return one pyarrow Table.

    `columns` maps each wanted column to its pyarrow type (for example `pa.int32()`), so the file is
    never held in memory as text and ids stay 32-bit. Set `multiline` for files whose quoted fields
    hold line breaks.
    """
    import pyarrow as pa
    from pyarrow import csv

    reader = csv.open_csv(
        str(path),
        read_options=csv.ReadOptions(block_size=block_bytes, use_threads=not multiline),
        parse_options=csv.ParseOptions(newlines_in_values=multiline),
        convert_options=csv.ConvertOptions(include_columns=list(columns), column_types=columns),
    )
    batches = list(reader)
    return pa.Table.from_batches(batches, schema=reader.schema)
