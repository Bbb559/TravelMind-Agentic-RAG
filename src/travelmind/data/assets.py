"""运行资产盘点。"""

from __future__ import annotations

import csv

from travelmind.config import get_settings


def inventory() -> dict:
    settings = get_settings()
    csv_rows = 0
    csv_columns: list[str] = []
    if settings.travel_csv_path.exists():
        rows, columns = _read_csv_inventory(settings.travel_csv_path)
        csv_rows = rows
        csv_columns = columns
    pdfs = sorted(path.name for path in settings.assets_dir.joinpath("gang_ao_pdf").glob("*.pdf"))
    return {
        "assets_dir": str(settings.assets_dir),
        "travel_csv": str(settings.travel_csv_path),
        "csv_rows": csv_rows,
        "csv_columns": csv_columns,
        "pdfs": pdfs,
        "graphrag_output_dir": str(settings.graphrag_output_dir),
        "multimodal_markdown_dir": str(settings.multimodal_markdown_dir),
    }


def _read_csv_inventory(path):
    for encoding in ("utf-8-sig", "gbk", "gb18030"):
        try:
            with path.open("r", encoding=encoding, newline="") as file:
                reader = csv.DictReader(file)
                columns = list(reader.fieldnames or [])
                rows = sum(1 for _ in reader)
                return rows, columns
        except UnicodeDecodeError:
            continue
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as file:
        reader = csv.DictReader(file)
        columns = list(reader.fieldnames or [])
        rows = sum(1 for _ in reader)
        return rows, columns
