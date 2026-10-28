"""scripts/diag_db.py 的冒烟测试：只读诊断必须能读真实 schema。"""
from __future__ import annotations

import importlib.util
import json

import numpy as np

from core.database import MemeDatabase

ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent


def _load_script():
    path = ROOT / "scripts" / "diag_db.py"
    spec = importlib.util.spec_from_file_location("diag_db", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _make_db(tmp_path) -> MemeDatabase:
    db = MemeDatabase(tmp_path / "memes.db", tmp_path / "memes.faiss", dim=4)
    for i, tag in enumerate(("happy", "happy", "sad")):
        vids = db.add_vectors(np.array([[1.0, 0.0, 0.0, 0.0]], dtype="float32"))
        db.upsert_meme(f"/tmp/m{i}.png", f"h{i}", tag, int(vids[0]), file_name=f"m{i}.png")
    db.save_index()
    return db


def test_inspect_reports_schema_and_counts(tmp_path):
    diag = _load_script()
    _make_db(tmp_path)
    info = diag.inspect(tmp_path / "memes.db", logs=5)
    assert info["counts"]["memes"] == 3
    assert info["counts"]["tags"] == 0
    assert info["vector_id"]["rows_with_id"] == 3
    assert [t["tag"] for t in info["tags"]] == ["happy", "sad"]
    assert info["tags"][0]["count"] == 2
    assert "search_log" in info["tables"]


def test_main_json_output(tmp_path, capsys):
    diag = _load_script()
    _make_db(tmp_path)
    code = diag.main(["--db", str(tmp_path / "memes.db"), "--json", "--logs", "0"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["counts"]["memes"] == 3


def test_main_missing_db_returns_2(tmp_path, capsys):
    diag = _load_script()
    assert diag.main(["--db", str(tmp_path / "nope.db")]) == 2
    assert "不存在" in capsys.readouterr().err
