import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _config_get_keys() -> set[str]:
    """用 AST 收集 main.py 里真实执行的 config.get(...) 键，忽略注释与字符串。"""
    # main.py 带 UTF-8 BOM，用 utf-8-sig 读取
    tree = ast.parse((ROOT / "main.py").read_text(encoding="utf-8-sig"))
    keys: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if not (isinstance(fn, ast.Attribute) and fn.attr == "get"):
            continue
        if isinstance(fn.value, ast.Attribute) and fn.value.attr == "config":
            if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                keys.add(node.args[0].value)
    return keys


def test_config_keys_consistent_with_schema():
    schema = json.loads((ROOT / "_conf_schema.json").read_text(encoding="utf-8"))
    schema_keys = {k for group in schema.values() for k in group.get("items", {}).keys()}
    code_keys = _config_get_keys()

    assert code_keys - schema_keys == set()
    assert "trigger_probability" not in code_keys
    assert "_temp_dim" not in code_keys
    assert "trigger_probability" not in schema_keys
