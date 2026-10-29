"""异常处理卫生检查：运行时代码里不允许"宽泛 except 静默吞掉"。

约定（2026-10 审计）：
- 捕获 Exception 时必须显式降级（记日志 / 返回兜底值 / 计数），不能只有 pass/continue；
- 能确定异常类型的（int()/json.loads()/Path.resolve() 等）应收窄到具体类型。
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNTIME_FILES = [ROOT / "main.py", *sorted((ROOT / "core").glob("*.py"))]


def _handler_label(handler: ast.ExceptHandler) -> str | None:
    if handler.type is None:
        return "bare"
    if isinstance(handler.type, ast.Name):
        return handler.type.id
    if isinstance(handler.type, ast.Tuple):
        names = []
        for elt in handler.type.elts:
            if not isinstance(elt, ast.Name):
                return None
            names.append(elt.id)
        return "|".join(names)
    return None


def _is_silent(body: list[ast.stmt]) -> bool:
    return bool(body) and all(isinstance(stmt, (ast.Pass, ast.Continue)) for stmt in body)


def _silent_broad_handlers() -> list[str]:
    offenders = []
    for path in RUNTIME_FILES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Try):
                continue
            for handler in node.handlers:
                label = _handler_label(handler)
                if label not in ("Exception", "BaseException", "bare"):
                    continue
                if _is_silent(handler.body):
                    rel = path.relative_to(ROOT).as_posix()
                    offenders.append(f"{rel}:{handler.lineno} ({label})")
    return offenders


def test_no_silent_broad_exception_handler():
    offenders = _silent_broad_handlers()
    assert not offenders, (
        "以下位置用宽泛 except 静默吞掉了异常，请加日志/降级说明或收窄类型：\n"
        + "\n".join(offenders)
    )


def test_all_runtime_files_are_scanned():
    names = {path.name for path in RUNTIME_FILES}
    assert {"main.py", "database.py", "embedder.py", "retriever.py", "dedup.py"} <= names
