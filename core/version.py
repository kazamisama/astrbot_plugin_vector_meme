"""版本号唯一来源。

metadata.yaml / README / CHANGELOG 里的版本由 scripts/gen_command_docs.py 写入、
由 tests/test_main.py::test_version_consistency 校验，避免三处各写一份后漂移。
"""

VERSION = "0.8.2"
