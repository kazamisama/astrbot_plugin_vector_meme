"""版本号唯一来源。

metadata.yaml / README / CHANGELOG 里的版本由 scripts/gen_command_docs.py 写入，
由 tests/test_command_spec.py 的版本一致性用例校验，避免多处各写一份后漂移。
"""

VERSION = "0.8.7"
