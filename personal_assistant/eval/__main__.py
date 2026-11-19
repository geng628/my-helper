"""支持 `python -m personal_assistant.eval` 直接运行评测命令行。"""
from personal_assistant.eval.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
