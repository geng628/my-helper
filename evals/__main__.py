"""支持 `python -m evals` 直接运行评测命令行。"""
from evals.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
