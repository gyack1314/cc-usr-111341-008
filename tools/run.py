#!/usr/bin/env python3 -I
"""隔离模式启动器：显式定位仓库根目录后转交 patent_data.cli。

用 -I 运行以避免从 CWD 加载非受信模块；路径在脚本内基于 __file__ 解析。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from patent_data.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
