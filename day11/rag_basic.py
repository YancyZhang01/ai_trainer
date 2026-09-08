"""兼容旧入口，实际 RAG 流程位于 day11_rag/main.py。"""

import sys
from pathlib import Path


DAY11_RAG_DIR = Path(__file__).resolve().parent.parent / "day11_rag"
sys.path.insert(0, str(DAY11_RAG_DIR))

from main import main


if __name__ == "__main__":
    main()
