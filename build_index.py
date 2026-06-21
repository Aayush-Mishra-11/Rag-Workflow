"""One-shot script: PDF -> chunks -> embeddings -> saved index.

Usage:
    python build_index.py                       # default PDF in cwd
    python build_index.py "path/to/other.pdf"   # explicit path
"""
from src.embed import build_index


if __name__ == "__main__":
    import sys

    pdf = sys.argv[1] if len(sys.argv) > 1 else "AWS Customer Agreement.pdf"
    build_index(pdf)