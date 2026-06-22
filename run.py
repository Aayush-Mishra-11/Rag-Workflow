"""Convenience launcher: `python run.py` starts uvicorn."""
from __future__ import annotations

import os
import sys


def main() -> None:
    import uvicorn

    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    print(f"Starting Vestaff RAG on http://{host}:{port}")
    print("Press Ctrl+C to stop.")
    uvicorn.run("src.api:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
