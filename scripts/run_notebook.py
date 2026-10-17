"""Execute the clean demo's Python cells without a Jupyter installation.

Usage from the repository root: python scripts/run_notebook.py
This runner is deliberately limited to the included plain-Python demo. It
does not implement IPython magics or write outputs back into the notebook.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    os.environ.setdefault("MPLBACKEND", "Agg")
    notebook = json.loads((root / "notebooks/cleaned_demo.ipynb").read_text(encoding="utf-8"))
    namespace = {"__name__": "__main__"}
    count = 0
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            exec(compile(source, f"cleaned_demo.ipynb:cell-{index}", "exec"), namespace)
            count += 1
    print(f"Executed {count} code cells successfully; notebook source remains unchanged.")


if __name__ == "__main__":
    main()
