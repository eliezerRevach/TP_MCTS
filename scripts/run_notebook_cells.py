"""Run chosen code cells of a notebook, in order, in one shared namespace -- the same as clicking
those boxes one after the other in Jupyter, but from a shell (e.g. over ssh on the GCE VM).

    python scripts/run_notebook_cells.py experiments.ipynb --cells 1 3 7
    python scripts/run_notebook_cells.py experiments.ipynb --cells 1 3 19 20 --set WORKERS=12

--list           print every cell's number and first line, then stop
--set NAME=VALUE after the cells before the first Script cell have run (setup + config), override a
                 config variable (VALUE is a Python literal); the notebook file is never changed

display() falls back to print when IPython is not installed.
"""

import argparse
import ast
import json
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook")
    ap.add_argument("--cells", type=int, nargs="*", default=[])
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE")
    args = ap.parse_args()
    for stream in (sys.stdout, sys.stderr):              # box-drawing / arrows on any console
        try:
            stream.reconfigure(errors="replace")
        except AttributeError:
            pass

    with open(args.notebook, encoding="utf-8") as fh:
        cells = json.load(fh)["cells"]
    if args.list:
        for i, c in enumerate(cells):
            src = "".join(c["source"]).strip()
            print(i, c["cell_type"], "|", src.splitlines()[0][:100] if src else "")
        return

    overrides = {}
    for item in args.set:
        name, _, value = item.partition("=")
        overrides[name.strip()] = ast.literal_eval(value.strip())

    try:
        from IPython.display import display
    except ImportError:
        def display(obj):
            print(obj.to_string() if hasattr(obj, "to_string") else obj)

    namespace = {"__name__": "__main__", "display": display}
    sys.argv = [args.notebook]
    for i in args.cells:
        cell = cells[i]
        if cell["cell_type"] != "code":
            print(f"[cell {i}] is {cell['cell_type']}, skipped", flush=True)
            continue
        src = "".join(cell["source"])
        if overrides and "_overrides_applied" not in namespace and "Shared config" not in src \
                and "Runtime:" not in src:
            namespace.update(overrides)                  # config has run: apply the overrides once
            namespace["_overrides_applied"] = True
            print(f"[overrides] {overrides}", flush=True)
        first = src.strip().splitlines()[0][:90] if src.strip() else ""
        print(f"\n===== [cell {i}] {first}", flush=True)
        started = time.perf_counter()
        exec(compile(src, f"<cell {i}>", "exec"), namespace)
        print(f"===== [cell {i}] done in {time.perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
