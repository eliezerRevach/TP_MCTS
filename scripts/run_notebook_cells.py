"""Run chosen code cells of a notebook, in order, in one shared namespace -- the same as clicking
those boxes one after the other in Jupyter, but from a shell (e.g. over ssh on the GCE VM).

    python scripts/run_notebook_cells.py experiments.ipynb --cells 1 3 7
    python scripts/run_notebook_cells.py experiments.ipynb --cells 1 3 19 20 --set WORKERS=12

--list           print every cell's number and first line, then stop
--set NAME=VALUE override a config variable after every cell, so it wins over the config box (a value
                 the config derives from it must be overridden too). VALUE is a Python literal; the
                 notebook file is never changed
--env NAME=VALUE set an environment variable after every cell (VALUE as text), so it wins over the
                 config box's os.environ lines, e.g. --env TP_MCTS_WILAO_PATTERN_GROWTH=llm_by_hand

display() falls back to print when IPython is not installed.
"""

import argparse
import ast
import json
import os
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook")
    ap.add_argument("--cells", type=int, nargs="*", default=[])
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE")
    ap.add_argument("--env", action="append", default=[], metavar="NAME=VALUE")
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

    env = {}
    for item in args.env:
        name, _, value = item.partition("=")
        env[name.strip()] = value.strip()

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
        first = src.strip().splitlines()[0][:90] if src.strip() else ""
        print(f"\n===== [cell {i}] {first}", flush=True)
        started = time.perf_counter()
        exec(compile(src, f"<cell {i}>", "exec"), namespace)
        if overrides:                                    # after every cell, so a config cell never wins
            namespace.update(overrides)
            print(f"[overrides] {overrides}", flush=True)
        if env:
            os.environ.update(env)
            print(f"[env] {env}", flush=True)
        print(f"===== [cell {i}] done in {time.perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
