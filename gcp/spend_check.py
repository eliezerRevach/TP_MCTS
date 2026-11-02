"""Daily spend check from the PC: latest budget readings, credits left, and whether the VM runs.

Pulls the budget notifications Google publishes to Pub/Sub (subscription tp-mcts-budget-monthly-check),
keeps the newest reading per budget in ~/.tp_mcts_spend.json, and prints one line per budget plus a
WARN line when something needs attention. Exit code 1 when there is a warning.

    python gcp/spend_check.py            # uses PROJECT_ID from gcp/config.env

Budgets (created 2026-11-02 on billing account 01CD31-CE9EEF-56776E, spend before credits):
  "tp-mcts monthly 300"             300 ILS a month; the VM's budget_guard.sh stops the VM at 100%
  "tp-mcts credits left under 200"  700 ILS since 2026-11-02 = credits 900 - 200; email only
"""
import base64
import datetime
import json
import os
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
STATE = pathlib.Path.home() / ".tp_mcts_spend.json"
SUBSCRIPTION = "tp-mcts-budget-monthly-check"
MONTHLY = "tp-mcts monthly 300"
CREDITS = "tp-mcts credits left under 200"
CREDITS_TOTAL = 900.0  # ILS at setup time
WARN_FRACTION = 0.8


def config():
    cfg = {}
    for line in (HERE / "config.env").read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            cfg[k.strip()] = v.strip()
    return cfg


def gcloud(*args):
    exe = "gcloud.cmd" if os.name == "nt" else "gcloud"
    return subprocess.run([exe, *args], capture_output=True, text=True)


def main():
    cfg = config()
    project = cfg["PROJECT_ID"]
    try:
        state = json.loads(STATE.read_text())
    except Exception:
        state = {}

    pulled = gcloud("pubsub", "subscriptions", "pull", SUBSCRIPTION, "--auto-ack", "--limit=500",
                    "--format=json", f"--project={project}")
    try:
        msgs = json.loads(pulled.stdout or "[]")
    except json.JSONDecodeError:
        msgs = []
    for m in msgs:
        try:
            d = json.loads(base64.b64decode(m["message"]["data"]))
        except Exception:
            continue
        name = d.get("budgetDisplayName")
        new = (d.get("costIntervalStart", ""), d.get("costAmount", 0))
        prev = state.get(name)
        if prev is None or new >= (prev.get("costIntervalStart", ""), prev.get("costAmount", 0)):
            state[name] = {k: d.get(k) for k in ("costAmount", "budgetAmount", "costIntervalStart", "currencyCode")}
            state[name]["seen"] = datetime.datetime.now().isoformat(timespec="minutes")
    STATE.write_text(json.dumps(state, indent=1))

    warns = []
    if pulled.returncode != 0:
        warns.append(f"could not pull budget notifications: {pulled.stderr.strip()[:200]}")
    month = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m")
    m = state.get(MONTHLY)
    if m and str(m.get("costIntervalStart", "")).startswith(month):
        cost, budget = float(m["costAmount"]), float(m["budgetAmount"])
        print(f"this month: {cost:.0f} / {budget:.0f} {m['currencyCode']} (as of {m['seen']})")
        if cost >= WARN_FRACTION * budget:
            warns.append(f"monthly spend {cost:.0f} of {budget:.0f} ILS (VM stops itself at 100%)")
    else:
        print("this month: no reading yet (Google sends one a few times a day once there is spend)")
    c = state.get(CREDITS)
    if c:
        left = CREDITS_TOTAL - float(c["costAmount"])
        print(f"credits left: ~{left:.0f} ILS (spent {float(c['costAmount']):.0f} since 2026-11-02, as of {c['seen']})")
        if left <= 200 + (1 - WARN_FRACTION) * 200:
            warns.append(f"credits left ~{left:.0f} ILS; under 200 the trial is nearly used up")

    vm = gcloud("compute", "instances", "list", f"--project={project}", "--format=value(name,status)")
    running = [l for l in vm.stdout.splitlines() if l.split() and l.split()[-1] == "RUNNING"]
    print("VMs running: " + (", ".join(l.split()[0] for l in running) if running else "none"))
    for w in warns:
        print("WARN " + w)
    return 1 if warns else 0


if __name__ == "__main__":
    sys.exit(main())
