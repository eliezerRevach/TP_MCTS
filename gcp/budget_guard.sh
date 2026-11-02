#!/usr/bin/env bash
# Stop this VM when the month's spend reaches the monthly budget, so a forgotten run can never
# outspend the cap. GCP budgets only alert; this is what turns the alert into a stop.
#
# Reads the budget notifications Google publishes to Pub/Sub (several times a day) from the
# subscription tp-mcts-budget-vm-guard. Each message carries costAmount, budgetAmount and
# costIntervalStart for one budget. The last value per budget is kept in STATE, so a VM started
# again in an over-budget month shuts down at once, before a new notification arrives.
# Only CAP_BUDGET stops the VM; the "credits left under 200" budget only emails.
#
# Needs the VM created with --scopes=cloud-platform (default compute SA can pull Pub/Sub).
# Installed as a systemd service by gcp/install-budget-guard.sh. Log: /var/log/tp-mcts-budget-guard.log

SUBSCRIPTION="${SUBSCRIPTION:-tp-mcts-budget-vm-guard}"
CAP_BUDGET="${CAP_BUDGET:-tp-mcts monthly 300}"
CHECK_MINUTES="${CHECK_MINUTES:-10}"
STATE=/var/lib/tp-mcts-budget-guard.json
LOG=/var/log/tp-mcts-budget-guard.log

log() { echo "$(date -Is) $*" >> "$LOG"; }

# reads pulled messages on stdin, updates STATE, prints "over|ok <cost> <budget>" for CAP_BUDGET
read -r -d '' PARSE <<'EOF'
import base64, json, sys, datetime
state_path, cap = sys.argv[1], sys.argv[2]
try:
    state = json.load(open(state_path))
except Exception:
    state = {}
try:
    msgs = json.load(sys.stdin)
except Exception:
    msgs = []
for m in msgs:
    try:
        d = json.loads(base64.b64decode(m["message"]["data"]))
    except Exception:
        continue
    name = d.get("budgetDisplayName")
    prev = state.get(name)
    # keep the newest reading; within a month cost only grows
    if prev is None or (d.get("costIntervalStart", ""), d.get("costAmount", 0)) >= (prev.get("costIntervalStart", ""), prev.get("costAmount", 0)):
        state[name] = {k: d.get(k) for k in ("costAmount", "budgetAmount", "costIntervalStart", "currencyCode")}
json.dump(state, open(state_path, "w"))
r = state.get(cap)
month = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m")
if not r or not str(r.get("costIntervalStart", "")).startswith(month):
    print("ok 0 ?")
else:
    cost, budget = float(r["costAmount"]), float(r["budgetAmount"])
    print(("over" if cost >= budget else "ok"), cost, budget)
EOF

check() {
  gcloud pubsub subscriptions pull "$SUBSCRIPTION" --auto-ack --limit=100 --format=json 2>>"$LOG" \
    | python3 -c "$PARSE" "$STATE" "$CAP_BUDGET"
}

log "budget guard started: stop when '${CAP_BUDGET}' is reached (check every ${CHECK_MINUTES} min)"
while true; do
  read -r verdict cost budget <<< "$(check)"
  if [ "$verdict" = "over" ]; then
    log "monthly spend ${cost} >= budget ${budget}: shutting down"
    shutdown -h now
    exit 0
  fi
  sleep $((CHECK_MINUTES * 60))
done
