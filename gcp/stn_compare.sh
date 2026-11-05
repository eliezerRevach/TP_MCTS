#!/usr/bin/env bash
# The STN rules (windows_lao.py, "rules = stn") against the windows PDB of the table (blur 3), on the small rows:
#   win3  : TP_MCTS_WILAO_RULES=windows, blur 3   (the benchmark table's setting)
#   stn   : TP_MCTS_WILAO_RULES=stn,     blur 0   (the state keeps the path's STN)
#   stnlb : TP_MCTS_WILAO_RULES=stn_lb,  blur 0   (the STN only gives the LB: one state per (F, Q))
# Phase 1 builds every (row, config) table into its own cache, all at once (build time, states, h(s0));
# phase 2 runs Script 3 (experiments.ipynb cells 1 3 19: 100 runs, 12 workers) one scenario at a time, so
# every config runs under the same load. The table settings are the notebook's (as gcp/table_fill.sh).
#   bash gcp/stn_compare.sh   -> results/stn_compare/{prebuild,log,csv}/ and results/stn_compare/summary.txt
set -u
cd /opt/tp_mcts
PY=.venv/bin/python
OUT=results/stn_compare
mkdir -p $OUT/prebuild $OUT/log $OUT/csv
PREBUILD_TIMEOUT=${PREBUILD_TIMEOUT:-7200}
MEM_KB=${MEM_KB:-31457280}                      # 30 GB address-space cap per build

ROWS=("stuck_car|1|0|10" "stuck_car|1|0|15" "prob_conc|1|7|10" "prob_conc|1|7|15" "nasa_rover|1|0|25")
CONFIGS=("win3|windows|3" "stn|stn|0" "stnlb|stn_lb|0")

wilao_env() {   # rules blur cache
  export PYTHONPATH=/opt/tp_mcts PYTHONIOENCODING=utf-8
  export TP_MCTS_WILAO_OFFLINE_SECONDS=0 TP_MCTS_WILAO_OFFLINE_EXTEND=1 TP_MCTS_WILAO_PATTERN_GOALS=1
  export TP_MCTS_WILAO_PATTERN_FACTS=10 TP_MCTS_WILAO_MISS=one TP_MCTS_WILAO_QUERY_SECONDS=0.05
  export TP_MCTS_WILAO_QUERY_EXPANSIONS=200 TP_MCTS_WILAO_AGG=same TP_MCTS_WILAO_GROUPING=1
  export TP_MCTS_WILAO_NO_STN_PASS=0 TP_MCTS_WILAO_EXACT_STATICS=1
  export TP_MCTS_WILAO_LOGIC_SOURCES=random TP_MCTS_WILAO_LOGIC_ROLLOUTS=20 TP_MCTS_WILAO_LOGIC_DEPTH=60
  export TP_MCTS_WILAO_LOGIC_RANK=frequency TP_MCTS_WILAO_PATTERN_GROWTH=pairs
  export TP_MCTS_WILAO_RULES=$1 TP_MCTS_WILAO_BLUR=$2 TP_MCTS_WILAO_TABLE_CACHE=$3
}

prebuild() {   # domain obj garb dl tag rules blur
  local name=$1_$2_$3_d$4_$5
  ( wilao_env $6 $7 /opt/tp_mcts/$OUT/cache_$5; ulimit -v $MEM_KB
    timeout $PREBUILD_TIMEOUT $PY -u scripts/prebuild_wilao_tables.py --domain $1 --object_amount $2 \
      --garbage_amount $3 --deadlines $4 ) > $OUT/prebuild/$name.log 2>&1 < /dev/null
  echo "[$(date +%H:%M:%S)] prebuild $name: $(grep -c '^\[prebuild\]' $OUT/prebuild/$name.log) line(s)"
}

script3() {   # domain obj garb dl tag rules blur
  local name=$1_$2_$3_d$4_$5
  [ -f $OUT/csv/$name.csv ] && { echo "[skip] $name (csv exists)"; return; }
  grep -q '^\[prebuild\]' $OUT/prebuild/$name.log 2>/dev/null || { echo "[skip] $name (no table)"; return; }
  echo "[$(date +%H:%M:%S)] mcts $name"
  $PY -u scripts/run_notebook_cells.py experiments.ipynb --cells 1 3 19 \
    --set "DOMAINS=['$1']" --set "OBJECTS=[$2]" --set "GARBAGE_AMOUNTS=[$3]" --set "DEADLINES=[$4]" \
    --set "HEURISTICS=['windows_ilao_pdb']" --set "SCRIPT3_CSV='/opt/tp_mcts/$OUT/csv/$name.tmp.csv'" \
    --env TP_MCTS_WILAO_RULES=$6 --env TP_MCTS_WILAO_BLUR=$7 --env TP_MCTS_WILAO_PATTERN_GROWTH=pairs \
    --env TP_MCTS_WILAO_TABLE_CACHE=/opt/tp_mcts/$OUT/cache_$5 \
    > $OUT/log/$name.log 2>&1 < /dev/null
  [ -f $OUT/csv/$name.tmp.csv ] && mv $OUT/csv/$name.tmp.csv $OUT/csv/$name.csv
  echo "[$(date +%H:%M:%S)] done $name $(grep -o 'success=[0-9]*/[0-9]*' $OUT/log/$name.log | tail -1)"
}

echo "[$(date +%H:%M:%S)] phase 1: tables"
for row in "${ROWS[@]}"; do
  IFS='|' read -r dom obj garb dl <<< "$row"
  for cfg in "${CONFIGS[@]}"; do
    IFS='|' read -r tag rules blur <<< "$cfg"
    prebuild $dom $obj $garb $dl $tag $rules $blur &
  done
done
wait
echo "[$(date +%H:%M:%S)] phase 2: TP-MCTS"
for row in "${ROWS[@]}"; do
  IFS='|' read -r dom obj garb dl <<< "$row"
  for cfg in "${CONFIGS[@]}"; do
    IFS='|' read -r tag rules blur <<< "$cfg"
    script3 $dom $obj $garb $dl $tag $rules $blur
  done
done

$PY - <<'PY' > $OUT/summary.txt
import csv, glob, json, os
OUT = "results/stn_compare"
print("row | config | build s | table states | h(s0) | score | wall s")
for log in sorted(glob.glob(f"{OUT}/prebuild/*.log")):
    name = os.path.basename(log)[:-4]
    pre = [l for l in open(log) if l.startswith("[prebuild]")]
    b = json.loads(pre[-1][len("[prebuild] "):]) if pre else None
    states = sum(x for x in (b["report"].get("table_states") or []) if x) if b else None
    rows = list(csv.DictReader(open(f"{OUT}/csv/{name}.csv"))) if os.path.exists(f"{OUT}/csv/{name}.csv") else []
    score = f'{rows[0]["amount_success"]}/{rows[0]["runs_total"]}' if rows else "-"
    wall = rows[0]["wall_time_sec"] if rows else "-"
    row, tag = name.rsplit("_", 1)
    print(f"{row} | {tag} | {b['build_seconds'] if b else 'failed'} | {states} | {b['h_s0'] if b else '-'} | {score} | {wall}")
PY
cat $OUT/summary.txt
echo "[$(date +%H:%M:%S)] all done"
