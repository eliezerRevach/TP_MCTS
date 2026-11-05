#!/usr/bin/env bash
# The full STN rules (TP_MCTS_WILAO_RULES=stn, blur 0) on SMALLER hand-picked patterns, against the windows PDB, on
# nasa_rover 1 object at D 25 -- so the STN's table builds in about the windows table's time, and TP-MCTS decides:
#   win3_pairs  : windows, blur 3, the automatic pairs patterns (the benchmark table's setting)
#   <tag>_<growth> for growth in llm_by_hand_nasa_one_rock | llm_by_hand_nasa_one_joint (hand_patterns.py):
#       stn  : the STN rules, blur 0      win0 : windows, blur 0      win3 : windows, blur 3
# Phase 1 builds every table (windows ones together, the two STN ones one after the other, 40 GB each);
# phase 2 runs Script 3 (experiments.ipynb cells 1 3 19, 100 runs) with 6 workers -- 12 workers x ~10 GB of nasa
# tables ran the 96 GB VM out of memory -- one scenario at a time, so every config runs under the same load.
#   bash gcp/stn_small.sh   -> results/stn_small/{prebuild,log,csv}/ and results/stn_small/summary.txt
set -u
cd /opt/tp_mcts
PY=.venv/bin/python
OUT=results/stn_small
mkdir -p $OUT/prebuild $OUT/log $OUT/csv
DOM=nasa_rover; OBJ=1; GARB=0; DL=${STN_SMALL_DEADLINE:-25}
WORKERS=${STN_SMALL_WORKERS:-6}
TIMECMD=""; [ -x /usr/bin/time ] && TIMECMD="/usr/bin/time -v"

wilao_env() {   # rules blur growth cache
  export PYTHONPATH=/opt/tp_mcts PYTHONIOENCODING=utf-8
  export TP_MCTS_WILAO_OFFLINE_SECONDS=0 TP_MCTS_WILAO_OFFLINE_EXTEND=1 TP_MCTS_WILAO_PATTERN_GOALS=1
  export TP_MCTS_WILAO_PATTERN_FACTS=10 TP_MCTS_WILAO_MISS=one TP_MCTS_WILAO_QUERY_SECONDS=0.05
  export TP_MCTS_WILAO_QUERY_EXPANSIONS=200 TP_MCTS_WILAO_AGG=same TP_MCTS_WILAO_GROUPING=1
  export TP_MCTS_WILAO_NO_STN_PASS=0 TP_MCTS_WILAO_EXACT_STATICS=1
  export TP_MCTS_WILAO_LOGIC_SOURCES=random TP_MCTS_WILAO_LOGIC_ROLLOUTS=20 TP_MCTS_WILAO_LOGIC_DEPTH=60
  export TP_MCTS_WILAO_LOGIC_RANK=frequency
  export TP_MCTS_WILAO_RULES=$1 TP_MCTS_WILAO_BLUR=$2 TP_MCTS_WILAO_PATTERN_GROWTH=$3 TP_MCTS_WILAO_TABLE_CACHE=$4
}

# name | rules | blur | growth
CONFIGS=("win3_pairs|windows|3|pairs"
         "stn_rock|stn|0|llm_by_hand_nasa_one_rock" "win0_rock|windows|0|llm_by_hand_nasa_one_rock"
         "win3_rock|windows|3|llm_by_hand_nasa_one_rock"
         "stn_joint|stn|0|llm_by_hand_nasa_one_joint" "win0_joint|windows|0|llm_by_hand_nasa_one_joint"
         "win3_joint|windows|3|llm_by_hand_nasa_one_joint")

prebuild() {   # name rules blur growth mem_kb timeout
  ( wilao_env $2 $3 $4 /opt/tp_mcts/$OUT/cache_$1; ulimit -v $5
    $TIMECMD timeout $6 $PY -u scripts/prebuild_wilao_tables.py --domain $DOM --object_amount $OBJ \
      --garbage_amount $GARB --deadlines $DL ) > $OUT/prebuild/$1.log 2>&1 < /dev/null
  echo "[$(date +%H:%M:%S)] prebuild $1: $(grep -c '^\[prebuild\]' $OUT/prebuild/$1.log) line(s)," \
       "max RSS $(grep -o 'Maximum resident set size (kbytes): [0-9]*' $OUT/prebuild/$1.log | grep -o '[0-9]*$') kB"
}

script3() {   # name rules blur growth
  [ -f $OUT/csv/$1.csv ] && { echo "[skip] $1 (csv exists)"; return; }
  grep -q '^\[prebuild\]' $OUT/prebuild/$1.log 2>/dev/null || { echo "[skip] $1 (no table)"; return; }
  echo "[$(date +%H:%M:%S)] mcts $1"
  $PY -u scripts/run_notebook_cells.py experiments.ipynb --cells 1 3 19 \
    --set "DOMAINS=['$DOM']" --set "OBJECTS=[$OBJ]" --set "GARBAGE_AMOUNTS=[$GARB]" --set "DEADLINES=[$DL]" \
    --set "HEURISTICS=['windows_ilao_pdb']" --set "WORKERS=$WORKERS" \
    --set "SCRIPT3_CSV='/opt/tp_mcts/$OUT/csv/$1.tmp.csv'" \
    --env TP_MCTS_WILAO_RULES=$2 --env TP_MCTS_WILAO_BLUR=$3 --env TP_MCTS_WILAO_PATTERN_GROWTH=$4 \
    --env TP_MCTS_WILAO_TABLE_CACHE=/opt/tp_mcts/$OUT/cache_$1 \
    > $OUT/log/$1.log 2>&1 < /dev/null
  [ -f $OUT/csv/$1.tmp.csv ] && mv $OUT/csv/$1.tmp.csv $OUT/csv/$1.csv
  echo "[$(date +%H:%M:%S)] done $1 $(grep -o 'success=[0-9]*/[0-9]*' $OUT/log/$1.log | tail -1)"
}

echo "[$(date +%H:%M:%S)] phase 1: tables (D $DL)"
for cfg in "${CONFIGS[@]}"; do
  IFS='|' read -r name rules blur growth <<< "$cfg"
  [ "$rules" = windows ] && prebuild $name $rules $blur $growth 16777216 3600 &
done
( for cfg in "${CONFIGS[@]}"; do
    IFS='|' read -r name rules blur growth <<< "$cfg"
    [ "$rules" = stn ] && prebuild $name $rules $blur $growth 41943040 3600
  done ) &
wait
echo "[$(date +%H:%M:%S)] phase 2: TP-MCTS ($WORKERS workers)"
for cfg in "${CONFIGS[@]}"; do
  IFS='|' read -r name rules blur growth <<< "$cfg"
  script3 $name $rules $blur $growth
done

$PY - <<'PY' > $OUT/summary.txt
import csv, glob, json, os
OUT = "results/stn_small"
print("config | build s | table states (per pattern) | h(s0) | score | wall s")
for log in sorted(glob.glob(f"{OUT}/prebuild/*.log")):
    name = os.path.basename(log)[:-4]
    pre = [l for l in open(log) if l.startswith("[prebuild]")]
    b = json.loads(pre[-1][len("[prebuild] "):]) if pre else None
    ts = b["report"].get("table_states") if b else None
    rows = list(csv.DictReader(open(f"{OUT}/csv/{name}.csv"))) if os.path.exists(f"{OUT}/csv/{name}.csv") else []
    score = f'{rows[0]["amount_success"]}/{rows[0]["runs_total"]}' if rows else "-"
    wall = rows[0]["wall_time_sec"] if rows else "-"
    print(f"{name} | {b['build_seconds'] if b else 'failed'} | {ts} | {b['h_s0'] if b else '-'} | {score} | {wall}")
PY
cat $OUT/summary.txt
echo "[$(date +%H:%M:%S)] all done"
