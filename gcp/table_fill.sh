#!/usr/bin/env bash
# Fill the TP-MCTS table (windows PDB vs PTRPG, + the LLM hand-picked windows PDB) on the VM.
# Two lanes, started together:
#   bash gcp/table_fill.sh prebuild   # ONE core: builds the windows PDB tables of the heavy rows into
#                                     # the cache, one (row, deadline) at a time; writes a marker per build
#   bash gcp/table_fill.sh mcts       # Script 3 (experiments.ipynb cells 1 3 19) per row: PTRPG rows first
#                                     # (no table needed), then each windows row once its marker is there
# Every setting comes from the notebook config (cell 3) except what --set / --env override below:
# the scenario (domain, objects, garbage, deadline), the heuristic, the output CSV, the pattern growth
# (pairs = automatic, llm_by_hand = hand_patterns.py) and the cache folder (fresh for this table).
# Outputs: results/table_fill/{csv,log,marks}/  -- one CSV + log per (row, heuristic, deadline).
set -u
cd /opt/tp_mcts
PY=.venv/bin/python
OUT=results/table_fill
CACHE=/opt/tp_mcts/results/wilao_cache_table
mkdir -p $OUT/csv $OUT/log $OUT/marks $CACHE
PREBUILD_TIMEOUT=${PREBUILD_TIMEOUT:-14400}     # seconds per (row, deadline) build
PREBUILD_MEM_KB=${PREBUILD_MEM_KB:-62914560}    # 60 GB address-space cap on a build

# row tag | domain | objects | garbage | deadlines
ROWS_LIGHT=(
  "simple10|simple|1|10|10 15 30"
  "simple11|simple|1|11|10 15 30"
  "simple12|simple|1|12|10 15 30"
  "simple13|simple|1|13|10 15 30"
  "simple15|simple|1|15|10 15 30"
  "conc|conc|1|0|10 15 30"
  "hosting1|hosting|1|1|10 15 30"
)
ROWS_HEAVY=(
  "nasa1|nasa_rover|1|0|25 35 50"
  "stuck2|stuck_car|2|0|10 20 30"
  "ms2|machine_shop|2|0|17 27 40"
  "ms3|machine_shop|3|0|17 27 40"
  "nasa3|nasa_rover|3|0|25 35 50"
  "ms4|machine_shop|4|0|17 27 40"
)
# windows rows that need a prebuilt table: tag|growth  (nasa2 llm = the LLM column at D 35, 50)
WINDOWS_HEAVY=(
  "nasa1|pairs" "nasa1|llm_by_hand" "nasa2|llm_by_hand"
  "stuck2|pairs" "stuck2|llm_by_hand"
  "ms2|pairs" "ms2|llm_by_hand"
  "ms3|pairs" "ms3|llm_by_hand"
  "nasa3|pairs" "nasa3|llm_by_hand"
  "ms4|pairs" "ms4|llm_by_hand"
)

row_of() {   # tag -> "domain|objects|garbage|deadlines"
  local tag=$1 r
  [ "$tag" = nasa2 ] && { echo "nasa_rover|2|0|35 50"; return; }
  for r in "${ROWS_LIGHT[@]}" "${ROWS_HEAVY[@]}"; do
    [ "${r%%|*}" = "$tag" ] && { echo "${r#*|}"; return; }
  done
}

# the notebook's windows settings (cell 3), so a prebuilt table has the key the workers look up
wilao_env() {
  export PYTHONPATH=/opt/tp_mcts PYTHONIOENCODING=utf-8
  export TP_MCTS_WILAO_OFFLINE_SECONDS=0 TP_MCTS_WILAO_OFFLINE_EXTEND=1 TP_MCTS_WILAO_PATTERN_GOALS=1
  export TP_MCTS_WILAO_PATTERN_FACTS=10 TP_MCTS_WILAO_MISS=one TP_MCTS_WILAO_QUERY_SECONDS=0.05
  export TP_MCTS_WILAO_QUERY_EXPANSIONS=200 TP_MCTS_WILAO_AGG=same TP_MCTS_WILAO_GROUPING=1
  export TP_MCTS_WILAO_NO_STN_PASS=0 TP_MCTS_WILAO_BLUR=3 TP_MCTS_WILAO_EXACT_STATICS=1
  export TP_MCTS_WILAO_LOGIC_SOURCES=random TP_MCTS_WILAO_LOGIC_ROLLOUTS=20 TP_MCTS_WILAO_LOGIC_DEPTH=60
  export TP_MCTS_WILAO_LOGIC_RANK=frequency TP_MCTS_WILAO_TABLE_CACHE=$CACHE
}

script3() {  # tag domain objects garbage deadline heuristic growth
  local tag=$1 dom=$2 obj=$3 garb=$4 dl=$5 heur=$6 growth=$7
  local name=${tag}_${heur}_${growth}_d${dl}
  [ -f $OUT/csv/$name.csv ] && { echo "[skip] $name (csv exists)"; return; }
  echo "[$(date +%H:%M:%S)] start $name"
  $PY -u scripts/run_notebook_cells.py experiments.ipynb --cells 1 3 19 \
    --set "DOMAINS=['$dom']" --set "OBJECTS=[$obj]" --set "GARBAGE_AMOUNTS=[$garb]" --set "DEADLINES=[$dl]" \
    --set "HEURISTICS=['$heur']" --set "SCRIPT3_CSV='/opt/tp_mcts/$OUT/csv/$name.tmp.csv'" \
    --env TP_MCTS_WILAO_PATTERN_GROWTH=$growth --env TP_MCTS_WILAO_TABLE_CACHE=$CACHE \
    > $OUT/log/$name.log 2>&1 < /dev/null
  local rc=$?
  if [ -f $OUT/csv/$name.tmp.csv ]; then mv $OUT/csv/$name.tmp.csv $OUT/csv/$name.csv; fi
  echo "[$(date +%H:%M:%S)] done  $name rc=$rc $(grep -o 'success=[0-9]*/[0-9]*' $OUT/log/$name.log | tail -1)"
}

lane_prebuild() {
  local item tag growth row dom obj garb dls dl mark
  for item in "${WINDOWS_HEAVY[@]}"; do
    tag=${item%%|*}; growth=${item#*|}
    IFS='|' read -r dom obj garb dls <<< "$(row_of $tag)"
    for dl in $dls; do
      mark=$OUT/marks/${tag}_${growth}_d${dl}
      [ -f $mark.ok ] || [ -f $mark.fail ] && continue
      echo "[$(date +%H:%M:%S)] prebuild $tag $growth d$dl"
      ( wilao_env; export TP_MCTS_WILAO_PATTERN_GROWTH=$growth; ulimit -v $PREBUILD_MEM_KB
        timeout $PREBUILD_TIMEOUT $PY -u scripts/prebuild_wilao_tables.py --domain $dom --object_amount $obj \
          --garbage_amount $garb --deadlines $dl ) > $mark.log 2>&1 < /dev/null
      if grep -q '^\[prebuild\]' $mark.log; then touch $mark.ok; else touch $mark.fail; fi
      echo "[$(date +%H:%M:%S)]   -> $(ls $mark.ok $mark.fail 2>/dev/null)"
    done
  done
}

lane_mcts() {
  local r tag dom obj garb dls dl item growth mark
  for r in "${ROWS_LIGHT[@]}"; do          # light rows: tables build in seconds inside the run
    IFS='|' read -r tag dom obj garb dls <<< "$r"
    for dl in $dls; do
      script3 $tag $dom $obj $garb $dl ptrpg_old pairs
      script3 $tag $dom $obj $garb $dl windows_ilao_pdb pairs
    done
  done
  for r in "${ROWS_HEAVY[@]}"; do          # PTRPG of the heavy rows while their tables build
    IFS='|' read -r tag dom obj garb dls <<< "$r"
    for dl in $dls; do script3 $tag $dom $obj $garb $dl ptrpg_old pairs; done
  done
  for item in "${WINDOWS_HEAVY[@]}"; do    # windows rows once their table is built
    tag=${item%%|*}; growth=${item#*|}
    IFS='|' read -r dom obj garb dls <<< "$(row_of $tag)"
    for dl in $dls; do
      mark=$OUT/marks/${tag}_${growth}_d${dl}
      while [ ! -f $mark.ok ] && [ ! -f $mark.fail ]; do sleep 30; done
      if [ -f $mark.fail ]; then echo "[skip] $tag $growth d$dl: table build failed ($mark.log)"; continue; fi
      script3 $tag $dom $obj $garb $dl windows_ilao_pdb $growth
    done
  done
  echo "[$(date +%H:%M:%S)] mcts lane finished"
}

case "${1:-}" in
  prebuild) lane_prebuild ;;
  mcts) lane_mcts ;;
  *) echo "usage: $0 prebuild|mcts"; exit 2 ;;
esac
