#!/usr/bin/env bash
# Layers vs intervals (TP_MCTS_WILAO_VI) on nasa_rover 2 and 3 objects: the table's windows settings at blur
# VI_BLUR (3), Auto (pairs) and LLM (llm_by_hand) patterns, deadline VI_DEADLINE (25). Each run builds the
# forward graph once and times both backward passes on it, comparing every cell (scripts/compare_wilao_vi.py).
# No table cache, so every table is really built.
#   bash gcp/vi_compare.sh      -> results/vi_compare/<tag>.log, one [vi] line per table + a [vi-total] line
set -u
cd /opt/tp_mcts
PY=.venv/bin/python
OUT=results/vi_compare
BLUR=${VI_BLUR:-3}
DL=${VI_DEADLINE:-25}
MEM_KB=${VI_MEM_KB:-41943040}                   # 40 GB address-space cap per run
mkdir -p $OUT

run() {   # tag growth objects
  ( export PYTHONPATH=/opt/tp_mcts PYTHONIOENCODING=utf-8
    export TP_MCTS_WILAO_OFFLINE_SECONDS=0 TP_MCTS_WILAO_OFFLINE_EXTEND=1 TP_MCTS_WILAO_PATTERN_GOALS=1
    export TP_MCTS_WILAO_PATTERN_FACTS=10 TP_MCTS_WILAO_MISS=one TP_MCTS_WILAO_QUERY_SECONDS=0.05
    export TP_MCTS_WILAO_QUERY_EXPANSIONS=200 TP_MCTS_WILAO_AGG=same TP_MCTS_WILAO_GROUPING=1
    export TP_MCTS_WILAO_NO_STN_PASS=0 TP_MCTS_WILAO_BLUR=$BLUR TP_MCTS_WILAO_EXACT_STATICS=1
    export TP_MCTS_WILAO_LOGIC_SOURCES=random TP_MCTS_WILAO_LOGIC_ROLLOUTS=20 TP_MCTS_WILAO_LOGIC_DEPTH=60
    export TP_MCTS_WILAO_LOGIC_RANK=frequency TP_MCTS_WILAO_PATTERN_GROWTH=$2 TP_MCTS_WILAO_REPORT=0
    unset TP_MCTS_WILAO_TABLE_CACHE
    ulimit -v $MEM_KB
    exec $PY -u scripts/compare_wilao_vi.py --domain nasa_rover --object_amount $3 --deadline $DL
  ) > $OUT/$1_d$DL.log 2>&1 < /dev/null
  echo "[$(date +%H:%M:%S)] $1 d$DL finished: $(grep -c '^\[vi-total\]' $OUT/$1_d$DL.log) total line(s)"
}

echo "[$(date +%H:%M:%S)] start: blur $BLUR, deadline $DL"
run nasa2_pairs pairs 2 &
run nasa2_llm llm_by_hand 2 &
run nasa3_pairs pairs 3 &
run nasa3_llm llm_by_hand 3 &
wait
echo "[$(date +%H:%M:%S)] all done"
