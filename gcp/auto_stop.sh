#!/usr/bin/env bash
# Stop this VM once it has been idle for IDLE_MINUTES in a row, so a run left alone (PC off, chat
# closed) never keeps billing after it ends. Results stay on the disk (/opt/tp_mcts/results).
#
# Busy = a python process is working inside /opt/tp_mcts (an experiment, its workers, a notebook
# kernel), OR someone is connected over SSH (Cursor Remote-SSH, a terminal, a Claude check-in). Shutting down from inside = "stopped" on GCE: no CPU/RAM billing.
#
# Installed as a systemd service by gcp/install-auto-stop.sh (runs at every boot).
# Log: /var/log/tp-mcts-auto-stop.log

IDLE_MINUTES="${IDLE_MINUTES:-15}"
LOG=/var/log/tp-mcts-auto-stop.log
idle=0

log() { echo "$(date -Is) $*" >> "$LOG"; }

busy() {
  # a python process working inside the repo: experiments, their workers, notebook kernels
  # (by working folder, not command line -- this script's own command line names the repo)
  for pid in $(pgrep python); do
    case "$(readlink "/proc/$pid/cwd" 2>/dev/null)" in
      /opt/tp_mcts|/opt/tp_mcts/*) echo "job"; return 0 ;;
    esac
  done
  if [ "$(ss -Htn state established '( sport = :22 )' | wc -l)" -gt 0 ]; then
    echo "ssh"; return 0
  fi
  return 1
}

log "auto-stop started: shut down after ${IDLE_MINUTES} idle minutes"
while true; do
  sleep 60
  if why=$(busy); then
    [ "$idle" -gt 0 ] && log "busy again ($why) after ${idle} idle minutes"
    idle=0
  else
    idle=$((idle + 1))
    log "idle ${idle}/${IDLE_MINUTES}"
    if [ "$idle" -ge "$IDLE_MINUTES" ]; then
      log "idle for ${IDLE_MINUTES} minutes: shutting down"
      shutdown -h now
      exit 0
    fi
  fi
done
