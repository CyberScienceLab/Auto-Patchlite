#!/usr/bin/env bash
set -euo pipefail
MODEL="ollama"
MODEL_NAME="phi3:mini"
OUTDIR=""
MASTER_REPORT="phi3_all_prompts.report.jsonl"
PROBE_FILE="garak_all_probes.txt"

# Minimal detectors that are unlikely to crash (keeps everything you want logged).
DETECTORS="always.Pass,continuation.Continuation,divergence.RepeatDiverges,dan.*,promptinject.*,malwaregen.*,exploitation.*,encoding.*,xss.*,lmrc.*,unsafe_content.*,mitigation.*,shields.*"

# Clear master file (or comment this line if you want to append)
: > "$MASTER_REPORT"

while IFS= read -r PROBE || [ -n "$PROBE" ]; do
  # skip blank / comment lines
  [[ -z "$PROBE" || "$PROBE" =~ ^# ]] && continue

  # sanitize probe name for filename
  SAFE_NAME=$(echo "$PROBE" | tr '/:.* ' '_' | tr -c 'A-Za-z0-9_-' '_')
  PREFIX="phi3_${SAFE_NAME}_$(date +%s)"

  echo "=== running probe: $PROBE → report prefix: $PREFIX ==="

  # run garak for this single probe (capture stdout/stderr into a run-specific log)
  garak -m "$MODEL" -n "$MODEL_NAME" \
    --probes "$PROBE" \
    --detectors "$DETECTORS" \
    --report_prefix "$PREFIX" 2>&1 | tee "$HOME/garak_run_${SAFE_NAME}.log" || true

  # find the generated report file (garak names it <prefix>.report.jsonl in garak_runs)
  REPORT_PATH="$(ls -1t "$OUTDIR" | grep "^${PREFIX}\.report\.jsonl$" | head -n1 || true)"
  if [[ -n "$REPORT_PATH" ]]; then
    REPORT_FULL="$OUTDIR/$REPORT_PATH"
    # append to master report
    cat "$REPORT_FULL" >> "$MASTER_REPORT"
  else
    echo "WARN: report not found for $PROBE (prefix $PREFIX). Check log: $HOME/garak_run_${SAFE_NAME}.log"
  fi

  # short sleep so Ollama/daemon isn't hammered (tweak as needed)
  sleep 0.5
done < "$PROBE_FILE"

echo "Done. Combined hitlog: $MASTER_REPORT"