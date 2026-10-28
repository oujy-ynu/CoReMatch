#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_TIMESTAMP="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
RESULTS_DIR="$PROJECT_DIR/results"
TEST_ARGS=()
while (( $# )); do
  case "$1" in
    --output-dir)
      RESULTS_DIR="${2:?--output-dir requires a directory}"
      shift 2
      ;;
    --output-dir=*)
      RESULTS_DIR="${1#*=}"
      shift
      ;;
    --help|-h)
      exec bash "$PROJECT_DIR/scripts/test/test_aids.sh" --help
      ;;
    *)
      TEST_ARGS+=("$1")
      shift
      ;;
  esac
done
RESULT_CSVS=()
for entry in aids:AIDS700nef linux:LINUX imdb:IMDBMulti; do
  script_name="${entry%%:*}"
  dataset="${entry#*:}"
  output_dir="$RESULTS_DIR/${dataset}_${RUN_TIMESTAMP}"
  bash "$PROJECT_DIR/scripts/test/test_${script_name}.sh" \
    "${TEST_ARGS[@]}" --output-dir "$output_dir"
  RESULT_CSVS+=("$output_dir/test_result.csv")
done
"${PYTHON_BIN:-python}" -B - "$RESULTS_DIR/summary_${RUN_TIMESTAMP}.csv" "${RESULT_CSVS[@]}" <<'PY'
import csv
import sys
from pathlib import Path

destination = Path(sys.argv[1])
rows = []
for result_csv in sys.argv[2:]:
    with Path(result_csv).open(newline="") as stream:
        rows.extend(csv.DictReader(stream))
with destination.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(destination.read_text(), end="")
PY
