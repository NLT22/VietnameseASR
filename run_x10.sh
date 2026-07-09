#!/usr/bin/env bash
# Backward-compatible wrapper. run.sh is now the single unified runner and
# already defaults to the x10 matched-split pipeline, so this just forwards all
# arguments. Prefer calling run.sh directly:  bash run.sh [...]
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$here/run.sh" "$@"
