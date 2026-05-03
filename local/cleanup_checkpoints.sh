#!/usr/bin/env bash
set -euo pipefail

keep_epochs="30 50"
dry_run=0
delete_global_checkpoints=1

usage() {
  cat <<EOF
Usage: $0 [--keep-epochs "30 50"] [--dry-run] [--no-delete-global-checkpoints] EXP_DIR...

Deletes checkpoint files to save disk space.
Keeps:
  - epoch-N.pt where N is in --keep-epochs
  - best-*.pt
  - pretrained.pt
  - averaged*.pt
  - jit_script.pt

By default it also deletes checkpoint-*.pt files.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --keep-epochs)
      keep_epochs="$2"
      shift 2
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    --no-delete-global-checkpoints)
      delete_global_checkpoints=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
    *)
      break
      ;;
  esac
done

if [ "$#" -eq 0 ]; then
  usage >&2
  exit 1
fi

should_keep_epoch() {
  local epoch="$1"
  local keep
  for keep in ${keep_epochs}; do
    if [ "${epoch}" = "${keep}" ]; then
      return 0
    fi
  done
  return 1
}

delete_file() {
  local path="$1"
  if [ "${dry_run}" = "1" ]; then
    printf "DRY-RUN delete %s\n" "${path}"
  else
    rm -f -- "${path}"
    printf "Deleted %s\n" "${path}"
  fi
}

for exp_dir in "$@"; do
  if [ ! -d "${exp_dir}" ]; then
    echo "Skip missing exp dir: ${exp_dir}" >&2
    continue
  fi

  while IFS= read -r -d '' path; do
    filename="$(basename -- "${path}")"
    case "${filename}" in
      epoch-*.pt)
        epoch="${filename#epoch-}"
        epoch="${epoch%.pt}"
        if should_keep_epoch "${epoch}"; then
          printf "Keep %s\n" "${path}"
        else
          delete_file "${path}"
        fi
        ;;
      checkpoint-*.pt)
        if [ "${delete_global_checkpoints}" = "1" ]; then
          delete_file "${path}"
        else
          printf "Keep %s\n" "${path}"
        fi
        ;;
      best-*.pt|pretrained.pt|averaged*.pt|jit_script.pt)
        printf "Keep %s\n" "${path}"
        ;;
    esac
  done < <(
    find "${exp_dir}" -maxdepth 1 -type f \
      \( -name 'epoch-*.pt' -o -name 'checkpoint-*.pt' -o -name 'best-*.pt' -o -name 'pretrained.pt' -o -name 'averaged*.pt' -o -name 'jit_script.pt' \) \
      -print0
  )
done
