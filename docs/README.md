# Documentation

Narrative docs for VietnameseASR. (The `README.md` and `CLAUDE.md` at the recipe
root stay there — `README.md` is the entry point, `CLAUDE.md` is auto-loaded by
Claude Code. Component docs live next to their code: `deploy/jetson_nano/README.md`,
`live_ui/README.md`, `local/tts/README.md`. Raw experiment output stays in
`results/*.md`.)

## Start here

| you want to… | read |
| --- | --- |
| **understand the project** (plain, with analogies) | [TEACHING_NOTES_VI.md](TEACHING_NOTES_VI.md) — tiếng Việt, dễ hiểu |
| the same, in full technical detail from theory to code (English) | [TEACHING_NOTES.md](TEACHING_NOTES.md) |
| the same, full detail + every config + code map (Vietnamese) | [TEACHING_NOTES_VI_chi_tiet.md](TEACHING_NOTES_VI_chi_tiet.md) |
| **the numbers** — WER, experiments, what's deployed | [RESULTS.md](RESULTS.md) |
| old experiments whose checkpoints were deleted | [ARCHIVED_EXPERIMENTS.md](ARCHIVED_EXPERIMENTS.md) |
| goals, design decisions, environment gotchas | [PROJECT_NOTES.md](PROJECT_NOTES.md) |
| the speaker-diarization / speaker-ID research | [DIARIZATION.md](DIARIZATION.md) |

## The three teaching notes — which one?

They cover the same material for different readers:

- **`TEACHING_NOTES_VI.md`** — short, plain Vietnamese with everyday analogies.
  Read this first if you just want to *understand* how it works.
- **`TEACHING_NOTES.md`** — full English reference, down to tensor shapes, loss
  math, current pipeline stages, and code file map. The canonical detailed
  version.
- **`TEACHING_NOTES_VI_chi_tiet.md`** — Vietnamese equivalent of the English one,
  plus the complete config-parameter tables and current code map.

Kept separate on purpose (different language / depth); not merged.

`_superseded/` now holds only historical redirect stubs. The old theory/code
content was merged into the active teaching notes above.
