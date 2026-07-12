# Superseded: theory/code practical notes

This file used to contain the practical code walkthrough for the ASR pipeline.
Its useful content has been merged into the active teaching notes:

- `docs/TEACHING_NOTES.md` — English canonical theory-to-code reference.
- `docs/TEACHING_NOTES_VI_chi_tiet.md` — Vietnamese detailed theory-to-code reference.

Do not use the old version from git history as an implementation guide. It
mentioned removed CTC/JIT files and an older export flow. The current path is:

```text
run.sh stage -2..16
  -> transcripts_main/
  -> data/manifests_main/
  -> fbank_main/
  -> ASR/zipformer/train.py
  -> local/export_for_jetson.sh
  -> deploy/jetson_nano/model_medium_epoch30_avg10/
```

The active model is a streaming Zipformer-Transducer exported as ONNX
encoder/decoder/joiner graphs and decoded with the custom classic beam search.
