#!/usr/bin/env python3
"""
Real-time microphone ASR với Silero VAD.

Usage:
    source ~/test-icefall/bin/activate
    cd /home/trung/icefall/egs/vi_asr_corpus

    # Streaming mode (low latency, partial results while speaking):
    python3 mic_streaming_asr.py --mode streaming

    # Non-streaming mode (higher accuracy, result after each utterance):
    python3 mic_streaming_asr.py --mode full

Press Ctrl+C to stop.
"""

import sys
import queue
import argparse
import time
import torch
import soundfile as sf
import sounddevice as sd
import k2
from lhotse import Fbank, FbankConfig
from silero_vad import load_silero_vad

# ── Model paths ───────────────────────────────────────────────────────────────
STREAMING_MODEL = "ASR/zipformer/exp_bpe100_small_streaming_scratch30_streaming/jit_script_chunk_32_left_256.pt"
FULL_MODEL      = "ASR/zipformer/exp_bpe100_small_scratch30/jit_script.pt"
TOKENS_PATH     = "data/lang_bpe_100/tokens.txt"

# ── Audio / VAD config ────────────────────────────────────────────────────────
SAMPLE_RATE  = 16000
VAD_CHUNK    = 512       # silero-vad yêu cầu đúng 512 samples @ 16kHz = 32ms
MEL_BINS     = 80

VAD_THRESHOLD_ON      = 0.6    # prob > 0.6 → speech
VAD_THRESHOLD_OFF     = 0.35   # prob < 0.35 → silence
SILENCE_CHUNKS_TO_END = 25     # 25 × 32ms = 800ms im lặng → kết thúc câu
MIN_SPEECH_CHUNKS     = 6      # ít nhất 6 chunk speech liên tiếp → bắt đầu utterance
BEAM_SIZE             = 4
SAVE_AUDIO_DIR        = None   # None = không lưu; đặt qua --save-audio
# ──────────────────────────────────────────────────────────────────────────────


_fbank_extractor = Fbank(FbankConfig(num_mel_bins=MEL_BINS))


def compute_fbank(waveform: torch.Tensor) -> torch.Tensor:
    """waveform: (N,) float32 → fbank: (T, 80) — same extractor as training."""
    feats = _fbank_extractor.extract(waveform.numpy(), sampling_rate=SAMPLE_RATE)
    return torch.from_numpy(feats)


def init_beams(blank_id, context_size):
    """Khởi tạo beam search với 1 hypothesis rỗng."""
    return [([blank_id] * context_size, 0.0)]  # list of (ys, log_prob)


def beam_search_step(decoder, joiner, enc_frame, beams, context_size, blank_id, device):
    """
    Xử lý 1 encoder frame, cập nhật danh sách beam hypotheses.
    enc_frame: (C,)
    beams: list of (ys, log_prob)
    Returns: pruned list of (ys, log_prob)
    """
    num_hyps = len(beams)

    # Batch tất cả decoder inputs
    dec_input = torch.tensor(
        [h[0][-context_size:] for h in beams],
        dtype=torch.int32, device=device,
    )  # (num_hyps, context_size)
    dec_out = decoder(dec_input, False).squeeze(1)  # (num_hyps, D)

    # Batch joiner
    enc_rep = enc_frame.unsqueeze(0).expand(num_hyps, -1)  # (num_hyps, C)
    logits  = joiner(enc_rep, dec_out)                     # (num_hyps, vocab_size)
    log_probs = logits.log_softmax(dim=-1)

    vocab_size = log_probs.size(-1)

    # Cộng điểm hiện tại của từng hypothesis
    hyp_scores = torch.tensor([h[1] for h in beams], device=device).unsqueeze(1)
    log_probs  = log_probs + hyp_scores  # (num_hyps, vocab_size)

    # Lấy top-BEAM_SIZE candidates từ tất cả (hyp × token)
    topk_vals, topk_idx = log_probs.reshape(-1).topk(BEAM_SIZE)

    new_beams = {}  # tuple(ys) → (ys, log_prob)  — dedup bằng dict
    for val, idx in zip(topk_vals.tolist(), topk_idx.tolist()):
        hyp_idx  = idx // vocab_size
        token_id = idx % vocab_size
        old_ys   = beams[hyp_idx][0]
        new_ys   = old_ys if token_id == blank_id else old_ys + [token_id]
        key = tuple(new_ys)
        if key not in new_beams or new_beams[key][1] < val:
            new_beams[key] = (new_ys, val)

    # Sort và giữ top-BEAM_SIZE
    return sorted(new_beams.values(), key=lambda h: h[1], reverse=True)[:BEAM_SIZE]


def decode_hyp(ys, context_size, token_table):
    return "".join(token_table[i] for i in ys[context_size:]).replace("▁", " ").strip()


def clear_line(msg=""):
    sys.stdout.write(f"\r\033[K{msg}")
    sys.stdout.flush()


def maybe_save_audio(sample_buf: torch.Tensor):
    """Lưu utterance ra wav nếu --save-audio được bật."""
    if SAVE_AUDIO_DIR is None:
        return
    import os
    os.makedirs(SAVE_AUDIO_DIR, exist_ok=True)
    fname = os.path.join(SAVE_AUDIO_DIR, f"utt_{time.strftime('%H%M%S_%f')[:12]}.wav")
    sf.write(fname, sample_buf.numpy(), SAMPLE_RATE)
    print(f"  [saved: {fname}]", file=sys.stderr)


# ── Streaming inference ───────────────────────────────────────────────────────

@torch.no_grad()
def run_streaming(encoder, decoder, joiner, token_table, device, audio_queue):
    context_size = decoder.context_size
    blank_id     = decoder.blank_id
    chunk_length = encoder.chunk_size * 2
    T            = chunk_length + encoder.pad_length

    print(f"  Streaming: chunk_size={encoder.chunk_size} pad={encoder.pad_length} "
          f"T={T} out={chunk_length}  beam={BEAM_SIZE}  (latency ~{T*10}ms)")

    def reset():
        s  = encoder.get_init_states(device=device)
        b  = init_beams(blank_id, context_size)
        fb = torch.zeros(0, MEL_BINS)
        return s, b, fb

    states, beams, fbank_buf = reset()
    utterance_active = False
    silence_count    = 0
    speech_count     = 0
    pre_buf          = []
    sample_buf       = torch.zeros(0)

    clear_line("🎤  Listening...")

    while True:
        try:
            chunk_np = audio_queue.get(timeout=2.0)
        except queue.Empty:
            continue

        chunk      = torch.from_numpy(chunk_np)
        sample_buf = torch.cat([sample_buf, chunk])

        while sample_buf.size(0) >= VAD_CHUNK:
            window     = sample_buf[:VAD_CHUNK]
            sample_buf = sample_buf[VAD_CHUNK:]

            vad_prob  = vad_model(window, SAMPLE_RATE).item()
            is_speech = vad_prob >= VAD_THRESHOLD_ON

            if is_speech:
                silence_count = 0
                speech_count += 1
                pre_buf.append(window)
                if len(pre_buf) > MIN_SPEECH_CHUNKS:
                    pre_buf.pop(0)

                if not utterance_active and speech_count >= MIN_SPEECH_CHUNKS:
                    utterance_active = True
                    clear_line("● REC  ")
                    for w in pre_buf:
                        fbank_buf = torch.cat([fbank_buf, compute_fbank(w)], dim=0)
                    pre_buf = []
                elif utterance_active:
                    fbank_buf = torch.cat([fbank_buf, compute_fbank(window)], dim=0)
            else:
                speech_count = 0
                pre_buf      = []
                if utterance_active:
                    silence_count += 1
                    fbank_buf = torch.cat([fbank_buf, compute_fbank(window)], dim=0)

            # Kết thúc utterance
            if utterance_active and silence_count >= SILENCE_CHUNKS_TO_END:
                while fbank_buf.size(0) >= T:
                    fi            = fbank_buf[:T].unsqueeze(0).to(device)
                    xl            = torch.tensor([T], dtype=torch.int32, device=device)
                    eo, _, states = encoder(features=fi, feature_lengths=xl, states=states)
                    fbank_buf     = fbank_buf[chunk_length:]
                    for t in range(eo.size(1)):
                        beams = beam_search_step(decoder, joiner, eo[0, t], beams,
                                                 context_size, blank_id, device)

                text = decode_hyp(beams[0][0], context_size, token_table)
                if text:
                    sys.stdout.write(f"\r\033[K▶  {text}\n")
                    sys.stdout.flush()

                states, beams, fbank_buf = reset()
                utterance_active = False
                silence_count    = 0
                speech_count     = 0
                clear_line("🎤  Listening...")
                continue

            # Mid-utterance: chạy encoder từng chunk
            if utterance_active:
                changed = False
                while fbank_buf.size(0) >= T:
                    fi            = fbank_buf[:T].unsqueeze(0).to(device)
                    xl            = torch.tensor([T], dtype=torch.int32, device=device)
                    eo, _, states = encoder(features=fi, feature_lengths=xl, states=states)
                    fbank_buf     = fbank_buf[chunk_length:]
                    for t in range(eo.size(1)):
                        beams = beam_search_step(decoder, joiner, eo[0, t], beams,
                                                 context_size, blank_id, device)
                    changed = True

                if changed:
                    text = decode_hyp(beams[0][0], context_size, token_table)
                    if text:
                        clear_line(f"● {text}")


# ── Full (non-streaming) inference ───────────────────────────────────────────

@torch.no_grad()
def run_full(encoder, decoder, joiner, token_table, device, audio_queue):
    context_size = decoder.context_size
    blank_id     = decoder.blank_id

    def encode_full(fbank_buf):
        feat   = fbank_buf.unsqueeze(0).to(device)
        x_lens = torch.tensor([fbank_buf.size(0)], dtype=torch.int32, device=device)
        enc_out, _ = encoder(feat, x_lens)
        return enc_out.squeeze(0)  # (T', C)

    utterance_active = False
    silence_count    = 0
    speech_count     = 0
    pre_buf          = []
    pre_raw          = []   # raw samples tương ứng với pre_buf (để save audio)
    fbank_buf        = torch.zeros(0, MEL_BINS)
    raw_buf          = torch.zeros(0)   # raw samples của utterance hiện tại
    sample_buf       = torch.zeros(0)

    print(f"  Full-context: encode whole utterance after speech ends  beam={BEAM_SIZE}")
    clear_line("🎤  Listening...")

    while True:
        try:
            chunk_np = audio_queue.get(timeout=2.0)
        except queue.Empty:
            continue

        chunk      = torch.from_numpy(chunk_np)
        sample_buf = torch.cat([sample_buf, chunk])

        while sample_buf.size(0) >= VAD_CHUNK:
            window     = sample_buf[:VAD_CHUNK]
            sample_buf = sample_buf[VAD_CHUNK:]

            vad_prob  = vad_model(window, SAMPLE_RATE).item()
            is_speech = vad_prob >= VAD_THRESHOLD_ON

            if is_speech:
                silence_count = 0
                speech_count += 1
                pre_buf.append(window)
                pre_raw.append(window)
                if len(pre_buf) > MIN_SPEECH_CHUNKS:
                    pre_buf.pop(0)
                    pre_raw.pop(0)

                if not utterance_active and speech_count >= MIN_SPEECH_CHUNKS:
                    utterance_active = True
                    clear_line("● REC  ")
                    for w in pre_buf:
                        fbank_buf = torch.cat([fbank_buf, compute_fbank(w)], dim=0)
                    raw_buf = torch.cat(pre_raw)
                    pre_buf = []
                    pre_raw = []
                elif utterance_active:
                    fbank_buf = torch.cat([fbank_buf, compute_fbank(window)], dim=0)
                    raw_buf   = torch.cat([raw_buf, window])
            else:
                speech_count = 0
                pre_buf      = []
                pre_raw      = []
                if utterance_active:
                    silence_count += 1
                    fbank_buf = torch.cat([fbank_buf, compute_fbank(window)], dim=0)
                    raw_buf   = torch.cat([raw_buf, window])

            # Kết thúc utterance → encode 1 lần, beam search toàn bộ
            if utterance_active and silence_count >= SILENCE_CHUNKS_TO_END:
                maybe_save_audio(raw_buf)
                if fbank_buf.size(0) > 0:
                    enc_out = encode_full(fbank_buf)
                    beams   = init_beams(blank_id, context_size)
                    for t in range(enc_out.size(0)):
                        beams = beam_search_step(decoder, joiner, enc_out[t], beams,
                                                 context_size, blank_id, device)
                    text = decode_hyp(beams[0][0], context_size, token_table)
                    if text:
                        sys.stdout.write(f"\r\033[K▶  {text}\n")
                        sys.stdout.flush()

                fbank_buf        = torch.zeros(0, MEL_BINS)
                raw_buf          = torch.zeros(0)
                utterance_active = False
                silence_count    = 0
                speech_count     = 0
                clear_line("🎤  Listening...")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global BEAM_SIZE, SAVE_AUDIO_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["streaming", "full"], default="streaming",
        help="streaming: partial results while speaking | full: result after utterance ends",
    )
    parser.add_argument("--device", type=int, default=None,
                        help="Sounddevice input device index (default: system default)")
    parser.add_argument("--beam", type=int, default=BEAM_SIZE,
                        help=f"Beam size (default: {BEAM_SIZE})")
    parser.add_argument("--save-audio", type=str, default=None, metavar="DIR",
                        help="Lưu từng utterance ra wav vào thư mục DIR")
    args = parser.parse_args()
    BEAM_SIZE      = args.beam
    SAVE_AUDIO_DIR = args.save_audio
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Mode: {args.mode}  |  Beam: {BEAM_SIZE}")

    model_path = STREAMING_MODEL if args.mode == "streaming" else FULL_MODEL
    print(f"Loading model: {model_path}")
    asr_model = torch.jit.load(model_path, map_location=device)
    asr_model.eval()
    encoder = asr_model.encoder
    decoder = asr_model.decoder
    joiner  = asr_model.joiner

    token_table = k2.SymbolTable.from_file(TOKENS_PATH)

    print("Loading Silero VAD...")
    global vad_model
    vad_model = load_silero_vad()
    vad_model.eval()

    if args.device is not None:
        sd.default.device = args.device
    dev_info = sd.query_devices(kind="input")
    print(f"Mic: [{dev_info['index']}] {dev_info['name']}")
    print(f"VAD: on>{VAD_THRESHOLD_ON} silence={SILENCE_CHUNKS_TO_END}×32ms "
          f"min_speech={MIN_SPEECH_CHUNKS}×32ms\n")

    audio_queue = queue.Queue()

    def mic_callback(indata, frames, time_info, status):
        if status:
            print(f"\n[mic] {status}", file=sys.stderr)
        audio_queue.put(indata[:, 0].copy())

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=VAD_CHUNK,
        device=args.device,
        callback=mic_callback,
    ):
        if args.mode == "streaming":
            run_streaming(encoder, decoder, joiner, token_table, device, audio_queue)
        else:
            run_full(encoder, decoder, joiner, token_table, device, audio_queue)


if __name__ == "__main__":
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_set_profiling_mode(False)
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nStopped.")
