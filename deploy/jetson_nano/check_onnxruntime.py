#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import onnxruntime as ort


def run_provider(model: Path, provider: str):
    sess = ort.InferenceSession(str(model), providers=[provider])
    inputs = {
        sess.get_inputs()[0].name: np.zeros((1, 100, 80), dtype=np.float32),
        sess.get_inputs()[1].name: np.array([100], dtype=np.int64),
    }
    outputs = sess.run(None, inputs)
    print(f"{provider}: ok")
    print(f"  actual providers: {sess.get_providers()}")
    print(f"  outputs: {[tuple(o.shape) for o in outputs]}")


def main():
    model = Path(__file__).resolve().parent / "model" / "encoder.onnx"
    print(f"onnxruntime: {ort.__version__}")
    print(f"available providers: {ort.get_available_providers()}")
    print(f"device: {ort.get_device()}")
    run_provider(model, "CPUExecutionProvider")
    if "CUDAExecutionProvider" in ort.get_available_providers():
        run_provider(model, "CUDAExecutionProvider")
    else:
        print("CUDAExecutionProvider: not available")


if __name__ == "__main__":
    main()
