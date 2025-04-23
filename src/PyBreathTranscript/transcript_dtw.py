"""letter_recognizer.py – 200 ms window DTW recognizer + API for /audio_chunk

This module can be used in **two** ways:
1. Run it as a script – it opens the microphone and prints letters / "_" for
   silence every 200 ms.
2. Import it and call ``recognizer.process_chunk(chunk_bytes, sample_rate)``
   from your Flask/FastAPI ``/audio_chunk`` route.

Usage inside route (Flask example)
---------------------------------
```python
from letter_recognizer import get_recognizer

@app.post("/audio_chunk")
def audio_chunk():
    chunk = request.data  # 200 ms of raw PCM int16 mono 16 kHz
    letter = get_recognizer().process_chunk(chunk)
    return {"letter": letter}
```
"""

from __future__ import annotations
import os
import struct
import math
import threading
from typing import Dict, List
import pyaudio
import pickle

# --------------------------------------------------------------------- #
#                            Configuration
# --------------------------------------------------------------------- #
RATE: int = 16_000                     # recognizer working sample rate (Hz)
CHANNELS: int = 1
FORMAT = pyaudio.paInt16
CHUNK_FRAMES: int = 512                # microphone read size (≈32 ms)

SEGMENT_DURATION: float = 0.2          # one letter window (sec)
SEGMENT_SAMPLES: int = int(RATE * SEGMENT_DURATION)  # 3 200 samples
SEGMENT_BYTES: int = SEGMENT_SAMPLES * 2             # int16 → 2 bytes

SILENCE_THRESHOLD: int = 500           # max abs amplitude regarded silent

# Feature extraction
FRAME_MS: int = 10
FRAME_LEN: int = int(RATE * FRAME_MS / 1000)         # 160 samples
HOP_LEN: int = FRAME_LEN                              # no overlap

# DTW parameters
DTW_REL_BAND: float = 0.15            # Sakoe–Chiba band width (15 %)

# Reference WAV folder (16 kHz mono PCM files named A.wav, B.wav …)
REFERENCE_DIR: str = "letters_source_files"

# --------------------------------------------------------------------- #
#                            Helper functions
# --------------------------------------------------------------------- #

def downsample(ints: List[int], src_rate: int) -> List[int]:
    """Naïve decimation to recognizer sample rate."""
    factor = max(1, src_rate // RATE)
    return ints[::factor]


def frame_rms(ints: List[int]) -> List[float]:
    """Compute RMS energy every 10 ms."""
    out: List[float] = []
    for i in range(0, len(ints) - FRAME_LEN + 1, HOP_LEN):
        window = ints[i : i + FRAME_LEN]
        energy = sum(s * s for s in window) / FRAME_LEN
        out.append(math.sqrt(energy))
    return out or [0.0]


def dtw_cost(seq1: List[float], seq2: List[float], band: int | None = None) -> float:
    """Band‑limited DTW (falls back to full matrix if band is None)."""
    n, m = len(seq1), len(seq2)
    if band is None:
        band = max(n, m)
    band = max(band, abs(n - m))
    INF = float("inf")
    D = [[INF] * (m + 1) for _ in range(n + 1)]
    D[0][0] = 0.0
    for i in range(1, n + 1):
        j_start = max(1, i - band)
        j_end = min(m, i + band)
        for j in range(j_start, j_end + 1):
            cost = abs(seq1[i - 1] - seq2[j - 1])
            D[i][j] = cost + min(D[i - 1][j], D[i][j - 1], D[i - 1][j - 1])
    return D[n][m]


def is_silent(buf: bytes) -> bool:
    samples = struct.unpack(f"<{len(buf)//2}h", buf)
    return max(abs(s) for s in samples) < SILENCE_THRESHOLD

# --------------------------------------------------------------------- #
#                           Reference loading
# --------------------------------------------------------------------- #

def _load_references() -> Dict[str, List[float]]:
    pkl_path = os.path.join(os.path.dirname(__file__), "references.pkl")
    if not os.path.isfile(pkl_path):
        raise RuntimeError(f"Reference file not found: {pkl_path}")
    with open(pkl_path, "rb") as f:
        refs = pickle.load(f)
    return refs

# Legacy version
# def _load_references(path: str) -> Dict[str, List[float]]:
    # refs: Dict[str, List[float]] = {}
    # for fname in os.listdir(path):
    #     if not fname.lower().endswith(".wav"):
    #         continue
    #     letter = os.path.splitext(fname)[0].upper()
    #     wav_path = os.path.join(path, fname)
    #     with wave.open(wav_path, "rb") as wf:
    #         src_rate = wf.getframerate()
    #         raw = wf.readframes(wf.getnframes())
    #     ints = struct.unpack(f"<{len(raw)//2}h", raw)
    #     ints = downsample(list(ints), src_rate)
    #     refs[letter] = frame_rms(ints)
    # if not refs:
    #     raise RuntimeError(f"No reference WAVs found in {path!r}")
    #     # Save references to external file
    # with open('references.pkl', 'wb') as f:
    #     pickle.dump(refs, f)
    # return refs

# --------------------------------------------------------------------- #
#                        Recognizer class / API
# --------------------------------------------------------------------- #

class LetterRecognizer:
    """Instantiate **once** and reuse for every `/audio_chunk` call."""

    def __init__(self, reference_dir: str = REFERENCE_DIR):
        self.refs = _load_references()
        self.segment_bytes = SEGMENT_BYTES
        self.lock = threading.Lock()  # safe if multiple requests in parallel

    # --------------------------------------------------------------- #
    #  Public method for server code
    # --------------------------------------------------------------- #

    def process_chunk(self, chunk: bytes, sample_rate: int | None = None) -> str:
        """Return a single letter or "_" for silence.

        * ``chunk`` – raw **little‑endian int16 mono** PCM.
        * ``sample_rate`` – if provided and ≠ RATE, the function will decimate
          the audio before classification.
        """
        # 1. Make sure length is exactly 200 ms worth of samples after any
        #    resampling.
        if sample_rate and sample_rate != RATE:
            ints_in = struct.unpack(f"<{len(chunk)//2}h", chunk)
            ints_ds = downsample(list(ints_in), sample_rate)
            chunk = struct.pack(f"<{len(ints_ds)}h", *ints_ds)
        if len(chunk) < self.segment_bytes:
            # zero‑pad
            chunk = chunk + b"\x00" * (self.segment_bytes - len(chunk))
        elif len(chunk) > self.segment_bytes:
            chunk = chunk[: self.segment_bytes]

        # 2. Silence?
        if is_silent(chunk):
            return "_"

        # 3. Extract feature & DTW classify
        ints = struct.unpack(f"<{len(chunk)//2}h", chunk)
        feat = frame_rms(list(ints))
        best_letter, best_cost = "?", float("inf")
        for letter, ref in self.refs.items():
            band = int(max(len(feat), len(ref)) * DTW_REL_BAND)
            cost = dtw_cost(feat, ref, band)
            if math.isinf(cost):
                cost = dtw_cost(feat, ref, None)
            if cost < best_cost:
                best_cost, best_letter = cost, letter
        return best_letter

# ------------------------- Singleton accessor ------------------------ #

_recognizer_instance: LetterRecognizer | None = None


def get_recognizer() -> LetterRecognizer:
    global _recognizer_instance
    if _recognizer_instance is None:
        _recognizer_instance = LetterRecognizer()
    return _recognizer_instance

# --------------------------------------------------------------------- #
#                         CLI: demo via microphone
# --------------------------------------------------------------------- #

def _stream_demo():
    pa = pyaudio.PyAudio()
    stream = pa.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True,
                     frames_per_buffer=CHUNK_FRAMES)
    recognizer = get_recognizer()
    buffer = bytearray()
    print("[Demo] Speak letters – '_' printed for silence. Ctrl+C to stop.")
    try:
        while True:
            data = stream.read(CHUNK_FRAMES)
            buffer.extend(data)
            while len(buffer) >= SEGMENT_BYTES:
                seg = bytes(buffer[:SEGMENT_BYTES])
                del buffer[:SEGMENT_BYTES]
                print(recognizer.process_chunk(seg), end="", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()
        print("\nStopped.")

# --------------------------------------------------------------------- #
#                              Entrypoint
# --------------------------------------------------------------------- #

if __name__ == "__main__":
    print("Loading references from", REFERENCE_DIR)
    _ = get_recognizer()
    _stream_demo()
