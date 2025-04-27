"""transcrpit.py – 200 ms DTW recognizer + AudioSegment support

This module now accepts **either** raw bytes **or** `pydub.AudioSegment`
objects in `LetterRecognizer.process_chunk()`, so your Flask/Socket.IO
route can keep passing the `AudioSegment` it already builds.

Key additions
-------------
• `_audiosegment_to_bytes()` – converts any AudioSegment to **mono 16 kHz**
  int16 PCM bytes (padding or trimming to 200 ms as needed).
• Extended type handling in `process_chunk()` – transparently accepts
  bytes/bytearray/AudioSegment.
• A convenience constant `WINDOW_MS = 200` derived from `SEGMENT_DURATION`
  for clarity.

No changes are required in your `/audio_chunk` route other than removing
`sample_rate` – the recognizer now figures that out itself.
"""

from __future__ import annotations
import os
import wave
import struct
import math
import threading
from typing import Dict, List, Union
import pickle
import PyBreathTranscript
import importlib.resources as pkg_resources


import pyaudio

try:
    from pydub import AudioSegment  # optional runtime dependency
except ImportError:
    AudioSegment = None  # type: ignore

# --------------------------------------------------------------------- #
#                            Configuration
# --------------------------------------------------------------------- #
RATE: int = 16_000                     # working sample rate (Hz)
CHANNELS: int = 1                      # mono
FORMAT = pyaudio.paInt16
CHUNK_FRAMES: int = 512               # microphone read size (≈32 ms)

SEGMENT_DURATION: float = 0.2         # one letter window (sec)
WINDOW_MS: int = int(SEGMENT_DURATION * 1000)  # 200 ms
SEGMENT_SAMPLES: int = int(RATE * SEGMENT_DURATION)  # 3 200 samples
SEGMENT_BYTES: int = SEGMENT_SAMPLES * 2             # int16 → 2 bytes

SILENCE_THRESHOLD: int = 250          # max abs amplitude regarded silent

# Feature extraction
FRAME_MS: int = 10
FRAME_LEN: int = int(RATE * FRAME_MS / 1000)        # 160 samples
HOP_LEN: int = FRAME_LEN                             # no overlap

# DTW parameters
DTW_REL_BAND: float = 0.15           # Sakoe‑Chiba band width (15 %)

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


def _audiosegment_to_bytes(seg: "AudioSegment") -> bytes:  # type: ignore
    """Ensure segment is mono 16 kHz PCM and exactly 200 ms long."""
    if seg.channels != 1:
        seg = seg.set_channels(1)
    if seg.frame_rate != RATE:
        seg = seg.set_frame_rate(RATE)
    # pad/trim to 200 ms
    if len(seg) < WINDOW_MS:
        padding = AudioSegment.silent(duration=WINDOW_MS - len(seg), frame_rate=RATE)
        seg = seg + padding
    elif len(seg) > WINDOW_MS:
        seg = seg[:WINDOW_MS]
    raw = seg.raw_data
    if len(raw) < SEGMENT_BYTES:
        raw += b"\x00" * (SEGMENT_BYTES - len(raw))
    return raw[:SEGMENT_BYTES]

# --------------------------------------------------------------------- #
#                           Reference loading
# --------------------------------------------------------------------- #

def _load_references(path: str) -> Dict[str, List[float]]:
    with pkg_resources.files(PyBreathTranscript).joinpath("references.pkl").open("rb") as f:
        refs = pickle.load(f)
    return refs

# --------------------------------------------------------------------- #
#                        Recognizer class / API
# --------------------------------------------------------------------- #

class LetterRecognizer:
    """Instantiate once and reuse – thread‑safe."""

    def __init__(self, reference_dir: str = REFERENCE_DIR):
        self.refs = _load_references(reference_dir)
        self.segment_bytes = SEGMENT_BYTES
        self.lock = threading.Lock()

    # ------------------------------------------------------------- #
    #  Public method for server code
    # ------------------------------------------------------------- #

    def process_chunk(self, chunk: Union[bytes, bytearray, "AudioSegment"],
                      sample_rate: int | None = None) -> str:
        """Return a letter ("_" for silence) given **200 ms** of audio.

        Accepts raw PCM bytes/bytearray **or** a pydub ``AudioSegment``.
        ``sample_rate`` is optional for the bytes case; the AudioSegment case
        is resampled internally.
        """
        # 0. Convert AudioSegment → bytes if needed
        if AudioSegment is not None and isinstance(chunk, AudioSegment):
            chunk = _audiosegment_to_bytes(chunk)
            sample_rate = RATE  # already converted
        elif not isinstance(chunk, (bytes, bytearray)):
            raise TypeError("chunk must be bytes, bytearray, or AudioSegment")

        chunk_bytes: bytes = bytes(chunk)

        # 1. Resample bytes if sample_rate differs
        if sample_rate and sample_rate != RATE:
            ints_in = struct.unpack(f"<{len(chunk_bytes)//2}h", chunk_bytes)
            ints_ds = downsample(list(ints_in), sample_rate)
            chunk_bytes = struct.pack(f"<{len(ints_ds)}h", *ints_ds)

        # 2. Ensure exact 200 ms length
        if len(chunk_bytes) < self.segment_bytes:
            chunk_bytes += b"\x00" * (self.segment_bytes - len(chunk_bytes))
        elif len(chunk_bytes) > self.segment_bytes:
            chunk_bytes = chunk_bytes[: self.segment_bytes]

        # 3. Silence check
        if is_silent(chunk_bytes):
            return "_"

        # 4. Extract features & DTW classify
        ints = struct.unpack(f"<{len(chunk_bytes)//2}h", chunk_bytes)
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
                print(recognizer.process_chunk(seg, RATE), end="", flush=True)
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
