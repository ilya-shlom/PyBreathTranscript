# PyBreathTranscript
[![PyPi](https://img.shields.io/badge/pypi-0.7.0-blue)](https://test.pypi.org/project/PyBreathTranscript/)



This package is used for transcribing breath recordings (captured using an in-ear microphone) into text symbols.

Check [Full project](https://github.com/ilya-shlom/breathing-analysis) to see this library in action.

## Quickstart

Install the package:

```bash
pip install -i https://test.pypi.org/simple/ PyBreathTranscript
```

or

```bash
pip install git+https://github.com/ilya-shlom/PyBreathTranscript.git
```

Example usage:

```python
from PyBreathTranscript import transcript as bt

result = bt.transcript("breath.wav", bt.FINGERPRINT)
print(result)
```

## Usage

Import library into your project using

```python
from PyBreathTranscript import transcript as bt
```

To transcribe a recording of your breath, use the `bt.transcript(file_path, method)` function:

```python
transcript = bt.transcript("path/to/your/recording.wav", bt.FINGERPRINT)
```

`file_path: str` - path of your recording.

`method: int` - method used to transcribe an audio file. There are two methods available now acceptable by constants:

1. `bt.FINGERPRINT` – method based on an audio fingerprinting algorithm. This is the recommended and default method.

2. `bt.WAVEFORM` – compares waveforms and returns the best-matching symbol. This approach is more realistic but less distinctive, recommended for demonstration purposes only.

_______
_Please note:_ this library is created for educational and demonstration purposes. For the best experience, an in-ear microphone based on the [BOYA BY-M1 Lavalier microphone](https://www.boyamic.com/product/lavalier-microphone-by-m1) is recommended.

This project is a part of my graduation work, Cyrillic symbols are one of requirements for the system.
