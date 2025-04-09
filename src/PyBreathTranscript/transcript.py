import librosa
from pydub import AudioSegment
import joblib
import importlib.resources as pkg_resources

from PyBreathTranscript.audio_fingerprinting import *
from PyBreathTranscript.waveform_comparing import *
import PyBreathTranscript


FINGERPRINT = 1
WAVEFORM = 2
SPECTROGRAM = 3 # TBA

with pkg_resources.files(PyBreathTranscript).joinpath("audio_data.pkl").open("rb") as f:
    letters = joblib.load(f)

def load_audio(file_path):
    """
    Load the audio file
    """
    signal, sr = librosa.load(file_path, sr=None)
    return signal, sr


def transcript_chunk(file_path: str, method=FINGERPRINT) -> str:
    """
     Transcribes one chunk of breath using the specified method.

    ### Parameters
        **file_path (str)**: Path to the file that needs to be transcribed.
        **method (str)**: The transcription method to use (default: 'FINGERPRINT').

    ### Methods
        **FINGERPRINT (default)**: Uses Audio Fingerprinting method for transcription
        **WAVEFORM**: Uses waveforms comparison for transcription
    
    ### Returns
        **str**: The transcript.
    """
    if method == FINGERPRINT:
            breath_fragment_fp = extract_audio_fingerprint(file_path)
            letters["similarity"] = letters["fingerprint"].apply(
                lambda fingerprint: compute_continuous_similarity(breath_fragment_fp, fingerprint)
            )
    elif method == WAVEFORM:
        breath_fragment_for_comparing, sr1 = load_audio(file_path)
        letters["similarity"] = letters["audio_data"].apply(
            lambda audio_data: compare_waveforms(breath_fragment_for_comparing, audio_data)
        )
        # Find the letter with the highest similarity
    best_letter = letters.loc[letters["similarity"].idxmax(), "letter"]
    return best_letter


def transcript(file_path: str, method=FINGERPRINT) -> str:
    """
     Transcribes full recording of breath using the specified method.

    ### Parameters
        **file_path (str)**: Path to the file that needs to be transcribed.
        **method (str)**: The transcription method to use (default: 'FINGERPRINT').

    ### Methods
        **FINGERPRINT (default)**: Uses Audio Fingerprinting method for transcription
        **WAVEFORM**: Uses waveforms comparison for transcription
    
    ### Returns
        **str**: The transcript.
    """
    # TODO: Add spectrogram comparison method
    transcript = ""
    breath_audio, sr = load_audio(file_path)
    duration = librosa.get_duration(y=breath_audio, sr=sr) * 1000
    breath = AudioSegment.from_wav(file_path)


    for x in range(0, int(duration), 200):
        breath_fragment = breath[x:x+200]
        # TODO: use breath_fragment for comparing, remove reudant file saving
        temp_output = "temp_fragment.wav"
        breath_fragment.export(temp_output, format='wav')

        # Creating transcript of breath
        best_letter = transcript_chunk(temp_output, method)
        transcript += best_letter

    return transcript
