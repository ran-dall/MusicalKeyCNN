"""Audio decoding and resampling helpers."""

from pathlib import Path

import librosa
import numpy as np
from torchcodec.decoders import AudioDecoder


def load_audio_mono(
    path: str | Path,
    *,
    sample_rate: int = 44_100,
) -> np.ndarray:
    """Decode an audio file to a mono ``float32`` waveform.

    TorchCodec owns decoding and channel conversion. Librosa owns sample-rate
    conversion so the resampling implementation is explicit and reproducible.

    Args:
        path: Audio file to decode.
        sample_rate: Desired output sample rate in Hz.

    Returns:
        A one-dimensional NumPy array containing mono audio at ``sample_rate``.
    """
    decoder = AudioDecoder(path, num_channels=1)
    samples = decoder.get_all_samples()

    waveform = samples.data.squeeze(0).cpu().numpy().astype(
        np.float32,
        copy=False,
    )

    if samples.sample_rate != sample_rate:
        waveform = librosa.resample(
            y=waveform,
            orig_sr=samples.sample_rate,
            target_sr=sample_rate,
            res_type="soxr_hq",
        ).astype(np.float32, copy=False)

    return waveform
