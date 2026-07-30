"""Tests for the TorchCodec + librosa audio pipeline."""

from __future__ import annotations

import ast
import math
import wave
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
AUDIO_UTILS = ROOT / "audio_utils.py"


def _dotted_name(node: ast.AST) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _keywords(call: ast.Call) -> dict[str, ast.AST]:
    return {keyword.arg: keyword.value for keyword in call.keywords if keyword.arg}


def _literal(node: ast.AST) -> object:
    try:
        return ast.literal_eval(node)
    except (TypeError, ValueError):
        return None


def _write_stereo_wav(path: Path, *, sample_rate: int, seconds: float = 3.0) -> None:
    sample_count = int(sample_rate * seconds)
    time = np.arange(sample_count, dtype=np.float64) / sample_rate
    left = 0.25 * np.sin(2 * math.pi * 220.0 * time)
    right = 0.20 * np.sin(2 * math.pi * 329.63 * time)
    stereo = np.stack((left, right), axis=1)
    pcm = np.clip(stereo * 32767.0, -32768, 32767).astype("<i2")

    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def test_source_contract() -> None:
    """TorchCodec decodes to mono and librosa owns all resampling."""
    application_sources = (
        ROOT / "audio_utils.py",
        ROOT / "predict_keys.py",
        ROOT / "preprocess_data.py",
    )
    removed_dependency = "torch" + "audio"
    for path in application_sources:
        source = path.read_text(encoding="utf-8")
        assert removed_dependency not in source.lower(), (
            f"Removed audio dependency remains in {path}"
        )

    source = AUDIO_UTILS.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(AUDIO_UTILS))
    decoder_calls: list[ast.Call] = []
    resample_calls: list[ast.Call] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _dotted_name(node.func)
        if name == "AudioDecoder":
            decoder_calls.append(node)
        elif name == "librosa.resample":
            resample_calls.append(node)

    assert len(decoder_calls) == 1
    decoder_keywords = _keywords(decoder_calls[0])
    assert _literal(decoder_keywords["num_channels"]) == 1
    assert "sample_rate" not in decoder_keywords

    assert len(resample_calls) == 1
    resample_keywords = _keywords(resample_calls[0])
    assert {"y", "orig_sr", "target_sr", "res_type"} <= resample_keywords.keys()
    assert _literal(resample_keywords["res_type"]) == "soxr_hq"


def test_44k_audio_skips_resampling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("torchcodec")
    audio_utils = pytest.importorskip("audio_utils")

    audio_path = tmp_path / "stereo-44k.wav"
    _write_stereo_wav(audio_path, sample_rate=44_100)

    def unexpected_resample(*args: object, **kwargs: object) -> np.ndarray:
        raise AssertionError("44.1 kHz input should not be resampled")

    monkeypatch.setattr(audio_utils.librosa, "resample", unexpected_resample)
    waveform = audio_utils.load_audio_mono(audio_path, sample_rate=44_100)

    assert waveform.dtype == np.float32
    assert waveform.ndim == 1
    assert waveform.size == 44_100 * 3
    assert bool(np.isfinite(waveform).all())


def test_real_48k_stereo_decode_and_resample(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise real TorchCodec decoding and confirm librosa handles 48k -> 44.1k."""
    pytest.importorskip("torchcodec")
    audio_utils = pytest.importorskip("audio_utils")

    audio_path = tmp_path / "stereo-48k.wav"
    _write_stereo_wav(audio_path, sample_rate=48_000)

    original_resample = audio_utils.librosa.resample
    calls: list[dict[str, object]] = []

    def recording_resample(*args: object, **kwargs: object) -> np.ndarray:
        calls.append(dict(kwargs))
        return original_resample(*args, **kwargs)

    monkeypatch.setattr(audio_utils.librosa, "resample", recording_resample)
    waveform = audio_utils.load_audio_mono(audio_path, sample_rate=44_100)

    assert len(calls) == 1
    assert calls[0]["orig_sr"] == 48_000
    assert calls[0]["target_sr"] == 44_100
    assert calls[0]["res_type"] == "soxr_hq"
    assert waveform.dtype == np.float32
    assert waveform.ndim == 1
    assert waveform.size == 44_100 * 3
    assert bool(np.isfinite(waveform).all())


def test_prediction_preprocessing(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    predict_keys = pytest.importorskip("predict_keys")

    audio_path = tmp_path / "stereo-48k.wav"
    _write_stereo_wav(audio_path, sample_rate=48_000)
    spectrogram = predict_keys.preprocess_mp3(audio_path, sample_rate=44_100)

    assert spectrogram.dtype == torch.float32
    assert spectrogram.ndim == 3
    assert spectrogram.shape[0] == 1
    assert spectrogram.shape[1] == 105
    assert spectrogram.shape[2] > 0
    assert bool(torch.isfinite(spectrogram).all())


def test_checkpoint_forward_pass(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    predict_keys = pytest.importorskip("predict_keys")

    checkpoint = ROOT / "checkpoints" / "keynet.pt"
    if not checkpoint.exists():
        pytest.skip("checkpoints/keynet.pt is not available")

    audio_path = tmp_path / "stereo-48k.wav"
    _write_stereo_wav(audio_path, sample_rate=48_000)
    spectrogram = predict_keys.preprocess_mp3(audio_path, sample_rate=44_100)

    device = torch.device("cpu")
    model = predict_keys.load_model(checkpoint, device)
    with torch.no_grad():
        logits = model(spectrogram.unsqueeze(0))

    assert tuple(logits.shape) == (1, 24)
    assert bool(torch.isfinite(logits).all())
