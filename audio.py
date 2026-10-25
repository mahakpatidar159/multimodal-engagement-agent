"""Audio Module: windowed spectral energy + VAD. No speech recognition, no content analysis,
no waveform retained after feature extraction (privacy-preserving)."""
import wave
import numpy as np
import pandas as pd

CLASSES = ["Silence", "Instructor Lecture", "Student Discussion", "High Ambient Distraction"]


def load_wav(file) -> tuple:
    """Read PCM WAV (8/16/32-bit) -> mono float32 in [-1,1], sample rate."""
    with wave.open(file, "rb") as w:
        sr, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(n)
    dt = {1: np.uint8, 2: np.int16, 4: np.int32}.get(sw)
    if dt is None:
        raise ValueError("Unsupported WAV bit-depth; use 8/16/32-bit PCM.")
    x = np.frombuffer(raw, dtype=dt).astype(np.float32)
    x = (x - 128) / 128 if sw == 1 else x / float(2 ** (8 * sw - 1))
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x, sr


def analyze(y, sr, window_s=10, frame_ms=25) -> pd.DataFrame:
    fl = int(sr * frame_ms / 1000)
    n = len(y) // fl
    if n < 2:
        raise ValueError("Audio too short to analyse.")
    fr = y[: n * fl].reshape(n, fl) * np.hanning(fl)
    rms = np.sqrt((fr ** 2).mean(1)) + 1e-9
    db = 20 * np.log10(rms)
    spec = np.abs(np.fft.rfft(fr, axis=1)) ** 2 + 1e-12
    freqs = np.fft.rfftfreq(fl, 1 / sr)
    band = (freqs >= 300) & (freqs <= 3400)
    band_ratio = spec[:, band].sum(1) / spec.sum(1)
    flat = np.exp(np.log(spec).mean(1)) / spec.mean(1)  # spectral flatness: ~1 = noise
    floor = np.percentile(db, 10)
    voiced = (db > floor + 10) & (band_ratio > 0.5) & (flat < 0.3)  # energy + spectral VAD
    per = int(window_s * 1000 / frame_ms)
    rows = []
    for i in range(0, n - per // 2, per):
        s = slice(i, i + per)
        d, v, br, fl_ = db[s], voiced[s], band_ratio[s], flat[s]
        active = d.mean() - floor
        vf = v.mean()
        trans = np.abs(np.diff(v.astype(int))).sum() / window_s  # turn-taking rate
        if active < 6 and vf < 0.1:
            c = "Silence"
        elif active > 50 or (vf < 0.3 and active > 12) or br.mean() < 0.4:
            c = "High Ambient Distraction"
        elif vf >= 0.3:
            c = "Student Discussion" if (trans >= 0.5 or vf < 0.55) else "Instructor Lecture"
        else:
            c = "Silence"
        rows.append({"window": i // per, "energy_db": round(float(d.mean()), 1), "voice_frac": round(float(vf), 2),
                     "turn_rate": round(float(trans), 2), "class": c})
    return pd.DataFrame(rows)
