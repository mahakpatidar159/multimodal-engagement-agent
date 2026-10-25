"""Multimodal Agent: transparent rule-based fusion -> Engagement Index (0-100)."""
import numpy as np
import pandas as pd

AUDIO_SCORE = {"Silence": 30, "Instructor Lecture": 70, "Student Discussion": 100, "High Ambient Distraction": 0}


def fuse(vis: pd.DataFrame, aud: pd.DataFrame, window_s=10, w_vis=0.6, smooth=3) -> pd.DataFrame:
    n = int(max(vis.window.max() if len(vis) else -1, aud.window.max() if len(aud) else -1)) + 1
    df = pd.DataFrame({"window": range(n)})
    df = df.merge(vis[["window", "visual", "faces"]], on="window", how="left") if len(vis) else df.assign(visual=np.nan, faces=0)
    df = df.merge(aud[["window", "class", "energy_db"]], on="window", how="left") if len(aud) else df.assign(**{"class": None, "energy_db": np.nan})
    scores, reasons = [], []
    for _, r in df.iterrows():
        v, c = r.visual, r["class"]
        a = AUDIO_SCORE.get(c, np.nan)
        why = []
        if c == "Silence" and v >= 70:
            a = 60; why.append("quiet focused work (attentive + silent)")
        if np.isnan(v) and np.isnan(a):
            scores.append(np.nan); reasons.append("no data"); continue
        if np.isnan(v):
            s = a; why.append("audio only")
        elif np.isnan(a):
            s = v; why.append("vision only")
        else:
            s = w_vis * v + (1 - w_vis) * a
            if c == "Student Discussion" and v >= 70:
                s += 5; why.append("attentive and interactive")
            if c == "High Ambient Distraction" and v < 50:
                s -= 10; why.append("noise with attention drop")
            if c == "Instructor Lecture" and v < 40:
                s -= 5; why.append("passive lecture, low attention")
            if v < 40 and not why:
                why.append("low visual attentiveness")
        scores.append(s); reasons.append("; ".join(why) or "balanced signals")
    df["raw"] = np.clip(scores, 0, 100)
    df["engagement"] = df.raw.rolling(smooth, center=True, min_periods=1).mean()
    df["reason"] = reasons
    df["minute"] = df.window * window_s / 60
    return df
