"""Synthetic demo session so the app is explorable without any media files."""
import numpy as np
import pandas as pd
from vision import fmt_time


def demo_session(minutes=40, window_s=10, seed=7):
    rng = np.random.default_rng(seed)
    n = minutes * 60 // window_s
    m = np.arange(n) * window_s / 60
    seg = [(0, 11, "Instructor Lecture", 78), (11, 20, "Instructor Lecture", 28), (20, 26, "Student Discussion", 84),
           (26, 33, "High Ambient Distraction", 35), (33, 36, "Silence", 80), (36, 40, "Student Discussion", 76)]
    cls, vis = [], []
    for x in m:
        for a, b, c, v in seg:
            if a <= x < b:
                cls.append(c); vis.append(np.clip(v + rng.normal(0, 6), 0, 100)); break
    aud = pd.DataFrame({"window": range(n), "class": cls, "energy_db": rng.normal(-30, 4, n).round(1),
                        "voice_frac": rng.uniform(0.2, 0.9, n).round(2), "turn_rate": rng.uniform(0, 2, n).round(2)})
    v = pd.DataFrame({"window": range(n), "visual": vis, "faces": rng.integers(24, 29, n)})
    total = minutes * 60
    rows = []
    for i in range(1, 29):
        r = rng.uniform(0.82, 1.0) if i <= 24 else rng.uniform(0.15, 0.55)
        first = rng.uniform(0, 120) if r > 0.7 else rng.uniform(0, total * 0.5)
        dur = r * total
        rows.append({"Student ID": f"S{i:02d}", "First Seen": fmt_time(first), "Last Seen": fmt_time(min(total, first + dur + 60)),
                     "Total Duration": fmt_time(dur), "Presence %": round(r * 100, 1), "Status": "Present" if r >= 0.7 else "Absent"})
    return pd.DataFrame(rows), v, aud
