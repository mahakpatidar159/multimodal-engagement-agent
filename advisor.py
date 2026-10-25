"""Teaching Suggestion Engine: heuristic, collective-level, non-punitive prompts."""
import numpy as np


def _runs(mask):
    out, s = [], None
    for i, m in enumerate(list(mask) + [False]):
        if m and s is None:
            s = i
        if not m and s is not None:
            out.append((s, i - 1)); s = None
    return out


def suggest(df, window_s=10, low=45, sustain_min=5):
    min_w = lambda m: max(1, int(round(m * 60 / window_s)))
    out = []

    def add(a, b, sev, title, msg, why):
        out.append({"start_min": round(a * window_s / 60, 1), "end_min": round((b + 1) * window_s / 60, 1),
                    "severity": sev, "title": title, "suggestion": msg, "why": why})

    e = df.engagement.fillna(50)
    for a, b in _runs(e < low):
        if b - a + 1 >= min_w(sustain_min):
            d = (b - a + 1) * window_s / 60
            add(a, b, "High", "Sustained engagement dip",
                f"Collective engagement dropped below {low}% for {d:.0f} minutes — recommend launching a quick poll, "
                "interactive discussion, or a 2-minute stretch break.", f"Engagement Index < {low}% across {d:.1f} min.")
    c = df["class"].fillna("")
    for a, b in _runs(c == "High Ambient Distraction"):
        if b - a + 1 >= min_w(2):
            add(a, b, "Medium", "Elevated classroom noise",
                "Ambient noise stayed high; consider a short reset cue, clearer group-work norms, or a brief pause.",
                "Audio classified as High Ambient Distraction for 2+ minutes.")
    for a, b in _runs((c == "Silence") & (df.visual.fillna(50) < 60)):
        if b - a + 1 >= min_w(3):
            add(a, b, "Medium", "Quiet and inattentive",
                "Extended silence with weak attention — try a think-pair-share or an open question to re-activate the room.",
                "Silence with visual attentiveness < 60% for 3+ minutes.")
    for a, b in _runs(c == "Instructor Lecture"):
        if b - a + 1 >= min_w(10):
            add(a, b, "Low", "Long uninterrupted lecture",
                "Over 10 minutes of continuous lecture — consider inserting a concept check or short activity.",
                "Instructor-led audio with no student interaction for 10+ minutes.")
    for a, b in _runs(e >= 70):
        if b - a + 1 >= min_w(5):
            add(a, b, "Positive", "Engagement peak",
                "Strong, sustained engagement — note what format worked here and reuse it.", "Engagement Index ≥ 70% for 5+ minutes.")
    drops = np.where(e.diff(3).fillna(0) <= -25)[0]
    if len(drops):
        i = int(drops[0])
        add(i - 2, i, "Low", "Sudden drop", "Engagement fell sharply — check pacing, clarity, or a possible interruption.",
            "Index fell ≥ 25 points within 3 windows.")
    return sorted(out, key=lambda x: x["start_min"])
