"""Vision Module: face detection, ID tracking, sustained-presence attendance, attentiveness.
Runs fully offline (OpenCV Haar cascades). No frames or face crops are ever stored."""
import os

import cv2
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_cascade(filename: str) -> "cv2.CascadeClassifier":
    """Load a Haar cascade without assuming cv2.data exists (Streamlit Cloud, OpenCV builds).
    Search order: 1) cascades/ folder shipped in the repo, 2) cv2.data.haarcascades,
    3) <cv2 package dir>/data, 4) next to vision.py."""
    candidates = [os.path.join(_HERE, "cascades", filename)]
    data = getattr(cv2, "data", None)
    if data is not None and getattr(data, "haarcascades", None):
        candidates.append(os.path.join(data.haarcascades, filename))
    cv2_dir = os.path.dirname(getattr(cv2, "__file__", "") or "")
    if cv2_dir:
        candidates.append(os.path.join(cv2_dir, "data", filename))
    candidates.append(os.path.join(_HERE, filename))
    for path in candidates:
        if os.path.isfile(path):
            clf = cv2.CascadeClassifier(path)
            if not clf.empty():
                return clf
    raise RuntimeError(
        f"Could not load Haar cascade '{filename}'. Add it to a 'cascades/' folder next to vision.py. "
        f"Searched: {candidates}"
    )


_FRONT = _load_cascade("haarcascade_frontalface_default.xml")
_PROF = _load_cascade("haarcascade_profileface.xml")


def fmt_time(s: float) -> str:
    s = int(max(0, s))
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


class Track:
    def __init__(self, tid, box, idx, t):
        self.id, self.box, self.first, self.last = tid, box, t, t
        self.seen = [idx]
        self.cy = []  # face-centre y history -> posture baseline

    def update(self, box, idx, t):
        self.box, self.last = box, t
        self.seen.append(idx)


class VisionEngine:
    """Stateful tracker. Call process(frame, t_seconds) per sampled frame."""

    def __init__(self, max_gap=3.0, min_face=40):
        self.max_gap, self.min_face = max_gap, min_face
        self.tracks, self.frame_times, self.records = {}, [], []
        self._next = 1

    # -- detection (histogram equalisation handles lighting variation) --
    def detect(self, bgr):
        gray = cv2.equalizeHist(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
        ms = (self.min_face, self.min_face)
        front = [tuple(map(int, b)) for b in _FRONT.detectMultiScale(gray, 1.1, 5, minSize=ms)]
        prof = []
        w = gray.shape[1]
        for b in _PROF.detectMultiScale(gray, 1.1, 5, minSize=ms):
            prof.append(tuple(map(int, b)))
        for x, y, bw, bh in _PROF.detectMultiScale(cv2.flip(gray, 1), 1.1, 5, minSize=ms):
            prof.append((int(w - x - bw), int(y), int(bw), int(bh)))
        out = [(b, True) for b in front]
        for b in prof:
            if all(_iou(b, f) < 0.3 for f, _ in out):
                out.append((b, False))  # profile only = head turned away
        return out

    def process(self, bgr, t):
        """Track faces in one frame; returns annotated copy for transient display."""
        idx = len(self.frame_times)
        self.frame_times.append(float(t))
        dets = self.detect(bgr)
        free = {k: tr for k, tr in self.tracks.items() if t - tr.last <= self.max_gap}
        vis = bgr.copy()
        for box, frontal in dets:
            best, bs = None, 0.0
            for k, tr in free.items():
                s = _iou(box, tr.box)
                cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
                tx, ty = tr.box[0] + tr.box[2] / 2, tr.box[1] + tr.box[3] / 2
                if s < 0.15 and np.hypot(cx - tx, cy - ty) < 0.8 * box[2]:
                    s = 0.15  # centroid fallback for fast movement
                if s > bs:
                    best, bs = k, s
            if best is None:
                tr = Track(self._next, box, idx, t)
                self.tracks[self._next] = tr
                self._next += 1
            else:
                tr = free.pop(best)
                tr.update(box, idx, t)
            cy = box[1] + box[3] / 2
            tr.cy.append(cy)
            base = np.median(tr.cy[:10])
            upright = (cy - base) / box[3] < 0.4  # face sinking > 0.4 face-heights = slumping
            self.records.append((t, tr.id, bool(frontal), bool(upright)))
            col = (60, 180, 75) if frontal and upright else (0, 140, 255)
            x, y, w, h = box
            cv2.rectangle(vis, (x, y), (x + w, y + h), col, 2)
            cv2.putText(vis, f"S{tr.id:02d}", (x, max(15, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        return vis

    # -- sustained presence attendance --
    def attendance(self, threshold=0.7):
        ft = np.array(self.frame_times)
        if len(ft) == 0:
            return pd.DataFrame(columns=["Student ID", "First Seen", "Last Seen", "Total Duration", "Presence %", "Status"])
        dts = np.diff(ft)
        step = float(np.median(dts)) if len(dts) else 1.0
        dt = np.append(np.minimum(dts, self.max_gap), step) if len(dts) else np.array([step])
        session = max(ft[-1] - ft[0] + step, 1e-6)
        rows = []
        for tr in self.tracks.values():
            if len(ft) > 1 and len(tr.seen) < 2:
                continue  # single-frame blips are noise
            dur = float(dt[tr.seen].sum())
            ratio = min(dur / session, 1.0)
            rows.append({
                "Student ID": f"S{tr.id:02d}", "First Seen": fmt_time(tr.first - ft[0]),
                "Last Seen": fmt_time(tr.last - ft[0]), "Total Duration": fmt_time(dur),
                "Presence %": round(ratio * 100, 1),
                "Status": "Present" if ratio >= threshold else "Absent",
            })
        return pd.DataFrame(rows, columns=["Student ID", "First Seen", "Last Seen", "Total Duration", "Presence %", "Status"])

    def timeline(self, window_s=10):
        """Per-window visual attentiveness (0-100): share of faces frontal AND upright."""
        if not self.records:
            return pd.DataFrame(columns=["window", "visual", "faces"])
        df = pd.DataFrame(self.records, columns=["t", "id", "frontal", "upright"])
        df["window"] = ((df.t - self.frame_times[0]) // window_s).astype(int)
        df["att"] = (df.frontal & df.upright).astype(float)
        g = df.groupby("window").agg(visual=("att", "mean"), faces=("id", "nunique")).reset_index()
        g["visual"] *= 100
        return g