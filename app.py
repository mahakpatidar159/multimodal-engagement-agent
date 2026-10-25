"""Multimodal Attendance & Engagement Agent - Skill Wallet Capstone Project 5.
100% offline, in-memory processing. Collective feedback, not individual surveillance."""
import tempfile, time
from datetime import datetime
import cv2, numpy as np, pandas as pd
import plotly.graph_objects as go
import streamlit as st

from vision import VisionEngine
import audio as A
from fusion import fuse
from advisor import suggest
from demo import demo_session

st.set_page_config(page_title="Multimodal Attendance & Engagement Agent", page_icon="🎓", layout="wide")
st.markdown("""<style>
.block-container{padding-top:1.5rem}
.hero{background:linear-gradient(90deg,#1e3a8a,#2563eb);color:#fff;padding:1.1rem 1.4rem;border-radius:14px;margin-bottom:1rem}
.hero h1{margin:0;font-size:1.6rem}.hero p{margin:.2rem 0 0;opacity:.9}
.card{border-left:5px solid #2563eb;background:#f8fafc;padding:.7rem 1rem;border-radius:8px;margin-bottom:.6rem;color:#0f172a}
.High{border-color:#dc2626}.Medium{border-color:#f59e0b}.Low{border-color:#64748b}.Positive{border-color:#16a34a}
</style>""", unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>🎓 Multimodal Attendance & Engagement Agent</h1>'
            '<p>🔒 Edge-only processing: no video or audio leaves this device. Engagement is reported collectively, never to rank or punish individuals.</p></div>',
            unsafe_allow_html=True)

S = st.session_state
for k, v in {"eng": VisionEngine(), "att": None, "vis": pd.DataFrame(), "aud": pd.DataFrame(), "t0": None, "demo": False}.items():
    S.setdefault(k, v)

with st.sidebar:
    st.header("⚙️ Settings")
    thr = st.slider("Sustained presence threshold", 0.5, 0.95, 0.70, 0.05)
    win = st.select_slider("Analysis window (s)", [5, 10, 15, 30], 10)
    sample = st.slider("Video sampling interval (s)", 0.5, 5.0, 1.0, 0.5)
    w_vis = st.slider("Weight: visual vs audio", 0.2, 0.9, 0.6, 0.05)
    low = st.slider("Low-engagement threshold (%)", 20, 70, 45)
    sustain = st.slider("Sustained dip trigger (min)", 1, 10, 5)
    st.divider()
    if st.button("▶ Load demo session", width="stretch"):
        S.att, S.vis, S.aud = demo_session(window_s=win); S.demo = True
    if st.button("🗑 Reset session", width="stretch"):
        for k in ("eng", "att", "vis", "aud", "t0", "demo"):
            S.pop(k, None)
        st.rerun()
    st.caption("No database, no cloud calls. Data lives only in this browser session.")


def refresh_vision():
    S.att = S.eng.attendance(thr)
    S.vis = S.eng.timeline(win)


tabs = st.tabs(["👥 Attendance", "🎙 Audio Dynamics", "🧠 Engagement Fusion", "💡 Teaching Suggestions", "📊 Session Report"])

# ---------------- Tab 1 ----------------
with tabs[0]:
    st.subheader("Presence-based attendance")
    src = st.radio("Source", ["Upload video", "Upload photo", "Webcam snapshot"], horizontal=True)
    try:
        if src == "Upload video":
            f = st.file_uploader("Classroom video", type=["mp4", "avi", "mov", "mkv"])
            if f and st.button("Process video"):
                S.eng, S.demo = VisionEngine(), False
                with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
                    tmp.write(f.read()); tmp.flush()
                    cap = cv2.VideoCapture(tmp.name)
                    fps = cap.get(cv2.CAP_PROP_FPS) or 25
                    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                    step, bar, last, i = max(1, int(fps * sample)), st.progress(0.0), None, 0
                    while True:
                        ok, fr = cap.read()
                        if not ok:
                            break
                        if i % step == 0:
                            if fr.shape[1] > 960:
                                fr = cv2.resize(fr, (960, int(fr.shape[0] * 960 / fr.shape[1])))
                            last = S.eng.process(fr, i / fps)
                        i += 1
                        if total and i % (step * 5) == 0:
                            bar.progress(min(i / total, 1.0))
                    cap.release(); bar.progress(1.0)
                if last is not None:
                    st.image(last[:, :, ::-1], caption="Last processed frame (not stored)")
                refresh_vision()
        else:
            img = st.file_uploader("Classroom photo", type=["jpg", "jpeg", "png"]) if src == "Upload photo" else st.camera_input("Capture")
            if img is not None and st.button("Analyse frame"):
                fr = cv2.imdecode(np.frombuffer(img.getvalue(), np.uint8), cv2.IMREAD_COLOR)
                if src == "Upload photo":
                    S.eng = VisionEngine()
                    t = 0.0
                else:
                    S.t0 = S.t0 or time.time(); t = time.time() - S.t0
                st.image(S.eng.process(fr, t)[:, :, ::-1], caption="Annotated frame (not stored)")
                refresh_vision()
                if src == "Upload photo":
                    st.info("A single photo is one time-step, so everyone detected counts as 100% present. Use video or repeated snapshots for sustained-presence logic.")
    except Exception as e:
        st.error(f"Vision processing failed: {e}")
    if S.att is not None and len(S.att):
        a = S.att
        c = st.columns(3)
        c[0].metric("Detected IDs", len(a)); c[1].metric("Present", int((a.Status == "Present").sum()))
        c[2].metric("Absent / transient", int((a.Status == "Absent").sum()))
        st.dataframe(a, width="stretch", hide_index=True)
        st.download_button("⬇ Download attendance CSV", a.to_csv(index=False).encode(), "attendance_register.csv", "text/csv")
        st.caption(f"Status = Present only if detected for ≥ {int(thr*100)}% of session time; brief pass-throughs stay Absent. IDs are anonymous tracker labels.")
    else:
        st.info("Process a video/photo, or load the demo session from the sidebar.")

# ---------------- Tab 2 ----------------
with tabs[1]:
    st.subheader("Acoustic interaction & noise dynamics")
    st.caption("Spectral energy + voice-activity detection only. No speech recognition; audio is discarded after feature extraction.")
    up = st.file_uploader("Session audio (PCM WAV)", type=["wav"])
    rec = st.audio_input("…or record") if hasattr(st, "audio_input") else None
    src_a = up or rec
    if src_a is not None and st.button("Analyse audio"):
        try:
            y, sr = A.load_wav(src_a)
            S.aud, S.demo = A.analyze(y, sr, win), False
        except Exception as e:
            st.error(f"Audio analysis failed: {e}")
    if len(S.aud):
        d = S.aud
        cols = {"Silence": "#94a3b8", "Instructor Lecture": "#2563eb", "Student Discussion": "#16a34a", "High Ambient Distraction": "#dc2626"}
        fig = go.Figure(go.Bar(x=d.window * win / 60, y=d.energy_db - d.energy_db.min() + 1, marker_color=[cols[c] for c in d["class"]],
                               customdata=d["class"], hovertemplate="%{x:.1f} min<br>%{customdata}<extra></extra>"))
        fig.update_layout(height=320, xaxis_title="Minutes", yaxis_title="Relative energy (dB)", margin=dict(t=10))
        st.plotly_chart(fig, width="stretch")
        dist = (d["class"].value_counts(normalize=True) * 100).reindex(A.CLASSES).fillna(0).round(1)
        st.bar_chart(dist)
        st.caption("Colors: grey Silence · blue Instructor Lecture · green Student Discussion · red High Ambient Distraction")
        st.dataframe(d, width="stretch", hide_index=True)
    else:
        st.info("Upload a WAV file or load the demo session.")

# ---------------- Tab 3 ----------------
F = fuse(S.vis, S.aud, win, w_vis) if (len(S.vis) or len(S.aud)) else pd.DataFrame()
with tabs[2]:
    st.subheader("Multimodal fusion & Engagement Index")
    if len(F):
        fig = go.Figure(go.Scatter(x=F.minute, y=F.engagement, mode="lines", fill="tozeroy", line=dict(color="#2563eb"), name="Engagement"))
        fig.add_hline(y=low, line_dash="dash", line_color="#dc2626", annotation_text=f"Low ({low}%)")
        pk, dp = F.engagement.idxmax(), F.engagement.idxmin()
        fig.add_trace(go.Scatter(x=[F.minute[pk]], y=[F.engagement[pk]], mode="markers+text", text=["Peak"], textposition="top center", marker=dict(color="#16a34a", size=11), name="Peak"))
        fig.add_trace(go.Scatter(x=[F.minute[dp]], y=[F.engagement[dp]], mode="markers+text", text=["Dip"], textposition="bottom center", marker=dict(color="#dc2626", size=11), name="Dip"))
        fig.update_layout(height=380, yaxis=dict(range=[0, 105], title="Engagement Index"), xaxis_title="Minutes", margin=dict(t=10))
        st.plotly_chart(fig, width="stretch")
        st.markdown("**Unified time-step matrix** (every score is explained by the rule that produced it)")
        st.dataframe(F[["minute", "visual", "class", "energy_db", "raw", "engagement", "reason"]].round(1), width="stretch", hide_index=True)
    else:
        st.info("Provide vision and/or audio data (or load the demo) to compute engagement.")

# ---------------- Tab 4 ----------------
SUG = suggest(F, win, low, sustain) if len(F) else []
with tabs[3]:
    st.subheader("AI teaching suggestions")
    st.caption("Advisory only, about the whole class. Prompts are explainable and never identify individual students.")
    if not len(F):
        st.info("No session data yet.")
    elif not SUG:
        st.success("No intervention needed: engagement stayed healthy.")
    for s in SUG:
        st.markdown(f'<div class="card {s["severity"]}"><b>{s["title"]}</b> · {s["start_min"]}–{s["end_min"]} min · <i>{s["severity"]}</i><br>'
                    f'{s["suggestion"]}<br><small>Why: {s["why"]}</small></div>', unsafe_allow_html=True)

# ---------------- Tab 5 ----------------
with tabs[4]:
    st.subheader("Session analytics & educator report")
    if len(F):
        e = F.engagement.dropna()
        att = S.att if S.att is not None else pd.DataFrame()
        att_rate = (att.Status == "Present").mean() * 100 if len(att) else float("nan")
        part = (F["class"] == "Student Discussion").mean() * 100
        noise = (F["class"] == "High Ambient Distraction").mean() * 100
        hi, lo = (e >= 70).mean() * 100, (e < low).mean() * 100
        k = st.columns(5)
        k[0].metric("Avg engagement", f"{e.mean():.0f}/100"); k[1].metric("Attendance rate", f"{att_rate:.0f}%" if len(att) else "n/a")
        k[2].metric("Participation", f"{part:.0f}%"); k[3].metric("High-engagement time", f"{hi:.0f}%"); k[4].metric("Distraction noise", f"{noise:.0f}%")
        st.line_chart(F.set_index("minute").engagement)
        peak_m, dip_m = F.minute[e.idxmax()], F.minute[e.idxmin()]
        rpt = [f"# Session Summary Report", f"Generated: {datetime.now():%Y-%m-%d %H:%M}", "",
               "## Key KPIs", f"- Average Engagement Index: {e.mean():.1f}/100", f"- Peak: {e.max():.0f} at minute {peak_m:.1f}; lowest: {e.min():.0f} at minute {dip_m:.1f}",
               f"- Time in high engagement (>=70): {hi:.0f}% | below {low}: {lo:.0f}%", f"- Participation (student discussion windows): {part:.0f}%",
               f"- Ambient distraction windows: {noise:.0f}%", f"- Attendance rate: {att_rate:.0f}% (sustained presence >= {int(thr*100)}%)" if len(att) else "- Attendance: n/a",
               "", "## Recommendations"] + ([f"- [{s['severity']}] {s['start_min']}-{s['end_min']} min: {s['suggestion']}" for s in SUG] or ["- None needed."]) + \
              ["", "## Student self-awareness note", "Collective focus was strongest during interactive segments; consider reviewing material from low-engagement periods.",
               "", "_Privacy: all processing was local; no individual engagement scores were produced._"]
        txt = "\n".join(rpt)
        c1, c2 = st.columns(2)
        c1.download_button("⬇ Report (Markdown)", txt, "session_report.md"); c2.download_button("⬇ Timeline CSV", F.to_csv(index=False).encode(), "engagement_timeline.csv")
        with st.expander("Preview report"):
            st.markdown(txt)
    else:
        st.info("Run an analysis or load the demo to see the report.")
