"""Promo Finder: the editor's screen.

    streamlit run app.py

Pick a show (the fictional demo, an uploaded .zip, or a saved run), run it with rules
only (free) or with an AI model (your key), then review the ranked promos: read the
exact lines of each clip, accept or reject it, and download the sheet.

Keys typed here are used for this session's calls only: never saved, never logged,
never put in the environment (on a shared server that would leak them to other users).
"""
from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from promo_finder.config import load_config
from promo_finder.llm.client import LLMError
from promo_finder.llm.providers import api_key
from promo_finder.pipeline import Pipeline
from promo_finder.uploads import save_uploads

ROOT = Path(__file__).parent
DEMO = ROOT / "samples" / "demo_show"
OUT = ROOT / "outputs"

# Rough per-item costs measured on a real gpt-5-mini run (50 episodes, 150 promos: $0.72).
COST_PER_EPISODE, COST_PER_PROMO = 0.005, 0.003

st.set_page_config(page_title="Promo Finder", page_icon="🎬", layout="wide")


# ---- helpers ------------------------------------------------------------------------------
def saved_runs() -> list[Path]:
    return sorted((p.parent for p in OUT.glob("*/*/promos.json")), key=lambda p: p.stat().st_mtime, reverse=True)


def load_run(d: Path) -> dict:
    return {"dir": d, "meta": json.loads((d / "run_meta.json").read_text()),
            "promos": json.loads((d / "promos.json").read_text())}


def seg_text(p: dict) -> str:
    f = lambda t: f"{int(t // 60):02d}:{t % 60:05.2f}"
    return " | ".join(f"Ep{s['episode']} {f(s['start'])}-{f(s['end'])}" for s in p["segments"])


def feedback_path(run_dir: Path) -> Path:
    return run_dir / "feedback.jsonl"


def load_feedback(run_dir: Path) -> dict[str, dict]:
    fb = {}
    if feedback_path(run_dir).exists():
        for line in feedback_path(run_dir).read_text().splitlines():
            r = json.loads(line)
            fb[r["promo_id"]] = r  # latest decision wins
    return fb


def save_feedback(run_dir: Path, run_id: str, promo_id: str, decision: str, note: str) -> None:
    rec = {"ts": datetime.now().isoformat(timespec="seconds"), "run_id": run_id,
           "promo_id": promo_id, "decision": decision, "note": note}
    with open(feedback_path(run_dir), "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ---- sidebar: what to run and how ---------------------------------------------------------
st.sidebar.title("🎬 Promo Finder")
st.sidebar.caption("Finds 20-60s promo clips in a microdrama screenplay and ranks them.")

runs = saved_runs()
sources = ["Demo show (fictional)", "Upload episodes"] + (["Open a saved run"] if runs else [])
source = st.sidebar.radio("Show", sources)

input_path = None
if source == "Demo show (fictional)":
    st.sidebar.caption("3 short original episodes in the same script format. Not Kuku content.")
    input_path = DEMO
elif source == "Upload episodes":
    ups = st.sidebar.file_uploader("Episode .txt files, or a .zip of them (any mix)", type=["txt", "zip"],
                                   accept_multiple_files=True)
    if ups:
        try:
            input_path = save_uploads(ups)
            st.sidebar.caption(f"{len(list(input_path.rglob('*.txt')))} episode files ready.")
        except Exception as e:
            st.sidebar.error(f"Could not read the upload: {e}")
else:
    pick = st.sidebar.selectbox("Saved run", runs, format_func=lambda p: f"{p.parent.name} / {p.name}")
    if st.sidebar.button("Open", use_container_width=True):
        st.session_state["run"] = load_run(pick)

if source != "Open a saved run":
    st.sidebar.divider()
    mode = st.sidebar.radio("Engine", ["Rules only (free, no key)", "AI models (needs a key)"])
    live = mode.startswith("AI")
    keys: dict[str, str] = {}
    vendor, model, reasoning, max_cands = "auto", "", None, 150
    if live:
        vendor = st.sidebar.selectbox("Vendor", ["openai", "anthropic", "gemini"])
        env_key = api_key(vendor)
        typed = st.sidebar.text_input(f"{vendor} API key", type="password",
                                      help="Used for this session only. Never saved or logged.",
                                      placeholder="using the key from .env" if env_key else "paste a key")
        if typed:
            keys[vendor] = typed.strip()
        model = st.sidebar.text_input("Model override (optional)", placeholder=f"{vendor}:model-name")
        reasoning = st.sidebar.selectbox("Thinking effort", [None, "minimal", "low", "medium", "high"],
                                         format_func=lambda x: "model default" if x is None else x)
        max_cands = st.sidebar.slider("Promos to judge (cost control)", 20, 150, 80, 10)
        n_eps = len(list(Path(input_path).rglob("*.txt"))) if input_path else 0
        st.sidebar.caption(f"Estimated cost: about ${n_eps * COST_PER_EPISODE + max_cands * COST_PER_PROMO:.2f} "
                           f"on a small model ({n_eps} episodes, {max_cands} promos). Reruns are cached.")
        if not (keys or env_key):
            st.sidebar.warning("Add a key to run with AI.")

    run_clicked = st.sidebar.button("Find promos", type="primary", use_container_width=True,
                                    disabled=input_path is None or (live and not (keys or api_key(vendor))))
    if run_clicked:
        overrides = {"llm.mode": "live" if live else "mock", "paths.output_dir": str(OUT),
                     "promo.max_candidates": max_cands}
        if live:
            overrides.update({"llm.preset": vendor, "llm.reasoning": reasoning})
            if model:
                overrides["llm.models"] = {k: model for k in ("default", "show_bible", "miner", "judge")}
        cfg = load_config(overrides=overrides)
        cfg.llm.api_keys = keys  # in memory only (excluded from every dump)
        script_thread = threading.get_ident()
        log: list[str] = []
        with st.status("Finding promos...", expanded=True) as status:
            def progress(msg: str) -> None:
                log.append(msg)
                if threading.get_ident() == script_thread:  # Streamlit can only draw from this thread
                    status.write(msg)
            try:
                t0 = time.time()
                res = Pipeline(cfg, str(input_path), progress=progress).run()
                status.update(label=f"Done in {time.time() - t0:.0f}s", state="complete", expanded=False)
                st.session_state["run"] = load_run(res["out_dir"])
            except (LLMError, FileNotFoundError, ValueError) as e:
                status.update(label="Run failed", state="error")
                st.error(str(e))
            except Exception as e:  # show, don't crash the page
                status.update(label="Run failed", state="error")
                st.exception(e)


# ---- main area --------------------------------------------------------------------------------
run = st.session_state.get("run")
if not run:
    st.title("Promo Finder")
    st.markdown(
        "Pick a show on the left and press **Find promos**.\n\n"
        "- **Rules only** is free and instant: keyword and timing rules pick the moments.\n"
        "- **AI models** read each episode, find hooks and cliffhangers by line id, and judge every candidate "
        "as a stranger would. Code sets every timestamp and checks every rule (20-60s, hook in 3s, no cut "
        "mid-line).\n\nThen review: read each clip's exact lines, accept or reject, and download the sheet.")
    st.stop()

meta, promos, run_dir = run["meta"], run["promos"], run["dir"]
usage = meta["llm_usage"]["_total"]
st.title(meta["show"])
c = st.columns(5)
c[0].metric("Episodes", meta["episodes"])
c[1].metric("Runtime (min)", meta["runtime_min"])
c[2].metric("Promos ranked", len(promos))
c[3].metric("Engine", "rules" if meta["mode"] == "mock" else "AI")
c[4].metric("LLM cost", f"${usage['cost_usd']:.2f}")
if meta["mode"] == "mock":
    st.info("Rules-only run: scores are rule-based estimates, not a model's judgement.")

tab_review, tab_dl, tab_meta = st.tabs(["Review promos", "Download", "Run details"])

with tab_review:
    fb = load_feedback(run_dir)
    f1, f2, f3 = st.columns([2, 2, 3])
    order = f1.radio("Order", ["Posting order", "Quality order"], horizontal=True,
                     help="Posting = quality + early-episode funnel + variety. Quality = promo quality alone.")
    tiers = sorted({p["tier"] for p in promos})
    chosen = f2.multiselect("Tiers", tiers, default=[t for t in tiers if t != "backlog"] or tiers)
    show_n = f3.slider("Show top", 5, len(promos), min(20, len(promos)))
    key = "rank" if order == "Posting order" else "quality_rank"
    view = sorted([p for p in promos if p["tier"] in chosen],
                  key=lambda p: p.get(key) or p["rank"])[:show_n]

    table = pd.DataFrame([{
        "#": p["rank"], "quality #": p.get("quality_rank"), "tier": p["tier"], "id": p["id"],
        "type": p["template"], "secs": round(p["duration"], 1), "segments": seg_text(p),
        "hook": p["hook"], "cliffhanger": p["cliffhanger"], "score": p["scores"]["total"],
        "review": fb.get(p["id"], {}).get("decision", "")} for p in view])
    st.dataframe(table, hide_index=True, use_container_width=True)

    decided = [r for r in fb.values()]
    if decided:
        acc = sum(r["decision"] == "accept" for r in decided)
        st.caption(f"Reviewed {len(decided)} promos: {acc} accepted "
                   f"({100 * acc / len(decided):.0f}% acceptance, the main success metric once editors use it).")

    for p in view:
        done = fb.get(p["id"], {}).get("decision")
        badge = {"accept": "✅", "reject": "❌", "edit": "✏️"}.get(done, "")
        with st.expander(f"{badge} #{p['rank']} · {p['id']} · {p['template']} · {p['duration']:.0f}s · "
                         f"{p['hook'][:70]}"):
            a, b = st.columns([3, 2])
            with a:
                st.markdown(f"**Segments:** {seg_text(p)}  \n**Hook:** {p['hook']}  \n"
                            f"**Cliffhanger:** {p['cliffhanger']}  \n**Why it works:** {p['why']}  \n"
                            f"**Why this rank:** {p['rank_reason']}")
                if p.get("cold_summary"):
                    st.markdown(f"**Cold viewer read:** {p['cold_summary']}")
                if p.get("confusions"):
                    st.markdown("**Could confuse a stranger:** " + "; ".join(p["confusions"]))
                if p.get("flags"):
                    st.markdown("**Flags:** " + ", ".join(f"`{f}`" for f in p["flags"]))
                if p.get("clip_lines"):
                    st.code("\n".join(p["clip_lines"]), language=None, wrap_lines=True)
            with b:
                s = p["scores"]
                st.markdown("**Scores (0-1)**")
                st.dataframe(pd.DataFrame({k: [round(s.get(k, 0), 2)] for k in
                                           ("hook", "cliffhanger", "standalone", "emotion", "funnel")}),
                             hide_index=True)
                with st.form(f"fb-{p['id']}"):
                    opts = ["accept", "reject", "edit"]
                    decision = st.radio("Editor decision", opts, horizontal=True,
                                        index=opts.index(done) if done in opts else 0,
                                        format_func={"accept": "Accept", "reject": "Reject",
                                                     "edit": "Needs edit"}.get)
                    note = st.text_input("Note (why, or the better cut)", value=fb.get(p["id"], {}).get("note", ""))
                    if st.form_submit_button("Save review"):
                        save_feedback(run_dir, meta["run_id"], p["id"], decision, note)
                        st.rerun()

with tab_dl:
    st.markdown("Everything for this run, as written to its output folder.")
    for name, mime in (("promos_ranked.csv", "text/csv"), ("promos_ranked.md", "text/markdown"),
                       ("promos.json", "application/json"), ("rejected.json", "application/json"),
                       ("run_meta.json", "application/json"), ("feedback.jsonl", "application/json")):
        f = run_dir / name
        if f.exists():
            st.download_button(f"Download {name}", f.read_bytes(), file_name=name, mime=mime, key=f"dl-{name}")

with tab_meta:
    st.markdown(f"**Run:** `{meta['run_id']}` · **mode:** {meta['mode']} · **models:** {meta['models']}")
    st.markdown("**Seconds per stage**")
    st.bar_chart(pd.Series(meta["stage_seconds"]))
    st.markdown("**LLM usage by task**")
    st.dataframe(pd.DataFrame({k: v for k, v in meta["llm_usage"].items() if not k.startswith("_")}).T,
                 use_container_width=True)
    st.json(usage, expanded=False)
    if meta.get("parse_warnings"):
        st.markdown("**Script parse warnings**")
        st.write(meta["parse_warnings"])
