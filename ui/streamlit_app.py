"""Streamlit front end. A thin client over `novaml.service.NovaML`.

    streamlit run ui/streamlit_app.py
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pandas as pd
import streamlit as st

from novaml.config import get_settings
from novaml.service import InvalidResume, NovaML, RunResult
from novaml.tools.ml import to_display

st.set_page_config(page_title="NovaML", layout="wide")


@st.cache_resource
def service() -> NovaML:
    return NovaML(get_settings())


def sidebar() -> None:
    svc = service()
    with st.sidebar:
        st.header("New run")
        st.caption(f"LLM: {svc.settings.llm_provider} · model: {svc.settings.resolved_model() or 'policies only'}")
        up = st.file_uploader("Dataset", type=[e.lstrip(".") for e in svc.settings.allowed_extensions])
        target = st.text_input("Target column")
        pt = st.selectbox("Problem type", ["auto", "classification", "regression"])
        if st.button("Run", type="primary", disabled=not (up and target)):
            upload_dir = svc.settings.data_dir / "uploads"
            upload_dir.mkdir(parents=True, exist_ok=True)
            # Never trust the client filename for the on-disk path.
            dest = upload_dir / f"{uuid.uuid4().hex}{Path(up.name).suffix.lower()}"
            dest.write_bytes(up.getbuffer())
            with st.spinner("Agents are working…"):
                st.session_state.result = svc.start(dest, target.strip(), None if pt == "auto" else pt)
            st.rerun()
        st.divider()
        rid = st.text_input("Open run by id")
        if st.button("Open") and rid:
            try:
                st.session_state.result = svc.get(rid.strip())
            except KeyError:
                st.error("run not found")
            st.rerun()


def approval(r: RunResult) -> None:
    st.info("Human review: approve the models to train.")
    st.write(r.pending.get("reasoning", ""))
    with st.form("approve"):
        picks = [m for m in r.pending["candidates"] if st.checkbox(m, value=True)]
        if st.form_submit_button("Approve and train"):
            try:
                with st.spinner("Training, evaluating, packaging…"):
                    st.session_state.result = service().resume(r.run_id, picks)
            except InvalidResume as e:
                st.error(str(e))
                return
            st.rerun()


def results(r: RunResult) -> None:
    s = r.state
    pt, metric = s["problem_type"], s["metric"]
    card = s.get("model_card", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best model", s["best"]["model"])
    c2.metric(f"CV {metric}", f"{to_display(metric, pt, s['best']['cv_score']):.4f}")
    if metric in s["holdout"]["model"]:
        c3.metric(f"Holdout {metric}", f"{s['holdout']['model'][metric]:.4f}")
    c4.metric("LLM cost", f"${card.get('llm', {}).get('cost_usd', 0):.4f}")
    for w in card.get("warnings", []):
        st.warning(w)

    tabs = st.tabs(["Leaderboard", "Reasoning", "Data", "Features", "LLM calls", "Timeline", "Model card"])
    with tabs[0]:
        lb = pd.DataFrame(s.get("leaderboard", []))
        if not lb.empty and "cv_score" in lb:
            lb["cv"] = lb["cv_score"].map(lambda v: to_display(metric, pt, v) if pd.notna(v) else None)
            cols = [c for c in ["round", "model", "mode", "cv", "cv_std", "overfit_gap", "seconds", "params", "error"] if c in lb]
            st.dataframe(lb[cols], use_container_width=True)
        st.json(s.get("holdout", {}))
    with tabs[1]:
        plan = s.get("plan", {})
        st.subheader("Plan")
        st.write(plan.get("rationale", ""))
        for r in plan.get("risks", []):
            st.warning(r)
        analysis = s.get("analysis") or {}
        st.subheader(f"Analysis ({analysis.get('source', 'n/a')}, {analysis.get('code_steps', 0)} code steps)")
        for x in analysis.get("insights", []):
            st.markdown(f"- {x}")
        st.subheader("Critic rounds")
        crit = [{k: c.get(k) for k in ("round", "decision", "reasoning")} for c in s.get("critiques", [])]
        st.dataframe(pd.DataFrame(crit), use_container_width=True)
    with tabs[2]:
        prof = s["profile"]
        st.write(f"{prof['n_rows']} training rows · {prof['n_features']} features · {prof['total_missing']} missing cells")
        st.dataframe(pd.DataFrame(prof["columns"]).T, use_container_width=True)
    with tabs[3]:
        st.json(s.get("feature_plan", {}))
        for w in s.get("feature_warnings", []):
            st.caption(w)
    with tabs[4]:
        st.dataframe(pd.DataFrame(s.get("llm_calls", [])), use_container_width=True)
    with tabs[5]:
        st.dataframe(pd.DataFrame(s.get("events", [])), use_container_width=True)
        st.code("\n".join(s.get("messages", [])))
    with tabs[6]:
        st.json(card)
        st.caption(f"Serve it: novaml serve {s.get('bundle_dir')}")


def main() -> None:
    st.title("NovaML")
    sidebar()
    r: RunResult | None = st.session_state.get("result")
    if r is None:
        st.write("Upload a dataset, name the target column, and run.")
        return
    st.caption(f"run `{r.run_id}` · status **{r.status}**")
    if r.status == "failed":
        for e in r.errors:
            st.error(e)
    elif r.status == "awaiting_approval":
        approval(r)
    elif r.status == "completed":
        results(r)
    with st.expander("Agent log"):
        st.code("\n".join(r.state.get("messages", [])))


main()
