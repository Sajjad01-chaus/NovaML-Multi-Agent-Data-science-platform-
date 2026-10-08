"""Streamlit front end.

    streamlit run ui/streamlit_app.py                       # in-process (single user, dev)
    NOVAML_API_URL=http://localhost:8080 streamlit run ...  # thin client over the HTTP API

Both modes go through the same small `Backend` interface, so the screens don't care
whether runs execute in this process or on a pool of workers.
"""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from novaml.tools.ml import to_display

st.set_page_config(page_title="NovaML", layout="wide")


@dataclass
class RunView:
    run_id: str
    status: str
    state: dict[str, Any]
    pending: dict[str, Any] | None = None
    errors: list[str] = field(default_factory=list)


class LocalBackend:
    label = "in-process"

    def __init__(self):
        from novaml.config import get_settings
        from novaml.service import NovaML

        self.svc = NovaML(get_settings())

    def caption(self) -> str:
        s = self.svc.settings
        return f"LLM: {s.effective_provider()} · model: {s.resolved_model() or 'policies only'}"

    def start(self, data: bytes, filename: str, target: str, problem_type: str | None) -> RunView:
        upload_dir = self.svc.settings.data_dir / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        dest = upload_dir / f"{uuid.uuid4().hex}{Path(filename).suffix.lower()}"  # never trust client filenames
        dest.write_bytes(data)
        return self._view(self.svc.start(dest, target, problem_type))

    def get(self, run_id: str) -> RunView:
        return self._view(self.svc.get(run_id))

    def approve(self, run_id: str, models: list[str]) -> RunView:
        return self._view(self.svc.resume(run_id, models))

    def model_zip(self, run_id: str) -> bytes | None:
        from novaml.artifacts import make_store

        return make_store(self.svc.settings, run_id).zip_dir("bundle")

    @staticmethod
    def _view(r) -> RunView:
        return RunView(r.run_id, r.status, r.state, r.pending, r.errors)


class ApiBackend:
    label = "api"

    def __init__(self, url: str):
        from novaml.client import NovaMLClient

        self.url = url
        self.c = NovaMLClient(url)

    def caption(self) -> str:
        return f"API: {self.url}"

    def start(self, data: bytes, filename: str, target: str, problem_type: str | None) -> RunView:
        ds = self.c.upload(data=data, filename=filename)
        run = self.c.create_run(ds["dataset_id"], target, problem_type)
        return self._follow(run["id"])

    def get(self, run_id: str) -> RunView:
        return self._view(self.c.get_run(run_id))

    def approve(self, run_id: str, models: list[str]) -> RunView:
        self.c.approve(run_id, models)
        return self._follow(run_id)

    def model_zip(self, run_id: str) -> bytes | None:
        r = self.c.http.get(f"/v1/runs/{run_id}/model")
        return r.content if r.status_code == 200 else None

    def _follow(self, run_id: str) -> RunView:
        """Show live progress from the event stream until the run settles."""
        box = st.empty()
        lines: list[str] = []
        for ev in self.c.events(run_id):
            data = ev.get("data") or {}
            if ev["event"] == "agent_done":
                lines += data.get("messages", []) or [f"[{data.get('node')}] done"]
            elif ev["event"] == "status":
                lines.append(f"status: {data.get('status')}")
            box.code("\n".join(lines[-25:]) or "queued…")
        time.sleep(0.2)
        return self.get(run_id)

    @staticmethod
    def _view(run: dict) -> RunView:
        detail = run.get("detail") or {}
        state = {**detail, "run_id": run["id"]}
        errors = detail.get("errors") or ([run["error"]] if run.get("error") else [])
        return RunView(run["id"], run["status"], state, detail.get("pending"), errors)


@st.cache_resource
def backend():
    url = os.getenv("NOVAML_API_URL")
    return ApiBackend(url) if url else LocalBackend()


def sidebar() -> None:
    b = backend()
    with st.sidebar:
        st.header("New run")
        st.caption(b.caption())
        up = st.file_uploader("Dataset", type=["csv", "xlsx", "xls", "parquet"])
        target = st.text_input("Target column")
        pt = st.selectbox("Problem type", ["auto", "classification", "regression"])
        if st.button("Run", type="primary", disabled=not (up and target)):
            with st.spinner("Agents are working…"):
                try:
                    st.session_state.result = b.start(bytes(up.getbuffer()), up.name, target.strip(), None if pt == "auto" else pt)
                except Exception as e:
                    st.error(str(e))
                    return
            st.rerun()
        st.divider()
        rid = st.text_input("Open run by id")
        if st.button("Open") and rid:
            try:
                st.session_state.result = b.get(rid.strip())
            except Exception:
                st.error("run not found")
            st.rerun()


def approval(r: RunView) -> None:
    st.info("Human review: approve the models to train.")
    st.write(r.pending.get("reasoning", ""))
    with st.form("approve"):
        picks = [m for m in r.pending["candidates"] if st.checkbox(m, value=True)]
        if st.form_submit_button("Approve and train"):
            try:
                with st.spinner("Training, evaluating, packaging…"):
                    st.session_state.result = backend().approve(r.run_id, picks)
            except Exception as e:
                st.error(str(e))
                return
            st.rerun()


def results(r: RunView) -> None:
    s = r.state
    pt, metric = s["problem_type"], s["metric"]
    card = s.get("model_card") or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best model", s["best"]["model"])
    c2.metric(f"CV {metric}", f"{to_display(metric, pt, s['best']['cv_score']):.4f}")
    if metric in s["holdout"]["model"]:
        c3.metric(f"Holdout {metric}", f"{s['holdout']['model'][metric]:.4f}")
    c4.metric("LLM cost", f"${(card.get('llm') or {}).get('cost_usd', 0):.4f}")
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
        plan = s.get("plan") or {}
        st.subheader("Plan")
        st.write(plan.get("rationale", ""))
        for risk in plan.get("risks", []):
            st.warning(risk)
        analysis = s.get("analysis") or {}
        st.subheader(f"Analysis ({analysis.get('source', 'n/a')}, {analysis.get('code_steps', 0)} code steps)")
        for x in analysis.get("insights", []):
            st.markdown(f"- {x}")
        st.subheader("Critic rounds")
        crit = [{k: c.get(k) for k in ("round", "decision", "reasoning")} for c in s.get("critiques", [])]
        st.dataframe(pd.DataFrame(crit), use_container_width=True)
    with tabs[2]:
        prof = s.get("profile")
        if prof:
            st.write(f"{prof['n_rows']} training rows · {prof['n_features']} features · {prof['total_missing']} missing cells")
            st.dataframe(pd.DataFrame(prof["columns"]).T, use_container_width=True)
    with tabs[3]:
        st.json(s.get("feature_plan") or {})
        for w in s.get("feature_warnings") or []:
            st.caption(w)
    with tabs[4]:
        st.dataframe(pd.DataFrame(s.get("llm_calls", [])), use_container_width=True)
    with tabs[5]:
        st.dataframe(pd.DataFrame(s.get("events", [])), use_container_width=True)
        st.code("\n".join(s.get("messages", [])))
    with tabs[6]:
        st.json(card)
        data = backend().model_zip(r.run_id)
        if data:
            st.download_button("Download model bundle (.zip)", data, file_name=f"novaml-{r.run_id}.zip")
            st.caption("Serve it: unzip, then `novaml serve <dir>`")


def main() -> None:
    st.title("NovaML")
    sidebar()
    r: RunView | None = st.session_state.get("result")
    if r is None:
        st.write("Upload a dataset, name the target column, and run.")
        return
    st.caption(f"run `{r.run_id}` · status **{r.status}**")
    if r.status in ("failed", "cancelled"):
        for e in r.errors:
            st.error(e)
    elif r.status == "awaiting_approval" and r.pending:
        approval(r)
    elif r.status == "completed":
        results(r)
    else:
        st.info(f"Run is {r.status}. Reopen it by id to refresh.")
    with st.expander("Agent log"):
        st.code("\n".join(r.state.get("messages", [])))


main()
