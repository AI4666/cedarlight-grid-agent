"""Streamlit UI showing the six stages of the cycle.  Run:  PYTHONPATH=src streamlit run ui/app.py
Talks to the agent in-process (no API key needed). Human approval buttons call the same policy-gated code path."""
import json
from datetime import datetime, timezone

import streamlit as st

from agent.scenarios import SCENARIOS, build
from agent.service import AgentService

st.set_page_config(page_title="CedarLight Grid Agent", layout="wide")
if "svc" not in st.session_state:
    st.session_state.svc = AgentService()
    st.session_state.last = None
svc: AgentService = st.session_state.svc

st.title("CedarLight Grid Load & Maintenance Planning Agent")
st.caption("Synthetic data only. High-impact actions are recommendations that need human approval.")
state = svc.state()
c1, c2, c3 = st.columns(3)
c1.metric("Kill switch", "ACTIVE" if state["kill_switch_active"] else "off")
c2.metric("Risk model", state["model"]["source"])
c3.metric("Pending approvals", len(state["pending_approvals"]))

with st.sidebar:
    name = st.selectbox("Scenario", list(SCENARIOS))
    st.write(SCENARIOS[name])
    if st.button("Run scenario", type="primary"):
        st.session_state.last = svc.run_scenario(name)
    st.divider()
    custom = st.text_area("Or paste a JSON list of percepts", height=120)
    if st.button("Run custom percepts") and custom.strip():
        try:
            st.session_state.last = svc.step(json.loads(custom))
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")
    st.divider()
    if st.button("KILL SWITCH", type="secondary"):
        svc.kill("ui-operator", "pressed in UI")
        st.rerun()

res = st.session_state.last
if res:
    t = st.tabs(["1 Percepts", "2 Propositions", "3 Knowledge base", "4 Inference", "5 Entailment", "6 Action", "Audit"])
    with t[0]:
        st.dataframe([{k: p[k] for k in ("source", "entity_id", "validation_status", "quality", "validation_notes")} for p in res["percepts"]])
        if res["rejected"]:
            st.warning(f"{len(res['rejected'])} input(s) rejected"); st.json(res["rejected"])
    with t[1]:
        st.dataframe([{"proposition": f"{'' if p['truth'] else '~'}{p['predicate']}({','.join(p['args'])})",
                       "confidence": p["confidence"], "contradicts": len(p["contradicts"])} for p in res["propositions"]])
    with t[2]:
        st.json(svc.state()["knowledge"])
        if res["conflicts"]:
            st.error("Knowledge conflicts"); st.json(res["conflicts"])
    with t[3]:
        for tr in res["traces"]:
            with st.expander(f"{tr['conclusion']}  (p={tr.get('probability')})"):
                st.json(tr)
    with t[4]:
        st.dataframe([{"query": e["query"], "status": e["status"], "support": len(e["supporting_ids"]),
                       "conflicts": len(e["conflicting_ids"]), "model_conf": e["model_confidence"]} for e in res["entailments"]])
    with t[5]:
        for p in res["proposals"]:
            st.write(f"**{p['tool']}** — tier {p['tier']} — {p['approval']}")
            st.caption(p["justification"])
        for m in res["escalations"]:
            st.warning(m)
    with t[6]:
        st.dataframe(svc.audit(200))

st.subheader("Pending human approvals")
for p in state["pending_approvals"]:
    with st.container(border=True):
        st.write(f"**{p['tool']}** `{p['action_id']}`"); st.caption(p["justification"]); st.json(p["arguments"])
        who = st.text_input("Your name", key=f"who{p['action_id']}")
        a, b = st.columns(2)
        if a.button("Approve", key=f"a{p['action_id']}") and who.strip():
            st.json(svc.approve(p["action_id"], who.strip())); st.rerun()
        if b.button("Reject", key=f"r{p['action_id']}") and who.strip():
            svc.reject(p["action_id"], who.strip()); st.rerun()
