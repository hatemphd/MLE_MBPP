"""Blind reviewer UI for `human_review_sample.csv`.

Run from the project root:

    uv run --with streamlit streamlit run human_edit/app.py
    uv run --with streamlit streamlit run human_edit/app.py -- --file path/to/human_review_sample.csv

Each answer is saved to the CSV immediately, so you can close the browser and continue later.
When finished: `uv run trust-review --output-dir artifacts_pilot_openai`.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from datetime import datetime

import streamlit as st
from checks import (
    DEFAULT_FILE,
    MACHINE_NOTES,
    load_reviews,
    make_executor,
    save_reviews,
)
from checks import load_problems as _load_problems
from checks import run_checks as _run_checks

ANSWERS = {"yes": "Yes, trustworthy", "no": "No, not trustworthy", "": "Not reviewed yet"}
TEST_ICONS = {"pass": "✅ pass", "fail": "❌ fail (assertion is false)", "timeout": "⏱️ timeout"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default=str(DEFAULT_FILE), help="review CSV to edit")
    return parser.parse_known_args(sys.argv[1:])[0]


def backup_once(path):
    """One timestamped copy per session, before the first edit."""
    if not st.session_state.get("backed_up"):
        stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        shutil.copy2(path, f"{path}.{stamp}.bak")
        st.session_state.backed_up = True


@st.cache_data
def load_problems(folder):
    return _load_problems(folder)


@st.cache_resource
def get_executor():
    return make_executor()


def run_checks(code, setup, lines):
    """Lines starting with `assert` are tests; any other non-empty line is an expression to evaluate."""
    return _run_checks(get_executor(), code, setup, lines)


def show_checks(result):
    if result["status"] != "ok":
        st.error(f"The code fails before any check runs: **{result['status']}**  \n{result['error']}")
        return
    if not result["tests"] and not result["probes"]:
        st.success("The code loads without errors (nothing to check).")
    for src, outcome in result["tests"]:
        icon = TEST_ICONS.get(outcome, f"💥 {outcome.replace('error:', 'error: ')}")
        st.markdown(f"{icon}  \n`{src}`")
    for src, value in result["probes"]:
        st.markdown(f"`{src}` → `{value}`")


def default_checks(problem):
    """Starting point for the reviewer's own checks: the probe inputs derived from the example test."""
    probes = problem.get("probes") or []
    lines = ["# One check per line: an `assert ...` or an expression to see its value"]
    lines += probes[:4] if probes else [f"{problem.get('function_name') or 'func'}(...)"]
    return "\n".join(lines)


def try_code_panel(i, row, problem, reveal):
    st.markdown("**Try the code** (runs in a sandboxed process with time limits)")
    code, setup = row["generated_code"] or "", problem.get("setup_code", "")
    results = st.session_state.setdefault("check_results", {})
    visible = problem.get("visible_tests") or []

    buttons = st.columns(3)
    if buttons[0].button("▶ Run code", key=f"run-{i}", use_container_width=True,
                         help="Does the code load at all (syntax errors, crashes at import)?"):
        results[(i, "run")] = run_checks(code, setup, [])
    if buttons[1].button("▶ Run example test", key=f"visible-{i}", use_container_width=True,
                         disabled=not visible, help="The one test the model was shown."):
        results[(i, "visible")] = run_checks(code, setup, visible)
    hidden = problem.get("hidden_tests") or []
    if buttons[2].button("▶ Run hidden tests", key=f"hidden-{i}", use_container_width=True,
                         disabled=not (reveal and hidden),
                         help="Turn on 'Reveal' first: these decide the automated label."):
        results[(i, "hidden")] = run_checks(code, setup, hidden)

    for kind, title in (("run", "Run code"), ("visible", "Example test"), ("hidden", "Hidden tests")):
        if (i, kind) in results and (kind != "hidden" or reveal):
            with st.container(border=True):
                st.caption(title)
                show_checks(results[(i, kind)])

    own = st.text_area("Your own checks (edge cases: empty, zero, negative, one element, duplicates...). "
                       "Saved with your answer.",
                       value=row.get("human_checks") or default_checks(problem), height=140, key=f"own-{i}")
    if st.button("▶ Run my checks", key=f"own-run-{i}"):
        results[(i, "own")] = run_checks(code, setup, own.splitlines())
    if (i, "own") in results:
        with st.container(border=True):
            st.caption("Your checks")
            show_checks(results[(i, "own")])


def is_machine_filled(frame):
    return frame["human_notes"].str.startswith(MACHINE_NOTES)


def is_answered(frame):
    return frame["human_trustworthy"].isin(["yes", "no"]) & ~is_machine_filled(frame)


def go_to(index, total):
    st.session_state.row = max(0, min(index, total - 1))


def main():
    st.set_page_config(page_title="Human review", layout="wide")
    path = os.path.abspath(parse_args().file)
    if not os.path.exists(path):
        st.error(f"File not found: {path}. Run `uv run trust-features --output-dir ...` first.")
        st.stop()

    frame = load_reviews(path)
    total = len(frame)
    problems = load_problems(os.path.dirname(path))
    answered = is_answered(frame)
    st.session_state.setdefault("row", int((~answered).idxmax()) if not answered.all() else 0)

    with st.sidebar:
        st.header("Progress")
        st.progress(answered.sum() / max(total, 1), text=f"{answered.sum()} / {total} reviewed")
        counts = frame.loc[answered, "human_trustworthy"].value_counts()
        st.write(f"Yes: {counts.get('yes', 0)}  ·  No: {counts.get('no', 0)}")
        machine = int(is_machine_filled(frame).sum())
        if machine:
            st.info(f"{machine} rows were filled automatically. They are not "
                    "counted as human reviews; answering one yourself replaces the automatic answer.")
        st.divider()
        only_open = st.toggle("Show only unreviewed rows", value=False)
        reveal = st.toggle("Reveal automated label and hidden tests", value=False,
                           help="Keep this off while deciding, so the review stays blind.")
        st.divider()
        st.caption(f"Editing `{path}`")
        st.caption("Answers are saved immediately. A backup copy is made before the first edit.")

    order = list(frame.index[~answered]) if only_open else list(frame.index)
    if not order:
        st.success("All rows are reviewed. Run `uv run trust-review --output-dir artifacts_pilot_openai`.")
        st.stop()
    if st.session_state.row not in order:
        st.session_state.row = order[0]
    position = order.index(st.session_state.row)

    nav = st.columns([1, 1, 1, 3])
    if nav[0].button("◀ Previous", disabled=position == 0, use_container_width=True):
        st.session_state.row = order[position - 1]
        st.rerun()
    if nav[1].button("Next ▶", disabled=position == len(order) - 1, use_container_width=True):
        st.session_state.row = order[position + 1]
        st.rerun()
    if nav[2].button("Next unreviewed", disabled=answered.all(), use_container_width=True):
        later = [i for i in frame.index[~answered] if i > st.session_state.row]
        st.session_state.row = later[0] if later else int(frame.index[~answered][0])
        st.rerun()
    jump = nav[3].number_input("Go to row", min_value=1, max_value=total,
                               value=st.session_state.row + 1, step=1)
    if jump - 1 != st.session_state.row:
        go_to(jump - 1, total)
        st.rerun()

    i = st.session_state.row
    row = frame.loc[i]
    problem = problems.get(str(row["task_id"]), {})
    status = "reviewed" if answered[i] else "not reviewed"
    st.subheader(f"Row {i + 1} of {total}  ·  task {row['task_id']}  ·  {status}")
    st.caption(f"Candidate `{row['candidate_id']}`  ·  model `{row['model_name']}`")

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Problem**")
        st.info(row["problem_text"])
        if problem.get("visible_tests"):
            st.markdown("**Example test given to the model**")
            st.code("\n".join(problem["visible_tests"]), language="python")
        st.markdown("**Generated code**")
        st.code(row["generated_code"] or "(empty)", language="python", line_numbers=True)
        st.divider()
        try_code_panel(i, row, problem, reveal)

    with right:
        st.markdown("**Your judgement**")
        st.caption("Would you approve this code for the task as described? Judge correctness for all "
                   "reasonable inputs and obvious security problems, not style.")
        current = row["human_trustworthy"] if answered[i] else ""
        with st.form(key=f"review-{i}"):
            choice = st.radio("Trustworthy?", list(ANSWERS), format_func=ANSWERS.get,
                              index=list(ANSWERS).index(current))
            notes = "" if row["human_notes"].startswith(MACHINE_NOTES) else row["human_notes"]
            notes = st.text_area("Notes (why, especially for 'no' or an ambiguous problem)", value=notes,
                                 height=120)
            col_a, col_b = st.columns(2)
            save = col_a.form_submit_button("Save", use_container_width=True)
            save_next = col_b.form_submit_button("Save and next ▶", type="primary", use_container_width=True)
        if save or save_next:
            backup_once(path)
            frame.loc[i, "human_trustworthy"] = choice
            frame.loc[i, "human_notes"] = notes.strip()
            frame.loc[i, "human_checks"] = st.session_state.get(f"own-{i}", "").strip()
            save_reviews(frame, path)
            if save_next:
                later = [j for j in order if j > i]
                if later:
                    st.session_state.row = later[0]
            st.toast(f"Saved row {i + 1}")
            st.rerun()

        if reveal:
            st.divider()
            st.markdown("**Automated result (hidden while reviewing blind)**")
            label = "trustworthy" if str(row["label_trustworthy"]) in ("1", "True", "true") else "not trustworthy"
            st.write(f"Execution status: `{row['exec_status']}`  ·  automated label: **{label}**")
            if problem.get("hidden_tests"):
                st.code("\n".join(problem["hidden_tests"]), language="python")


main()
