"""Tests of the assistant's modes (fast / smart / ultra / auto) and the agent team inside the Router, with scripted
models: one test per role (understander, planner, parallel builders, assembler, checker, fixer, critic, reporter),
the mode presets and auto's choice, how a client picks a mode (model name, "mode" field, X-OOA-Mode header,
GET /v1/models) and one end-to-end run through the HTTP server.
Run: python test_modes.py PLATFORM_REPO   (or OOA_PLATFORM=... python -m pytest test_modes.py)"""
import json, os, sys, threading, time, urllib.request
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import test_layers as TL                                    # noqa: E402  (shared scripted models and flows)
from test_layers import S, F, CAT, plan_answer, PlanAware, LOOP_WHOLE, LOOP_REQ, _loop_parts  # noqa: E402

CSV_TO_EXCEL, CSV_REQ = TL.CSV_TO_EXCEL, TL.CSV_REQ


def router(local, cloud=None, examples=None, prompt_format="context", mode=None, **policy):
    return S.Router(TL.CATALOG, TL.PLATFORM, S.Policy(**policy), local, cloud, examples, prompt_format, mode=mode)


class BatchAware(PlanAware):
    """PlanAware with generate_batch, as NanochatLocal has: records each batch's size and the items' sampling."""
    def __init__(self, answer, stops=None, info=None):
        super().__init__(answer, stops)
        self.batches, self.samples, self.deadlines = [], [], []
        if info is not None:
            self.info = info

    def generate_batch(self, items, deadline=None):
        self.batches.append(len(items))
        self.deadlines.append(deadline)
        texts, stats = [], []
        for it in items:
            self.samples.append((it.get("temperature"), it.get("seed")))
            self._item = it
            texts.append(self.generate(it["prompt"], it.get("attempt", 0), it.get("max_tokens") or 2500,
                                       it.get("on_plan"), it.get("prefix")))
            stats.append(dict(self.last_stats))
        self.last_batch_stats = stats
        return texts


def agents(res):
    return [a["agent"] for a in res.get("agents") or []]


# ------------------------------------------------------------------------------------------- presets and auto
def test_mode_presets():
    base = S.Policy(allow_escalation=False, mask_pii=True, store_logs=True)
    fast, smart, ultra = (base.for_mode(m) for m in ("fast", "smart", "ultra"))
    assert fast.mode == "fast" and fast.local_retries == 0 and not fast.use_parts and fast.fixer_rounds == 0
    assert smart.local_retries == 2 and smart.use_parts and smart.parallel_parts and smart.fixer_rounds == 2
    assert not smart.critic and not smart.understander and smart.best_of == 1
    assert ultra.critic and ultra.understander and ultra.best_of == 4 and ultra.fixer_rounds == 3
    assert ultra.local_retries == 3 and ultra.decompose_lines < smart.decompose_lines
    assert 0.2 <= ultra.sample_temperature <= 0.3
    assert ultra.stop_after <= 120 and ultra.time_limit <= 150 < 180          # Studio waits 180 s
    for p in (fast, smart, ultra):                                          # the client's data settings stay
        assert p.store_logs and p.mask_pii and not p.allow_escalation
    try:
        base.for_mode("turbo")
        raise AssertionError("an unknown mode must be refused")
    except ValueError:
        pass
    try:
        S.Router(TL.CATALOG, TL.PLATFORM, base, mode="turbo")
        raise AssertionError("an unknown default mode must be refused")
    except ValueError:
        pass


def test_auto_choice():
    pick = lambda r, rw=False: S.choose_mode(r, rw)[0]
    assert pick("Copy all PDF files from the inbox folder to the archive folder") == "fast"
    assert pick("Read a CSV file into a table, then write the table to an Excel sheet") == "fast"
    assert pick("For each Excel file in the reports folder, read the first sheet and email it") == "smart"   # loop
    assert pick("If the file report.pdf exists, delete it") == "smart"                                     # condition
    assert pick("Read the csv file, write it to excel, then zip the excel file") == "smart"                # 3 parts
    assert pick(LOOP_REQ) == "ultra"                                                                       # 6 parts
    assert pick("word " * 61 + "file") == "ultra"                                                         # long
    assert pick("do the usual thing for the boss") == "smart"                         # vague, older model
    assert pick("do the usual thing for the boss", True) == "ultra"                   # vague, model restates
    assert pick("excel file padh ke unpaid orders filter karo") == "fast"
    mode, why = S.choose_mode(LOOP_REQ)
    assert why.startswith("a long process") and isinstance(why, str)


def test_request_mode_and_model_names():
    rm = S.request_mode
    assert rm({"model": "ooa-assistant"}) == (None, None)                     # the server's default mode
    assert rm({"model": "ooa-flow-assistant"}) == ("auto", None)
    assert rm({"model": "ooa-flow-assistant-fast"}) == ("fast", None)
    assert rm({"model": "ooa-flow-assistant-ultra"}, "smart") == ("smart", None)    # the header wins over the name
    assert rm({"model": "ooa-flow-assistant-ultra", "mode": "Fast"}, "smart") == ("fast", None)   # the field first
    m, err = rm({"mode": "turbo"})
    assert m is None and "fast" in err
    names = [d["id"] for d in S.models_list()["data"]]
    assert names == ["ooa-assistant", "ooa-flow-assistant", "ooa-flow-assistant-fast", "ooa-flow-assistant-smart",
                     "ooa-flow-assistant-ultra"]


# ------------------------------------------------------------------------------------------- one test per role
def test_planner_and_single_builder_fast():
    """fast: the planner writes the plan and goes on to the flow itself (one builder), one attempt, no parts."""
    m = BatchAware(lambda req, p, a: plan_answer(CSV_TO_EXCEL))
    res = router(m, mode="fast").run(CSV_REQ)
    assert res["verified"] and res["mode"] == "fast" and len(m.prompts) == 1
    assert agents(res) == ["planner", "checker", "reporter"], agents(res)
    assert res["report"].startswith("Mode: Fast, ") and "Agents: planner, checker, reporter." in res["report"]
    # one attempt only, and a long plan is never cut into parts
    m2 = BatchAware(lambda req, p, a: "Plan:\n" + "\n".join(F.plan_from_flow(LOOP_WHOLE, CAT)) + "\nFlow:\n{")
    res = router(m2, mode="fast").run(LOOP_REQ)
    assert not res["verified"] and len(m2.prompts) == 1 and not m2.options[0]["on_plan"]


def _loop_answer(fail_part3_first=False):
    plan = F.plan_from_flow(LOOP_WHOLE, CAT)
    parts = F.plan_parts(plan)
    p1, p2, p3 = _loop_parts()
    reqs = {F.chunk_request(parts[0]["chunk"]): p1, F.chunk_request(parts[2]["chunk"]): p3}
    seen = {"p3": 0}

    def answer(req, prompt, attempt):
        if req == LOOP_REQ:
            return plan_answer(LOOP_WHOLE)
        if req.startswith("With the current file: "):
            return plan_answer(p2)
        flow = reqs[req]
        if flow is p3:
            seen["p3"] += 1
            if fail_part3_first and seen["p3"] == 1:
                return "Plan:\n1. Log (core/log)\nFlow:\n{\"steps\":[{\"action\":\"ooa.nothing/x\"}]}"
        return plan_answer(flow)
    return answer, seen


def test_parallel_builders_and_assembler():
    """smart: the plan stops at "Flow:", the top-level parts are built at once in one batch, the part inside the
    loop after the part that writes the loop; the assembler merges them into the whole flow."""
    answer, _ = _loop_answer()
    m = BatchAware(answer)
    res = router(m, mode="smart").run(LOOP_REQ)
    assert res["verified"] and res.get("built_in_parts") and res["mode"] == "smart", res
    assert m.batches == [1, 2, 1], m.batches                     # planner; parts 1 + 3 together; part 2 (inside)
    assert m.options[0]["on_plan"] and all(not o["on_plan"] for o in m.options[1:])
    assert all(d is not None for d in m.deadlines)               # a running generation is cut at the time limit
    assert F.outline(json.loads(res["flow"])) == F.outline(LOOP_WHOLE)
    assert agents(res) == ["planner", "checker", "builders", "checker", "builders", "checker", "assembler",
                           "reporter"], agents(res)
    b = [a for a in res["agents"] if a["agent"] == "builders"]
    assert b[0]["parts"] == 2 and b[0]["batch"] == 2 and "builders (2 parts at once)" in res["report"]


def test_fixer_rebuilds_only_failed_parts():
    answer, seen = _loop_answer(fail_part3_first=True)
    m = BatchAware(answer)
    res = router(m, mode="smart").run(LOOP_REQ)
    assert res["verified"] and res.get("built_in_parts"), res
    assert m.batches == [1, 2, 1, 1] and seen["p3"] == 2, m.batches      # the fixer's batch has part 3 only
    fixer = [a for a in res["agents"] if a["agent"] == "fixer"]
    assert len(fixer) == 1 and fixer[0]["parts"] == 1 and fixer[0]["round"] == 1
    assert "### Problems" in m.prompts[-1] and "### Previous flow" in m.prompts[-1]
    # with no fixer round the part's failure is the answer's problem
    answer, _ = _loop_answer(fail_part3_first=True)
    m2 = BatchAware(answer)
    r = router(m2, local_retries=0, parts_first=True)
    r.policy = r.policy.for_mode("smart")
    from dataclasses import replace
    r.policy = replace(r.policy, fixer_rounds=0, local_retries=0)
    r._agents, r._mode = [], {"mode": "smart", "asked": "smart", "why": "chosen"}
    res = r._run(LOOP_REQ)
    assert not res["verified"] and any("Part 3 of 3" in p for p in res["problems"]), res["problems"]


def test_assembler_links_shared_variables():
    p1 = {"kind": "TaskBot", "id": "a", "name": "a",
          "variables": [{"name": "csv_path", "type": "String", "direction": "input"},
                        {"name": "orders_table", "type": "Table", "direction": "local"}],
          "steps": [{"id": "s1", "action": "ooa.csv/read", "args": {"path": "$csv_path$"}, "out": {"table": "orders_table"}}]}
    p2 = {"kind": "TaskBot", "id": "b", "name": "b",
          "variables": [{"name": "table", "type": "Table", "direction": "input"},
                        {"name": "xlsx", "type": "String", "direction": "input"}],
          "steps": [{"id": "s1", "action": "ooa.excel/write_table", "args": {"table": "$table$", "note": "${table}"}}]}
    p3 = {"kind": "TaskBot", "id": "c", "name": "c",
          "variables": [{"name": "rows", "type": "Table", "direction": "input"}],
          "steps": [{"id": "s1", "action": "core/log", "args": {"message": "$rows$ and $rows"}}]}
    parts, links = F.link_part_variables([p1, p2, p3])
    assert links == [(2, "table", "orders_table"), (3, "rows", "orders_table")], links
    assert parts[1]["steps"][0]["args"] == {"table": "$orders_table$", "note": "${orders_table}"}
    assert [v["name"] for v in parts[1]["variables"]] == ["xlsx"]          # the input it read is not declared again
    assert parts[2]["steps"][0]["args"]["message"] == "$orders_table$ and $orders_table"
    # a text input with no name in common stays the user's input; variables of the open agent are never linked
    p4 = {"variables": [{"name": "recipient", "type": "String", "direction": "input"}], "steps": []}
    assert F.link_part_variables([p1, p4])[1] == []
    assert F.link_part_variables([p1, p2], known=[{"name": "table"}])[1] == []
    assert F.part_dependencies([{"inside": False}, {"inside": True}, {"inside": 2}, {"inside": True},
                                {"inside": False}]) == [None, 0, 1, 0, None]


def test_checker_takes_the_first_verified_sample():
    """ultra: best-of-4 in one batch (one prompt, the extra samples at the sample temperature); the first verified
    answer is kept."""
    calls = {"n": 0}

    def answer(req, prompt, attempt):
        calls["n"] += 1
        return "garbage {" if calls["n"] == 1 else plan_answer(CSV_TO_EXCEL)
    m = BatchAware(answer)
    res = router(m, mode="ultra").run(CSV_REQ)
    assert res["verified"] and m.batches[0] == 4, m.batches
    assert m.samples[0][0] is None and all(t == 0.25 for t, _ in m.samples[1:4])
    assert len({s for _, s in m.samples[1:4]}) == 3                       # their own seeds
    assert [a.get("ok") for a in res["attempts"]][:2] == [False, True]


class ClearAware(BatchAware):
    """Answers a "Clear request: " start with a restatement line, then its usual plan-first answer."""
    clear = "Read the CSV file and write it to an Excel sheet"

    def generate(self, prompt, attempt=0, max_tokens=2500, on_plan=None, prefix=None):
        self.asked_prefix = getattr(self, "asked_prefix", []) + [prefix]
        if prefix == F.CLEAR_HEAD + " ":
            return prefix + self.clear + "\n" + super().generate(prompt, attempt, max_tokens, on_plan, None)
        return super().generate(prompt, attempt, max_tokens, on_plan, prefix)


def test_understander():
    """ultra on a model trained with rewrite rows: the samples but the last are started with "Clear request: " and the
    restatement reaches Studio's notes; an older model gets no such start; a question (from the free sample) ends
    the request."""
    answer = lambda req, prompt, attempt: plan_answer(CSV_TO_EXCEL)
    m = ClearAware(answer, info={"rewrite": True})
    res = router(m, mode="ultra").run(CSV_REQ, assistant=True)
    assert m.asked_prefix[:4] == [F.CLEAR_HEAD + " "] * 3 + [None], m.asked_prefix      # the last sample is free
    assert res["understood"] == ClearAware.clear, res.get("understood")
    notes = json.loads(res["flow"])["notes"]
    assert notes.startswith(S.UNDERSTOOD + "Read the CSV file") and "Mode: Ultra" in notes, notes
    assert "What it does" not in notes                       # the restatement says it already
    assert agents(res)[0] == "understander + planner"
    old = ClearAware(answer)
    res = router(old, mode="ultra").run(CSV_REQ)
    assert old.asked_prefix[0] is None and agents(res)[0] == "planner" and "understood" not in res
    asks = BatchAware(lambda r, p, a: "Question: Which folder are the invoices in?", info={"rewrite": True})
    res = router(asks, mode="ultra").run("sort out my invoices")
    assert res["question"] == "Which folder are the invoices in?" and not res["verified"]


def test_best_of_never_trades_a_verified_flow_for_a_question():
    """A batch with one verified flow and questions in the other samples returns the flow (ultra is never worse than
    fast); a question ends the request only when no sample of the batch verified, also without the understander."""
    q = "Question: Which CSV file should I read?"

    def mixed(req, prompt, attempt, n=[0]):
        n[0] += 1
        return plan_answer(CSV_TO_EXCEL) if n[0] == 1 else q
    m = BatchAware(mixed)
    res = router(m, mode="ultra").run(CSV_REQ)
    assert res["verified"] and not res.get("question") and m.batches[0] == 4, res

    def late_flow(req, prompt, attempt, n=[0]):
        n[0] += 1
        return plan_answer(CSV_TO_EXCEL) if n[0] == 4 else q
    m = BatchAware(late_flow)
    res = router(m, mode="ultra").run(CSV_REQ)
    assert res["verified"] and not res.get("question"), res
    m = BatchAware(lambda r, p, a: q)
    res = router(m, mode="ultra").run(CSV_REQ)
    assert res["question"] == "Which CSV file should I read?" and not res["verified"]
    m = BatchAware(lambda r, p, a: q)
    assert router(m, mode="fast").run(CSV_REQ)["question"] == "Which CSV file should I read?"

    def fixer_asks(req, prompt, attempt, n=[0]):        # a retry after a planned but broken flow never asks
        n[0] += 1
        return "Plan:\n1. Read the CSV (ooa.csv/read)\nFlow:\n{bad" if n[0] == 1 else q
    m = BatchAware(fixer_asks)
    res = router(m, mode="smart").run(CSV_REQ)
    assert not res.get("question") and not res["verified"] and len(m.prompts) == 3, (res, len(m.prompts))


EMAIL = {"kind": "TaskBot", "id": "mail", "name": "mail",
         "variables": [{"name": "manager", "type": "String", "direction": "input", "required": True}],
         "steps": [{"id": "s1", "action": "ooa.email/send", "description": "Email the sheet to the manager",
                    "args": {"to": "$manager$", "subject": "The sheet"}}]}


def test_critic_adds_a_missing_piece_once():
    req = CSV_REQ + ", then email the sheet to the manager"
    missing = F.request_clauses(req)[-1]
    assert missing == "email the sheet to the manager"

    def answer(r, prompt, attempt):
        return plan_answer(EMAIL) if r == missing else plan_answer(CSV_TO_EXCEL)
    rt = router(BatchAware(answer), mode="ultra")
    assert F.uncovered_clauses(req, CSV_TO_EXCEL, rt.index) == [missing]
    assert F.uncovered_clauses(CSV_REQ, CSV_TO_EXCEL, rt.index) == []
    res = rt.run(req)
    assert res["verified"] and res["critic_added"] == [missing], res
    assert F.actions_of(json.loads(res["flow"]))[-1] == "ooa.email/send"
    assert "critic" in agents(res) and "Added steps the first answer missed" in res["report"]
    # smart has no critic
    res = router(BatchAware(answer), mode="smart").run(req)
    assert res["verified"] and "critic" not in agents(res) and "ooa.email/send" not in res["flow"]
    # a piece the builders cannot make: the flow stays, the report says what to check
    rt = router(BatchAware(lambda r, p, a: "garbage {" if r == missing else plan_answer(CSV_TO_EXCEL)), mode="ultra")
    res = rt.run(req)
    assert res["verified"] and res["critic_missing"] == [missing] and "Check that it also does" in res["report"]


def test_reporter_and_studio_contract():
    """The reporter's summary ends Studio's notes; the Studio shape is unchanged (variables, steps, notes, warnings);
    the reply names the model asked for and the mode that ran."""
    rt = router(BatchAware(lambda r, p, a: plan_answer(CSV_TO_EXCEL)), mode="auto")
    for model, ran in (("ooa-flow-assistant-fast", "fast"), ("ooa-flow-assistant-smart", "smart"),
                       ("ooa-flow-assistant-ultra", "ultra"), ("ooa-flow-assistant", "fast"), ("ooa-assistant", "fast")):
        body = TL.studio_body(CSV_REQ)
        body["model"] = model
        code, out = S.chat_completion(rt, body)
        ans = json.loads(out["choices"][0]["message"]["content"])
        assert code == 200 and set(ans) == {"variables", "steps", "notes", "warnings"}, ans
        assert out["model"] == (model if model != "ooa-assistant" else "ooa-assistant")
        assert out["ooa"]["mode"] == ran and ("Mode: " + S.MODE_LABELS[ran]) in ans["notes"], (model, ans["notes"])
        assert ans["notes"].startswith(S.STUDIO_NOTE) and ans["warnings"] == []
        assert "What it does: " in ans["notes"] and "Assumed: you give " in ans["notes"]
        assert out["ooa"]["mode_asked"] == ("auto" if model in ("ooa-flow-assistant", "ooa-assistant") else ran)
    code, out = S.chat_completion(rt, {**TL.studio_body(CSV_REQ), "mode": "warp"})
    assert code == 400 and out["error"]["type"] == "invalid_request_error"
    # a failure says which mode ran too
    bad = router(BatchAware(lambda r, p, a: "garbage {"), mode="fast")
    code, out = S.chat_completion(bad, TL.studio_body(CSV_REQ))
    notes = json.loads(out["choices"][0]["message"]["content"])["notes"]
    assert notes.startswith("No flow was returned") and "Mode: Fast" in notes
    # without a mode (a router made without one), nothing changes
    plain = router(BatchAware(lambda r, p, a: plan_answer(CSV_TO_EXCEL)))
    code, out = S.chat_completion(plain, TL.studio_body(CSV_REQ))
    assert "Mode:" not in out["choices"][0]["message"]["content"] and "mode" not in out["ooa"]


def test_old_backends_keep_working():
    """A backend without generate_batch (or without the newer options): the team runs one call after another; a
    model given plain prompts (trained before context) runs the old layers."""
    answer, _ = _loop_answer()
    m = PlanAware(answer)
    res = router(m, mode="smart").run(LOOP_REQ)
    assert res["verified"] and res.get("built_in_parts") and "builders (one after another)" in agents(res), agents(res)
    s = TL.Scripted(lambda r, p, a: json.dumps(CSV_TO_EXCEL))
    res = router(s, prompt_format="plain", mode="ultra").run(CSV_REQ)
    assert res["verified"] and res["mode"] == "ultra" and s.prompts == [CSV_REQ]


def test_time_limits():
    """stop_after: no new attempt starts after it; time_limit: the deadline given to a running batch, never more than
    HARD_GRACE seconds after the caller's deadline (a request that waited in the queue has less time left)."""
    bad = lambda r, p, a: "Plan:\n1. Log it (core/log)\nFlow:\n{bad"
    m = BatchAware(bad)
    rt = router(m, mode="smart")
    t = time.time()
    res = rt.run(CSV_REQ)
    assert not res["verified"] and t + 139 < m.deadlines[0] <= time.time() + 140      # no caller deadline
    m = BatchAware(bad)
    rt = router(m, mode="smart")
    dl = time.time() - 1                                  # e.g. 121 s spent waiting for the model
    res = rt.run(CSV_REQ, deadline=dl)
    assert not res["verified"] and len(m.prompts) == 1 and S.OUT_OF_TIME in res["problems"]
    assert m.deadlines[0] == dl + S.HARD_GRACE, (m.deadlines[0], dl)
    m = BatchAware(bad)
    dl = time.time() + 120 - 60                           # arrived 60 s ago with a 120 s budget
    router(m, mode="ultra").run(CSV_REQ, deadline=dl)
    assert m.deadlines[0] == dl + S.HARD_GRACE and m.deadlines[0] - (dl - 120) < 180     # inside Studio's 180 s


def test_end_to_end_server():
    """Through the HTTP server: /v1/models, a mode by header and by model name, the agents and report in "ooa"."""
    answer, _ = _loop_answer()
    rt = router(BatchAware(answer), mode="auto")
    srv = S.serve(rt, 0)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{port}"
        models = json.loads(urllib.request.urlopen(base + "/v1/models", timeout=10).read())
        assert "ooa-flow-assistant-ultra" in [d["id"] for d in models["data"]]

        def post(body, headers=None):
            req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                         {"content-type": "application/json", **(headers or {})})
            return json.loads(urllib.request.urlopen(req, timeout=30).read())
        out = post({"model": "ooa-assistant", "messages": [{"role": "user", "content": LOOP_REQ}]},
                   {"X-OOA-Mode": "smart"})
        assert out["ooa"]["verified"] and out["ooa"]["mode"] == "smart" and out["ooa"]["built_in_parts"]
        assert [a["agent"] for a in out["ooa"]["agents"]][:3] == ["planner", "checker", "builders"]
        out = post({"model": "ooa-flow-assistant", "messages": [{"role": "user", "content": LOOP_REQ}]})
        assert out["ooa"]["mode"] == "ultra" and out["ooa"]["mode_asked"] == "auto" and out["model"] == "ooa-flow-assistant"
        assert out["ooa"]["report"].startswith("Mode: Ultra (picked by Auto: a long process")
    finally:
        srv.shutdown()


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Exception as e:  # noqa: BLE001
                fails += 1
                import traceback
                traceback.print_exc()
                print("FAIL", name, type(e).__name__, e)
    print("ALL MODE TESTS PASSED" if not fails else f"{fails} FAILED")
    sys.exit(1 if fails else 0)
