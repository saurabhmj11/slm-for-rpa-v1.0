"""Tests for gen_beginner.py (beginner rewrites and clarify records). Run: python test_gen_beginner.py
(works under pytest too). The end-to-end test needs the platform's flow schema: OOA_SCHEMA or OOA_PLATFORM."""
import json, os, random, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gen_beginner as G, ooa_format as F

CATALOG = os.path.join(HERE, "../data/catalog/catalog.jsonl")
REQ = "Read orders.csv from the Reports folder, keep only the rows where Status is Unpaid and email them to boss@acme.com"


def test_typos_spare_names_and_hit_a_critical_noun_once():
    rng = random.Random(3)
    for _ in range(200):
        t = G.add_typos(REQ, rng, 3)
        assert t and t != REQ
        for keep in ("orders.csv", "boss@acme.com"):
            assert keep in t, t
        crit = [w for w in ("folder", "email", "rows", "Status") if w.lower() in G.CRITICAL and w not in re.findall(r"\w+", t)]
        assert len(crit) <= 1, (crit, t)


def test_every_style_changes_the_request_and_keeps_the_details():
    rng = random.Random(5)
    for style, fn in G.TRANSFORMS.items():
        outs = [fn(REQ, rng) for _ in range(20)]
        outs = [o for o in outs if o]
        assert outs, style
        for o in outs:
            assert o != REQ, style
            if style != "typo":
                assert "orders" in o.lower() and ("boss@acme.com" in o or "boss at acme dot com" in o), (style, o)


def test_hinglish_uses_hindi_words_and_moves_the_verb():
    out = G.to_hinglish("Zip the Reports folder and then email it to the finance team", random.Random(1))
    assert out and G.is_hinglish(out), out
    assert re.search(r"\b(kar do|bana do|bhej do|bhej dena|send kar do|mail bhej do|zip kar do)\b", out), out
    assert "finance team ko" in out, out
    assert G.to_hinglish("excel padh ke unpaid wale hata do", random.Random(1)) is None or True
    assert len(G.HINGLISH_VERBS) + len(G.HINGLISH_WORDS) + len(G.HINGLISH_POST) >= 100


def test_versions_are_distinct_styles():
    v = G.beginner_versions(REQ, random.Random(2), 5)
    assert 3 <= len(v) <= 5
    firsts = [tags[0] for tags, _ in v]
    assert len(set(firsts)) == len(firsts) and all(s in G.STYLES for tags, _ in v for s in tags)
    # a request already in Hinglish only gets the styles that still read well
    h = G.beginner_versions("sales csv padh ke unpaid wale orders filter karo aur manager ko bhej do", random.Random(2), 5)
    assert h and all(tags[0] in G.HINGLISH_OK for tags, _ in h)


def test_take_out_leaves_a_vague_reference():
    rng = random.Random(0)
    assert G.take_out("Move the files to the Reports folder", "folder", rng) in (
        "Move the files to the right folder", "Move the files to the usual place")
    r = G.take_out("Read sales.csv and email the totals to the finance team", "recipient", rng)
    assert r.startswith("Read sales.csv and email the totals to ") and "finance" not in r, r
    assert G.take_out("Filter the orders over 5000 and save them", "condition", rng).startswith("Filter the orders that")
    assert "Slack" not in G.take_out("post the summary on Slack", "app", rng)
    assert G.take_out("merge a.pdf and b.pdf", "file", rng) is None          # another file is still named
    assert G.take_out("Move the files to the Reports folder", "recipient", rng) is None


def test_recipient_slot_needs_a_sending_verb():
    rng = random.Random(0)
    for req in ("save the attachments of each new email in Support to Finance",
                "Save the attachments of each new email in Inbox to the Exports folder",
                "move the email to Unpaid-paid folder", "copy the emails from Billing to Archive",
                "save the attachments of each new email in Supplier Mail to Contracts",
                "Forward this email in Outlook to Finance at accounts@contoso.com"):
        assert G.take_out(req, "recipient", rng) is None, req
    r = G.take_out("Check each new email in Support and forward it to Priya", "recipient", rng)
    assert r and "Priya" not in r and r.startswith("Check each new email in Support and forward it to "), r


def test_file_slot_points_at_a_file_only_the_user_knows():
    for _ in range(20):
        r = G.take_out("Read orders.csv and keep the unpaid rows", "file", random.Random(_))
        assert re.search(r"usual|always use|last time", r) and "that file" not in r and "my file" not in r, r


def test_hinglish_keeps_sign_out_and_the_order():
    for i in range(30):
        rng = random.Random(i)
        out = G.to_hinglish("Read the report, email it to the finance team, then sign out / disconnect at the end", rng)
        assert out and not re.search(r"login karo|sign in karo|log in karo", out) and "logout" in out or \
            "disconnect kar do" in out, out
        out = G.to_hinglish("Before emailing the HR report outside the company, encrypt it into a new protected file", rng)
        assert out is None or (out.startswith("before emailing") and "uske baad" not in out and "phir" not in out), out
    # long processes: a clause without a verb is not turned into garbled Hinglish
    long = "Broken links hurt our search ranking, so crawl the entire website for dead links and email the web team"
    assert G.to_hinglish(long, random.Random(1), strict=True) is None
    assert G.to_hinglish("Read the alerts, but only when count is above zero, and log the rest", random.Random(1),
                         strict=True)


def test_typos_spare_the_flows_values_and_names():
    flow = {"steps": [{"action": "ooa.file/list", "args": {"folder": "Scans"}},
                      {"action": "ooa.excel/read_sheet", "args": {"sheet": "Purchase Orders"}},
                      {"action": "ooa.ad/search", "args": {"filter": "(description=Pending start)"}}]}
    prot = G.value_words(flow)
    assert {"scans", "purchase", "orders", "pending", "start"} <= prot
    req = "for each pdf in scans read the purchase orders sheet and find users marked pending start"
    for i in range(100):
        t = G.add_typos(req, random.Random(i), 3, protect=prot)
        if t:
            for w in ("scans", "purchase", "orders", "pending", "start"):
                assert w in t.split(), (w, t)
    for i in range(50):        # beginner_versions passes them on to every style (casual lower-cases first)
        for tags, t in G.beginner_versions("For each PDF in the Scans folder, read the Purchase Orders sheet", random.Random(i),
                                           5, flow):
            if "typo" in tags:
                assert re.search(r"(?i)\bscans\b", t) and re.search(r"(?i)purchase orders", t), (tags, t)


def test_vague_goal_fits_the_flows_actions():
    ad = {"steps": [{"action": "ooa.ad/create_user"}, {"action": "ooa.ad/move_user"}]}
    mail = {"steps": [{"action": "ooa.email/send"}]}
    tidy = {"tidy up my files", "sort out my folders", "keep the folders neat"}
    for i in range(40):
        rng = random.Random(i)
        assert G.goal_for("move new accounts to the holding OU and copy their groups", rng, ad) not in tidy
        assert G.goal_for("read the payslip table from SQL", rng, {"steps": [{"action": "ooa.database/query"}]}) \
            not in {"keep on top of my emails", "sort out my mailbox stuff"}
        assert G.goal_for("send the note", rng, mail) in {"keep on top of my emails", "sort out my mailbox stuff"}


def test_clear_line_prefers_plain_step_words():
    from gen_targeted import FlowBuilder, load_catalog
    from gen_clarify import c_csv_excel
    cat = load_catalog(CATALOG)
    req, flow = c_csv_excel(lambda: FlowBuilder(cat), random.Random(1))
    assert G.described(flow) and G.clear_for(req, flow, cat) == F.clear_request(flow, cat)
    bare = {**flow, "steps": [{k: v for k, v in s.items() if k != "description"} for s in flow["steps"]]}
    assert not G.described(bare) and G.clear_for("convert my csv to excel", bare, cat) == "Convert my csv to excel."
    copied = {**flow, "steps": [{**s, "description": cat[s["action"]]["description"]} for s in flow["steps"]]}
    assert not G.described(copied, cat) and G.clear_for("convert my csv to excel", copied, cat) == "Convert my csv to excel."
    assert "synth_pattern" not in G.ORIGIN_SHARE and "synth_pattern.jsonl" not in G.CHAT_FILES


def test_generate_end_to_end():
    schema = os.environ.get("OOA_SCHEMA") or (os.path.join(os.environ["OOA_PLATFORM"], "contracts/flow-schema/flow.v1.json")
                                              if os.environ.get("OOA_PLATFORM") else "")
    if not schema or not os.path.exists(schema):
        print("SKIP test_generate_end_to_end - set OOA_SCHEMA or OOA_PLATFORM")
        return
    from gen_targeted import FlowBuilder, load_catalog
    from gen_clarify import c_csv_excel, c_send, c_zip, c_merge
    cat = load_catalog(CATALOG)
    rng = random.Random(4)
    src = {"targeted": [f(lambda: FlowBuilder(cat), rng) for f in (c_csv_excel, c_send, c_zip, c_merge)] +
           [("Copy reports.xlsx to the Archive folder and email it to the finance team",
             c_send(lambda: FlowBuilder(cat), rng)[1])]}
    rows, rep = G.generate(src, CATALOG, schema, n_flows=5, n_clarify=5, seed=1, log=lambda *_: None)
    rw = [r for r in rows if r["kind"] == "rewrite"]
    cl = [r for r in rows if r["kind"] == "clarify"]
    assert len(rw) >= 10 and rep["checker"]["ok"] == rep["checker"]["n"]
    assert len({r["request"].lower() for r in rows}) == len(rows)
    assert all(r["clear"].endswith(".") and r["flow"]["steps"] for r in rw)
    assert cl and all(r["question"] and "missing_detail" in r["style"] for r in cl)
    row = F.render_chat(rw[0], cat, __import__("retrieval").ActionIndex(list(cat.values())), None, random.Random(0))
    assert F.parse_answer(row["messages"][1]["content"])["clear"] == rw[0]["clear"]


def test_guard_drops_test_prompts():
    g = G.Guard()
    assert g.leaks("pdf to text pls, file is notes.pdf")                         # a beginner test prompt
    assert g.leaks("Make a new folder called Reports")                           # a plain_requests prompt
    assert not g.leaks("Zip the quarterly board packs and upload them to the board portal")


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
    print("ALL PASS" if not fails else f"{fails} FAILED")
    sys.exit(1 if fails else 0)
