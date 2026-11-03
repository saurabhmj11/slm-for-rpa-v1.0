"""
Logistics SLM — Multi-turn conversation generator (memory training + benchmark).

Builds long ChatML conversations whose correct answers depend on facts stated
many turns earlier:

  * user name                ("hi, I'm Priya" ... "what's my name?")
  * home warehouse           ("my warehouse is LA Port" ... "check pallets in my warehouse")
  * route preference         ("always quote me on LA-SF" ... "how much to ship pallets?")
  * shipment facts           (track SHP1023 ... "what was the status of SHP1023?")
  * references               (track SHP1023 ... "initiate a return for that shipment")
  * corrections              ("sorry, I meant SHP1042 not SHP1024")
  * state updates            (status changes; the latest one must be recalled)

The assistant <thought> explicitly resolves every reference
(e.g. "that shipment = SHP1023"), and every <tool_call> carries its arguments,
so correct memory use is directly visible and scoreable.

Entity split: shipment IDs whose number ends in 7 are reserved for the
benchmark, so the model must copy IDs from context instead of memorising them.

Usage:
    py data_gen_multiturn.py                    # train/val conversations
    py data_gen_multiturn.py --bench --n 15     # held-out memory benchmark
"""
import argparse
import json
import os
import random

# --- Entities -----------------------------------------------------------------
LOCATIONS = ["Chicago Warehouse", "NY Hub", "LA Port", "Dallas Transit Center", "Miami Dock",
             "Seattle Depot", "Denver Yard", "Atlanta Crossdock"]
ITEMS = ["forklift parts", "packaging tape", "pallets", "safety gear", "conveyor belts",
         "shrink wrap", "barcode scanners", "steel racks"]
ROUTES = ["NY-BOS", "CHI-DAL", "LA-SF", "MIA-ATL", "SEA-DEN", "ATL-NY"]
SUPPLIERS = ["GlobalPack", "HeavyLift Inc", "SafeTransit Co", "QuickShip LLC"]
DOCUMENTS = ["shipping label", "commercial invoice", "bill of lading", "customs declaration"]
NAMES = ["Priya", "Rahul", "Ananya", "Arjun", "Meera", "Vikram", "Sara", "John", "Maria", "Chen",
         "Fatima", "Omar", "Lucas", "Emma", "Kenji", "Aisha", "Diego", "Olga", "Ravi", "Sneha",
         "Tom", "Lena", "Ivan", "Zara"]

STATUSES = ["in transit", "delayed due to weather", "delivered", "held at customs", "out for delivery"]
STOCK_LEVELS = ["high (500+ units)", "medium (100-500 units)", "low (<100 units)", "out of stock"]
TRAFFIC_CONDS = ["clear skies and no traffic", "heavy traffic delays", "severe weather warning",
                 "accident reported, expect delays"]
COST_ESTIMATES = ["$450.00", "$1,200.00", "$320.50", "$2,100.00", "$875.25"]
PERFORMANCE_METRICS = ["98% on-time", "85% on-time (needs improvement)", "92% on-time"]
CAPACITY_LEVELS = ["80% full", "20% full (plenty of space)", "100% full (at capacity)"]
EXTRACTED_DATA = ["total_weight: 450kg", "invoice_amount: $1200", "destination: LA Port", "sender: GlobalPack"]

ALL_IDS = [f"SHP{n}" for n in range(1000, 10000)]
TRAIN_IDS = [s for s in ALL_IDS if not s.endswith("7")]
TEST_IDS = [s for s in ALL_IDS if s.endswith("7")]

BENCH_TYPES = ["recall", "reference", "correction", "state_update", "name", "home", "preference"]


def asst(thought, route, tool=None, obs=None, action=""):
    s = f"<thought> {thought} <route> {route} "
    if tool is not None:
        s += f"<tool_call> {tool} <observation> {obs} "
    return s + f"<action> {action}"


def render_turn(u, a):
    return f"<|im_start|> user\n{u}<|im_end|>\n<|im_start|> assistant\n{a}<end>\n"


class Conversation:
    """Stateful simulator. Each turn method appends (user, assistant) and updates memory state."""

    def __init__(self, rng, id_pool):
        self.rng = rng
        self.id_pool = id_pool
        self.turns = []
        self.name = None
        self.home = None
        self.pref_route = None
        self.ship_facts = {}        # sid -> (status, loc), latest tracked value
        self.mentioned = []         # every shipment id mentioned so far
        self.last_ship = None       # referent of "that shipment" / "it"
        self.last_explicit = None   # (kind, sid) if the previous turn named a shipment explicitly

    # ---- helpers ----
    def c(self, xs):
        return self.rng.choice(xs)

    def _add(self, u, a, explicit=None):
        self.turns.append((u, a))
        self.last_explicit = explicit

    def _mention(self, sid):
        if sid not in self.mentioned:
            self.mentioned.append(sid)
        self.last_ship = sid

    def fresh_id(self):
        while True:
            sid = self.c(self.id_pool)
            if sid not in self.mentioned:
                return sid

    def text(self):
        return "".join(render_turn(u, a) for u, a in self.turns)

    # ---- shipment turns (mode: explicit | ref | correct) ----
    def _ship_prefix(self, mode, sid, old, explicit_msgs, ref_msgs, explicit_thought):
        if mode == "explicit":
            return self.c(explicit_msgs), explicit_thought
        if mode == "ref":
            return self.c(ref_msgs), f"that shipment = {sid}"
        u = self.c([f"sorry, I meant {sid} not {old}", f"wait, wrong number, it's {sid}",
                    f"correction: {sid}, not {old}"])
        return u, f"correction: {old} -> {sid}"

    def track(self, sid, mode="explicit", old=None, status=None):
        status = status or self.c(STATUSES)
        loc = self.c(LOCATIONS)
        u, th = self._ship_prefix(
            mode, sid, old,
            [f"what is the status of shipment {sid} ?", f"track {sid} for me",
             f"where is {sid} right now?", f"any update on {sid}?"],
            ["track that shipment again", "what's the status of that shipment now?",
             "check on it again", "where is that one now?"],
            "user wants to track a shipment")
        a = asst(th, "tracking_agent", f"get_shipment_status {sid}", f"{status} | {loc}",
                 f"Shipment {sid} is currently {status} at {loc}.")
        self.ship_facts.pop(sid, None)
        self.ship_facts[sid] = (status, loc)
        self._mention(sid)
        self._add(u, a, None if mode == "ref" else ("track", sid))

    def ret(self, sid, mode="explicit", old=None):
        u, th = self._ship_prefix(
            mode, sid, old,
            [f"process a return for damaged goods on {sid}", f"initiate return for {sid}",
             f"the customer wants to send back {sid}, start a return"],
            ["initiate a return for it", "start a return for that shipment", "process a return on that one"],
            "user wants to process a return")
        a = asst(th, "returns_agent", f"initiate_return {sid}", "return_auth_generated",
                 f"Return authorization for {sid} has been successfully generated.")
        self._mention(sid)
        self._add(u, a, None if mode == "ref" else ("ret", sid))

    def ocr(self, sid, mode="explicit", old=None):
        doc = self.c(DOCUMENTS)
        data = self.c(EXTRACTED_DATA)
        u, th = self._ship_prefix(
            mode, sid, old,
            [f"scan this {doc} for {sid}", f"extract data from {sid} {doc}"],
            [f"scan the {doc} for that shipment", f"extract data from its {doc}"],
            "user wants to extract text from document")
        a = asst(th, "ocr_agent", f"process_document {sid} | {doc}", data,
                 f"The extracted data from the {doc} of {sid} is: {data}.")
        self._mention(sid)
        self._add(u, a, None if mode == "ref" else ("ocr", sid))

    def correction(self):
        kind, old = self.last_explicit
        new = self.fresh_id()
        {"track": self.track, "ret": self.ret, "ocr": self.ocr}[kind](new, mode="correct", old=old)

    def recall(self, sid):
        status, loc = self.ship_facts[sid]
        u = self.c([f"remind me, what was the status of {sid}?", f"what did you say about {sid} earlier?",
                    f"where was {sid} last time you checked?"])
        a = asst(f"recall {sid} from earlier", "memory",
                 action=f"Last time I checked, shipment {sid} was {status} at {loc}.")
        self._mention(sid)
        self._add(u, a)

    def filter_status(self):
        present = [s for s, _ in self.ship_facts.values()]
        status = self.c(present) if self.rng.random() < 0.7 else self.c(STATUSES)
        matches = [sid for sid, (s, _) in self.ship_facts.items() if s == status]
        u = self.c([f"which of my shipments are {status}?", f"did any shipment show {status}?"])
        if not matches:
            act = f"None of the shipments we checked are {status}."
        else:
            act = f"{', '.join(matches)} {'is' if len(matches) == 1 else 'are'} {status}."
        a = asst(f"scan tracked shipments for {status}", "memory", action=act)
        self._add(u, a)

    # ---- non-shipment skills ----
    def inventory(self, use_home=False):
        item = self.c(ITEMS)
        stock = self.c(STOCK_LEVELS)
        if use_home:
            loc = self.home
            u = self.c([f"check inventory for {item} in my warehouse", f"do we have any {item} left at my warehouse?",
                        f"how many {item} are in my home warehouse?"])
            th = f"my warehouse = {loc}"
        else:
            loc = self.c(LOCATIONS)
            u = self.c([f"check inventory for {item} in {loc}", f"do we have any {item} left at the {loc}?"])
            th = "user wants to check stock levels"
        a = asst(th, "inventory_agent", f"check_stock {item} | {loc}", stock,
                 f"The inventory for {item} at {loc} is currently {stock}.")
        self._add(u, a)

    def capacity(self, use_home=False):
        cap = self.c(CAPACITY_LEVELS)
        if use_home:
            loc = self.home
            u = self.c(["do we have space in my warehouse?", "how full is my warehouse?",
                        "what is the current capacity of my home warehouse?"])
            th = f"my warehouse = {loc}"
        else:
            loc = self.c(LOCATIONS)
            u = self.c([f"do we have space in {loc}?", f"what is the current capacity of {loc}?"])
            th = "user wants warehouse capacity"
        a = asst(th, "capacity_agent", f"check_capacity {loc}", cap, f"The {loc} is currently {cap}.")
        self._add(u, a)

    def cost(self, use_pref=False):
        item = self.c(ITEMS)
        est = self.c(COST_ESTIMATES)
        if use_pref:
            route = self.pref_route
            u = self.c([f"how much to ship {item}?", f"estimate freight cost for {item}", f"quote me for {item}"])
            th = f"preferred route = {route}"
        else:
            route = self.c(ROUTES)
            u = self.c([f"estimate freight cost for {item} on {route}", f"how much to ship {item} via {route}?"])
            th = "user wants freight cost estimation"
        a = asst(th, "cost_agent", f"get_freight_estimate {item} | {route}", est,
                 f"The estimated freight cost to ship {item} on route {route} is {est}.")
        self._add(u, a)

    def routing(self):
        route = self.c(ROUTES)
        cond = self.c(TRAFFIC_CONDS)
        u = self.c([f"are there any alerts for route {route} ?", f"how is the traffic on {route} today?"])
        a = asst("user wants route conditions", "routing_agent", f"get_route_status {route}", cond,
                 f"The current status for route {route} is: {cond}.")
        self._add(u, a)

    def supplier(self):
        sup = self.c(SUPPLIERS)
        met = self.c(PERFORMANCE_METRICS)
        u = self.c([f"what is the on-time delivery rate for {sup}?", f"supplier performance report for {sup}"])
        a = asst("user wants supplier analytics", "supplier_agent", f"get_supplier_metrics {sup}", met,
                 f"The performance metric for {sup} is: {met}.")
        self._add(u, a)

    # ---- user profile memory ----
    def set_name(self):
        name = self.c([n for n in NAMES if n != self.name])
        if self.name is None:
            u = self.c([f"hi, my name is {name}", f"i'm {name} by the way", f"hello, this is {name} from operations",
                        f"you can call me {name}"])
            th = "user introduced themselves"
        else:
            u = self.c([f"actually, call me {name}", f"please use the name {name} for me from now on"])
            th = "user changed their name"
        self.name = name
        self._add(u, asst(th, "memory", action=f"Nice to meet you, {name}. How can I help with your shipments today?"))

    def ask_name(self):
        u = self.c(["what's my name?", "do you remember my name?", "who am i again?", "remind me what name i gave you"])
        self._add(u, asst("recall user name", "memory", action=f"Your name is {self.name}."))

    def set_home(self):
        loc = self.c([x for x in LOCATIONS if x != self.home])
        if self.home is None:
            u = self.c([f"my home warehouse is {loc}", f"i work out of {loc}", f"note that my warehouse is {loc}"])
            th = "user set home warehouse"
        else:
            u = self.c([f"actually my warehouse is now {loc}", f"i moved, my warehouse is {loc} now"])
            th = "user updated home warehouse"
        self.home = loc
        self._add(u, asst(th, "memory", action=f"Noted, your home warehouse is {loc}."))

    def ask_home(self):
        u = self.c(["which warehouse is mine?", "what did i say my warehouse was?"])
        self._add(u, asst("recall home warehouse", "memory", action=f"Your home warehouse is {self.home}."))

    def set_pref(self):
        route = self.c([r for r in ROUTES if r != self.pref_route])
        if self.pref_route is None:
            u = self.c([f"always quote me on route {route}", f"from now on use route {route} for my quotes",
                        f"i only ship via {route}"])
            th = "user set route preference"
        else:
            u = self.c([f"change my default route to {route}", f"actually, quote me on {route} from now on"])
            th = "user updated route preference"
        self.pref_route = route
        self._add(u, asst(th, "memory", action=f"Got it, I will use route {route} for your quotes."))

    # ---- random turn selection with constraints ----
    def random_turn(self, forbid=frozenset(), ship_w=1.0, probe_w=1.0):
        """Pick a valid random turn. `forbid` tags: 'ship' (no shipment mentions at all),
        'ship:<ID>' (never mention that ID), 'name', 'home', 'pref'."""
        no_ship = "ship" in forbid
        has_id_forbid = any(t.startswith("ship:") for t in forbid)

        def ok(sid):
            return not no_ship and f"ship:{sid}" not in forbid

        opts = [
            (1.0, self.inventory), (1.0, self.capacity), (1.0, self.routing),
            (1.0, self.cost), (0.8, self.supplier),
        ]
        if not no_ship:
            known = [s for s in self.mentioned if ok(s)]

            def pick():
                return self.c(known) if known and self.rng.random() < 0.3 else self.fresh_id()
            opts += [(1.2 * ship_w, lambda: self.track(pick())),
                     (0.6 * ship_w, lambda: self.ret(pick())),
                     (0.5 * ship_w, lambda: self.ocr(pick()))]
            if self.last_ship and ok(self.last_ship):
                ls = self.last_ship
                opts += [(0.8 * probe_w, lambda: self.c([self.track, self.ret, self.ocr])(ls, mode="ref"))]
            tracked = [s for s in self.ship_facts if ok(s)]
            if tracked:
                opts += [(0.9 * probe_w, lambda: self.recall(self.c(tracked)))]
            if self.ship_facts and not has_id_forbid:
                opts += [(0.25 * probe_w, self.filter_status)]
            if self.last_explicit and ok(self.last_explicit[1]):
                opts += [(0.4 * ship_w, self.correction)]
        if "name" not in forbid:
            opts += [(0.3 if self.name is None else 0.08, self.set_name)]
            if self.name:
                opts += [(0.5 * probe_w, self.ask_name)]
        if "home" not in forbid:
            opts += [(0.3 if self.home is None else 0.08, self.set_home)]
            if self.home:
                opts += [(0.5 * probe_w, lambda: self.inventory(use_home=True)),
                         (0.4 * probe_w, lambda: self.capacity(use_home=True)),
                         (0.2 * probe_w, self.ask_home)]
        if "pref" not in forbid:
            opts += [(0.3 if self.pref_route is None else 0.08, self.set_pref)]
            if self.pref_route:
                opts += [(0.6 * probe_w, lambda: self.cost(use_pref=True))]

        r = self.rng.random() * sum(w for w, _ in opts)
        for w, fn in opts:
            r -= w
            if r <= 0:
                fn()
                return
        opts[-1][1]()


# --- Probe episodes (exact fact -> probe distance) ----------------------------
def probe_episode(rng, id_pool, ptype, distance, n_pre=None, n_post=0):
    """Return (Conversation, probe_turn_index). The probe turn is exactly `distance`
    turns after the turn that planted the needed fact."""
    c = Conversation(rng, id_pool)
    for _ in range(rng.randint(0, 3) if n_pre is None else n_pre):
        c.random_turn(forbid={"name", "home", "pref"} if ptype in ("name", "home", "preference") else set())

    if ptype == "recall":
        x = c.fresh_id()
        c.track(x)
        forbid, probe = {f"ship:{x}"}, (lambda: c.recall(x))
    elif ptype == "state_update":
        x = c.fresh_id()
        c.track(x)
        for _ in range(rng.randint(1, 3)):
            c.random_turn(forbid={f"ship:{x}"})
        new_status = rng.choice([s for s in STATUSES if s != c.ship_facts[x][0]])
        c.track(x, status=new_status)
        forbid, probe = {f"ship:{x}"}, (lambda: c.recall(x))
    elif ptype == "reference":
        rng.choice([c.track, c.ret, c.ocr])(c.fresh_id())
        forbid = {"ship"}
        probe = lambda: rng.choice([c.track, c.ret, c.ocr])(c.last_ship, mode="ref")  # noqa: E731
    elif ptype == "correction":
        rng.choice([c.track, c.ret, c.ocr])(c.fresh_id())
        c.correction()
        forbid = {"ship"}
        probe = lambda: rng.choice([c.track, c.ret, c.ocr])(c.last_ship, mode="ref")  # noqa: E731
    elif ptype == "name":
        c.set_name()
        forbid, probe = {"name"}, c.ask_name
    elif ptype == "home":
        c.set_home()
        forbid = {"home"}
        probe = lambda: rng.choice([c.inventory, c.capacity])(use_home=True)  # noqa: E731
    elif ptype == "preference":
        c.set_pref()
        forbid, probe = {"pref"}, (lambda: c.cost(use_pref=True))
    else:
        raise ValueError(ptype)

    for _ in range(distance - 1):
        c.random_turn(forbid=forbid)
    probe_idx = len(c.turns)
    probe()
    for _ in range(n_post):
        c.random_turn()
    return c, probe_idx


# --- Training conversations ---------------------------------------------------
def train_conversation(rng):
    if rng.random() < 0.4:
        ptype = rng.choice(BENCH_TYPES)
        distance = min(40, int(rng.expovariate(1 / 10)) + 1)
        c, _ = probe_episode(rng, TRAIN_IDS, ptype, distance, n_post=rng.randint(0, 6))
        return c
    c = Conversation(rng, TRAIN_IDS)
    ship_w = rng.uniform(0.15, 1.5)   # sparse-shipment conversations create long-range references
    n_turns = rng.randint(2, 40)
    for _ in range(n_turns):
        c.random_turn(ship_w=ship_w, probe_w=1.5)
    return c


def build_bench(n_per_cell, distances, seed=1234):
    rng = random.Random(seed)
    items = []
    for ptype in BENCH_TYPES:
        for d in distances:
            for _ in range(n_per_cell):
                c, pi = probe_episode(rng, TEST_IDS, ptype, d)
                u, gold = c.turns[pi]
                history = "".join(render_turn(a, b) for a, b in c.turns[:pi])
                history += f"<|im_start|> user\n{u}<|im_end|>\n<|im_start|> assistant\n"
                items.append({"type": ptype, "distance": d, "history": history, "gold": gold})
    return items


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n_train", type=int, default=15000)
    p.add_argument("--n_val", type=int, default=500)
    p.add_argument("--bench", action="store_true", help="write the held-out memory benchmark instead")
    p.add_argument("--n", type=int, default=15, help="bench items per (type, distance) cell")
    p.add_argument("--distances", type=str, default="1,2,4,8,16,32")
    a = p.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))

    if a.bench:
        dists = [int(x) for x in a.distances.split(",")]
        items = build_bench(a.n, dists)
        path = os.path.join(here, "bench_memory.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it) + "\n")
        print(f"wrote {len(items)} benchmark probes ({len(BENCH_TYPES)} types x {dists}) -> {path}")
    else:
        for split, n, seed in [("train", a.n_train, 42), ("val", a.n_val, 7)]:
            rng = random.Random(seed)
            path = os.path.join(here, f"conv_{split}.jsonl")
            turns = 0
            with open(path, "w", encoding="utf-8") as f:
                for _ in range(n):
                    c = train_conversation(rng)
                    turns += len(c.turns)
                    f.write(json.dumps({"text": c.text()}) + "\n")
            print(f"{split}: {n:,} conversations, avg {turns / n:.1f} turns -> {path}")
