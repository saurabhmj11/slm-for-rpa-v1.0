"""Permissive pretraining data for a new base model, written in nanochat's parquet shard format.

    python pretrain_data.py --out DIR --train-chars 3e9 --platform PLATFORM --render SFT_RENDER --catalog CAT

Replaces ClimbMix (CC BY-NC 4.0, non-commercial) with:
  web     FineWeb-Edu, sample-10BT (HuggingFaceFW/fineweb-edu at a pinned dataset revision; ODC-By 1.0, its use also
          subject to the Common Crawl Terms of Use). Read straight from Hugging Face by HTTP range requests, one row
          group (1,000 documents, about 4.9M characters) at a time, so nothing larger than a shard is ever kept; a
          local folder of the same parquet files works too (--web-dir).
  domain  (--domain-share of the characters, default 15%) the OOA platform's own docs, code / JSON (contracts,
          examples, templates, reference, platform sources) and action catalog, plus the rendered training
          conversations (sft_render: requests, retrieval context, plans and flow JSON). The owner's own material.

Domain clean-up, in this order: files that look like key material are never read (names / extensions); a document
holding a hard secret (private key block, cloud / chat / API key formats, JWT, storage account key) is dropped;
password / secret / token / api_key literals are replaced by <redacted>; exact duplicates (whitespace-normalised) are
dropped; rendered rows whose request is too close to an eval prompt (eval/beginner_score.too_close, the same check the
SFT data uses) or equal to a held-out val row's request are dropped, and so is any domain document that quotes an
eval prompt verbatim.

nanochat reads every *.parquet in its data folder ($NANOCHAT_BASE_DIR/base_data_climbmix at the pinned commit; the
folder name is nanochat's, the content is ours) in name order, the LAST one being the validation split: train shards
are shard_00000.parquet, shard_00001.parquet, ...; the validation shard is shard_99999.parquet (FineWeb row groups
never used for training + the held-out rendered val rows at the same domain share). One column "text", row groups of
1,024 documents (nanochat's DDP loader strides over row groups). Domain pieces are at most 8,000 characters, below
tok_train's 10,000-character document cap, so the tokenizer sees them whole.

Resumable and deterministic: the domain stream does not depend on --train-chars (sources are interleaved by weight,
each a fixed number of shuffled passes), so a rerun with a larger --train-chars keeps every finished shard (checked by
sha256 against manifest.json) and appends; a rerun with other settings or other domain sources refuses to mix them.
data_card.json / DATA_CARD.md record every source (licence, revision, file hashes, characters, repeats) and the
measured shares.
"""
import argparse
import hashlib
import io
import json
import os
import random
import re
import sys
import time
from pathlib import Path

FINEWEB = {
    "name": "FineWeb-Edu sample-10BT",
    "repo": "HuggingFaceFW/fineweb-edu",
    "revision": "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",     # dataset commit of 2025-07-11 (pinned)
    "folder": "sample/10BT",
    "files": 14,                     # 000_00000 .. 013_00000.parquet: 13 x 726 + 183 row groups, ~47B characters
    "licence": "ODC-By 1.0 (Open Data Commons Attribution License); use also subject to the Common Crawl Terms of Use",
    "licence_url": "https://opendatacommons.org/licenses/by/1-0/",
    "terms_url": "https://commoncrawl.org/terms-of-use",
    "card": "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu",
    "attribution": "FineWeb-Edu by Hugging Face (Lozhkov, Ben Allal, von Werra, Wolf, 2024), ODC-By 1.0",
}
HF = "https://huggingface.co"
VAL_NAME = "shard_99999.parquet"         # sorts after every train shard: nanochat's validation split
DOC_CHARS = 8000                         # domain pieces stay below tok_train's 10,000-character document cap
ROW_GROUP = 1024
DOMAIN_WEIGHTS = {"render": 0.55, "code": 0.20, "docs": 0.15, "catalog": 0.10}
MAX_REPEAT = {"render": 3, "code": 4, "docs": 4, "catalog": 6}  # passes over a source's documents, at most
DOMAIN_ORDER = ("catalog", "code", "docs", "render")             # dedupe keeps the first copy in this order
CODE_EXT = {".java", ".ts", ".tsx", ".kts", ".json", ".py", ".toml", ".js", ".md", ".css", ".html", ".csv", ".yml",
            ".yaml", ".xml", ".sql", ".txt", ".ps1"}
DOCS_EXT = {".md", ".txt", ".py", ".json", ".js"}
CODE_DIRS = ("contracts", "examples", "templates", "reference", "platform")
SKIP_DIRS = {".git", "node_modules", "build", "dist", "out", "target", ".gradle", "__pycache__", ".idea", ".venv", "venv"}
# key material is never read, whatever it is called inside
SKIP_NAME = re.compile(r"(?i)^(\.env.*|id_[rd]sa.*|.*\.(pem|key|p12|pfx|jks|keystore|crt|cer|der|kdbx)|"
                       r"credentials?(\..*)?|secrets?(\..*)?|.*\.secret(s)?\..*)$")
SECRET = re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----|AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{36}"
                    r"|github_pat_[A-Za-z0-9_]{40,}|xox[baprs]-[A-Za-z0-9-]{10,}|sk-[A-Za-z0-9_-]{32,}"
                    r"|AIza[0-9A-Za-z_-]{35}|hooks\.slack\.com/services/T[A-Z0-9]+/|AccountKey=[A-Za-z0-9+/=]{40,}"
                    r"|eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}")
SOFT_SECRET = re.compile(r"(?i)((?:password|passwd|pwd|secret|api[_-]?key|access[_-]?token|auth[_-]?token|"
                         r"client[_-]?secret)[\"']?\s*[:=]\s*[\"'])([^\"'\s$<{][^\"'\n]{5,})([\"'])")
MAX_FILE = 512 * 1024
RENDER_SKIP = ("catalog.jsonl", "examples.jsonl", "render_report.json")
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]                   # .../rpa model


# ------------------------------------------------------------------------------------------------ web (FineWeb-Edu)
class HttpRangeFile(io.RawIOBase):
    """A remote file read by HTTP range requests (seekable), so pyarrow reads one row group's text column without
    downloading the 2 GB file. Retries with backoff; the signed CDN address is renewed when it expires."""

    def __init__(self, url, session=None, retries=6):
        import requests
        self.url0, self.s, self.retries = url, session or requests.Session(), retries
        self._resolve()
        self.p = 0

    def _resolve(self):
        for k in range(self.retries):
            try:
                r = self.s.head(self.url0, allow_redirects=True, timeout=60)
                r.raise_for_status()
                self.url, self.n = r.url, int(r.headers["content-length"])
                return
            except Exception as e:                                       # noqa: BLE001
                if k == self.retries - 1:
                    raise
                print(f"  (retry {k + 1}: {e})", flush=True)
                time.sleep(2 ** k)

    def readable(self):
        return True

    def seekable(self):
        return True

    def seek(self, o, whence=0):
        self.p = o if whence == 0 else self.p + o if whence == 1 else self.n + o
        return self.p

    def tell(self):
        return self.p

    def readinto(self, b):
        if self.p >= self.n:
            return 0
        end = min(self.n, self.p + len(b)) - 1
        for k in range(self.retries):
            try:
                r = self.s.get(self.url, headers={"Range": f"bytes={self.p}-{end}"}, timeout=180)
                if r.status_code in (403, 410):                          # signed address expired: ask again
                    self._resolve()
                    raise IOError(f"HTTP {r.status_code}")
                r.raise_for_status()
                d = r.content
                if r.status_code != 206 or len(d) != end - self.p + 1:
                    raise IOError(f"short read {len(d)} (HTTP {r.status_code})")
                b[:len(d)] = d
                self.p += len(d)
                return len(d)
            except Exception as e:                                       # noqa: BLE001
                if k == self.retries - 1:
                    raise
                print(f"  (retry {k + 1}: {e})", flush=True)
                time.sleep(2 ** k)
        return 0


class WebSource:
    """FineWeb-Edu row groups as units (file index, row group). Train units run forward from file 0, row group 0;
    the validation units are the last val_groups row groups of file 0 and are never used for training."""

    def __init__(self, web_dir=None, val_groups=2, files=None):
        self.web_dir = Path(web_dir) if web_dir else None
        if self.web_dir:
            self.paths = sorted(str(p) for p in self.web_dir.glob("*.parquet"))
            if not self.paths:
                raise SystemExit(f"no parquet files in {self.web_dir}")
        else:
            n = files or FINEWEB["files"]
            self.paths = [f"{HF}/datasets/{FINEWEB['repo']}/resolve/{FINEWEB['revision']}/{FINEWEB['folder']}/"
                          f"{i:03d}_00000.parquet" for i in range(n)]
        self.val_groups = val_groups
        self._pf, self._groups = {}, {}

    def pf(self, i):
        import pyarrow.parquet as pq
        if i not in self._pf:
            src = self.paths[i]
            f = src if self.web_dir else io.BufferedReader(HttpRangeFile(src), 1 << 20)
            self._pf[i] = pq.ParquetFile(f)
            self._groups[i] = self._pf[i].metadata.num_row_groups
        return self._pf[i]

    def groups(self, i):
        self.pf(i)
        return self._groups[i]

    def val_units(self):
        g = self.groups(0)
        if g <= self.val_groups:
            raise SystemExit(f"the first web file has only {g} row groups; need more than --val-groups {self.val_groups}")
        return [(0, r) for r in range(g - self.val_groups, g)]

    def next_unit(self, u):
        """The train unit after u (None = the first one); skips the validation units."""
        f, r = (0, -1) if u is None else u
        while True:
            r += 1
            limit = self.groups(f) - (self.val_groups if f == 0 else 0)
            if r >= limit:
                f, r = f + 1, -1
                if f >= len(self.paths):
                    raise SystemExit("out of web data: all FineWeb-Edu files used (raise --web-files)")
                continue
            return (f, r)

    def texts(self, u):
        t = self.pf(u[0]).read_row_group(u[1], columns=["text"]).column("text").to_pylist()
        return [x for x in t if x and x.strip()]


def fineweb_hashes():
    """{file name: sha256 of the parquet file} from the Hugging Face API at the pinned revision ({} offline)."""
    try:
        import requests
        r = requests.get(f"{HF}/api/datasets/{FINEWEB['repo']}/tree/{FINEWEB['revision']}/{FINEWEB['folder']}",
                         timeout=60)
        r.raise_for_status()
        return {Path(x["path"]).name: (x.get("lfs") or {}).get("oid") for x in r.json() if x.get("type") == "file"}
    except Exception as e:                                               # noqa: BLE001
        print("  (could not read the FineWeb-Edu file hashes:", e, ")")
        return {}


# ------------------------------------------------------------------------------------------------ clean-up
def scrub(text, stats):
    """None when the text holds a hard secret (the document is dropped), else the text with password / secret /
    token literals replaced by <redacted>."""
    if SECRET.search(text):
        stats["dropped_secret"] = stats.get("dropped_secret", 0) + 1
        return None
    out, n = SOFT_SECRET.subn(lambda m: m.group(1) + "<redacted>" + m.group(3), text)
    if n:
        stats["redacted_literals"] = stats.get("redacted_literals", 0) + n
    return out


def norm(text):
    return " ".join(str(text).lower().split())


def eval_prompts(eval_dir=None):
    """The prompts of every eval/*.jsonl set (nothing close to them may be trained on)."""
    d = Path(eval_dir) if eval_dir else REPO / "eval"
    out = []
    for f in sorted(d.glob("*.jsonl")) if d.exists() else []:
        for line in f.open(encoding="utf-8"):
            if line.strip():
                try:
                    p = json.loads(line).get("prompt")
                except (ValueError, AttributeError):
                    continue
                if isinstance(p, str) and p.strip():
                    out.append(p)
    return out


def _too_close():
    sys.path.insert(0, str(REPO / "eval"))
    try:
        from beginner_score import too_close, content_words
    finally:
        sys.path.pop(0)
    return too_close, content_words


class EvalGuard:
    """too_close(eval prompt, training request) for every eval prompt, with a content-word index so a request is
    only compared with the prompts it shares words with; quoted() finds an eval prompt copied into a document."""

    def __init__(self, prompts):
        self.prompts = list(prompts)
        self.too_close, self.cw = _too_close()
        self.idx = {}
        for i, p in enumerate(self.prompts):
            for w in self.cw(p):
                self.idx.setdefault(w, set()).add(i)
        self.small = [i for i, p in enumerate(self.prompts) if len(self.cw(p)) < 2]
        self.quotes = [norm(p) for p in self.prompts if len(norm(p)) >= 25]
        self._cache = {}

    def close(self, request):
        if request in self._cache:
            return self._cache[request]
        cand = set(self.small)
        for w in self.cw(request):
            cand |= self.idx.get(w, set())
        hit = next((self.prompts[i] for i in sorted(cand) if self.too_close(self.prompts[i], request)[0]), None)
        self._cache[request] = hit
        return hit

    def quoted(self, text):
        t = norm(text)
        return any(q in t for q in self.quotes)


# ------------------------------------------------------------------------------------------------ domain
def _pieces(text, limit=DOC_CHARS):
    """text cut at line ends into pieces of at most limit characters."""
    if len(text) <= limit:
        return [text] if text.strip() else []
    out, cur, n = [], [], 0
    for line in text.splitlines(True):
        while len(line) > limit:                       # one very long line (minified JSON): hard cut
            if cur:
                out.append("".join(cur))
                cur, n = [], 0
            out.append(line[:limit])
            line = line[limit:]
        if n + len(line) > limit and cur:
            out.append("".join(cur))
            cur, n = [], 0
        cur.append(line)
        n += len(line)
    if cur:
        out.append("".join(cur))
    return [p for p in out if p.strip()]


def _sha_files(paths):
    h = hashlib.sha256()
    for p in sorted(paths, key=str):
        h.update(str(Path(p).name).encode())
        h.update(Path(p).read_bytes())
    return h.hexdigest()


def _repo_files(root, sub, exts):
    base = Path(root) / sub
    if not base.exists():
        return []
    out = []
    for dp, dns, fns in os.walk(base):
        dns[:] = sorted(d for d in dns if d not in SKIP_DIRS and not d.startswith("."))
        for fn in sorted(fns):
            p = Path(dp) / fn
            if p.suffix.lower() in exts and not SKIP_NAME.match(fn) and p.stat().st_size <= MAX_FILE:
                out.append(p)
    return out


def _read_texts(paths, root, stats):
    docs = []
    for p in paths:
        try:
            t = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            stats["skipped_binary"] = stats.get("skipped_binary", 0) + 1
            continue
        t = scrub(t, stats)
        if t is None:
            continue
        rel = p.relative_to(root).as_posix()
        docs += _pieces(t if p.suffix == ".md" else f"// file: {rel}\n{t}")
    return docs


def conversation_text(row):
    """A rendered row as plain text: the request (with its context) then the answer (plan + flow)."""
    return "\n\n".join((m.get("content") or "").strip() for m in row.get("messages", []) if isinstance(m, dict)).strip()


def row_request(row):
    """The user's request of a rendered row (first user turn, before its '### Context')."""
    u = next((m for m in row.get("messages", []) if isinstance(m, dict) and m.get("role") == "user"), None)
    c = u.get("content") if u else None
    return c.split("\n### Context")[0].strip() if isinstance(c, str) else ""


def _render_files(d):
    """(train files, held-out files) of a rendered data folder. Held out: val*, eval*, rl_val*; never used: rl_train
    (RL prompts without answers), catalog / examples (covered by the catalog source)."""
    train, held = [], []
    for f in sorted(Path(d).glob("*.jsonl")):
        if f.name.startswith(("val", "eval", "rl_val")):
            held.append(f)
        elif f.name in RENDER_SKIP or f.name.startswith("rl_"):
            continue
        else:
            train.append(f)
    return train, held


def _jsonl(f):
    for line in f.open(encoding="utf-8"):
        if line.strip():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if isinstance(r, dict):
                yield r


def domain_sources(platform=None, render_dirs=(), catalog=None, eval_dir=None, log=print):
    """{source: {"docs": [...], "files": [...], "info": {...}}} for render / code / docs / catalog (scrubbed,
    deduplicated, eval prompts and val rows excluded), and the held-out domain validation documents."""
    src, val = {}, []
    if platform and Path(platform).exists():
        root = Path(platform)
        st = {}
        dpaths = _repo_files(root, "docs", DOCS_EXT)
        src["docs"] = {"docs": _read_texts(dpaths, root, st), "files": dpaths, "info": st}
        st = {}
        cpaths = [p for sub in CODE_DIRS for p in _repo_files(root, sub, CODE_EXT)]
        cpaths += [p for p in (root / "README.md", root / "SECURITY.md", root / "module-map.json") if p.exists()]
        src["code"] = {"docs": _read_texts(cpaths, root, st), "files": cpaths, "info": st}
    if catalog and Path(catalog).exists():
        lines = [l for l in Path(catalog).read_text(encoding="utf-8").splitlines() if l.strip()]
        docs = ["\n".join(lines[i:i + 40]) for i in range(0, len(lines), 40)]
        # an index of every action id with its name, the way the retrieval context lists them
        idx = []
        for l in lines:
            try:
                r = json.loads(l)
                idx.append(f"- {r['id']}: {r.get('name', '')} ({r.get('toolkit', '')})")
            except (ValueError, KeyError, TypeError):
                pass
        docs += ["\n".join(idx[i:i + 200]) for i in range(0, len(idx), 200)]
        st = {"actions": len(idx)}
        docs = [d for d in (scrub(x, st) for x in docs) if d]
        src["catalog"] = {"docs": [p for d in docs for p in _pieces(d)], "files": [Path(catalog)], "info": st}
    # rendered conversations: held-out rows first (their requests are excluded from training)
    guard = EvalGuard(eval_prompts(eval_dir))
    rfiles, held_files, held_req = [], [], set()
    for d in render_dirs or ():
        if Path(d).exists():
            t, h = _render_files(d)
            rfiles += t
            held_files += h
    for f in held_files:
        for r in _jsonl(f):
            t = conversation_text(r)
            if t:
                val.append(t)
                held_req.add(norm(row_request(r)))
    st = {"rows": 0, "excluded_eval_close": 0, "excluded_val_request": 0}
    rdocs = []
    for f in rfiles:
        for r in _jsonl(f):
            st["rows"] += 1
            req = row_request(r)
            if req and norm(req) in held_req:
                st["excluded_val_request"] += 1
                continue
            if req and guard.close(req):
                st["excluded_eval_close"] += 1
                continue
            t = conversation_text(r)
            t = scrub(t, st) if t else None
            if t:
                rdocs += _pieces(t)
    if rfiles:
        st["file_names"] = [f"{f.parent.name}/{f.name}" for f in rfiles]
        st["held_out_files"] = [f"{f.parent.name}/{f.name}" for f in held_files]
        src["render"] = {"docs": rdocs, "files": rfiles, "info": st}
    # documents quoting an eval prompt, then exact duplicates across all domain sources
    seen = set()
    for k in [k for k in DOMAIN_ORDER if k in src]:
        keep, quoted, dup = [], 0, 0
        for d in src[k]["docs"]:
            if guard.quoted(d):
                quoted += 1
                continue
            h = hashlib.sha256(norm(d).encode()).digest()
            if h in seen:
                dup += 1
                continue
            seen.add(h)
            keep.append(d)
        src[k]["docs"] = keep
        src[k]["info"].update(excluded_eval_quote=quoted, dropped_duplicates=dup)
    vstat = {}
    val = [v for v in (scrub(x, vstat) for x in val) if v and not guard.quoted(v)]
    return src, val


def domain_fingerprint(sources):
    """sha256 over every domain document in order: a resumed build must see exactly the same domain stream."""
    h = hashlib.sha256()
    for k in sorted(sources):
        h.update(k.encode())
        for d in sources[k]["docs"]:
            h.update(hashlib.sha256(d.encode()).digest())
    return h.hexdigest()


def domain_stream(sources, seed=1, weights=None, max_repeat=None):
    """The whole domain stream as [(source, document)], independent of how much of it a build uses: each source is
    max_repeat[source] passes over its documents (each pass shuffled), and the sources are interleaved so every
    prefix holds them at their weights (the next document always comes from the source furthest below its share;
    a source whose passes are used up drops out and the others fill in)."""
    weights = dict(DOMAIN_WEIGHTS, **(weights or {}))
    max_repeat = dict(MAX_REPEAT, **(max_repeat or {}))
    queues = {}
    for k in sorted(sources):
        docs = sources[k]["docs"]
        if not docs or weights.get(k, 0) <= 0:
            continue
        q = []
        for p in range(max_repeat.get(k, 3)):
            order = docs[:]
            random.Random(f"{seed}:{k}:{p}").shuffle(order)
            q += order
        queues[k] = q
    pos = {k: 0 for k in queues}
    used = {k: 0 for k in queues}
    out = []
    while True:
        live = [k for k in queues if pos[k] < len(queues[k])]
        if not live:
            return out
        k = min(live, key=lambda s: (used[s] / weights[s], s))
        d = queues[k][pos[k]]
        pos[k] += 1
        used[k] += len(d)
        out.append((k, d))


def stream_report(sources, stream, n):
    """Per source: unique characters, characters used by the first n stream documents, repeats (used / unique)."""
    rep = {k: {"docs": len(v["docs"]), "unique_chars": sum(map(len, v["docs"])), "used_chars": 0}
           for k, v in sources.items()}
    for k, d in stream[:n]:
        rep[k]["used_chars"] += len(d)
    tot = sum(r["used_chars"] for r in rep.values()) or 1
    for r in rep.values():
        r["repeats"] = round(r["used_chars"] / max(1, r["unique_chars"]), 2)
        r["share_of_domain"] = round(r["used_chars"] / tot, 3)
    return rep


# ------------------------------------------------------------------------------------------------ shards
def write_shard(path, texts):
    import pyarrow as pa
    import pyarrow.parquet as pq
    tmp = str(path) + ".tmp"
    pq.write_table(pa.table({"text": texts}), tmp, row_group_size=ROW_GROUP, compression="zstd")
    os.replace(tmp, path)
    return _sha(path)


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


SETTING_KEYS = ("shard_chars", "domain_share", "seed", "web", "val_groups", "domain")


def build(out, train_chars, shard_chars=250_000_000, domain_share=0.15, platform=None, render_dirs=(), catalog=None,
          web_dir=None, web_files=None, val_groups=2, seed=1, eval_dir=None, log=print):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    man_f = out / "manifest.json"
    man = json.loads(man_f.read_text()) if man_f.exists() else {}
    sources, val_docs = domain_sources(platform, render_dirs, catalog, eval_dir, log)
    settings = {"train_chars": int(train_chars), "shard_chars": int(shard_chars), "domain_share": domain_share,
                "seed": seed, "web": "local:" + str(web_dir) if web_dir else FINEWEB["revision"],
                "val_groups": val_groups, "domain": domain_fingerprint(sources)}
    if man and {k: man["settings"].get(k) for k in SETTING_KEYS} != {k: settings[k] for k in SETTING_KEYS}:
        raise SystemExit(f"{out} holds shards built with other settings or other domain sources "
                         f"({ {k: man['settings'].get(k) for k in SETTING_KEYS} }); use another folder")
    n_shards = max(1, -(-int(train_chars) // int(shard_chars)))
    log("domain sources: " + ", ".join(f"{k} {len(v['docs'])} docs / {sum(map(len, v['docs'])) / 1e6:.1f}M chars"
                                       for k, v in sorted(sources.items())) + f"; held-out val rows {len(val_docs)}")
    dom = domain_stream(sources, seed)
    web = WebSource(web_dir, val_groups, web_files)
    # finished shards stay (checked by hash); the build continues after the last good one
    keep = []
    for s in man.get("shards", []):
        p = out / s["file"]
        if len(keep) < n_shards and p.exists() and _sha(p) == s["sha256"]:
            keep.append(s)
        else:
            break
    shards = keep
    for f in sorted(out.glob("shard_*.parquet")):           # a shard of an unfinished or larger earlier run
        if f.name != VAL_NAME and f.name not in {s["file"] for s in shards}:
            f.unlink()
    if shards:
        log(f"  keeping {len(shards)} finished shard(s)")
    unit = tuple(shards[-1]["web_units"][-1]) if shards else None
    dpos = shards[-1]["domain_end"] if shards else 0
    warned = False
    for k in range(len(shards), n_shards):
        t0 = time.time()
        texts, wchars, units = [], 0, []
        while wchars < shard_chars * (1 - domain_share):
            unit = web.next_unit(unit)
            t = web.texts(unit)
            texts += t
            wchars += sum(map(len, t))
            units.append(list(unit))
        dchars, d0 = 0, dpos
        while dpos < len(dom) and dchars < wchars * domain_share / (1 - domain_share):
            texts.append(dom[dpos][1])
            dchars += len(dom[dpos][1])
            dpos += 1
        if dpos >= len(dom) and not warned and domain_share > 0:
            log("  WARNING: the domain stream is used up (every source at its max passes); the domain share falls")
            warned = True
        random.Random(seed * 100003 + k).shuffle(texts)
        name = f"shard_{k:05d}.parquet"
        sha = write_shard(out / name, texts)
        shards.append({"file": name, "sha256": sha, "docs": len(texts), "web_chars": wchars, "domain_chars": dchars,
                       "domain_docs": dpos - d0, "domain_end": dpos, "web_units": units})
        man = {"settings": settings, "shards": shards}
        man_f.write_text(json.dumps(man, indent=1))
        log(f"  {name}: {len(texts):,} docs, web {wchars / 1e6:.0f}M + domain {dchars / 1e6:.1f}M chars "
            f"({100 * dchars / max(1, wchars + dchars):.1f}%), {time.time() - t0:.0f} s")
    # validation shard: unseen FineWeb row groups + held-out rendered rows at the same share
    vtexts, vw = [], 0
    for u in web.val_units():
        t = web.texts(u)
        vtexts += t
        vw += sum(map(len, t))
    vd, want = 0, vw * domain_share / (1 - domain_share)
    for t in val_docs:
        if vd >= want:
            break
        vtexts.append(t)
        vd += len(t)
    random.Random(seed).shuffle(vtexts)
    vsha = write_shard(out / VAL_NAME, vtexts)
    settings["train_chars"] = int(train_chars)
    man = {"settings": settings, "shards": shards,
           "val": {"file": VAL_NAME, "sha256": vsha, "docs": len(vtexts), "web_chars": vw, "domain_chars": vd,
                   "web_units": [list(u) for u in web.val_units()]}}
    man_f.write_text(json.dumps(man, indent=1))
    card = data_card(man, sources, stream_report(sources, dom, dpos), web_dir)
    card["totals"]["shard_bytes"] = sum((out / s["file"]).stat().st_size for s in shards)
    card["totals"]["bytes_per_1e9_chars"] = int(card["totals"]["shard_bytes"] / max(1, card["totals"]["train_chars"]) * 1e9)
    (out / "data_card.json").write_text(json.dumps(card, indent=1))
    (out / "DATA_CARD.md").write_text(card_markdown(card), encoding="utf-8")
    log(f"data: {len(shards)} train shards, {card['totals']['train_chars'] / 1e9:.2f}B chars, domain share "
        f"{100 * card['totals']['domain_share']:.1f}%; validation {VAL_NAME} ({vw / 1e6:.0f}M web + {vd / 1e6:.1f}M domain)")
    return card


LICENCES = {"render": "the owner's own training data (generated from the OneOfficeAutomation platform and catalog)",
            "code": "OneOfficeAutomation platform repository (the owner's own code)",
            "docs": "OneOfficeAutomation platform docs (the owner's own)",
            "catalog": "OOA action catalog built from the platform (the owner's own)"}


def data_card(man, sources, dom_rep, web_dir=None):
    tw = sum(s["web_chars"] for s in man["shards"])
    td = sum(s["domain_chars"] for s in man["shards"])
    units = [tuple(u) for s in man["shards"] for u in s["web_units"]]
    files = sorted({u[0] for u in units})
    if web_dir:
        web = {"name": "local parquet folder (test / offline copy)", "path": str(web_dir), "chars": tw,
               "used_row_groups": len(units), "licence": "as the files' source"}
    else:
        hashes = fineweb_hashes()
        web = dict(FINEWEB, used_row_groups=len(units), chars=tw,
                   files_used={f"{i:03d}_00000.parquet": hashes.get(f"{i:03d}_00000.parquet") for i in files})
    dom = {}
    for k, v in sources.items():
        dom[k] = dict(dom_rep.get(k, {}), licence=LICENCES.get(k, "the owner's own"), files=len(v["files"]),
                      sha256=_sha_files(v["files"]) if v["files"] else None, **v.get("info", {}))
    return {"built": time.strftime("%Y-%m-%d %H:%M"), "settings": man["settings"], "web": web, "domain": dom,
            "not_included": ["ClimbMix (CC BY-NC 4.0)", "SmolTalk or any other third-party chat / instruction data", "any eval/*.jsonl prompt or a "
                             "request too close to one", "held-out val rows", "files that look like key material"],
            "totals": {"train_chars": tw + td, "web_chars": tw, "domain_chars": td,
                       "domain_share": td / max(1, tw + td), "train_shards": len(man["shards"])},
            "val": {k: v for k, v in man["val"].items() if k != "web_units"},
            "shards": [{k: s[k] for k in ("file", "sha256", "docs", "web_chars", "domain_chars")} for s in man["shards"]]}


def card_markdown(card):
    t, w = card["totals"], card["web"]
    lines = ["# Pretraining data card (generated by pretrain_data.py)", "",
             f"Built {card['built']}; {t['train_shards']} train shards, {t['train_chars'] / 1e6:,.0f}M characters "
             f"({t['web_chars'] / 1e6:,.0f}M web + {t['domain_chars'] / 1e6:,.1f}M domain = "
             f"{100 * t['domain_share']:.1f}% domain); validation {card['val']['file']} "
             f"({card['val']['web_chars'] / 1e6:,.1f}M web + {card['val']['domain_chars'] / 1e6:,.2f}M domain).", "",
             "## Web", "", f"- {w['name']}: {w.get('licence', '')}",
             f"- revision {w.get('revision', w.get('path', ''))}, {w.get('used_row_groups', 0)} row groups"]
    if w.get("attribution"):
        lines.append(f"- attribution: {w['attribution']} ({w['card']})")
    for f, h in (w.get("files_used") or {}).items():
        lines.append(f"- {f} sha256 {h}")
    lines += ["", "## Domain", "", "| source | docs | unique chars | used chars | repeats | share of domain | files sha256 |",
              "|---|---|---|---|---|---|---|"]
    for k, d in sorted(card["domain"].items()):
        lines.append(f"| {k} | {d.get('docs', 0):,} | {d.get('unique_chars', 0):,} | {d.get('used_chars', 0):,} | "
                     f"{d.get('repeats', 0)} | {d.get('share_of_domain', 0)} | {(d.get('sha256') or '')[:16]} |")
    lines += ["", "Clean-up per source:", ""]
    for k, d in sorted(card["domain"].items()):
        info = {x: d[x] for x in ("dropped_secret", "redacted_literals", "dropped_duplicates", "excluded_eval_close",
                                  "excluded_eval_quote", "excluded_val_request", "skipped_binary") if d.get(x)}
        lines.append(f"- {k}: {d['licence']}; " + (", ".join(f"{a} {b}" for a, b in info.items()) or "nothing removed"))
    lines += ["", "## Not included", ""] + [f"- {x}" for x in card["not_included"]] + [""]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output folder (nanochat reads it as its dataset)")
    ap.add_argument("--train-chars", type=float, required=True, help="characters of training text in total")
    ap.add_argument("--shard-chars", type=float, default=250e6)
    ap.add_argument("--domain-share", type=float, default=0.15)
    ap.add_argument("--platform", default=None, help="OneOfficeAutomation platform checkout (docs, code, contracts)")
    ap.add_argument("--render", action="append", default=[], help="rendered SFT data folder (sft_render); repeatable")
    ap.add_argument("--catalog", default=None, help="catalog.jsonl")
    ap.add_argument("--eval-dir", default=None, help="eval sets to keep out of training (default: the repo's eval/)")
    ap.add_argument("--web-dir", default=None, help="local FineWeb-Edu parquet files instead of streaming them")
    ap.add_argument("--web-files", type=int, default=None)
    ap.add_argument("--val-groups", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(argv)
    build(a.out, a.train_chars, a.shard_chars, a.domain_share, a.platform, a.render, a.catalog, a.web_dir,
          a.web_files, a.val_groups, a.seed, a.eval_dir)


if __name__ == "__main__":
    sys.exit(main())
