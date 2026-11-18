"""Get the models and the v4 training data from Google Drive (folder ooa-nanochat) onto this PC, and prepare the data
exactly as Colab cells 6 / 6d do. Used by ooa_local_pc.ipynb (cells 3-4); everything lands in "rpa model"/train_data:

    train_data/sft_render/   the rendered data the model trains on (plan-first format, compact answers, gold dropout,
                             beginner rows): Drive sft_render/ when it is there and changed since the last download
                             (saves about 50 minutes); else rendered here from sft/ (+ sft_clean/)
    train_data/drive_model/  the newest trained model on Drive (chatsft_checkpoints/d8, else the newest kept
                             chatsft_checkpoints_before_*) + its tokenizer: where a top-up starts
    train_data/base/         the pretrained base model (base_checkpoints/d8 + optimizer) for the full training

Files Drive shows as "name (1).jsonl" are copies made by a second upload: they are reported and never used.

    python data_pc.py --rpa DIR --drive-id ID [--mode topup|full|install] [--refresh] [--seq 4096]
    python data_pc.py --rpa DIR --fake-drive FOLDER ...     a local folder laid out like the Drive folder (tests)
"""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import sft_setup as T                                          # noqa: E402

COPY_RE = re.compile(r" \(\d+\)(\.[^./]+)?$")                  # "long_train (1).jsonl": a second upload
ANSWER_FORMAT = "compact"                                      # Colab cell 1 ANSWER_FORMAT
GOLD_DROPOUT = 0.5                                             # Colab cell 1 GOLD_DROPOUT
FILTER_CAP = 60                                                # Colab cell 6: data_filter.py --cap 60


def row_chars(seq):
    return 6500 * seq // 2048                                   # Colab cell 1 ROW_CHARS


def row_tokens(seq):
    return seq - 28                                             # Colab cell 1 ROW_TOKENS


def is_copy(path):
    return bool(COPY_RE.search(path.rsplit("/", 1)[-1]))


class GdownDrive:
    """The Drive folder through gdown (shared as "anyone with the link"). files: {path: [file ids]}."""

    def __init__(self, folder_id, log=print):
        import gdown
        self.gdown = gdown
        log("listing the Drive folder (a minute) ...")
        listing = gdown.download_folder(id=folder_id, skip_download=True, quiet=True)
        self.files = {}
        for f in listing or []:
            self.files.setdefault(str(f.path).replace("\\", "/"), []).append(f.id)

    def download(self, path, dest, i=0):
        self.gdown.download(id=self.files[path][i], output=str(dest), quiet=True)


class FolderDrive:
    """A local folder laid out like the Drive folder (tests, or a Drive folder synced by Google Drive for desktop)."""

    def __init__(self, root, log=print):
        self.root = Path(root)
        self.files = {}
        for p in sorted(self.root.rglob("*")):
            if p.is_file():
                rel = p.relative_to(self.root).as_posix()
                self.files[rel] = [hashlib.sha1(f"{rel}:{p.stat().st_size}:{p.stat().st_mtime_ns}".encode()).hexdigest()]

    def download(self, path, dest, i=0):
        shutil.copyfile(self.root / path, dest)


class Sync:
    def __init__(self, drive, rpa, log=print):
        self.drive, self.rpa, self.log = drive, Path(rpa), log
        self.td = self.rpa / "train_data"
        self.downloaded = 0
        self.seen_f = self.td / ".drive_ids.json"
        self.seen = json.loads(self.seen_f.read_text(encoding="utf-8")) if self.seen_f.exists() else {}
        copies = sorted(p for p in drive.files if is_copy(p))
        for p in copies:
            log(f"WARNING: Drive copy with (n) in the name, ignored (delete it on Drive and keep one file): {p}")
        self.copies = copies

    def listed(self, folder):
        pre = folder.rstrip("/") + "/"
        return sorted(p for p in self.drive.files if p.startswith(pre) and "/" not in p[len(pre):] and not is_copy(p))

    def fetch(self, path, dest, refresh=False):
        """One file; when Drive holds two with the same name, the larger (fuller) one. -> dest or None."""
        dest = Path(dest)
        if path not in self.drive.files:
            return None
        ids = ",".join(self.drive.files[path])
        if dest.exists() and dest.stat().st_size > 0 and not refresh and self.seen.get(str(dest)) in (ids, None):
            if self.seen.get(str(dest)) is None:                # downloaded by an older notebook: adopt it
                self._remember(dest, ids)
            return dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        best, best_n = None, -1
        for i in range(len(self.drive.files[path])):
            tmp = dest.with_name(dest.name + f".part{i}")
            self.drive.download(path, tmp, i)
            n = tmp.stat().st_size
            if n > best_n:
                if best is not None:
                    best.unlink()
                best, best_n = tmp, n
            else:
                tmp.unlink()
        best.replace(dest)
        self.downloaded += 1
        self._remember(dest, ids)
        return dest

    def _remember(self, dest, ids):
        """Drive file ids of what is here: a file uploaded again on Drive gets a new id and is downloaded again."""
        self.seen[str(dest)] = ids
        self.seen_f.parent.mkdir(parents=True, exist_ok=True)
        self.seen_f.write_text(json.dumps(self.seen, indent=0), encoding="utf-8")

    def newest_pair(self, folder):
        steps = []
        for p in self.drive.files:
            m = re.fullmatch(re.escape(folder) + r"/model_(\d+)\.pt", p)
            if m and f"{folder}/meta_{m.group(1)}.json" in self.drive.files:
                steps.append(m.group(1))
        return max(steps, key=int) if steps else None

    # -- models ------------------------------------------------------------------------------------------
    def tokenizer(self, root):
        for name in ("tokenizer.pkl", "token_bytes.pt"):
            if not self.fetch(f"nanochat_base/tokenizer/{name}", Path(root) / "tokenizer" / name):
                raise SystemExit(f"{name} not found on Drive (nanochat_base/tokenizer)")

    def sft_model(self):
        """The newest trained model on Drive -> train_data/drive_model (re-downloaded only when Drive's changed).
        -> (folder on Drive, step) or None."""
        folders = ["nanochat_base/chatsft_checkpoints/d8"] + sorted(
            {p.rsplit("/", 1)[0] for p in self.drive.files
             if re.match(r"nanochat_base/chatsft_checkpoints_before_\d+_\d+/d8/", p)}, reverse=True)
        hit = next(((f, s) for f in folders if (s := self.newest_pair(f))), None)
        if not hit:
            self.log("no trained model on Drive (chatsft_checkpoints)")
            return None
        folder, s = hit
        root = self.td / "drive_model"
        d8 = root / "chatsft_checkpoints" / "d8"
        ident = f"{folder}/model_{s}.pt:{self.drive.files[f'{folder}/model_{s}.pt'][0]}"
        marker = root / ".drive_id"
        if not (marker.exists() and marker.read_text(encoding="utf-8") == ident and (d8 / f"model_{s}.pt").exists()):
            shutil.rmtree(d8, ignore_errors=True)
            self.log(f"downloading the trained model {folder} step {int(s)} (about 300 MB) ...")
            self.fetch(f"{folder}/model_{s}.pt", d8 / f"model_{s}.pt", refresh=True)
            self.fetch(f"{folder}/meta_{s}.json", d8 / f"meta_{s}.json", refresh=True)
            marker.write_text(ident, encoding="utf-8")
        else:
            self.log(f"trained model already here: {folder} step {int(s)}")
        self.tokenizer(root)
        if folder != "nanochat_base/chatsft_checkpoints/d8":
            self.log("note: Drive's chatsft_checkpoints has no finished model yet (a Colab run in progress?); "
                     f"using the newest kept one, {folder}")
        return folder, int(s)

    def base_model(self):
        s = self.newest_pair("nanochat_base/base_checkpoints/d8")
        if not s:
            raise SystemExit("no pretrained base model on Drive (nanochat_base/base_checkpoints/d8)")
        root = self.td / "base"
        d8 = root / "base_checkpoints" / "d8"
        for f in d8.glob("*") if d8.exists() else []:
            if not f.name.endswith((f"_{s}.pt", f"_{s}.json", f"_{s}_rank0.pt")):
                f.unlink()
        for name in (f"model_{s}.pt", f"meta_{s}.json", f"optim_{s}_rank0.pt"):
            self.fetch(f"nanochat_base/base_checkpoints/d8/{name}", d8 / name)
        self.tokenizer(root)
        self.log(f"pretrained base model: step {int(s)}")
        return int(s)

    # -- data -----------------------------------------------------------------------------------------------
    def render_from_drive(self, refresh=False):
        """Drive sft_render -> train_data/sft_render when Drive has one that differs from the local copy (its
        render_report.json changes with every render). -> "downloaded" | "up to date" | None (not on Drive)."""
        rep = "sft_render/render_report.json"
        files = self.listed("sft_render")
        if rep not in self.drive.files or "sft_render/examples.jsonl" not in files:
            return None
        out = self.td / "sft_render"
        tmp_rep = self.td / ".drive_render_report.json"
        self.fetch(rep, tmp_rep, refresh=True)
        sha = hashlib.sha1(tmp_rep.read_bytes()).hexdigest()
        marker = out / ".drive_report.sha1"
        if not refresh and marker.exists() and marker.read_text(encoding="utf-8") == sha \
                and (out / "examples.jsonl").exists():
            tmp_rep.unlink()
            return "up to date"
        new = self.td / "sft_render.download"
        shutil.rmtree(new, ignore_errors=True)
        new.mkdir(parents=True)
        self.log(f"downloading Drive sft_render ({len(files)} files, the data the Colab run trains on) ...")
        for p in files:
            if p == rep:
                continue
            self.fetch(p, new / p.rsplit("/", 1)[-1], refresh=True)
            self.log("  got", p)
        shutil.move(str(tmp_rep), new / "render_report.json")
        (new / ".drive_report.sha1").write_text(sha, encoding="utf-8")
        if out.exists():
            shutil.rmtree(out)
        new.rename(out)
        return "downloaded"

    def raw_data(self, refresh=False):
        """Drive sft/ (+ sft_clean/ when it has filter_report.json) -> train_data/sft, train_data/sft_clean."""
        files = [p for p in self.listed("sft") if p.endswith(".jsonl")]
        if "sft/catalog.jsonl" not in files:
            raise SystemExit("Drive has neither sft_render/ nor sft/catalog.jsonl: nothing to train on")
        for f in T.RENDER_EXTRA:
            if f"sft/{f}" not in files:
                self.log("MISSING in Drive sft/ (upload it):", f)
        for p in files:
            self.fetch(p, self.td / "sft" / p.rsplit("/", 1)[-1], refresh=refresh)
        clean = self.listed("sft_clean")
        has_clean = "sft_clean/filter_report.json" in clean
        if has_clean:
            for p in clean:
                self.fetch(p, self.td / "sft_clean" / p.rsplit("/", 1)[-1], refresh=refresh)
        self.log(f"sft/ ({len(files)} files)" + (" and sft_clean/" if has_clean else "") + " from Drive are here")
        return has_clean


def render_local(rpa, seq=4096, log=print, platform=None, filter_here=True):
    """Colab cells 6 (data_filter, when Drive had no sft_clean: filter_here) and 6d (render_dataset) on this PC, same
    flags. About an hour on 4 cores."""
    rpa = Path(rpa)
    td, tr = rpa / "train_data", rpa / "training"
    sft, clean, out = td / "sft", td / "sft_clean", td / "sft_render"
    if filter_here:
        shutil.rmtree(clean, ignore_errors=True)                 # always from the sft/ that is here now
    if not (clean / "filter_report.json").exists():
        plat = Path(platform) if platform else rpa / "platform"
        log("== cleaning the data (training/data_filter.py, as Colab cell 6) ...")
        r = subprocess.run([sys.executable, str(tr / "data_filter.py"), str(sft), str(clean), str(plat), str(rpa / "eval"),
                            "--cap", str(FILTER_CAP), "--max-chars", str(row_chars(seq))])
        if r.returncode != 0:
            raise SystemExit("data_filter.py failed (see above)")
    src = clean if (clean / "filter_report.json").exists() else sft
    extra = [str(sft / f) for f in T.RENDER_EXTRA if (sft / f).exists()]
    tmp = td / "sft_render.local"
    shutil.rmtree(tmp, ignore_errors=True)
    log(f"== rendering {src.name} + {len(extra)} new files into the plan-first format (render_dataset.py, as Colab "
        "cell 6d; 30-60 minutes) ...")
    r = subprocess.run([sys.executable, str(tr / "render_dataset.py"), str(src), str(tmp), "--catalog", str(sft / "catalog.jsonl"),
                        "--max-chars", str(row_chars(seq)), "--max-tokens", str(row_tokens(seq)),
                        "--answer-format", ANSWER_FORMAT, "--gold-dropout", str(GOLD_DROPOUT), "--extra", *extra])
    if r.returncode != 0:
        raise SystemExit("render_dataset.py failed (see above)")
    if (src / "train_catalog.jsonl").exists():                 # catalog question rows stay as they are
        shutil.copy2(src / "train_catalog.jsonl", tmp / "train_catalog.jsonl")
    (tmp / ".local_render").write_text(time.strftime("%Y-%m-%d %H:%M"), encoding="utf-8")
    shutil.rmtree(out, ignore_errors=True)
    tmp.rename(out)
    return out


def add_feedback(rpa, data, log=print):
    """The assistant's Accept / Edit feedback -> data/feedback_train.jsonl in the same rendered format
    (ingest_feedback.py records, rendered by render_dataset.py). Nothing when there is no feedback."""
    rpa, data = Path(rpa), Path(data)
    fb = rpa / "assistant" / "assistant_data" / "feedback.jsonl"
    (data / "feedback_train.jsonl").unlink(missing_ok=True)
    if not fb.exists() or not fb.read_text(encoding="utf-8").strip():
        return 0
    work = rpa / "train_data" / "feedback_work"
    shutil.rmtree(work, ignore_errors=True)
    (work / "empty").mkdir(parents=True)
    schema = rpa / "platform" / "contracts" / "flow-schema" / "flow.v1.json"
    r = subprocess.run([sys.executable, str(rpa / "training" / "ingest_feedback.py"), str(fb), str(work / "fb"),
                        str(data / "catalog.jsonl"), str(schema)], capture_output=True, text=True)
    rec = work / "fb" / "feedback_records.jsonl"
    if r.returncode != 0 or not rec.exists() or not rec.read_text(encoding="utf-8").strip():
        log("feedback: nothing usable", (r.stdout.strip() or r.stderr.strip())[-300:])
        return 0
    r = subprocess.run([sys.executable, str(rpa / "training" / "render_dataset.py"), str(work / "empty"), str(work / "out"),
                        "--catalog", str(data / "catalog.jsonl"), "--max-chars", str(row_chars(4096)),
                        "--max-tokens", str(row_tokens(4096)), "--answer-format", ANSWER_FORMAT,
                        "--gold-dropout", str(GOLD_DROPOUT), "--extra", str(rec)], capture_output=True, text=True)
    out = work / "out" / "feedback_records.jsonl"
    if r.returncode != 0 or not out.exists():
        log("feedback: render failed", r.stderr.strip()[-300:])
        return 0
    rows = [l for l in out.read_text(encoding="utf-8").splitlines(True) if l.strip()]
    (data / "feedback_train.jsonl").write_text("".join(rows), encoding="utf-8")
    shutil.rmtree(work, ignore_errors=True)
    log(f"feedback: {len(rows)} rows from your Accept / Edit answers -> {data / 'feedback_train.jsonl'}")
    return len(rows)


def sync(drive, rpa, mode="topup", refresh=False, seq=4096, log=print, platform=None):
    """Everything a mode needs. -> {"data": sft_render folder, "render": how it got here, "drive_model": ..., ...}"""
    s = Sync(drive, rpa, log)
    out = {"copies_ignored": s.copies}
    if mode in ("topup", "install"):
        out["drive_model"] = s.sft_model()
    if mode == "full":
        out["base_step"] = s.base_model()
    how = s.render_from_drive(refresh)
    if how is None:
        log("Drive has no sft_render/ (or it is incomplete): downloading sft/ and rendering here")
        before = s.downloaded
        has_clean = s.raw_data(refresh)
        mark = Path(rpa) / "train_data" / "sft_render" / ".local_render"
        if mark.exists() and s.downloaded == before and not refresh:
            how = "rendered here earlier; sft/ unchanged"
        else:
            render_local(rpa, seq, log, platform, filter_here=not has_clean)
            how = "rendered here"
    data = Path(rpa) / "train_data" / "sft_render"
    log(f"training data: {data} ({how})")
    out.update(data=str(data), render=how)
    if mode == "topup":                                        # the full run stays exactly the Colab data
        out["feedback_rows"] = add_feedback(rpa, data, log)
    else:
        (data / "feedback_train.jsonl").unlink(missing_ok=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rpa", required=True)
    ap.add_argument("--drive-id", default="")
    ap.add_argument("--fake-drive", default="", help="a local folder laid out like the Drive folder")
    ap.add_argument("--mode", default="topup", choices=["topup", "full", "install"])
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--platform", default=None)
    ap.add_argument("--feedback-only", action="store_true", help="only render the assistant's feedback into "
                                                                  "train_data/sft_render/feedback_train.jsonl")
    a = ap.parse_args(argv)
    if a.feedback_only:
        add_feedback(a.rpa, Path(a.rpa) / "train_data" / "sft_render")
        return
    drive = FolderDrive(a.fake_drive) if a.fake_drive else GdownDrive(a.drive_id)
    print(json.dumps(sync(drive, a.rpa, a.mode, a.refresh, a.seq, platform=a.platform), indent=1, default=str))


if __name__ == "__main__":
    main()
