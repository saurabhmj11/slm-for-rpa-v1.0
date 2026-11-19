"""First-run setup of the "rpa model" folder for ooa_local_pc.ipynb (cell 2), standard library only (it runs before
anything is installed), on Windows and Linux:

  - python_problems(): a Python the training cannot use (older than 3.10, 32-bit)
  - adopt_old(): an update zip carries the code only. When the previous "rpa model" folder was renamed (e.g. to
    "rpa model OLD") next to the new one, its platform/, nanochat_base/ (the installed model), sft/ (its context),
    train_data/ (the downloads) and assistant/assistant_data/ (your feedback) are copied into the new folder
  - ensure_nanochat(): nanochat at the pinned commit in "rpa model/nanochat": copied from the old folder, else
    `git clone` when Git is installed, else GitHub's zip of that commit (no Git needed)
  - platform_ok(): the platform's flow rules (platform/contracts/flow-schema/flow.v1.json) that cells 6 and 7 need

    python setup_pc.py --rpa DIR          the same from the command line
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

NANOCHAT_COMMIT = "92d63d4e8bb4df75c3b71618f31ddde2378b2bcd"   # = train_pc.NANOCHAT_COMMIT (checked by the tests)
NANOCHAT_REPO = "https://github.com/karpathy/nanochat"
NANOCHAT_ZIP = f"{NANOCHAT_REPO}/archive/{NANOCHAT_COMMIT}.zip"
MARKER = ".ooa_commit"                     # nanochat from the zip (no .git): the commit it holds
# what a renamed older "rpa model" folder holds that the code-only update zip does not; nanochat is handled by
# ensure_nanochat. local_run/ is not copied: a run in progress names its files by their full path in the old folder.
ADOPT = ["platform", "nanochat_base", "sft", "train_data", "assistant/assistant_data"]
SKIP = shutil.ignore_patterns("old_model_*", "base_checkpoints", "*.part*", "*.lock", "feedback_work",
                              "sft_render.download", "sft_render.local", "__pycache__")


def python_problems(version=None, bits64=None):
    """Why this Python cannot run the training (empty: it can)."""
    v = version or sys.version_info
    b = (sys.maxsize > 2**32) if bits64 is None else bits64
    out = []
    if tuple(v[:2]) < (3, 10):
        out.append(f"Python {v[0]}.{v[1]} is too old: PyTorch, gdown and this notebook need Python 3.10 or newer "
                   "(install Python 3.12 from https://www.python.org and pick it as the notebook's kernel)")
    if not b:
        out.append("this Python is 32-bit: PyTorch needs a 64-bit Python (install the 64-bit Python 3.12 from "
                   "https://www.python.org and pick it as the notebook's kernel)")
    return out


def path_notes(rpa, is_windows=None):
    """Warnings about where the folder is (Windows: long paths, OneDrive)."""
    is_windows = (os.name == "nt") if is_windows is None else is_windows
    s, out = str(rpa), []
    if is_windows and len(s) > 120:
        out.append(f"the folder path is {len(s)} characters long; Windows limits paths to 260 characters and the "
                   "model files sit about 90 characters deeper. If saving fails, move the folder to e.g. C:\\OOA")
    if is_windows and "onedrive" in s.lower():
        out.append("the folder is inside OneDrive: OneDrive uploads every saved training chunk (GBs) and can lock "
                   "files while it does. Better: move the folder to e.g. C:\\OOA (outside OneDrive)")
    return out


def nanochat_ok(nano):
    """A complete nanochat folder whose scripts/chat_sft.py is nanochat's own (not patched in place by an older
    tool: train_pc.py patches a copy)."""
    nano = Path(nano)
    if not all((nano / p).exists() for p in ("nanochat/engine.py", "nanochat/gpt.py", "scripts/chat_sft.py",
                                             "tasks/common.py")):
        return False
    try:
        return "OOADialogues" not in (nano / "scripts" / "chat_sft.py").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False


def _git():
    return shutil.which("git")


def _run(cmd, cwd=None):
    """-> (exit code, output); never raises for a missing program."""
    try:
        r = subprocess.run([str(c) for c in cmd], cwd=None if cwd is None else str(cwd), capture_output=True,
                           text=True, encoding="utf-8", errors="replace")
        return r.returncode, (r.stdout + r.stderr).strip()
    except OSError as ex:
        return 127, str(ex)


def nanochat_commit(nano):
    """The commit a nanochat folder holds: from git for a clone, from MARKER for the zip; None when unknown."""
    nano = Path(nano)
    if (nano / ".git").exists() and _git():
        rc, out = _run(["git", "-C", nano, "rev-parse", "HEAD"])
        if rc == 0 and out:
            return out.splitlines()[-1].strip()
    if (nano / MARKER).exists():
        return (nano / MARKER).read_text(encoding="utf-8").strip()
    return None


def old_folders(rpa):
    """Other "rpa model" folders next to this one (a renamed earlier copy), newest first."""
    rpa = Path(rpa).resolve()
    out = []
    try:
        sibs = list(rpa.parent.iterdir())
    except OSError:
        return out
    for p in sibs:
        try:
            if p.is_dir() and p.resolve() != rpa and ((p / "assistant" / "app.py").exists()
                                                      or (p / "nanochat_base").is_dir() or nanochat_ok(p / "nanochat")):
                out.append(p)
        except OSError:
            continue
    return sorted(out, key=lambda p: p.stat().st_mtime, reverse=True)


COPYING_MARK = ".ooa_copying"      # inside a folder while it is being copied from the older folder; removed when done


def copy_tree(src, dst, direct=None):
    """Copy src to dst. On Windows straight into dst (an editor such as VS Code / Antigravity watching the folder, or an
    antivirus, can hold new files for minutes, so renaming or deleting a temporary copy fails): a COPYING_MARK file
    sits in dst while it is copied, so a stopped copy is finished on the next run instead of being taken as complete.
    Elsewhere: copied to dst.copying, then renamed into place."""
    direct = (os.name == "nt") if direct is None else direct
    if direct:
        Path(dst).mkdir(parents=True, exist_ok=True)
        (Path(dst) / COPYING_MARK).write_text("copying from " + str(src), encoding="utf-8")
        shutil.copytree(src, dst, ignore=SKIP, dirs_exist_ok=True)
        (Path(dst) / COPYING_MARK).unlink(missing_ok=True)
        return
    tmp = Path(str(dst) + ".copying")
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(src, tmp, ignore=SKIP, dirs_exist_ok=True)
    try:
        replace(tmp, dst, tries=5)
    except PermissionError:
        shutil.copytree(tmp, dst, dirs_exist_ok=True)
        shutil.rmtree(tmp, ignore_errors=True)


def replace(src, dst, tries=10):
    """Rename src to dst. Windows: an antivirus or indexer briefly holding a new file makes a rename fail with
    'access denied'; wait and try again."""
    for i in range(tries):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if i == tries - 1:
                raise
            time.sleep(1 + i)


def adopt_old(rpa, log=print, olds=None):
    """Copy what the code-only update zip lacks from an older "rpa model" folder next to this one. -> names copied."""
    rpa = Path(rpa)
    olds = old_folders(rpa) if olds is None else olds
    done = []
    for item in ADOPT:
        dst = rpa / item
        if dst.exists() and any(p.name != COPYING_MARK for p in dst.iterdir()) and not (dst / COPYING_MARK).exists():
            continue                                  # already there (and not a copy that was stopped half-way)
        src = next((o / item for o in olds if (o / item).is_dir() and any((o / item).iterdir())), None)
        if src is None:
            continue
        log(f"copying {item} from the older folder {src.parent if '/' not in item else src.parent.parent} ...")
        leftover = Path(str(dst) + ".copying")       # from an earlier version's stopped copy; removed if Windows lets us
        if leftover.exists():
            shutil.rmtree(leftover, ignore_errors=True)
            if leftover.exists():
                log(f"  note: could not remove {leftover.name} (a program holds its files); delete it later by hand")
        dst.parent.mkdir(parents=True, exist_ok=True)
        copy_tree(src, dst)
        done.append(item)
    return done


def download(url, dest, log=print):
    """url -> dest with a progress line every 10 MB (system proxy settings and certificates)."""
    dest = Path(dest)
    tmp = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "ooa-local-pc"})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        n, mark = 0, 10 << 20
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)
            n += len(b)
            if n >= mark:
                log(f"  {n >> 20} MB")
                mark += 10 << 20
    replace(tmp, dest)
    return dest


def unpack_zip(zf, nano):
    """GitHub's zip (one top folder nanochat-<commit>/) -> nano; writes MARKER."""
    nano = Path(nano)
    work = nano.with_name(nano.name + ".unzip")
    shutil.rmtree(work, ignore_errors=True)
    with zipfile.ZipFile(zf) as z:
        z.extractall(work)
    tops = [p for p in work.iterdir() if p.is_dir()]
    if len(tops) != 1 or not nanochat_ok(tops[0]):
        shutil.rmtree(work, ignore_errors=True)
        raise SystemExit("the nanochat download is not the expected archive (no nanochat/engine.py in it)")
    (tops[0] / MARKER).write_text(NANOCHAT_COMMIT, encoding="utf-8")
    shutil.rmtree(nano, ignore_errors=True)
    replace(tops[0], nano)
    shutil.rmtree(work, ignore_errors=True)


def ensure_nanochat(rpa, log=print, olds=None, zip_url=NANOCHAT_ZIP, use_git=True):
    """nanochat at the pinned commit in rpa/nanochat. -> how it got there."""
    nano = Path(rpa) / "nanochat"
    if nanochat_ok(nano):
        have = nanochat_commit(nano)
        if have in (None, NANOCHAT_COMMIT):
            return "already here" + ("" if have else " (version not recorded: train_pc.py checks the script)")
        if (nano / ".git").exists() and _git() and use_git:
            log(f"nanochat is at {have[:7]}: switching to the pinned {NANOCHAT_COMMIT[:7]} ...")
            if _run(["git", "-C", nano, "cat-file", "-e", NANOCHAT_COMMIT])[0] != 0:
                _run(["git", "-C", nano, "fetch", "-q", "origin"])
            rc, out = _run(["git", "-C", nano, "checkout", "-q", "-f", NANOCHAT_COMMIT])
            if rc == 0:
                return "switched to the pinned commit"
            log("  git checkout failed:", out[-300:])
        log(f"nanochat is at another version ({have[:7]}): getting the pinned one again")
        shutil.rmtree(nano, ignore_errors=True)
    elif nano.exists():
        log("the nanochat folder is incomplete or was changed by an older tool: getting it again")
        shutil.rmtree(nano, ignore_errors=True)

    olds = old_folders(rpa) if olds is None else olds
    for o in olds:
        if nanochat_ok(o / "nanochat") and nanochat_commit(o / "nanochat") in (None, NANOCHAT_COMMIT):
            log(f"copying nanochat from the older folder {o} ...")
            tmp = Path(str(nano) + ".copying")
            shutil.rmtree(tmp, ignore_errors=True)
            shutil.copytree(o / "nanochat", tmp, ignore=shutil.ignore_patterns("__pycache__"))
            replace(tmp, nano)
            for f in ("chat_sft_ooa.py",):                       # written by train_pc.py, never part of nanochat
                (nano / "scripts" / f).unlink(missing_ok=True)
            if nanochat_ok(nano):
                return "copied from " + str(o)

    if use_git and _git():
        log(f"downloading nanochat with git ({NANOCHAT_COMMIT[:7]}) ...")
        tmp = Path(str(nano) + ".clone")
        shutil.rmtree(tmp, ignore_errors=True)
        rc, out = _run(["git", "clone", "-q", NANOCHAT_REPO + ".git", tmp])
        if rc == 0:
            rc, out = _run(["git", "-C", tmp, "checkout", "-q", "-f", NANOCHAT_COMMIT])
        if rc == 0 and nanochat_ok(tmp):
            replace(tmp, nano)
            return "cloned with git"
        log("  git did not work, downloading the zip instead:", out[-300:])
        shutil.rmtree(tmp, ignore_errors=True)
    else:
        log("Git is not installed (that is fine): downloading nanochat as a zip from GitHub ...")
    zf = Path(rpa) / f"nanochat-{NANOCHAT_COMMIT[:7]}.zip"
    try:
        if str(zip_url).startswith(("http://", "https://")):
            download(zip_url, zf, log)
        else:
            shutil.copyfile(zip_url, zf)                         # tests: a local zip
        unpack_zip(zf, nano)
    except (OSError, zipfile.BadZipFile) as ex:
        raise SystemExit(f"could not download nanochat ({ex}). Check the internet connection and run this cell "
                         f"again; or download {NANOCHAT_ZIP} in the browser, unzip it and rename the folder to "
                         f"{nano}") from None
    finally:
        Path(zf).unlink(missing_ok=True)
    return "downloaded the zip"


def platform_ok(rpa):
    return (Path(rpa) / "platform" / "contracts" / "flow-schema" / "flow.v1.json").exists()


PLATFORM_HELP = ("the platform files are missing ({p}). They come with ooa_assistant.zip (its 'rpa model/platform' "
                 "folder): copy that platform folder into {rpa}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rpa", default=str(Path(__file__).resolve().parent.parent.parent))
    a = ap.parse_args(argv)
    for p in python_problems() + path_notes(a.rpa):
        print("!", p)
    print("copied from an older folder:", adopt_old(a.rpa) or "nothing")
    print("nanochat:", ensure_nanochat(a.rpa))
    if not platform_ok(a.rpa):
        print("!", PLATFORM_HELP.format(p=Path(a.rpa) / "platform", rpa=a.rpa))
    return 0


if __name__ == "__main__":
    sys.exit(main())
