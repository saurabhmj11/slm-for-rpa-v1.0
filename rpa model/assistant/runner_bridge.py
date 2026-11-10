"""Run a flow on this PC with the platform's Python reference runner (package `oneofficeautomation-reference`,
import name `ooa`), inside a sandbox folder.

What this module does:
- `available`: whether the reference runner can be imported (from `<platform_repo>/reference` or an installed copy).
- `supported_actions()`: the action ids the local runner implements (from the registry manifests).
- `can_run(flow)`: unsupported actions, risky actions, reference `validate_flow` problems, path-guard problems and the
  flow's input variables the user must fill.
- `run(flow, inputs, allow_risky, timeout_s)`: runs the flow in a SUBPROCESS (`sys.executable -I`) with cwd and
  work_dir = the sandbox folder, killed after `timeout_s` seconds. Returns outputs, at most 200 events, the error and the
  files in the sandbox after the run.

RISKY actions (blocked unless the user confirms with allow_risky=True), kept simple and explicit:
- deleting files: `ooa.file/delete`
- overwriting: `ooa.file/copy` when its `overwrite` arg is set, `ooa.excel/add_sheet` when its `replace` arg is set
  (writing a text/JSON/CSV/Excel file is NOT risky: the path guard keeps it inside the sandbox)
- network: everything in `ooa.rest`
- AI: everything in `ooa.genai` and `ooa.agents`
- running other bots: `core/run_bot`
- anything whose package is not one of core, ooa.file, ooa.excel, ooa.datatable, ooa.json, ooa.string

PATH GUARD, two layers:
1. Before the run (`can_run`): every literal string in step args, variable defaults and run inputs that looks like a
   path must resolve inside the sandbox folder (absolute paths outside it and '..' escapes are refused; relative paths
   are resolved against the sandbox). This gives early, readable messages.
2. During the run (in the subprocess): every path an action opens goes through the runner's `resolve_path`, which is
   wrapped so a path outside the sandbox (built from an expression, a variable, parsed JSON, ...) fails the step with
   FILE.OUTSIDE_SANDBOX, and folder listings never return files outside it.
This is still a guard against mistakes, NOT a security boundary: the flow runs with the user's own permissions and
risky actions (network, AI, other bots) are only gated by the user's confirmation.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections import deque
from pathlib import Path, PureWindowsPath

SAFE_PACKAGES = {"core", "ooa.file", "ooa.excel", "ooa.datatable", "ooa.json", "ooa.string"}
RISKY_PACKAGES = {"ooa.rest", "ooa.genai", "ooa.agents"}
RISKY = {
    "ooa.file/delete",
    "core/run_bot",
    "ooa.rest/request",
    "ooa.genai/prompt", "ooa.genai/classify", "ooa.genai/summarize", "ooa.genai/extract_to_schema",
    "ooa.agents/run_agent",
}
# action id -> arg name: risky only when that arg is given and not literally false (overwrite-type behaviour)
RISKY_WHEN_ARG = {"ooa.file/copy": "overwrite", "ooa.excel/add_sheet": "replace"}

MAX_EVENTS = 200
_HEAD_EVENTS = 190          # events kept from the start of a run; the rest of MAX_EVENTS come from its end
_MAX_FILES = 500
_PATH_KEYS = re.compile(r"(path|folder|file|dir|source|target)", re.I)
_VAR_REF = re.compile(r"^\$[A-Za-z_][A-Za-z0-9_]*$")
_EXPR = re.compile(r"\$\{[^}]*\}")


def is_risky(action: str, args: dict | None = None) -> bool:
    pid = action.partition("/")[0]
    if action in RISKY or pid in RISKY_PACKAGES or pid not in SAFE_PACKAGES:
        return True
    arg = RISKY_WHEN_ARG.get(action)
    if arg and isinstance(args, dict) and arg in args and args[arg] not in (False, "false", "False", None, ""):
        return True
    return False


def _list(x):
    return x if isinstance(x, list) else []


def _walk(steps):
    for s in _list(steps):
        if not isinstance(s, dict):
            continue
        yield s
        for key in ("children", "else", "finally"):
            yield from _walk(s.get(key))
        for b in _list(s.get("elseIf")):
            if isinstance(b, dict):
                yield from _walk(b.get("children"))
        for c in _list(s.get("catch")):
            if isinstance(c, dict):
                yield from _walk(c.get("children"))


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _strings(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _strings(v)


def _looks_like_path(s: str, key: str = "") -> bool:
    s = s.strip()
    if not s or "://" in s or _VAR_REF.match(s):
        return False
    if key and _PATH_KEYS.search(key):
        return True
    if s.startswith("~") or s == ".." or re.match(r"^[A-Za-z]:[\\/]", s):
        return True
    return ("/" in s or "\\" in s) and bool(s.strip("/\\"))


def path_problem(s: str, sandbox: Path, key: str = "") -> str | None:
    """None if `s` is not path-like or stays inside the sandbox, else a plain reason."""
    if not isinstance(s, str) or not _looks_like_path(s, key):
        return None
    literal = _EXPR.sub("x", s.strip()).strip("'\"")   # keep the literal parts of "${...}" / quoted expressions
    if not literal or _VAR_REF.match(literal):
        return None
    if literal.startswith("~") or PureWindowsPath(literal).drive or PureWindowsPath(literal).root:
        p = Path(os.path.expanduser(literal)) if literal.startswith("~") else Path(literal)
    else:
        p = sandbox / literal
    try:
        p.resolve().relative_to(sandbox.resolve())
    except ValueError:
        how = "escapes the sandbox folder with '..'" if ".." in re.split(r"[\\/]", literal) \
            else "is outside the sandbox folder"
        return f"path '{s}' {how}"
    return None


def _import_reference(platform_repo: str | None):
    if platform_repo:
        ref = str(Path(platform_repo) / "reference")
        if Path(ref, "ooa").is_dir() and ref not in sys.path:
            sys.path.insert(0, ref)
    from ooa.packages import builtin_registry          # noqa: E402
    from ooa.runner.validate import validate_flow      # noqa: E402
    return builtin_registry, validate_flow


class RunnerBridge:
    def __init__(self, platform_repo: str | None, sandbox_dir: str):
        self.platform_repo = platform_repo
        self.sandbox = Path(sandbox_dir).resolve()
        self.sandbox.mkdir(parents=True, exist_ok=True)
        self.available = False
        self.unavailable_reason = None
        self._registry = None
        self._validate = None
        self._supported: set[str] = set()
        self._reference_dir = None
        try:
            builtin_registry, self._validate = _import_reference(platform_repo)
            self._registry = builtin_registry()
            for m in self._registry.all_manifests():
                for a in m.get("actions", []):
                    self._supported.add(f"{m['id']}/{a['id']}")
            import ooa
            self._reference_dir = str(Path(ooa.__file__).resolve().parents[1])
            self.available = True
        except Exception as e:  # missing package or its dependencies (jsonschema, cryptography)
            self.unavailable_reason = f"{type(e).__name__}: {e}"

    def supported_actions(self) -> set[str]:
        return set(self._supported)

    # ---------- checks ----------
    def _path_problems(self, flow: dict, inputs: dict | None = None) -> list[str]:
        out = []
        for v in _list(flow.get("variables")):
            if isinstance(v, dict) and "default" in v:
                for s in _strings(v["default"]):
                    msg = path_problem(s, self.sandbox, str(v.get("name", "")))
                    if msg:
                        out.append(f"variable {v.get('name')}: {msg}")
        for step in _walk(flow.get("steps")):
            args = step.get("args") if isinstance(step.get("args"), dict) else {}
            for k, val in args.items():
                for s in _strings(val):
                    msg = path_problem(s, self.sandbox, k)
                    if msg:
                        out.append(f"step {step.get('id')}: input '{k}': {msg}")
        for k, val in (inputs or {}).items():
            for s in _strings(val):
                msg = path_problem(s, self.sandbox, k)
                if msg:
                    out.append(f"input '{k}': {msg}")
        return out

    def can_run(self, flow: dict) -> dict:
        res = {"ok": False, "unsupported": [], "risky": [], "problems": [], "inputs": []}
        if not isinstance(flow, dict):
            res["problems"].append("The flow must be a JSON object.")
            return res
        for v in _list(flow.get("variables")):
            if isinstance(v, dict) and v.get("direction") in ("input", "in_out"):
                res["inputs"].append({"name": v.get("name"), "type": v.get("type"), "required": bool(v.get("required")),
                                      "default": v.get("default"), "description": v.get("description", "")})
        if not self.available:
            res["problems"].append(f"The local reference runner is not available ({self.unavailable_reason}).")
            return res
        unsupported_steps = set()
        for step in _walk(flow.get("steps")):
            act = step.get("action")
            if not isinstance(act, str):
                continue
            if act not in self._supported:
                unsupported_steps.add(step.get("id"))
                if act not in res["unsupported"]:
                    res["unsupported"].append(act)
            elif is_risky(act, step.get("args")) and act not in res["risky"]:
                res["risky"].append(act)
        try:
            problems = self._validate(flow, self._registry, raise_on_error=False)
        except Exception as e:
            problems = [{"stepId": None, "message": f"Validation crashed: {e}"}]
        for p in problems:
            if p.get("stepId") in unsupported_steps and p.get("path") == "action":
                continue        # already reported as unsupported
            where = f"step {p['stepId']}: " if p.get("stepId") else (f"{p['path']}: " if p.get("path") else "")
            fix = f" (fix: {p['suggestedFix']})" if p.get("suggestedFix") else ""
            res["problems"].append(f"{where}{p.get('message')}{fix}")
        res["problems"].extend(self._path_problems(flow))
        res["ok"] = not res["unsupported"] and not res["problems"]
        return res

    # ---------- run ----------
    def run(self, flow: dict, inputs: dict, allow_risky: bool = False, timeout_s: int = 60) -> dict:
        t0 = time.monotonic()
        base = {"ok": False, "outputs": {}, "events": [], "error": None, "seconds": 0.0, "files": []}
        inputs = inputs or {}
        check = self.can_run(flow)
        reason = None
        if not check["ok"]:
            parts = []
            if check["unsupported"]:
                parts.append("the local runner does not support " + ", ".join(check["unsupported"]))
            if check["problems"]:
                parts.append("cannot run locally: " + "; ".join(check["problems"]))
            reason = "; ".join(parts) or "cannot run locally"
        elif check["risky"] and not allow_risky:
            reason = "risky actions need your confirmation: " + ", ".join(check["risky"])
        else:
            bad = self._path_problems({}, inputs)
            if bad:
                reason = "cannot run locally: " + "; ".join(bad)
        if reason:
            base.update(refused=reason, error=reason, files=self._files())
            return base

        job = Path(tempfile.mkdtemp(prefix="ooa_run_"))
        try:
            (job / "job.json").write_text(json.dumps({
                "flow": flow, "inputs": inputs, "sandbox": str(self.sandbox), "reference": self._reference_dir,
                "events": str(job / "events.jsonl"), "result": str(job / "result.json"),
                "user_site": _user_site()}), encoding="utf-8")
            kw = {"start_new_session": True} if os.name != "nt" else \
                {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
            proc = subprocess.Popen([sys.executable, "-I", str(Path(__file__).resolve()), "--child", str(job / "job.json")],
                                    cwd=str(self.sandbox), stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kw)
            timed_out = False
            try:
                _, err = proc.communicate(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
                _kill(proc)
                _, err = proc.communicate()
            events = _read_events(job / "events.jsonl")
            result = None
            if (job / "result.json").is_file():
                try:
                    result = json.loads((job / "result.json").read_text(encoding="utf-8"))
                except ValueError:
                    result = None
            if result:
                events = (events[:_HEAD_EVENTS] + result.get("tail", []))[:MAX_EVENTS] \
                    if result.get("dropped") else events[:MAX_EVENTS]
                base["outputs"] = result.get("outputs") or {}
                base["ok"] = result.get("status") == "success"
                if not base["ok"]:
                    base["error"] = _error_text(result.get("error")) or f"run {result.get('status')}"
            else:
                events = events[:MAX_EVENTS]
                if timed_out:
                    base["error"] = f"Stopped: the run took longer than {timeout_s} s and was killed."
                else:
                    tail = (err or b"").decode("utf-8", "replace").strip().splitlines()[-5:]
                    base["error"] = "The runner process failed: " + (" | ".join(tail) or f"exit code {proc.returncode}")
            base["events"] = events
        finally:
            shutil.rmtree(job, ignore_errors=True)
        base["files"] = self._files()
        base["seconds"] = round(time.monotonic() - t0, 2)
        return base

    def _files(self) -> list[str]:
        out = []
        for p in sorted(self.sandbox.rglob("*")):
            if p.is_file():
                out.append(p.relative_to(self.sandbox).as_posix())
                if len(out) >= _MAX_FILES:
                    break
        return out


def _user_site() -> str | None:
    """The parent's user site-packages (pip --user installs). The child runs with -I, which hides it."""
    import site
    if not site.ENABLE_USER_SITE:
        return None
    usp = site.getusersitepackages()
    return usp if usp and os.path.isdir(usp) else None


def _install_sandbox_guard(sandbox: Path) -> None:
    """Child only: make the reference runner refuse any file path outside the sandbox at run time."""
    from ooa.sdk.errors import PackageError
    from ooa.sdk.package import ActionContext
    root = sandbox.resolve()

    def inside(path: Path) -> bool:
        try:
            path.resolve().relative_to(root)
            return True
        except (ValueError, OSError):
            return False

    class SandboxPath(type(Path())):
        def glob(self, pattern, *a, **kw):
            return (p for p in super().glob(pattern, *a, **kw) if inside(p))

        def rglob(self, pattern, *a, **kw):
            return (p for p in super().rglob(pattern, *a, **kw) if inside(p))

    original = ActionContext.resolve_path

    def resolve_path(self, p):
        path = original(self, p)
        full = path if path.is_absolute() else Path(self.work_dir) / path
        if not inside(full):
            raise PackageError(f"path '{p}' is outside the sandbox folder", code="FILE.OUTSIDE_SANDBOX",
                               suggested_fix="Use a path inside the sandbox folder (a relative path like data/in.csv).")
        return SandboxPath(full)

    ActionContext.resolve_path = resolve_path


def _kill(proc: subprocess.Popen) -> None:
    try:
        if os.name != "nt":
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except OSError:
            pass


def _error_text(err) -> str | None:
    if not err:
        return None
    if isinstance(err, str):
        return err
    text = f"{err.get('code', 'ERROR')}: {err.get('message', '')}"
    if err.get("stepId"):
        text = f"step {err['stepId']}: " + text
    if err.get("suggestedFix"):
        text += f" (fix: {err['suggestedFix']})"
    return text


def _read_events(path: Path) -> list[dict]:
    out = []
    if not path.is_file():
        return out
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                out.append(json.loads(line))
            except ValueError:
                pass            # last line may be cut short by a kill
    return out


def _event_view(e) -> dict:
    d = e.data or {}
    t = e.type
    if t == "step.started":
        msg = f"start {d.get('action')}"
    elif t == "step.finished":
        msg = f"done {d.get('action')} ({d.get('durationMs')} ms)"
    elif t == "step.failed":
        msg = f"failed {d.get('action')}: {d.get('code')}: {d.get('message')}"
    elif t == "step.retry":
        msg = f"retry {d.get('attempt')} after {d.get('code')}"
    elif t == "variable.set":
        msg = f"{d.get('name')} = {d.get('value')}"
    elif t == "log":
        msg = str(d.get("message", ""))
    elif t == "run.started":
        msg = f"run started ({d.get('automation')})"
    elif t == "run.finished":
        msg = f"run finished: {d.get('status')} ({d.get('durationMs')} ms)"
    else:
        msg = json.dumps(d, default=str)[:300]
    return {"type": t, "step": e.step_id, "message": msg[:500]}


def _child(job_path: str) -> int:
    """Runs inside the subprocess: execute the flow with the reference Platform and write events + result files."""
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    if job.get("reference") and job["reference"] not in sys.path:
        sys.path.insert(0, job["reference"])
    if job.get("user_site") and job["user_site"] not in sys.path:
        sys.path.append(job["user_site"])          # packages installed with pip --user (jsonschema, openpyxl, ...)
    from ooa.governance import Principal
    from ooa.platform import Platform
    from ooa.sdk.types import to_jsonable

    class CappedSink:
        def __init__(self, path):
            self.fh = open(path, "w", encoding="utf-8")
            self.n = 0
            self.tail = deque(maxlen=MAX_EVENTS - _HEAD_EVENTS)

        def emit(self, event):
            self.n += 1
            view = _event_view(event)
            if self.n <= _HEAD_EVENTS:
                self.fh.write(json.dumps(view, default=str) + "\n")
                self.fh.flush()
            else:
                self.tail.append(view)

    sink = CappedSink(job["events"])
    sandbox = Path(job["sandbox"])
    _install_sandbox_guard(sandbox)
    p = Platform(None, sink=sink, work_dir=sandbox)
    res = p.run(job["flow"], job["inputs"], Principal("assistant", roles=("Automation Developer",)))
    sink.fh.close()
    dropped = max(0, sink.n - _HEAD_EVENTS - len(sink.tail))
    Path(job["result"]).write_text(json.dumps(to_jsonable({
        "status": res.status, "outputs": res.outputs, "error": res.error,
        "tail": list(sink.tail) if sink.n > _HEAD_EVENTS else [], "dropped": sink.n > _HEAD_EVENTS,
        "droppedCount": dropped}), default=str), encoding="utf-8")
    return 0


if __name__ == "__main__" and len(sys.argv) == 3 and sys.argv[1] == "--child":
    sys.exit(_child(sys.argv[2]))
