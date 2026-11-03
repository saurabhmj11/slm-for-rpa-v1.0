"""Build the OOA action catalog from the platform's generated toolkit reference pages.

Usage:
    python build_catalog.py PLATFORM_REPO_DIR [OUT_JSONL]

Reads PLATFORM_REPO_DIR/docs/reference/toolkits/*.md (generated from `ooa manifest`) and writes one JSON object
per action / condition / loop type:
    {id, kind, toolkit, toolkit_id, name, description, inputs[], outputs[], notes}
Pure standard library; no platform install needed.
"""
import json
import re
import sys
from pathlib import Path

ENTRY = re.compile(r"^### (.+)$")
ID_LINE = re.compile(r"^`([^`]+)` \((action|condition|loop type)\)\s*$")
HEAD = re.compile(r"^# (.+)$")
TK_LINE = re.compile(r"^Toolkit `([^`]+)`, version ([^.]+(?:\.[^.]+)*)\.\s*(.*)$")


def split_row(line):
    # table cells; "\|" is an escaped pipe inside a cell
    cells = re.split(r"(?<!\\)\|", line.strip().strip("|"))
    return [c.strip().replace("\\|", "|") for c in cells]


def parse_table(lines, i):
    """lines[i] is the header row. Returns (rows as dicts, next index)."""
    header = split_row(lines[i])
    i += 2  # skip the --- separator
    rows = []
    while i < len(lines) and lines[i].startswith("|"):
        cells = split_row(lines[i])
        rows.append(dict(zip(header, cells)))
        i += 1
    return rows, i


def parse_file(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    toolkit = ""
    toolkit_id = ""
    entries = []
    cur = None
    i = 0
    while i < len(lines):
        line = lines[i]
        m = HEAD.match(line)
        if m and not toolkit:
            toolkit = m.group(1)
        m = TK_LINE.match(line)
        if m:
            toolkit_id = m.group(1)
        m = ENTRY.match(line)
        if m:
            cur = {"name": m.group(1), "toolkit": toolkit, "toolkit_id": toolkit_id,
                   "inputs": [], "outputs": [], "description": "", "notes": ""}
            entries.append(cur)
            i += 1
            continue
        if cur is not None:
            m = ID_LINE.match(line)
            if m:
                cur["id"], cur["kind"] = m.group(1), m.group(2)
                # description = next non-empty line
                j = i + 1
                while j < len(lines) and not lines[j].strip():
                    j += 1
                if j < len(lines) and not lines[j].startswith(("**", "|", "#")):
                    cur["description"] = lines[j].strip()
            elif line.strip() == "**Settings**":
                j = i + 1
                while j < len(lines) and not lines[j].startswith("|"):
                    j += 1
                rows, i = parse_table(lines, j)
                for r in rows:
                    cur["inputs"].append({
                        "name": r.get("Name", "").strip("`"),
                        "label": r.get("Field", ""),
                        "type": r.get("Type", ""),
                        "required": r.get("Required", "") == "Yes",
                        "default": r.get("Default", "").strip("`"),
                        "description": r.get("Description", ""),
                    })
                continue
            elif line.strip() == "**Results**":
                j = i + 1
                while j < len(lines) and not lines[j].startswith("|"):
                    j += 1
                rows, i = parse_table(lines, j)
                for r in rows:
                    cur["outputs"].append({
                        "name": r.get("Name", "").strip("`"),
                        "type": r.get("Type", ""),
                        "description": r.get("Description", ""),
                    })
                continue
            elif line.startswith("*") and line.endswith("*") and len(line) > 2:
                cur["notes"] = line.strip("*")
        i += 1
    return [e for e in entries if "id" in e]


def main():
    repo = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).with_name("catalog.jsonl")
    pages = sorted((repo / "docs/reference/toolkits").glob("*.md"))
    pages = [p for p in pages if p.name != "README.md"]
    total = 0
    with out.open("w", encoding="utf-8") as f:
        for p in pages:
            for e in parse_file(p):
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
                total += 1
    print(f"{total} entries from {len(pages)} toolkit pages -> {out}")


if __name__ == "__main__":
    main()
