"""Run the v1 evaluation matrix sequentially.

Usage (from the repository root):
    uv run python scripts/run_matrix.py              # run everything
    uv run python scripts/run_matrix.py --dry-run    # print the commands only
    uv run python scripts/run_matrix.py --resume     # skip labels already in the DB
    uv run python scripts/run_matrix.py --only v1-base-8b,v1-mcp-8b
    uv run python scripts/run_matrix.py --keep-going # do not stop at the first failure
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "evals" / "results" / "evals.sqlite"
LOG_PATH = ROOT / "evals" / "results" / "matrix.log"

RS_CASES = "research_synthesis,research_synthesis_para"

# (label, model, think, prompt variant, tools backend, repetitions, cases or None)
BATCHES: list[tuple[str, str, str, str, str, int, str | None]] = [
    ("v1-base-8b", "qwen3:8b", "off", "baseline", "local", 3, None),
    ("v1-wf-8b", "qwen3:8b", "off", "research_workflow", "local", 3, None),
    ("v1-mcp-8b", "qwen3:8b", "off", "baseline", "mcp", 3, None),
    ("v1-think-base-8b", "qwen3:8b", "on", "baseline", "local", 3, None),
    ("v1-think-wf-8b", "qwen3:8b", "on", "research_workflow", "local", 3, None),
    ("v1-rs-base-8b", "qwen3:8b", "off", "baseline", "local", 10, RS_CASES),
    ("v1-rs-wf-8b", "qwen3:8b", "off", "research_workflow", "local", 10, RS_CASES),
    ("v1-base-4b", "qwen3:4b", "off", "baseline", "local", 3, None),
    ("v1-base-14b", "qwen3:14b", "off", "baseline", "local", 3, None),
]

# (label, baseline label): one factor varies per line
COMPARISONS: list[tuple[str, str]] = [
    ("v1-wf-8b", "v1-base-8b"),
    ("v1-think-base-8b", "v1-base-8b"),
    ("v1-think-wf-8b", "v1-base-8b"),
    ("v1-think-wf-8b", "v1-wf-8b"),
    ("v1-mcp-8b", "v1-base-8b"),
    ("v1-base-4b", "v1-base-8b"),
    ("v1-base-14b", "v1-base-8b"),
    ("v1-rs-wf-8b", "v1-rs-base-8b"),
]


def ollama_base_url() -> str:
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    return host if "://" in host else f"http://{host}"


def preflight(models: set[str]) -> None:
    """Fail early (before hours of compute) if the environment is not ready."""
    base = ollama_base_url()
    try:
        version = httpx.get(f"{base}/api/version", timeout=5).json().get("version")
        installed = {m["name"] for m in httpx.get(f"{base}/api/tags", timeout=5).json()["models"]}
    except Exception as exc:  # noqa: BLE001 - any failure means "not ready"
        sys.exit(f"Ollama does not answer at {base} ({exc}). Start it and retry.")
    missing = sorted(models - installed)
    if missing:
        sys.exit("Missing models: " + ", ".join(missing) + "\n" + "\n".join(f"  ollama pull {m}" for m in missing))
    print(f"Ollama {version} OK, models present: {', '.join(sorted(models))}")

    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    if dirty:
        sys.exit("Uncommitted changes (the runs would be marked dirty):\n" + dirty)
    print("Git tree clean.")


def done_labels() -> set[str]:
    if not DB_PATH.exists():
        return set()
    try:
        with sqlite3.connect(DB_PATH) as con:
            return {row[0] for row in con.execute("SELECT label FROM batches")}
    except sqlite3.Error:
        return set()  # schema differs: do not skip anything


def stream(cmd: list[str], log) -> int:
    """Run a command, echo its output live and copy it to the log file."""
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    with subprocess.Popen(
        cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", env=env,
    ) as proc:
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        return proc.wait()


def batch_cmd(b: tuple[str, str, str, str, str, int, str | None]) -> list[str]:
    label, model, think, prompt, tools, repeat, cases = b
    cmd = [
        sys.executable, "-m", "evals.run", "--suite", "minimal",
        "--model", model, "--think", think, "--prompt-variant", prompt,
        "--tools", tools, "--repeat", str(repeat), "--label", label,
        "--require-clean",
    ]
    if cases:
        cmd += ["--cases", cases]
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true", help="skip labels already stored")
    parser.add_argument("--only", help="comma-separated labels to run")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--no-reports", action="store_true", help="skip the comparison reports")
    args = parser.parse_args()

    selected = [b for b in BATCHES if not args.only or b[0] in args.only.split(",")]
    if args.resume:
        already = done_labels()
        selected = [b for b in selected if b[0] not in already]
    if not selected:
        sys.exit("Nothing to run.")

    if args.dry_run:
        for b in selected:
            print(" ".join(batch_cmd(b)))
        return

    preflight({b[1] for b in selected})
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    started = time.monotonic()

    with LOG_PATH.open("a", encoding="utf-8") as log:
        for b in selected:
            header = f"\n=== {b[0]} ({b[1]}, think {b[2]}, {b[3]}, tools {b[4]}) ===\n"
            sys.stdout.write(header)
            log.write(header)
            t0 = time.monotonic()
            code = stream(batch_cmd(b), log)
            note = f"--- {b[0]}: exit {code}, {(time.monotonic() - t0) / 60:.1f} min ---\n"
            sys.stdout.write(note)
            log.write(note)
            if code != 0:
                failures.append(b[0])
                if not args.keep_going:
                    break

        if not failures and not args.no_reports:
            stored = done_labels()
            for label, baseline in COMPARISONS:
                if label in stored and baseline in stored:
                    stream([sys.executable, "-m", "evals.report", "--label", label, "--compare", baseline], log)

    print(f"\nTotal: {(time.monotonic() - started) / 60:.1f} min. Log: {LOG_PATH}")
    if failures:
        sys.exit("Failed batches: " + ", ".join(failures))


if __name__ == "__main__":
    main()