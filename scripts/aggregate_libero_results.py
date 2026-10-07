"""Aggregate LIBERO eval logs into a per-run summary table.

Parses the `[RESULTS_TABLE_START]...[RESULTS_TABLE_END]` blocks emitted by
`examples/libero/main.py` from one or more client log files (or a directory of
them) and prints a success-rate table per suite, plus a grand average.

Usage:
    uv run scripts/aggregate_libero_results.py results/libero/serial_<ts>/
    uv run scripts/aggregate_libero_results.py run1.log run2.log --name my_run
"""

import dataclasses
import pathlib
import re

import tyro

_LINE_RE = re.compile(
    r"task=(?P<task_id>\d+)\|(?P<desc>.*?)\|(?P<successes>\d+)/(?P<trials>\d+)\|(?P<rate>[\d.]+)"
)
_SUITE_RE = re.compile(r"suite=(?P<suite>\S+)")
_TOTAL_RE = re.compile(r"suite_total=(?P<s>\d+)/(?P<n>\d+)\|(?P<rate>[\d.]+)")


@dataclasses.dataclass
class SuiteResult:
    name: str
    tasks: list[tuple[int, str, int, int, float]]
    total_successes: int
    total_trials: int
    rate: float


def parse_log(path: pathlib.Path) -> list[SuiteResult]:
    """Parse all RESULTS_TABLE blocks from one log file."""
    text = path.read_text(errors="replace")
    blocks = re.findall(r"\[RESULTS_TABLE_START\](.*?)\[RESULTS_TABLE_END\]", text, re.S)
    suites: list[SuiteResult] = []
    for block in blocks:
        suite = "unknown"
        suite_match = _SUITE_RE.search(block)
        if suite_match:
            suite = suite_match.group("suite")
        tasks = [
            (int(m["task_id"]), m["desc"], int(m["successes"]), int(m["trials"]), float(m["rate"]))
            for m in _LINE_RE.finditer(block)
        ]
        total = _TOTAL_RE.search(block)
        if not tasks or total is None:
            continue
        suites.append(
            SuiteResult(
                name=suite,
                tasks=tasks,
                total_successes=int(total["s"]),
                total_trials=int(total["n"]),
                rate=float(total["rate"]),
            )
        )
    return suites


def main(log_paths: list[pathlib.Path], name: str | None = None) -> None:
    # Accept directories: glob any *.log / *.txt inside.
    files: list[pathlib.Path] = []
    for p in log_paths:
        if p.is_dir():
            files.extend(sorted(list(p.glob("*.log")) + list(p.glob("*.txt"))))
        else:
            files.append(p)
    all_suites: dict[str, SuiteResult] = {}
    for f in files:
        for suite in parse_log(f):
            all_suites[suite.name] = suite  # later logs override same-suite reruns
    if not all_suites:
        raise SystemExit(f"No [RESULTS_TABLE_*] blocks found in {len(files)} file(s).")

    run_name = name or (files[0].parent.name if len(files) == 1 else "run")
    print(f"\n=== LIBERO results: {run_name} ===")
    print(f"{'suite':<16}{'successes':>10}{'trials':>8}{'rate %':>9}")
    tot_s = tot_n = 0
    for suite in sorted(all_suites.values(), key=lambda s: s.name):
        print(f"{suite.name:<16}{suite.total_successes:>10}{suite.total_trials:>8}{suite.rate * 100:>9.2f}")
        tot_s += suite.total_successes
        tot_n += suite.total_trials
    avg = tot_s / tot_n if tot_n else 0.0
    print(f"{'TOTAL':<16}{tot_s:>10}{tot_n:>8}{avg * 100:>9.2f}")
    print(
        "\nReference: paper PLaW-VLA avg ≈ 97.4 | rebuttal stage3-only ≈ 94.05 | "
        "in-repo π0.5 baseline 96.85"
    )


if __name__ == "__main__":
    tyro.cli(main)
