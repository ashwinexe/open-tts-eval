"""Evaluate every sampled run in place and build the combined comparison.

Runs are the ``out/eval_multi/<provider>-<model>/`` directories produced by
``sample_multi.py`` (audio + manifest already colocated there). Each is assessed
into its own directory, so audio, manifest, results, and report live together
with relative audio paths. The measurement cache makes re-runs instant. Finally
a single ``compare_all`` report is written across all runs.

Run from the repo root:  python scripts/eval_multi.py
"""

from __future__ import annotations

from pathlib import Path

from tts_assess.config import load_config
from tts_assess.pipeline import run_assessment
from tts_assess.reporting.compare import run_comparison

REPO = Path(__file__).resolve().parent.parent
EVAL = REPO / "out" / "eval_multi"
CONFIG = load_config(REPO / "out" / "eval_multi.yml")

# Newest / most-important first, so partial results are still useful. Each entry
# is (run-directory name, comparison label).
RUNS: list[tuple[str, str]] = [
    ("inworld-inworld-tts-2-preview", "inworld-tts-2-preview"),
    ("inworld-inworld-tts-1.5-max", "inworld-1.5-max"),
    ("inworld-inworld-tts-1-max", "inworld-1-max"),
    ("elevenlabs-eleven_v3", "11l-v3"),
    ("elevenlabs-eleven_multilingual_v2", "11l-multilingual-v2"),
    ("hume-octave-2", "hume-octave-2"),
]


def main() -> None:
    done: list[tuple[Path, str]] = []
    for name, label in RUNS:
        run_dir = EVAL / name
        manifest = run_dir / "manifest.jsonl"
        if not manifest.exists():
            print(f"SKIP {name}: no manifest (run sample_multi.py first)", flush=True)
            continue
        print(f"EVAL {name} ...", flush=True)
        _rows, summary = run_assessment(manifest, run_dir, CONFIG)
        cache = summary.get("cache", {})
        print(
            f"  {label}: n={summary['sample_count']} pass={summary['pass_rate'] * 100:.1f}% "
            f"cache_hits={cache.get('hits')} computed={cache.get('misses')}",
            flush=True,
        )
        done.append((run_dir, label))

    if len(done) >= 2:
        run_comparison(
            [run_dir for run_dir, _ in done],
            EVAL / "compare_all",
            CONFIG,
            labels=[label for _, label in done],
        )
        print(f"wrote compare_all across {len(done)} runs", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
