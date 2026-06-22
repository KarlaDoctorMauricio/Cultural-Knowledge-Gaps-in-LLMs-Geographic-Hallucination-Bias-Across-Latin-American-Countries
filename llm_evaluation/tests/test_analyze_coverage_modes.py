import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.analyze_coverage import MODE_PRESETS, apply_mode_preset, parse_args


def test_mode_sample_preset():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=sorted(MODE_PRESETS), default=None)
    parser.add_argument("--source", default="manual")
    parser.add_argument("--n", type=int, default=999)
    parser.add_argument("--output-dir", type=Path, default=Path("manual"))
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    args = apply_mode_preset(parser.parse_args(["--mode", "sample"]))

    assert args.source == MODE_PRESETS["sample"]["source"]
    assert args.n == 0
    assert args.output_dir == MODE_PRESETS["sample"]["output_dir"]


def test_mode_full_preset():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=sorted(MODE_PRESETS), default=None)
    parser.add_argument("--source", default="manual")
    parser.add_argument("--n", type=int, default=999)
    parser.add_argument("--output-dir", type=Path, default=Path("manual"))
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    args = apply_mode_preset(parser.parse_args(["--mode", "full"]))

    assert args.n == 5000
    assert args.output_dir == MODE_PRESETS["full"]["output_dir"]


def test_no_mode_keeps_manual_defaults():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=sorted(MODE_PRESETS), default=None)
    parser.add_argument("--source", default="manual-source")
    parser.add_argument("--n", type=int, default=123)
    parser.add_argument("--output-dir", type=Path, default=Path("manual-out"))
    args = apply_mode_preset(parser.parse_args([]))

    assert args.mode is None
    assert args.source == "manual-source"
    assert args.n == 123
