#!/usr/bin/env python3
"""One front door for the Wan P12/L20 and cross-objective C1/B2 studies."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence

from .build_integrated_report import collect_evidence, render_report


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = PACKAGE_DIR / "study_manifest.json"
DEFAULT_PYTHON = Path("/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python")
CORE_MODELS = ("vjepa2", "videomae", "wan", "cogvideox", "f3r")


def load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text())


def command_text(command: Sequence[str]) -> str:
    return " ".join(str(part) for part in command)


def run_command(
    command: Sequence[str],
    dry_run: bool,
    environment_overrides: Mapping[str, str] | None = None,
) -> None:
    print(f"$ {command_text(command)}", flush=True)
    if dry_run:
        return
    environment = os.environ.copy()
    current = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(PROJECT_ROOT) if not current else f"{PROJECT_ROOT}:{current}"
    )
    environment["PROJECT_ROOT"] = str(PROJECT_ROOT)
    if environment_overrides:
        environment.update(environment_overrides)
    subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)


def describe() -> None:
    manifest = load_manifest()
    print(f"Study: {manifest['study_id']}")
    contract = manifest["fixed_contract"]
    print(
        "Fixed Wan contract: "
        f"prefix={contract['prefix']}, layer={contract['layer']}, "
        f"C1={contract['c1_samples']} samples/{contract['c1_scenes']} scenes, "
        f"B2={contract['b2_samples']} samples/{contract['b2_scenes']} scenes"
    )
    for name, analysis in manifest["analyses"].items():
        print(f"\n{name}\n  Question: {analysis['question']}")
        entrypoints = analysis.get("canonical_entrypoints") or [
            analysis.get("canonical_entrypoint") or analysis.get("entrypoint")
        ]
        for entrypoint in entrypoints:
            print(f"  Code: {entrypoint}")


def check_study() -> int:
    manifest = load_manifest()
    paths = [MANIFEST_PATH, PACKAGE_DIR / "README.md"]
    for analysis in manifest["analyses"].values():
        for key in ("canonical_entrypoint", "entrypoint", "template", "output"):
            if analysis.get(key):
                paths.append(PROJECT_ROOT / analysis[key])
        for key in ("canonical_entrypoints", "outputs"):
            paths.extend(PROJECT_ROOT / value for value in analysis.get(key, []))
    missing = sorted(str(path.relative_to(PROJECT_ROOT)) for path in paths if not path.exists())
    result = {
        "study_id": manifest["study_id"],
        "required_paths": len(paths),
        "missing_paths": missing,
        "evidence": None,
        "status": "FAIL" if missing else "PASS",
    }
    if not missing:
        try:
            evidence = collect_evidence(PROJECT_ROOT)
            result["evidence"] = {
                "prefix": evidence["contract"]["prefix"],
                "layer": evidence["contract"]["layer"],
                "p12_final_interpretation": evidence["strict"]["interpretation"],
                "c1_formal_shuffle_gain": evidence["strict"]["formal_gain"],
                "b2_remove_visible_delta": evidence["b2"]["remove_visible"]["delta"],
            }
        except Exception as error:
            result["status"] = "FAIL"
            result["evidence_error"] = f"{type(error).__name__}: {error}"
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print delegated commands without executing them.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("describe", help="Explain the study questions and code map.")
    subparsers.add_parser("check", help="Read-only validation of code and result artifacts.")

    replay = subparsers.add_parser("c1b2-replay", help="Replay fixed C1/B2 checkpoints.")
    replay.add_argument("--task", choices=("c1", "b2", "all"), default="all")
    replay.add_argument("--model", choices=(*CORE_MODELS, "all"), default="wan")
    subparsers.add_parser("c1b2-summarize", help="Build C1/B2 tables, plots, and report.")
    subparsers.add_parser("p12-prepare", help="Freeze P12 inputs and formal mappings.")
    subparsers.add_parser("p12-run", help="Run P12 interventions on the current GPU node.")

    summarize = subparsers.add_parser("p12-summarize", help="Build P12 gates and report.")
    summarize.add_argument("--job-id", required=True)
    summarize.add_argument("--tests-passed", action="store_true")
    subparsers.add_parser("integrate", help="Rebuild the integrated C1/B2 report.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    python = str(args.python)
    if args.command == "describe":
        describe()
        return
    if args.command == "check":
        raise SystemExit(check_study())
    if args.command == "c1b2-replay":
        models = CORE_MODELS if args.model == "all" else (args.model,)
        for model in models:
            run_command(
                [
                    python,
                    "analyses/probe_validity/run_probe_validity.py",
                    "--phase2-replay",
                    args.task,
                    "--model",
                    model,
                ],
                args.dry_run,
            )
        return
    if args.command == "c1b2-summarize":
        run_command(
            [python, "analyses/probe_validity/run_probe_validity.py", "--phase2-summarize"],
            args.dry_run,
        )
        return
    if args.command == "p12-prepare":
        run_command([python, "scripts/prepare_wan_p12_l20_action_validity.py"], args.dry_run)
        return
    if args.command == "p12-run":
        run_command(
            ["bash", "scripts/run_wan_p12_l20_action_validity_sco.sh"],
            args.dry_run,
            environment_overrides={"PY": python},
        )
        return
    if args.command == "p12-summarize":
        command = [
            python,
            "scripts/summarize_wan_p12_l20_action_validity.py",
            "--job-id",
            args.job_id,
        ]
        if args.tests_passed:
            command.append("--tests-passed")
        run_command(command, args.dry_run)
        return
    if args.command == "integrate":
        output = PROJECT_ROOT / "reports/wan_p12_l20_c1_b2_integrated_analysis.md"
        print(f"$ rebuild {output.relative_to(PROJECT_ROOT)}", flush=True)
        if not args.dry_run:
            evidence = collect_evidence(PROJECT_ROOT)
            output.write_text(render_report(evidence, PACKAGE_DIR), encoding="utf-8")
        return
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    main()
