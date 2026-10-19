import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_complete_route_selects_only_2721_row_classes(tmp_path):
    builder = load_script(
        "build_minimax_route", "scripts/build_minimax_bf16_production_route.py"
    )
    classes = []
    for prefix in builder.PREFIXES:
        manifest = tmp_path / f"p{prefix}.jsonl"
        manifest.write_text("{}\n")
        classes.extend(
            [
                {
                    "prefix": prefix,
                    "rows": 1694,
                    "path": str(tmp_path / f"old-p{prefix}.jsonl"),
                    "sha256": "old",
                },
                {
                    "prefix": prefix,
                    "rows": 2721,
                    "path": str(manifest),
                    "sha256": file_sha(manifest),
                },
            ]
        )
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "status": "FROZEN_NOT_SUBMITTED",
                "parent_manifest": {"rows": 2721},
                "availability_classes": classes,
            }
        )
    )
    selected = builder.select_complete_manifests(catalog)
    assert set(selected) == set(builder.PREFIXES)
    assert all(path.name == f"p{prefix}.jsonl" for prefix, (path, _) in selected.items())


def test_canary_counts_are_derived_from_route(tmp_path):
    submit = load_script(
        "submit_minimax", "scripts/submit_minimax_bf16_production_v3.py"
    )
    manifests = {}
    for prefix in (8, 12, 16, 24):
        path = tmp_path / f"p{prefix}.jsonl"
        rows = [
            {
                "scene_id": "scene-a",
                "streaming_window_id": window,
                "streaming_window_indices": list(range(28)),
            }
            for window in (0, 1)
        ]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        manifests[str(prefix)] = {
            "path": str(path),
            "sha256": file_sha(path),
            "rows": len(rows),
        }
    route = {
        "frozen_manifests": manifests,
        "streaming_layers": [10, 20, 30, 40, 49],
        "target_layers": [20, 30, 40, 49],
    }
    assert submit.expected_canary_counts(route, {"scene-a"}) == (40, 4)


def test_worker_log_is_valid_canary_fallback(tmp_path):
    submit = load_script(
        "submit_minimax_log", "scripts/submit_minimax_bf16_production_v3.py"
    )
    submit.CANARY_REPORT = tmp_path / "missing-summary.json"
    submit.CANARY_LOG = tmp_path / "worker.log"
    expected = {
        "status": "PASS",
        "streaming_payloads_checked": 80,
        "target_payloads_checked": 4,
    }
    submit.CANARY_LOG.write_text("worker preamble\n" + json.dumps(expected) + "\nCOMPLETE\n")
    report, evidence = submit.load_canary_report()
    assert report == expected
    assert evidence == submit.CANARY_LOG
