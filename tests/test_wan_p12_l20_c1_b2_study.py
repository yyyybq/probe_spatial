import json
import math
from pathlib import Path

from analyses.wan_p12_l20_c1_b2_study.build_integrated_report import (
    collect_evidence,
    render_report,
)
from analyses.probe_validity.build_representation_geometry_bridge import collect


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "analyses/wan_p12_l20_c1_b2_study"


def test_study_manifest_points_to_existing_sources_and_outputs():
    manifest = json.loads((PACKAGE / "study_manifest.json").read_text())
    assert manifest["fixed_contract"] == {
        "model": "wan",
        "prefix": 12,
        "layer": 20,
        "c1_samples": 276,
        "c1_scenes": 69,
        "b2_samples": 165,
        "b2_scenes": 65,
    }
    for analysis in manifest["analyses"].values():
        paths = []
        paths.extend(analysis.get("canonical_entrypoints", []))
        paths.extend(analysis.get("outputs", []))
        for key in ("canonical_entrypoint", "entrypoint", "template", "output"):
            if analysis.get(key):
                paths.append(analysis[key])
        assert paths
        assert all((ROOT / path).exists() for path in paths)


def test_integrated_evidence_matches_frozen_wan_results():
    evidence = collect_evidence(ROOT)
    assert evidence["contract"] == {"model": "wan", "prefix": 12, "layer": 20}
    strict = evidence["strict"]
    assert strict["interpretation"] == "WEAK_ACTION_SIGNAL"
    assert strict["phase1_gate"] == "WEAK_OR_NON_ACTION_SPECIFIC_SIGNAL"
    assert math.isclose(strict["formal_gain"], 0.052644927536231886)
    assert strict["formal_ci_low"] > 0
    assert strict["copy_ci_low"] < 0 < strict["copy_ci_high"]
    assert (strict["positive_classes"], strict["total_classes"]) == (12, 12)
    assert (strict["positive_scenes"], strict["total_scenes"]) == (59, 69)

    c1 = evidence["c1"]
    assert math.isclose(c1["legacy_shuffle_gain"], 0.0543478)
    assert c1["matched_ci_low"] < 0 < c1["matched_ci_high"]
    assert c1["same_scene_ci_low"] > 0

    b2 = evidence["b2"]
    assert b2["current_frame"]["ci_low"] > 0
    assert b2["remove_visible"]["ci_low"] > 0
    assert b2["query_only"]["ci_low"] > 0
    for control in ("query_mask", "query_swap", "shuffle", "reverse", "global_prior"):
        assert b2[control]["ci_low"] < 0 < b2[control]["ci_high"]

    geometry = evidence["geometry"]
    assert math.isclose(geometry["wan_copy_raw_cosine"], 0.978937566280365)
    assert math.isclose(geometry["wan_temporal_displacement"], 0.02106243371963501)
    assert geometry["dino_copy_centered_cosine"] > geometry["wan_copy_centered_cosine"]


def test_integrated_report_is_fully_substituted_and_explains_the_boundary():
    report = render_report(collect_evidence(ROOT), PACKAGE)
    assert "$formal_gain" not in report
    assert "+0.0526" in report
    assert "[-0.0109, +0.0761]" in report
    assert "ACTION_CONDITIONED_DYNAMICS = NOT_ESTABLISHED" in report
    assert "HISTORY_CONTENT_DEPENDENCE = SUPPORTED" in report
    assert "不是“Wan 完全没有动力学信息”" in report
    assert "Representation Geometry / Cosine Illusion Bridge" in report


def test_geometry_bridge_preserves_the_recovered_p12_distinction():
    rows = {str(row["model"]): row for row in collect()}
    wan = rows["wan"]
    dino = rows["dino"]
    assert wan["n"] == 276
    assert wan["raw_copy_target_cosine"] > dino["raw_copy_target_cosine"]
    assert dino["centered_copy_target_cosine"] > wan["centered_copy_target_cosine"]
    assert math.isclose(wan["full_minus_copy_r1"], 0.03260869565217392)
