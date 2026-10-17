from analyses.se3_protocol_v2.wan_vjepa2_v2_data import (
    WAN_TARGET_GAP_WORKLIST_SHA256,
    _gap_target_index_uri,
    _gap_target_uri,
)


def test_wan_target_delta_route_binds_worklist_and_layer_index():
    digest = WAN_TARGET_GAP_WORKLIST_SHA256[:16]
    expected_parent = (
        "aoss://baiqiao/InsScene-15K/"
        "FEAT_TARGET_CANONICAL_V2_GAP_DELTA_V1/wan/scannetpp/scene-a/"
        f".canonical_v2_target_delta/{digest}"
    )
    assert _gap_target_uri("wan", 15, "scene-a") == (
        f"{expected_parent}/feature_t749_layer15.delta.sft"
    )
    assert _gap_target_index_uri("wan", 15, "scene-a") == (
        f"{expected_parent}/target_indices_layer15.npy"
    )


def test_non_wan_target_gap_route_keeps_legacy_layout():
    assert _gap_target_uri("vjepa2", 11, "scene-a").endswith(
        "/vjepa2/scannetpp/scene-a/feature_layer11.sft"
    )
    assert _gap_target_index_uri("vjepa2", 11, "scene-a").endswith(
        "/vjepa2/scannetpp/scene-a/target_indices.npy"
    )
