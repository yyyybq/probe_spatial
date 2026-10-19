from pathlib import Path

from scripts.build_tstep_probe_matrix import ALL_TASKS, STREAM_TASKS, TARGET_TASKS


def test_movement_is_a_streaming_canonical_task() -> None:
    assert "streaming_movement" in STREAM_TASKS
    assert "streaming_movement" in ALL_TASKS
    assert "streaming_movement" not in TARGET_TASKS


def test_registered_movement_configs_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    config_dir = root / "configs/experiment/inscene15k_streaming"
    vfms = (
        "aether",
        "cogvideox",
        "dino",
        "f3r",
        "lingbot",
        "minimax_h3",
        "videomae",
        "vjepa2",
        "wan",
    )
    for vfm in vfms:
        assert (config_dir / f"streaming_movement_{vfm}_v1.yaml").is_file()
    assert (config_dir / "direct_vlm_streaming_movement_v1.yaml").is_file()
