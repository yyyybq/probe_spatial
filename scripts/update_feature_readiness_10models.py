"""Refresh the ten-model feature dashboard. Read-only toward data and scheduler.

No extraction, submission, cancellation, feature deletion or loader mutation.
Counts are unique scientific keys, not file counts or scheduler RUNNING counts.
"""
import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import re
import struct
import subprocess
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/main_v2_probe_v2_816_20260917"
DATA = Path("/mnt/umm/users/yinbaiqiao/InsScene-15K")
OUT = BASE / "feature_readiness_10models"
SCO = "/mnt/umm/users/yinbaiqiao/.sco/bin/sco"
MODELS = ["cogvideox", "videomae", "vjepa2", "lingbot", "dino", "f3r",
          "qwen2_5_vl", "aether", "wan", "minimax_h3"]
SOURCE_SHA = "ebc496eb0587cda372ccf29a7c5244ef454096151f73c8eed9511967ded1902d"
PROTOCOL_SHA = "215c720c8537f4dfae6243bedcc7339fbb7e3ba8a6ca2dcaf04f49cd4b738e4f"
ACTIVE = {"RUNNING", "STARTING", "QUEUEING", "PENDING", "CREATING", "QUEUED"}

def qwen_snapshot_status():
    path = ROOT.parent / '.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5'
    result = {'path': str(path), 'model_load_tested': False,
              'validation': 'index, tensor names, safetensors header and file length; not full payload hash'}
    try:
        index = load(path / 'model.safetensors.index.json')
        names = sorted(set(index['weight_map'].values()))
        assert len(names) == 5
        total = 0
        for name in names:
            assert Path(name).name == name
            p = path / name
            with p.open('rb') as f:
                header_size = struct.unpack('<Q', f.read(8))[0]
                assert 0 < header_size < 100_000_000
                header = json.loads(f.read(header_size))
            expected = {k for k,v in index['weight_map'].items() if v == name}
            assert expected <= header.keys()
            payload_size = max(v['data_offsets'][1] for k,v in header.items() if k != '__metadata__')
            assert p.stat().st_size == 8 + header_size + payload_size
            total += p.stat().st_size
        result.update(status='CACHE_FILES_COMPLETE', shards=len(names), tensor_count=len(index['weight_map']), bytes=total)
    except (OSError, ValueError, KeyError, AssertionError, struct.error) as exc:
        result.update(status='CACHE_NOT_CONFIRMED', error=repr(exc))
    return result

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def canonical(obj):
    return digest(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode())

def load(path):
    return json.loads(Path(path).read_text())

def publish(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".readiness-", dir=path.parent)
    with os.fdopen(fd, "w") as f:
        f.write(text)
    os.replace(tmp, path)

def latest_owner_jobs(jobs, model):
    latest = {}
    for job in jobs:
        if job["model"] != model:
            continue
        if job.get("command"):
            owner, category = job_owner(job['command'], job['display_name'])
            job = dict(job, owner=owner, category=category)
        key = (job["owner"], job["category"])
        if key not in latest or str(job["created_utc"]) > str(latest[key]["created_utc"]):
            latest[key] = job
    return list(latest.values())

def job_owner(cmd, display_name):
    scripts = re.findall(r"(?:bash|python\S*)\s+([^\s]+\.(?:sh|py))", cmd)
    text = (cmd + ' ' + display_name).replace('postcanary', '')
    category = 'canary' if 'canary' in text or '-can-' in display_name else 'production'
    owner = (scripts[-1] if category == 'canary' else Path(scripts[-1]).name) if scripts else display_name
    part = re.search(r"plans/(aether_target|aether_streaming)/batches/(part-\d+).json", cmd)
    if part:
        owner = '/'.join(part.groups())
    shard = re.search(r"SHARD_INDEX=(\d+)\s+SHARD_COUNT=(\d+)", cmd)
    if shard:
        owner += '/shard-' + '-of-'.join(shard.groups())
    return owner, category

def feature_status(stats, seven, model, errors):
    ready = all(s["remaining_without_verified_evidence"] == 0 for s in stats.values())
    conflicts = any(e.get("blocking", False) for e in errors)
    status = "READY" if ready and not conflicts else "PENDING" if seven or model == "aether" else "PARTIAL_EVIDENCE"
    return status, ready and seven and not conflicts

def scheduler_snapshot(refresh):
    path = OUT / "scheduler_snapshot.json"
    if not refresh:
        return load(path) if path.exists() else {"observed_utc": None, "jobs": [], "error": "NOT_QUERIED"}
    selected = {}
    for page in range(1, 4):
        result = subprocess.run([SCO, "acp", "jobs", "list", "--workspace-name", "aigc",
                                 "--page-size", "100", "--page-token", str(page), "-o", "json"],
                                capture_output=True, text=True, timeout=45, check=True)
        rows = json.loads(result.stdout)
        for row in rows:
            cmd = "\n".join(role.get("startup_script", "") for role in row.get("roles", []))
            if not any(token in cmd for token in ("atomic_feature_", "priority_layers_recovery",
                    "main_v2_aether_release", "wan_aether_streaming_official", "minimax_bf16",
                    "f3r_stream_canary", "f3r_target_canary", "qwen_snapshot_hydration", "canonical_v2_gap_recovery",
                    "vjepa_qwen_f3r_recovery_20260923")):
                continue
            display = row['display_name'].lower()
            if 'qwen2_5_vl' in display or '-qwen' in display:
                model = 'qwen2_5_vl'
            elif 'f3r' in display:
                model = 'f3r'
            elif 'vjepa2' in display:
                model = 'vjepa2'
            else:
                model = next((m for m in MODELS if m in cmd), None)
            if "minimax_bf16" in cmd:
                model = "minimax_h3"
            if model is None and "qwen" in cmd:
                model = "qwen2_5_vl"
            if model is None:
                continue
            owner, category = job_owner(cmd, row['display_name'])
            selected[row["name"]] = {
                "job_id": row["name"], "model": model, "state": row["state"],
                "created_utc": row.get("create_time"), "owner": owner, "category": category,
                "display_name": row["display_name"], "command": cmd,
                "pool": row.get("resource_pool", {}).get("display_name"),
            }
        if len(rows) < 100:
            break
    value = {"observed_utc": datetime.now(timezone.utc).isoformat(), "pages_queried": 3,
             "jobs": list(selected.values()), "error": None}
    publish(path, json.dumps(value, indent=2) + "\n")
    return value

def last_progress(logdir):
    observations = []
    if not logdir.is_dir():
        return observations
    for owner in logdir.iterdir():
        attempts = sorted(owner.glob("*/receipt.json"))
        if not attempts:
            continue
        path = attempts[-1]
        try:
            receipt = load(path)
            detail = []
            for log in path.parent.glob("gpu*.log"):
                with log.open("rb") as f:
                    f.seek(max(0, log.stat().st_size - 65536))
                    tail = f.read().decode(errors="replace")
                lines = [line for line in tail.splitlines() if "[INFO]" in line or "[ERROR]" in line]
                if lines:
                    detail.append(lines[-1][-400:])
            observations.append({"owner": owner.name, "receipt_status": receipt.get("status"),
                                 "path": str(path), "last_log_lines": detail})
        except (OSError, ValueError) as exc:
            observations.append({"owner": owner.name, "error": str(exc)})
    return observations

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-scheduler", action="store_true")
    args = parser.parse_args()
    scheduler = scheduler_snapshot(args.refresh_scheduler)
    scope_path = BASE / "targeted_reextract_worklists_20260922_r2/final_scope.json"
    scope = load(scope_path)
    queue_path = Path(scope["frozen_probing_experiments"]["path"])
    assert digest(queue_path.read_bytes()) == scope["frozen_probing_experiments"]["sha256"]
    queue = [json.loads(line) for line in queue_path.read_text().splitlines()]
    specs = {m: sorted({(r["disk_layer"], r["timestep"]) for r in queue if r["model"] == m}) for m in MODELS}
    specs.update(aether=[(1, 749)], wan=[(l, 749) for l in (0, 5, 10, 15, 20, 25, 29)],
                 minimax_h3=[(l, 250) for l in (10, 20, 30, 40, 49)])
    prefix_manifests = load(BASE / "execution_manifest_r3_receipt_union/manifest.json")["availability_classes"]
    wanted = {r["protocol"]["canonical_manifest_sha256"] for r in queue}
    stream_universe, future_universe, protocol_universe = {}, set(), set()
    for binding in prefix_manifests:
        if binding["sha256"] not in wanted:
            continue
        raw = Path(binding["path"]).read_bytes()
        assert digest(raw) == binding["sha256"]
        prefix = int(binding["prefix"])
        for line in raw.splitlines():
            row = json.loads(line)
            k = (row["scene_id"], int(row["streaming_window_id"]), prefix)
            stream_universe[k] = digest(json.dumps(row["streaming_indices"], separators=(",", ":")).encode())
            window = row["streaming_window_indices"]
            future_universe.update((k[0], int(window[prefix - 1 + h])) for h in (1, 2, 4))
            protocol_universe.update((k[0], int(window[prefix - 1 + h])) for h in (0, 1, 2, 4))
    assert len(stream_universe) == 10884 and len(future_universe) == 32350
    gaps = defaultdict(set)
    for binding in scope["worklists"]:
        path = Path(binding["worklist"])
        raw = path.read_bytes()
        assert digest(raw) == binding["file_sha256"]
        worklist = json.loads(raw)
        mode = "streaming" if worklist["mode"] == "streaming_prefix" else "target"
        for entry in worklist["entries"]:
            for layer in entry["layers"]:
                k = (int(layer), entry["scene"], int(entry["window_id"]), int(entry["prefix"])) if mode == "streaming" else (int(layer), entry["scene"], int(entry["frame_index"]))
                gaps[(binding["model"], mode)].add(k)
    # Existing exact-key positive remote gates supply Wan/Aether baseline keys.
    gate_root = ROOT / "artifacts/wan_aether_streaming_official_r1/overlay"
    gate_meta = load(gate_root / "manifest.json")
    gate_path = ROOT / gate_meta["route_file"]["path"]
    assert digest(gate_path.read_bytes()) == gate_meta["route_file"]["sha256"]
    gate_keys = defaultdict(set)
    with gzip.open(gate_path, "rt") as f:
        for line in f:
            row = json.loads(line); k = row["key"]
            if k[1] in ("wan", "aether") and row["status"] == "REFERENCE_SOURCE_READ_ONLY":
                assert "/FEAT_STREAMING_CANONICAL_V2_GAP_DELTA_V1/" in row["source_uri"]
                gate_keys[k[1]].add((int(k[4]), k[3], int(k[6]), int(k[7])))

    historical = load(BASE / "training_canary_r3_receipt_union/summary.json")
    historical_pass = {r["model"]: r for r in historical["passed"]}
    records, layer_rows, errors, evidence = [], [], [], []
    for model in MODELS:
        layers = [l for l, _t in specs[model]]
        # MiniMax's current full-sample route deliberately has four target layers.
        target_layers = layers
        mini_route = BASE / "minimax_bf16_v2_production_route_v2/route_receipt.json"
        if model == "minimax_h3" and mini_route.exists():
            target_layers = load(mini_route)["target_layers"]
        required = {
            "streaming": {(l, *k) for l in layers for k in stream_universe},
            "target": {(l, *k) for l in target_layers for k in future_universe},
        }
        seven = model in {r["model"] for r in queue}
        coverage, stats, progress = {}, {}, {}
        for mode, capital in (("streaming", "STREAMING"), ("target", "TARGET")):
            baseline = (required[mode] - gaps[(model, mode)]) if seven else (gate_keys[model] & required[mode] if mode == "streaming" else set())
            fresh = set()
            log_root = DATA / f"FEAT_{capital}_CANONICAL_V2_GAP_DELTA_V1/.windowfix_v2_gap_logs/{model}"
            progress[mode] = last_progress(log_root) if model != "minimax_h3" else []
            receipt_paths = set()
            if model != "minimax_h3":
                for p in log_root.glob("*/*/receipt.json"):
                    try:
                        receipt_paths.add(load(p)["remote_completion_receipt"])
                    except (OSError, ValueError, KeyError) as exc:
                        errors.append({"model": model, "path": str(p), "error": str(exc)})
            for name in sorted(receipt_paths):
                path = Path(name)
                if not path.is_file():
                    continue
                try:
                    raw = path.read_bytes(); value = json.loads(raw)
                    unhashed = dict(value)
                    assert unhashed.pop("content_sha256") == canonical(unhashed)
                    lineage = value["lineage"]
                    if lineage.get("vfm") != model or (lineage.get("timestep") not in {t for _l, t in specs[model]}):
                        continue
                    assert lineage["canonical_source_sha256"] == SOURCE_SHA
                    assert lineage["protocol_sha256"] == PROTOCOL_SHA
                    assert value["status"] == "PASS_REMOTE_VERIFIED"
                    assert value["remote"]["base_uri"].rstrip("/").endswith(f"/FEAT_{capital}_CANONICAL_V2_GAP_DELTA_V1/{model}/scannetpp")
                    received = set()
                    for obj in value["objects"].values():
                        identity = obj["canonical_unit_identity"]
                        assert canonical(identity) == obj["canonical_unit_identity_sha256"]
                        layer = identity.get("layer")
                        if layer not in layers:
                            continue
                        if mode == "streaming" and identity.get("kind") == "streaming_feature":
                            k = identity["gap_record_key"]
                            key = (k["scene"], int(k["window_id"]), int(k["prefix"]))
                            if key in stream_universe:
                                if "gap_record_raw_ids_sha256" not in identity:
                                    # Legacy receipts are not proof of corruption. The
                                    # frozen baseline already accounts for retained keys.
                                    continue
                                assert identity["gap_record_raw_ids_sha256"] == stream_universe[key]
                                received.add((layer, *key))
                        elif mode == "target" and identity.get("kind") == "target_delta_layer" and identity.get("payload_role") == "feature":
                            received.update((layer, identity["scene"], int(f)) for f in identity["target_indices"])
                    fresh.update(received & required[mode])
                    evidence.append({"model": model, "mode": mode, "path": name,
                                     "sha256": digest(raw), "required_keys_in_receipt": len(received & required[mode])})
                except (OSError, ValueError, KeyError, AssertionError) as exc:
                    errors.append({"model": model, "path": name, "error": type(exc).__name__ + ": " + str(exc),
                                   "blocking": isinstance(exc, AssertionError)})
            known = baseline | fresh
            coverage[mode] = known
            stats[mode] = {"required": len(required[mode]), "verified_available": len(known),
                           "remaining_without_verified_evidence": len(required[mode] - known),
                           "percent": round(100 * len(known) / len(required[mode]), 2),
                           "retained_baseline": len(baseline), "newly_verified_since_baseline": len(fresh - baseline),
                           "scope": "exact_queue_keys" if seven else "full_v2_reference_scope_not_original_528",
                           "not_evidence_of_absence": True}
            for layer in sorted({k[0] for k in required[mode]}):
                n = sum(k[0] == layer for k in required[mode]); done = sum(k[0] == layer for k in known)
                layer_rows.append({"model": model, "mode": mode, "disk_layer": layer,
                                   "required": n, "verified_available": done, "remaining": n - done})
        jobs = latest_owner_jobs(scheduler["jobs"], model)
        for job in jobs:
            if job['state'] != 'FAILED':
                continue
            # Read only the launcher/log explicitly named by this job. No
            # unrelated log search and no inference from old RUNNING receipts.
            scripts = re.findall(r'bash\s+([^\s]+\.sh)', job['command'])
            try:
                launcher = Path(scripts[-1]).read_text() if scripts else ''
                match = re.search(r'exec\s*>\s*(/[^\s]+)', launcher)
                if match:
                    log = Path(match.group(1))
                    with log.open('rb') as f:
                        f.seek(max(0, log.stat().st_size - 4000))
                        job['failure_log_tail'] = f.read().decode(errors='replace')
                    job['failure_log_path'] = str(log)
            except OSError:
                pass
        relevant_errors = [e for e in errors if e["model"] == model]
        status, ready_for_queue = feature_status(stats, seven, model, relevant_errors)
        notes = []
        if model == "qwen2_5_vl":
            cache = qwen_snapshot_status()
            notes.append(f"指定 Qwen2.5-VL-7B 快照：{cache['status']}；5 分片索引/头部/长度核对，不等同于 GPU 模型加载测试；{cache['path']}")
        if model == "f3r":
            for mode in ("stream", "target"):
                p = BASE / f"f3r_{mode}_canary_scene0_20260923_r2/receipts/f3r.json"
                notes.append(f"新版 {mode} canary：" + (str(load(p).get("status")) if p.exists() else "尚无 PASS receipt；参见 scheduler/log"))
        if model == "minimax_h3":
            notes.extend(["BF16_V2 独立 namespace；旧非 BF16 特征不计入", "完整样本 route_v2 已生成；当前进度查看 v4 canary，不再沿用旧 80/40 阻塞",
                          "此处 0 表示尚未计入正式完成证明，不表示磁盘上没有特征；target L10 不在当前提取 route"])
            log = BASE / "minimax_bf16_v2_production_extract_v4/shard_0_of_681/worker.log"
            if log.exists():
                tail = log.read_text()[-12000:]
                match = re.search(r'(\{\s*"created_utc".*?\n\})\s*COMPLETE shard=0/681', tail, re.S)
                if match:
                    verification = json.loads(match.group(1))
                    notes.append(f"v4 单场景本地校验 {verification['status']}：{verification['streaming_payloads_checked']} streaming payload、{verification['target_payloads_checked']} target payload；不等同于全量远端完成")
        if not seven:
            notes.append("不属于原 queue_528；" + ("已另列 Aether 24 项" if model == "aether" else "本清单不自动扩增 probing 任务"))
        if relevant_errors:
            notes.append(f"{len(relevant_errors)} 条证据读取/校验备注，详见 read_errors；仅身份冲突阻塞，旧记录不可读本身不推断缺文件")
        records.append({"model": model, "queue_528_units": sum(r["model"] == model for r in queue),
                        "additional_planned_units": 24 if model == "aether" else 0,
                        "disk_layers": layers, "target_layers": target_layers,
                        "feature_status": status, "feature_ready_for_queue_528": ready_for_queue,
                        "streaming": stats["streaming"], "target": stats["target"],
                        "historical_training_canary": historical_pass.get(model),
                        "current_full_route_training_acceptance": "NOT_ASSERTED_BY_COVERAGE_REPORT",
                        "jobs": jobs, "progress_logs": progress, "notes": notes})
        print(model, stats["streaming"]["verified_available"], stats["target"]["verified_available"], flush=True)

    stamp = datetime.now(timezone.utc).isoformat()
    report = {"updated_utc": stamp, "queue": {"path": str(queue_path), "sha256": digest(queue_path.read_bytes()), "units": 528},
              "refresh_command": "python3 scripts/update_feature_readiness_10models.py --refresh-scheduler",
              "policy": {"streaming_per_layer": 10884, "target_training_required_per_layer": 32350,
                         "target_extraction_including_anchor_per_layer": 37677,
                         "coverage_does_not_assert_loader_training_acceptance": True,
                         "missing_receipt_does_not_authorize_reextraction": True,
                         "baseline_scope": str(scope_path), "baseline_scope_sha256": digest(scope_path.read_bytes())},
              "models": records, "by_layer": layer_rows, "receipt_evidence": evidence,
              "read_errors": errors, "scheduler_observed_utc": scheduler.get("observed_utc"),
              "scheduler_error": scheduler.get("error")}
    raw = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    publish(OUT / "latest.json", raw)
    publish(OUT / "history" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".json"), raw)
    csv_rows = [{"model": r["model"], "queue_528_units": r["queue_528_units"], "feature_status": r["feature_status"],
                 "streaming_done": r["streaming"]["verified_available"], "streaming_total": r["streaming"]["required"],
                 "target_done": r["target"]["verified_available"], "target_total": r["target"]["required"],
                 "notes": "; ".join(r["notes"])} for r in records]
    for name, rows in (("latest.csv", csv_rows), ("by_layer.csv", layer_rows)):
        buf = io.StringIO(); writer = csv.DictWriter(buf, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        publish(OUT / name, buf.getvalue())
    md = ["# 十模型特征 readiness 与进度", "", f"更新时间：{stamp}；scheduler：{scheduler.get('observed_utc')}", "",
          "原 queue_528 保持 7 模型 528 项；Aether 另有 24 项，Wan/MiniMax 不自动加入。",
          "数字是已有验证证据覆盖的独立特征键。尚无完成记录不等于缺文件，也不授权重提。",
          "target 采用训练需要的 h1/h2/h4：每层 32,350；包含额外 anchor 的提取协议为 37,677，二者不混用。", "",
          "| 模型 | 原528任务数 | Streaming 已验证/需要 | Target 已验证/需要 | 特征状态 | 当前任务 |",
          "|---|---:|---:|---:|---|---|"]
    for r in records:
        states = Counter(j["state"] for j in r["jobs"] if j["state"] in ACTIVE)
        stext = f"{r['streaming']['verified_available']:,}/{r['streaming']['required']:,}"
        ttext = f"{r['target']['verified_available']:,}/{r['target']['required']:,}"
        if r['model'] == 'minimax_h3':
            stext = f"全量未核验 / {r['streaming']['required']:,}"
            ttext = f"全量未核验 / {r['target']['required']:,}"
        md.append(f"| {r['model']} | {r['queue_528_units']} | {stext} | {ttext} | {r['feature_status']} | " + (", ".join(f"{k}={v}" for k,v in sorted(states.items())) or "本次查询未见活跃任务") + " |")
    md.extend(["", "## 进度与阻塞", ""])
    for r in records:
        md.append(f"### {r['model']}"); md.append("")
        md.extend("- " + n for n in r["notes"])
        md.append("- 历史训练 canary：" + ("已通过，但仅覆盖当时的小样本/路径" if r["historical_training_canary"] else "无本表认可的记录"))
        for job in sorted(r["jobs"], key=lambda j: j["created_utc"] or "", reverse=True):
            md.append(f"- {job['job_id']}：{job['state']}；{job['category']}；{job['display_name']}")
            if job.get('failure_log_tail'):
                md.append('  - 最后日志：' + job['failure_log_tail'].strip().splitlines()[-1])
        for mode, rows in r['progress_logs'].items():
            for observation in rows:
                # Show recent extraction attempts; old RUNNING receipts are
                # retained in JSON but never presented as live scheduler state.
                if '/2026092' not in observation.get('path', ''):
                    continue
                lines = observation.get('last_log_lines', [])
                detail = ' / '.join(lines) or observation.get('receipt_status', '未知')
                md.append(f"- 最近日志（非 scheduler 状态）{mode}/{observation['owner']}：{detail}")
        md.append("")
    md.extend(["## 更新与明细", "", "```bash", report["refresh_command"], "```", "",
               "- latest.json：全部 job、最新提取日志、证据路径与 SHA、异常。",
               "- by_layer.csv：逐层已验证量及待确认量，便于优先跟踪 VJEPA2 L23、VideoMAE L23、LingBot L30。",
               "- history/：每次刷新快照。脚本仅更新此状态目录，不提交、取消、修改任何训练或特征。",
               "- READY 只表示特征覆盖；当前 loader 路径绑定和训练 canary 的验收单独进行，不把历史小样本 PASS 当全量可启动。", ""])
    publish(OUT / "README.md", "\n".join(md))
    print(json.dumps({"dashboard": str(OUT / "README.md"), "errors": len(errors), "updated": stamp}))

if __name__ == "__main__":
    main()
