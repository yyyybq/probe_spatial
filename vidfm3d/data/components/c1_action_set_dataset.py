"""Frozen same-anchor candidate-action sets for C3-from-C1 evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset


class C1ActionSetDataset(Dataset):
    """Load a frozen action-set artifact with no training behavior.

    The artifact is a ``torch.save``-ed list of records, or a dictionary whose
    ``records`` value is that list. Each record contains one shared C1 context
    and at least two matched action/target candidates. Tensor data are loaded
    eagerly so a missing or malformed artifact fails before evaluation starts.
    """

    REQUIRED_KEYS = frozenset({
        "action_set_id",
        "manifest_sample_id",
        "scene_path",
        "input_feat",
        "candidate_actions",
        "candidate_target_feat",
    })

    def __init__(self, path: str, split: str | None = None) -> None:
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"C1 action-set artifact not found: {self.path}")
        payload = torch.load(self.path, map_location="cpu", weights_only=False)
        records = payload.get("records") if isinstance(payload, dict) else payload
        if not isinstance(records, list) or not records:
            raise ValueError("C1 action-set artifact must contain a non-empty records list")
        if split is not None:
            records = [record for record in records if record.get("split") == split]
            if not records:
                raise ValueError(f"C1 action-set artifact contains no split={split!r} records")
        self.records = [self._validate_record(record, index) for index, record in enumerate(records)]
        ids = [record["action_set_id"] for record in self.records]
        if len(ids) != len(set(ids)):
            raise ValueError("C1 action-set artifact contains duplicate action_set_id values")

    @classmethod
    def _validate_record(cls, value: Any, index: int) -> Dict[str, Any]:
        if not isinstance(value, dict):
            raise TypeError(f"action-set record {index} is not a dictionary")
        missing = cls.REQUIRED_KEYS - set(value)
        if missing:
            raise ValueError(f"action-set record {index} is missing {sorted(missing)}")
        record = dict(value)
        record["action_set_id"] = str(record["action_set_id"])
        record["manifest_sample_id"] = str(record["manifest_sample_id"])
        record["scene_path"] = str(record["scene_path"])
        if not record["action_set_id"] or not record["manifest_sample_id"]:
            raise ValueError(f"action-set record {index} has empty lineage identifiers")

        context = record["input_feat"]
        actions = record["candidate_actions"]
        targets = record["candidate_target_feat"]
        if not all(isinstance(tensor, torch.Tensor) for tensor in (context, actions, targets)):
            raise TypeError(f"action-set record {index} context/actions/targets must be tensors")
        if context.ndim != 4:
            raise ValueError(f"action-set record {index} input_feat must have shape (T,H,W,C)")
        if actions.ndim != 2 or actions.shape[-1] != 9:
            raise ValueError(f"action-set record {index} candidate_actions must have shape (K,9)")
        candidates = int(actions.shape[0])
        if candidates < 2:
            raise ValueError(f"action-set record {index} needs at least two candidates")
        if targets.ndim not in {2, 4} or int(targets.shape[0]) != candidates:
            raise ValueError(
                f"action-set record {index} candidate_target_feat must have shape (K,C) or (K,H,W,C)"
            )
        if int(context.shape[-1]) != int(targets.shape[-1]):
            raise ValueError(f"action-set record {index} context/target channel dimensions differ")
        for name, tensor in (("input_feat", context), ("candidate_actions", actions),
                             ("candidate_target_feat", targets)):
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"action-set record {index} contains non-finite {name}")

        valid = record.get("candidate_valid", torch.ones(candidates, dtype=torch.bool))
        if not isinstance(valid, torch.Tensor) or valid.shape != (candidates,):
            raise ValueError(f"action-set record {index} candidate_valid must have shape (K,)")
        valid = valid.bool()
        if int(valid.sum().item()) < 2:
            raise ValueError(f"action-set record {index} has fewer than two valid candidates")
        record["candidate_valid"] = valid

        candidate_ids = record.get("candidate_ids", torch.arange(candidates))
        if not isinstance(candidate_ids, torch.Tensor) or candidate_ids.shape != (candidates,):
            raise ValueError(f"action-set record {index} candidate_ids must have shape (K,)")
        if len(set(candidate_ids.tolist())) != candidates:
            raise ValueError(f"action-set record {index} has duplicate candidate_ids")
        record["candidate_ids"] = candidate_ids

        equivalence = record.get("action_equivalence_group")
        if equivalence is not None:
            if not isinstance(equivalence, torch.Tensor) or equivalence.shape != (candidates,):
                raise ValueError(
                    f"action-set record {index} action_equivalence_group must be a tensor with shape (K,)"
                )
            if equivalence.dtype not in {
                torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8
            }:
                raise ValueError(
                    f"action-set record {index} action_equivalence_group must use integer class IDs"
                )
        return record

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.records[index]
