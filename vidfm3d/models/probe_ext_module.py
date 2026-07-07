"""ProbeExtensionLitModule.

A single LightningModule for the four diagnostic probes (A2/A3/B1/C1).
The probe type is selected via the ``probe_type`` hyperparameter and dispatches
to a different ``model_step`` implementation. Only one probe type runs per
training job — keep configs simple and isolate metrics.

Supported probe_type values:
    "view_consistency"  (A2)
    "abnormal"          (A3)
    "ego_belief"        (B1)
    "action_dynamics"   (C1)
    "path_integration"  (C2)
    "counterfactual"    (C3)
    "sae_spatial"       (SAE dictionary + spatial readouts)
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torchmetrics import MeanMetric

from vidfm3d.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def _resize_mask_to_feat(
    mask_hw: torch.Tensor,    # (B, S, H, W) bool/float
    H_f: int,
    W_f: int,
) -> torch.Tensor:
    """Nearest-neighbor downsample a per-pixel mask to feature resolution."""
    B, S, H, W = mask_hw.shape
    m = mask_hw.reshape(B * S, 1, H, W).float()
    m = F.interpolate(m, size=(H_f, W_f), mode="nearest")
    return m.reshape(B, S, H_f, W_f) > 0.5


class ProbeExtensionLitModule(LightningModule):
    PROBE_TYPES = {
        "view_consistency",
        "abnormal",
        "ego_belief",
        "ego_belief_v2",
        "action_dynamics",
        "path_integration",
        "counterfactual",
        "sae_spatial",
    }

    def __init__(
        self,
        probe: nn.Module,
        probe_type: str,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool = False,
        # A2 specific
        view_overlap_pos_threshold: float = 0.4,
        view_overlap_neg_threshold: float = 0.05,
        # B1 specific
        polar_loss_weights: Tuple[float, float, float] = (1.0, 1.0, 0.5),
        # C1 specific
        cosine_loss_weight: float = 1.0,
        mse_loss_weight: float = 1.0,
        # SAE specific
        sae_l1_weight: float = 1e-4,
        sae_overlap_weight: float = 0.2,
        sae_ego_weight: float = 0.2,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if probe_type not in self.PROBE_TYPES:
            raise ValueError(
                f"Unknown probe_type {probe_type!r}; must be one of {self.PROBE_TYPES}"
            )

        self.save_hyperparameters(logger=False, ignore=["probe"])
        self.probe = probe
        self.probe_type = probe_type
        self.kwargs = kwargs

        self.val_loss = MeanMetric()

    def _zero_loss(self) -> torch.Tensor:
        """Return a zero loss that is connected to the parameter graph so backward() is safe."""
        p = next(self.probe.parameters())
        return p.sum() * 0.0

    # ------------------------------------------------------------------ A2
    def _step_view_consistency(
        self, batch: Dict[str, torch.Tensor], train: bool
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        vfm_feat = batch["vfm_feat"]                   # (B, S, H_f, W_f, C)
        overlap_gt = batch["overlap_gt"]               # (B, S, S) in [0,1]
        B, S = vfm_feat.shape[:2]

        logits = self.probe(vfm_feat)                  # (B, S, S)

        # Build training pairs: ignore diagonal, only use clearly pos/neg pairs
        eye = torch.eye(S, dtype=torch.bool, device=logits.device).unsqueeze(0).expand(B, -1, -1)
        pos_mask = (overlap_gt >= self.hparams.view_overlap_pos_threshold) & (~eye)
        neg_mask = (overlap_gt <= self.hparams.view_overlap_neg_threshold) & (~eye)

        target = torch.zeros_like(overlap_gt)
        target[pos_mask] = 1.0
        sample_mask = pos_mask | neg_mask
        n_used = sample_mask.float().sum().clamp(min=1.0)
        n_pos = pos_mask.float().sum()
        n_neg = neg_mask.float().sum()

        bce = F.binary_cross_entropy_with_logits(
            logits, target, reduction="none"
        )
        loss = (bce * sample_mask.float()).sum() / n_used

        # also a regression loss over the full overlap matrix (excluding diagonal)
        soft = F.binary_cross_entropy_with_logits(
            logits, overlap_gt, reduction="none"
        )
        soft_loss = (soft * (~eye).float()).sum() / (~eye).float().sum().clamp(min=1.0)
        loss = loss + 0.5 * soft_loss

        with torch.no_grad():
            pred_prob = torch.sigmoid(logits)
            err = (pred_prob - overlap_gt).abs()
            mae = (err * (~eye).float()).sum() / (~eye).float().sum().clamp(min=1.0)
            # binary acc on hard pos/neg pairs
            pred_bin = (pred_prob > 0.5).float()
            correct = (pred_bin == target).float()
            acc = (correct * sample_mask.float()).sum() / n_used

        metrics = {
            "loss_overlap": loss.detach(),
            "overlap_mae": mae,
            "overlap_acc": acc,
            "overlap_n_used": n_used.detach(),
            "overlap_pos_frac": n_pos / n_used,
            "overlap_neg_frac": n_neg / n_used,
        }
        return loss, metrics

    # ------------------------------------------------------------------ A3
    def _step_abnormal(
        self, batch: Dict[str, torch.Tensor], train: bool
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Dataset packs both normal and shuffled feats per item; drop samples where
        # the shuffled-context feature was missing on disk to avoid leaking a trivial
        # "all-zeros == abnormal" shortcut.
        valid = batch.get("abnormal_feat_valid")
        if valid is None:
            valid = torch.ones(batch["vfm_feat"].shape[0], dtype=torch.bool,
                               device=batch["vfm_feat"].device)
        n_valid = int(valid.sum().item())
        device = batch["vfm_feat"].device
        if n_valid == 0:
            zero = self._zero_loss()
            return zero, {"loss_abnormal": zero.detach(),
                          "abnormal_n_valid": torch.tensor(0.0, device=device)}

        feat_normal = batch["vfm_feat"][valid]                 # (B', S, H_f, W_f, C)
        feat_shuf = batch["vfm_feat_shuffled"][valid]
        feat = torch.cat([feat_normal, feat_shuf], dim=0)      # (2B', S, ...)
        B = feat_normal.shape[0]
        labels = torch.cat(
            [torch.zeros(B, device=feat.device), torch.ones(B, device=feat.device)],
            dim=0,
        )

        logits = self.probe(feat)                              # (2B',)
        loss = F.binary_cross_entropy_with_logits(logits, labels)

        with torch.no_grad():
            probs = torch.sigmoid(logits)
            pred = (probs > 0.5).float()
            acc = (pred == labels).float().mean()

        metrics = {
            "loss_abnormal": loss.detach(),
            "abnormal_acc": acc,
            "abnormal_n_valid": torch.tensor(float(n_valid), device=device),
        }
        return loss, metrics

    # ------------------------------------------------------------------ B1
    def _step_ego_belief(
        self, batch: Dict[str, torch.Tensor], train: bool
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        valid = batch["hidden_obj_valid"]              # (B,) bool
        device = batch["vfm_feat"].device
        if valid.sum() == 0:
            zero = self._zero_loss()
            return zero, {"loss_ego": zero.detach(), "ego_n_valid": torch.tensor(0.0, device=device)}

        vfm_feat = batch["vfm_feat"][valid]            # (B', S, H_f, W_f, C)
        per_pix = batch["hidden_obj_mask"][valid]      # (B', S, H, W)
        polar_gt = batch["hidden_obj_polar"][valid]    # (B', 3)

        H_f, W_f = vfm_feat.shape[2], vfm_feat.shape[3]
        obj_mask_feat = _resize_mask_to_feat(per_pix, H_f, W_f)

        polar_pred = self.probe(vfm_feat, obj_mask_feat)  # (B', 3)

        w = torch.tensor(
            self.hparams.polar_loss_weights, device=polar_pred.device, dtype=polar_pred.dtype
        )
        delta = polar_pred - polar_gt
        # Azimuth lives on S1; wrap it before both optimization and reporting.
        # Use out-of-place cat to avoid inplace version-counter mismatch in autograd.
        delta_az = torch.atan2(torch.sin(delta[:, 0]), torch.cos(delta[:, 0]))
        delta = torch.cat([delta_az.unsqueeze(1), delta[:, 1:]], dim=1)
        per_dim_loss = F.smooth_l1_loss(delta, torch.zeros_like(delta), reduction="none") * w
        loss = per_dim_loss.mean()

        with torch.no_grad():
            ang_err = torch.abs(delta[:, :2]) * (180.0 / 3.14159265)
            log_dist_err = torch.abs(polar_pred[:, 2] - polar_gt[:, 2])

        metrics = {
            "loss_ego": loss.detach(),
            "ego_az_err_deg": ang_err[:, 0].mean(),
            "ego_el_err_deg": ang_err[:, 1].mean(),
            "ego_logd_err": log_dist_err.mean(),
            "ego_n_valid": torch.tensor(float(valid.sum().item()), device=device),
        }
        return loss, metrics

    # ------------------------------------------------------------------ B2
    def _step_ego_belief_v2(
        self, batch: Dict[str, torch.Tensor], train: bool
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        valid = batch["hidden_obj_valid"]
        device = batch["vfm_feat"].device
        if valid.sum() == 0:
            zero = self._zero_loss()
            return zero, {"loss_belief": zero.detach(), "belief_n_valid": torch.tensor(0.0, device=device)}

        vfm_feat = batch["vfm_feat"][valid]                 # (B', S, H_f, W_f, C)
        query_feat = batch["belief_query_feat"][valid]      # (B', C)
        polar_gt = batch["hidden_obj_polar"][valid]         # (B', 3)  (az, el, log_d)

        n_az = self.probe.n_az_bins
        n_el = self.probe.n_el_bins
        # Discretize ground truth azimuth in [-pi, pi] and elevation in [-pi/2, pi/2]
        import math
        az_bin = ((polar_gt[:, 0] + math.pi) / (2 * math.pi) * n_az).long().clamp(0, n_az - 1)
        el_bin = ((polar_gt[:, 1] + math.pi / 2) / math.pi * n_el).long().clamp(0, n_el - 1)
        joint_bin = az_bin * n_el + el_bin                   # (B',)

        out = self.probe(vfm_feat, query_feat)
        logits = out["logits"]                               # (B', n_az, n_el)
        log_dist_pred = out["log_dist"]                      # (B',)

        ce = F.cross_entropy(logits.reshape(-1, n_az * n_el), joint_bin)
        l1 = F.smooth_l1_loss(log_dist_pred, polar_gt[:, 2])
        loss = ce + 0.3 * l1

        with torch.no_grad():
            flat_logits = logits.reshape(-1, n_az * n_el)
            pred_bin = flat_logits.argmax(dim=-1)
            top1 = (pred_bin == joint_bin).float().mean()
            top3 = (flat_logits.topk(3, dim=-1).indices == joint_bin.unsqueeze(-1)).any(-1).float().mean()

            # Angular error from predicted bin centers
            pred_az = pred_bin // n_el
            pred_el = pred_bin % n_el
            az_center = (pred_az.float() + 0.5) * (2 * math.pi / n_az) - math.pi
            el_center = (pred_el.float() + 0.5) * (math.pi / n_el) - math.pi / 2
            # Spherical angular distance
            cos_ang = (
                torch.sin(el_center) * torch.sin(polar_gt[:, 1])
                + torch.cos(el_center) * torch.cos(polar_gt[:, 1])
                * torch.cos(az_center - polar_gt[:, 0])
            ).clamp(-1.0, 1.0)
            ang_err_deg = torch.acos(cos_ang) * (180.0 / math.pi)
            logd_err = (log_dist_pred - polar_gt[:, 2]).abs()

        metrics = {
            "loss_belief": loss.detach(),
            "belief_ce": ce.detach(),
            "belief_top1": top1,
            "belief_top3": top3,
            "belief_ang_err_deg": ang_err_deg.mean(),
            "belief_logd_err": logd_err.mean(),
            "belief_n_valid": torch.tensor(float(valid.sum().item()), device=device),
        }
        return loss, metrics

    # ------------------------------------------------------------------ C1
    def _step_action_dynamics(
        self, batch: Dict[str, torch.Tensor], train: bool
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        valid = batch.get("dyn_valid")
        if valid is None:
            valid = torch.ones(batch["input_feat"].shape[0], dtype=torch.bool,
                               device=batch["input_feat"].device)
        n_valid = int(valid.sum().item())
        device = batch["input_feat"].device
        if n_valid == 0:
            zero = self._zero_loss()
            return zero, {"loss_dyn": zero.detach(),
                          "dyn_n_valid": torch.tensor(0.0, device=device)}

        input_feat = batch["input_feat"][valid]        # (B', T, H_f, W_f, C)
        action = batch["action"][valid]                # (B', 9)
        target_feat = batch["target_feat"][valid]      # (B', H_f, W_f, C)

        target_pooled = target_feat.mean(dim=(1, 2))   # (B', C)
        pred = self.probe(input_feat, action)          # (B', C)

        mse = F.mse_loss(pred, target_pooled)
        cos = 1.0 - F.cosine_similarity(pred, target_pooled, dim=-1).mean()
        loss = (
            self.hparams.mse_loss_weight * mse
            + self.hparams.cosine_loss_weight * cos
        )

        with torch.no_grad():
            B = pred.shape[0]
            cos_sim_corr = F.cosine_similarity(pred, target_pooled, dim=-1).mean()
            if B >= 4:
                # in-batch retrieval rank of the correct target
                sim = F.normalize(pred, dim=-1) @ F.normalize(target_pooled, dim=-1).T
                rank = (sim.argsort(dim=-1, descending=True) ==
                        torch.arange(B, device=sim.device).unsqueeze(-1)
                        ).float().argmax(dim=-1)
                r1 = (rank == 0).float().mean()
                mean_rank = rank.float().mean()
            else:
                r1 = torch.tensor(float("nan"), device=pred.device)
                mean_rank = torch.tensor(float("nan"), device=pred.device)

        metrics = {
            "loss_dyn": loss.detach(),
            "dyn_mse": mse.detach(),
            "dyn_cos": cos_sim_corr,
            "dyn_inbatch_R@1": r1,
            "dyn_inbatch_mean_rank": mean_rank,
            "dyn_n_valid": torch.tensor(float(n_valid), device=device),
        }
        return loss, metrics

    def _masked_feature_prediction_loss(
        self,
        pred: torch.Tensor,          # (B, K, C)
        target_feat: torch.Tensor,   # (B, K, H_f, W_f, C)
        valid: torch.Tensor,         # (B, K) bool
        prefix: str,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        target = target_feat.mean(dim=(2, 3))
        valid = valid.bool()
        n_valid = int(valid.sum().item())
        device = pred.device
        if n_valid == 0:
            zero = self._zero_loss()
            return zero, {
                f"loss_{prefix}": zero.detach(),
                f"{prefix}_n_valid": torch.tensor(0.0, device=device),
            }

        pred_v = pred[valid]
        target_v = target[valid]
        mse = F.mse_loss(pred_v, target_v)
        cos_loss = 1.0 - F.cosine_similarity(pred_v, target_v, dim=-1).mean()
        loss = (
            self.hparams.mse_loss_weight * mse
            + self.hparams.cosine_loss_weight * cos_loss
        )

        with torch.no_grad():
            cos = F.cosine_similarity(pred_v, target_v, dim=-1).mean()
            if pred_v.shape[0] >= 4:
                sim = F.normalize(pred_v, dim=-1) @ F.normalize(target_v, dim=-1).T
                rank = (sim.argsort(dim=-1, descending=True) ==
                        torch.arange(pred_v.shape[0], device=sim.device).unsqueeze(-1)
                        ).float().argmax(dim=-1)
                r1 = (rank == 0).float().mean()
                mean_rank = rank.float().mean()
            else:
                r1 = torch.tensor(float("nan"), device=device)
                mean_rank = torch.tensor(float("nan"), device=device)

        return loss, {
            f"loss_{prefix}": loss.detach(),
            f"{prefix}_mse": mse.detach(),
            f"{prefix}_cos": cos,
            f"{prefix}_R@1": r1,
            f"{prefix}_mean_rank": mean_rank,
            f"{prefix}_n_valid": torch.tensor(float(n_valid), device=device),
        }

    # ------------------------------------------------------------------ C2
    def _step_path_integration(
        self, batch: Dict[str, torch.Tensor], train: bool
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        input_feat = batch["input_feat_seq"]
        actions = batch["path_actions"]
        target_feat = batch["target_feat_seq"]
        valid = batch.get("path_horizon_valid")
        if valid is None:
            valid = torch.ones(actions.shape[:2], dtype=torch.bool, device=actions.device)

        pred = self.probe(input_feat, actions)
        return self._masked_feature_prediction_loss(pred, target_feat, valid, "path")

    # ------------------------------------------------------------------ C3
    def _step_counterfactual(
        self, batch: Dict[str, torch.Tensor], train: bool
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        input_feat = batch["input_feat_seq"]
        actions = batch["counterfactual_actions"]
        target_feat = batch["target_feat_seq"]
        valid = batch.get("counterfactual_valid")
        if valid is None:
            valid = torch.ones(actions.shape[:2], dtype=torch.bool, device=actions.device)

        pred = self.probe(input_feat, actions)
        return self._masked_feature_prediction_loss(pred, target_feat, valid, "cf")

    # ------------------------------------------------------------------ SAE
    def _step_sae_spatial(
        self, batch: Dict[str, torch.Tensor], train: bool
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        vfm_feat = batch["vfm_feat"]
        out = self.probe(vfm_feat)
        tokens = out["sampled_tokens"]
        recon = out["recon"]
        sparse = out["sparse"]

        recon_mse = F.mse_loss(recon, tokens)
        rel_mse = recon_mse / tokens.pow(2).mean().clamp(min=1e-6)
        l1 = sparse.abs().mean()
        loss = recon_mse + self.hparams.sae_l1_weight * l1

        with torch.no_grad():
            active_frac = (sparse > 0).float().mean()
            l0 = (sparse > 0).float().sum(dim=-1).mean()

        metrics = {
            "loss_sae": loss.detach(),
            "sae_recon_mse": recon_mse.detach(),
            "sae_rel_mse": rel_mse.detach(),
            "sae_l1": l1.detach(),
            "sae_active_frac": active_frac,
            "sae_l0": l0,
        }

        frame_sparse = out["frame_sparse"]
        if getattr(self.probe, "use_overlap_readout", True) and "overlap_gt" in batch:
            overlap_gt = batch["overlap_gt"]
            B, S = overlap_gt.shape[:2]
            logits = self.probe.predict_overlap(frame_sparse)
            eye = torch.eye(S, dtype=torch.bool, device=logits.device).unsqueeze(0).expand(B, -1, -1)
            bce = F.binary_cross_entropy_with_logits(
                logits[~eye], overlap_gt[~eye].float()
            )
            loss = loss + self.hparams.sae_overlap_weight * bce
            with torch.no_grad():
                prob = torch.sigmoid(logits)
                mae = (prob[~eye] - overlap_gt[~eye]).abs().mean()
            metrics.update({
                "sae_overlap_bce": bce.detach(),
                "sae_overlap_mae": mae,
            })

        if getattr(self.probe, "use_ego_readout", True) and "hidden_obj_valid" in batch:
            valid = batch["hidden_obj_valid"].bool()
            if valid.any():
                query = batch.get("belief_query_feat")
                query = query[valid] if query is not None else None
                pred = self.probe.predict_ego(frame_sparse[valid], query)
                target = batch["hidden_obj_polar"][valid]
                ego_loss = F.smooth_l1_loss(pred, target)
                loss = loss + self.hparams.sae_ego_weight * ego_loss
                with torch.no_grad():
                    ang_err = torch.abs(pred[:, :2] - target[:, :2]) * (180.0 / 3.14159265)
                    logd_err = torch.abs(pred[:, 2] - target[:, 2]).mean()
                metrics.update({
                    "sae_ego_loss": ego_loss.detach(),
                    "sae_ego_az_err_deg": ang_err[:, 0].mean(),
                    "sae_ego_el_err_deg": ang_err[:, 1].mean(),
                    "sae_ego_logd_err": logd_err,
                    "sae_ego_n_valid": torch.tensor(float(valid.sum().item()), device=vfm_feat.device),
                })

        return loss, metrics

    # ------------------------------------------------------------------ dispatch
    def model_step(
        self, batch: Dict[str, torch.Tensor], train: bool
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Move tensors to device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=True)

        if self.probe_type == "view_consistency":
            return self._step_view_consistency(batch, train)
        if self.probe_type == "abnormal":
            return self._step_abnormal(batch, train)
        if self.probe_type == "ego_belief":
            return self._step_ego_belief(batch, train)
        if self.probe_type == "ego_belief_v2":
            return self._step_ego_belief_v2(batch, train)
        if self.probe_type == "action_dynamics":
            return self._step_action_dynamics(batch, train)
        if self.probe_type == "path_integration":
            return self._step_path_integration(batch, train)
        if self.probe_type == "counterfactual":
            return self._step_counterfactual(batch, train)
        if self.probe_type == "sae_spatial":
            return self._step_sae_spatial(batch, train)
        raise RuntimeError(f"Unhandled probe_type {self.probe_type}")

    # ------------------------------------------------------------------ Lightning hooks
    def training_step(self, batch, batch_idx):
        loss, metrics = self.model_step(batch, train=True)
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        for k, v in metrics.items():
            self.log(f"train/{k}", v, on_step=True, on_epoch=False, prog_bar=False)
        self.log(
            "trainer/lr",
            self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0],
            on_step=True, on_epoch=False, prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, metrics = self.model_step(batch, train=False)
        self.val_loss(loss)
        count_keys = {
            "view_consistency": "overlap_n_used",
            "abnormal": "abnormal_n_valid",
            "ego_belief": "ego_n_valid",
            "ego_belief_v2": "belief_n_valid",
            "action_dynamics": "dyn_n_valid",
            "path_integration": "path_n_valid",
            "counterfactual": "cf_n_valid",
        }
        count = metrics.get(count_keys.get(self.probe_type, ""), None)
        fallback_weight = next(
            (int(value.shape[0]) for value in batch.values()
             if isinstance(value, torch.Tensor) and value.ndim > 0),
            1,
        )
        batch_weight = max(int(count.item()), 1) if count is not None else fallback_weight
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True,
                 sync_dist=True, batch_size=batch_weight)
        for k, v in metrics.items():
            self.log(f"val/{k}", v, on_step=False, on_epoch=True, prog_bar=False,
                     sync_dist=True, batch_size=batch_weight)
        return loss

    def test_step(self, batch, batch_idx):
        loss, metrics = self.model_step(batch, train=False)
        fallback_weight = next(
            (int(value.shape[0]) for value in batch.values()
             if isinstance(value, torch.Tensor) and value.ndim > 0),
            1,
        )
        batch_weight = max(int(next((v.item() for k, v in metrics.items()
                                    if k.endswith("_n_valid")), fallback_weight)), 1)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True,
                 sync_dist=True, batch_size=batch_weight)
        for k, v in metrics.items():
            self.log(f"test/{k}", v, on_step=False, on_epoch=True, prog_bar=False,
                     sync_dist=True, batch_size=batch_weight)
        return loss

    def on_train_epoch_start(self) -> None:
        if hasattr(self.trainer.train_dataloader, "dataset") and hasattr(
            self.trainer.train_dataloader.dataset, "set_epoch"
        ):
            self.trainer.train_dataloader.dataset.set_epoch(self.current_epoch)
        if hasattr(self.trainer.train_dataloader, "sampler") and hasattr(
            self.trainer.train_dataloader.sampler, "set_epoch"
        ):
            self.trainer.train_dataloader.sampler.set_epoch(self.current_epoch)

    def on_validation_epoch_start(self) -> None:
        self.val_loss.reset()
        for loader in self.trainer.val_dataloaders:
            if hasattr(loader, "dataset") and hasattr(loader.dataset, "set_epoch"):
                loader.dataset.set_epoch(0)

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
