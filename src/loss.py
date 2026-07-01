import torch

from persistence import pd_from_graph, wasserstein_distance


def wasserstein_loss(
    pred_diags: list[torch.Tensor],
    target_diags: list[torch.Tensor],
    *,
    hom_dim: int = 2,
) -> dict[str, torch.Tensor]:
    """Wasserstein distances between predicted and target persistence diagrams."""
    terms = wasserstein_distance(pred_diags, target_diags, hom_dim)
    ref = pred_diags[0] if pred_diags else target_diags[0]
    zero = torch.zeros((), device=ref.device, dtype=ref.dtype)
    return {f"h{i}": terms[i] if i < len(terms) else zero for i in range(hom_dim)}


class MiniFoldTDALoss:
    """Wasserstein and VPD losses on persistence diagrams from distance matrices."""

    def __init__(self, config, h0rff=None, h1rff=None):
        self.config = config
        self.h0rff = h0rff
        self.h1rff = h1rff

    def _build_loss_fns(self, out, batch):
        loss_fns = {}
        cfg = self.config

        def add(name, fn):
            if cfg[name].enabled:
                loss_fns[name] = fn

        needs_diags = (
            cfg.wasserstein_h0.enabled
            or cfg.wasserstein_h1.enabled
            or cfg.vpd_h0.enabled
            or cfg.vpd_h1.enabled
        )
        if needs_diags:
            target_diags = pd_from_graph(batch["adj"], **cfg.pd)
            pred_diags = pd_from_graph(out["adj"], **cfg.pd)

        if cfg.wasserstein_h0.enabled or cfg.wasserstein_h1.enabled:
            wasserstein_terms = wasserstein_loss(
                pred_diags=pred_diags,
                target_diags=target_diags,
                hom_dim=cfg.pd.hom_dim,
            )
            if cfg.wasserstein_h0.enabled:
                add("wasserstein_h0", lambda: wasserstein_terms["h0"])
            if cfg.wasserstein_h1.enabled:
                add("wasserstein_h1", lambda: wasserstein_terms["h1"])

        if cfg.vpd_h0.enabled:
            if self.h0rff is None:
                raise ValueError("vpd_h0 loss is enabled but h0rff was not provided")
            add("vpd_h0", lambda: self.h0rff.vpd_loss(pred_diags[0], target_diags[0]))
        if cfg.vpd_h1.enabled:
            if self.h1rff is None:
                raise ValueError("vpd_h1 loss is enabled but h1rff was not provided")
            add("vpd_h1", lambda: self.h1rff.vpd_loss(pred_diags[1], target_diags[1]))

        return loss_fns

    def __call__(self, out, batch, _return_breakdown=False):
        loss_fns = self._build_loss_fns(out, batch)
        if not loss_fns:
            zero = torch.zeros((), device=out["adj"].device, dtype=out["adj"].dtype, requires_grad=True)
            if _return_breakdown:
                return zero, {}
            return zero

        cum_loss = 0.0
        losses = {}
        for loss_name, loss_fn in loss_fns.items():
            weight = self.config[loss_name].weight
            loss = loss_fn()
            if not torch.isfinite(loss).all():
                print(f"{loss_name} loss is NaN or Inf. Skipping...")
                loss = loss.new_zeros((), requires_grad=True)
            cum_loss = cum_loss + weight * loss
            losses[loss_name] = loss.detach().clone()

        losses["loss"] = cum_loss.detach().clone()

        if not _return_breakdown:
            return cum_loss
            
        return cum_loss, losses
