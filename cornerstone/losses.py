"""训练损失。

策略目标是稀疏的（top-K + 尾部质量），所以交叉熵要拆成两部分算：
top-K 逐项加权，尾部质量均摊到其余合法着法上。

之所以不能把 top-K 归一化后当成完整目标：训练早期先验接近均匀、
单步合法着法常有几百个，实测 top-32 只装了约 10% 的概率质量。
归一化等于在教网络集中到一组本质上任意的着法上。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_log_softmax(logits: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
    """只在合法着法上做 log_softmax；非法位置为 -inf。"""
    return torch.log_softmax(logits.masked_fill(~legal, float("-inf")), dim=-1)


def policy_loss(
    logits: torch.Tensor,      # [B, A]
    legal: torch.Tensor,       # [B, A] bool
    top_actions: torch.Tensor, # [B, K] int64，补位槽填的是合法动作
    top_probs: torch.Tensor,   # [B, K]，补位槽为 0
    rest_prob: torch.Tensor,   # [B]
    n_top: torch.Tensor,       # [B]
    n_legal: torch.Tensor,     # [B]
) -> torch.Tensor:
    logp = masked_log_softmax(logits, legal)

    lp_top = logp.gather(1, top_actions)                       # [B, K]
    head = (top_probs * lp_top).sum(dim=1)

    k = torch.arange(top_actions.shape[1], device=logits.device)
    valid = k[None, :] < n_top[:, None]

    # 尾部：把 rest_prob 均摊到「合法但不在 top-K 里」的动作上
    sum_legal = logp.masked_fill(~legal, 0.0).sum(dim=1)
    sum_top = torch.where(valid, lp_top, torch.zeros_like(lp_top)).sum(dim=1)
    tail_count = (n_legal - n_top).clamp(min=1).to(logits.dtype)
    tail = rest_prob * (sum_legal - sum_top) / tail_count

    return -(head + tail).mean()


def value_loss(wdl_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """胜/和/负三分类。和局是本项目里的真实结果，必须显式建模。"""
    return F.cross_entropy(wdl_logits, target)


def score_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """终局占格数差（已按 89 归一化）。辅助任务，信号比稀疏的三分类密集得多。"""
    return F.smooth_l1_loss(pred, target, beta=0.1)


def policy_entropy(logits: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
    logp = masked_log_softmax(logits, legal)
    p = logp.exp()
    return -(p * logp.masked_fill(~legal, 0.0)).sum(dim=1).mean()


def total_loss(out, batch, w_value: float = 1.0, w_score: float = 0.25) -> tuple:
    """out = (policy_logits, wdl_logits, score_pred)。返回 (总损失, 各项明细)。"""
    pol, wdl, sc = out
    legal = batch["legal"].bool()

    lp = policy_loss(pol, legal, batch["top_actions"], batch["top_probs"],
                     batch["rest_prob"], batch["n_top"], batch["n_legal"])
    lv = value_loss(wdl, batch["wdl"])
    ls = score_loss(sc, batch["score_diff"])
    loss = lp + w_value * lv + w_score * ls

    with torch.no_grad():
        acc = (wdl.argmax(dim=-1) == batch["wdl"]).float().mean()
        ent = policy_entropy(pol, legal)
    return loss, {
        "loss": loss.detach(),
        "policy": lp.detach(),
        "value": lv.detach(),
        "score": ls.detach(),
        "wdl_acc": acc,
        "policy_entropy": ent,
    }
