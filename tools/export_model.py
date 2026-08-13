#!/usr/bin/env python3
"""训练档 → 发布包，以及整个跑的收割。

    # 单份导出
    python3 tools/export_model.py ../runs/v4-bf16/ckpt/step00090189.pt --out /tmp/a.pt

    # 收割一个跑：全部导出到 model/，只留为继续训练而存在的训练档
    python3 tools/export_model.py harvest ../runs/v4-bf16          # 默认只看不做
    python3 tools/export_model.py harvest ../runs/v4-bf16 --yes    # 真干

收割做三件事，**顺序不能反**：

1. `ckpt/` 下每一份（含 `stable.pt`）导出到 `model/` 下同名文件
2. 逐份验证：训练档与发布包各建一个模型，同一批输入前向，**断言逐位相同**
3. **全部验证通过之后**，才删 `ckpt/` 里不在保留集合内的档

先验证再删，是因为这是不可逆操作而 `runs/` 不在 git 里。反过来写的话，
一旦导出有问题，等发现时原档已经没了。

保留集合**由用途机械推导**，不接受手工指定：

* `ckpt/stable.pt` —— WSD 的分叉点，它存在的唯一理由就是以后要从这里续训
* `ckpt/latest` 指向的那份 —— 续训入口

刻意不给 `--keep` 之类的口子：有了它每轮都会多留几份「以防万一」，
而一份 162.5 MiB，攒起来就是几十 GB。真要从某个中间档接着训，
就从它的发布包重起一条跑（权重完整，只是没有动量，等于换个 warmup）。
"""

import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import torch                                                    # noqa: E402

from cornerstone import _engine as E                            # noqa: E402
from cornerstone.export import build_release, write_release     # noqa: E402
from cornerstone.model import CornerNet, ModelConfig, load_weights  # noqa: E402


def _mb(n: int) -> str:
    return f"{n / 1e6:.1f} MB"


def _model_from_training_ckpt(path: str, device: str):
    """照训练/评测的老路径建模型，用来和发布包对拍。"""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**{k: v for k, v in (blob.get("model_config") or {}).items()
                         if k in ModelConfig.__dataclass_fields__})
    dev = torch.device(device)
    import contextlib
    ctx = torch.cuda.device(dev) if dev.type == "cuda" else contextlib.nullcontext()
    with ctx:
        m = CornerNet(cfg).to(dev)
        load_weights(m, blob["model"])
        m.to_param_dtype()
    return m.eval()


def _forward(model, device: str, seed: int = 7):
    import contextlib
    dev = torch.device(device)
    torch.manual_seed(seed)
    # 批大小取 8 的倍数：量化 GEMM 要求两个维度都被 32 整除，token 维是 B*196
    p = torch.randn(8, E.NUM_PLANES, E.BOARD_N, E.BOARD_N, device=dev)
    s = torch.randn(8, E.NUM_SCALARS, device=dev)
    ctx = torch.cuda.device(dev) if dev.type == "cuda" else contextlib.nullcontext()
    with ctx, torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16,
                                              enabled=dev.type == "cuda"):
        return [t.float().cpu() for t in model(p, s)]


def verify(ckpt: str, release: str, device: str) -> None:
    """训练档与发布包的前向必须逐位相同，否则抛错（收割就此中止，不删任何东西）。"""
    from cornerstone.export import load_release
    ref = _forward(_model_from_training_ckpt(ckpt, device), device)
    got = _forward(load_release(release, device)[0], device)
    for name, a, b in zip(("policy", "wdl", "score"), ref, got):
        if not torch.equal(a, b):
            raise SystemExit(
                f"验证失败：{os.path.basename(ckpt)} 的发布包 {name} 与训练档不逐位相同"
                f"（最大绝对差 {(a - b).abs().max():.3e}）。**没有删除任何文件。**")


def keep_set(ckpt_dir: str) -> set[str]:
    """为继续训练而存在的那些档。"""
    keep = set()
    if os.path.exists(os.path.join(ckpt_dir, "stable.pt")):
        keep.add("stable.pt")                       # WSD 的分叉点
    latest = os.path.join(ckpt_dir, "latest")
    if os.path.exists(latest):
        with open(latest) as f:
            keep.add(f.read().strip())              # 续训入口
    return keep


def harvest(run_dir: str, device: str, do_it: bool) -> int:
    ckpt_dir = os.path.join(run_dir, "ckpt")
    model_dir = os.path.join(run_dir, "model")
    if not os.path.isdir(ckpt_dir):
        raise SystemExit(f"{ckpt_dir} 不存在")

    names = sorted(f for f in os.listdir(ckpt_dir) if f.endswith(".pt"))
    if not names:
        raise SystemExit(f"{ckpt_dir} 里没有 .pt")
    keep = keep_set(ckpt_dir)
    drop = [n for n in names if n not in keep]

    print(f"跑目录 {run_dir}")
    print(f"  训练档 {len(names)} 份，共 {_mb(sum(os.path.getsize(os.path.join(ckpt_dir, n)) for n in names))}")
    print(f"  保留（为继续训练而存在）：{sorted(keep) or '（无）'}")
    print(f"  导出后删除：{len(drop)} 份")
    if not do_it:
        print("\n这是**预演**，什么都没做。确认无误后加 --yes 真跑。")
        return 0

    os.makedirs(model_dir, exist_ok=True)
    total = 0
    for i, n in enumerate(names, 1):
        src = os.path.join(ckpt_dir, n)
        dst = os.path.join(model_dir, n)
        size = write_release(src, dst, device)
        verify(src, dst, device)                    # 不过就 SystemExit，一个档都不删
        total += size
        print(f"  [{i:>2}/{len(names)}] {n}: {_mb(os.path.getsize(src))} -> "
              f"{_mb(size)}  验证通过", flush=True)

    # **全部验证通过之后**才动手删
    freed = 0
    for n in drop:
        p = os.path.join(ckpt_dir, n)
        freed += os.path.getsize(p)
        os.remove(p)
    print(f"\n发布包 {len(names)} 份共 {_mb(total)} -> {model_dir}")
    print(f"删除训练档 {len(drop)} 份，回收 {_mb(freed)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="单份 checkpoint 路径，或 harvest 时给跑目录")
    ap.add_argument("--out", help="单份导出的目标路径")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--yes", action="store_true", help="harvest 真执行（默认只预演）")
    args, rest = ap.parse_known_args()

    if args.target == "harvest":
        if not rest:
            raise SystemExit("用法: export_model.py harvest <跑目录> [--yes]")
        return harvest(rest[0], args.device, args.yes)

    out = args.out or os.path.join(os.path.dirname(args.target).replace(
        os.sep + "ckpt", os.sep + "model"), os.path.basename(args.target))
    size = write_release(args.target, out, args.device)
    verify(args.target, out, args.device)
    print(f"{args.target}  {_mb(os.path.getsize(args.target))} -> {out}  {_mb(size)}  验证通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
