"""Qwen3-0.6B **形状**的骨干层：28 层双向 Transformer。

**只借形状，不载任何预训练权重 —— 全部随机初始化。** 实验名前缀是 `qwen-`，
很容易被读成「从 Qwen3-0.6B 微调来的」，那是本项目最防的那类误读：
不报错、跑得通、结论全错。这里没有任何一处会去 HuggingFace 拉权重。

与 `model.py` 里的 `PolyBlock` 是两条并行的骨干，靠 `ModelConfig.arch` 选。
**这里只放层的实现**，输入侧的嵌入与三个头仍在 `CornerNet` 里，
所以量化、导出、权重分发、编译那一整套都原样复用，不必为新骨干改一遍。

照搬 Qwen3-0.6B 的三个形状特征：

* **GQA**：16 个 Q 头、8 个 KV 头，`head_dim=128`。
  注意 `16*128 = 2048 != hidden(1024)` —— 这是 Qwen3 的设计，不是笔误。
* **QK-Norm**：q 和 k 各自过一个 `head_dim` 上的 RMSNorm 再算注意力。
  这是 Qwen3 相对 Qwen2 的标志改动，随机初始化下它对训练稳定性尤其重要。
* **SwiGLU**，`intermediate=3072`，pre-norm 残差。

两处**有意的偏离**，都是因为输入不是文本：

1. **双向注意力**。棋盘上 240 个 token 同时可见，没有「未来」可言；
   因果 mask 会让第 k 个格子看不到它右边和下边的棋盘，那是纯粹的信息损失。
2. **2D RoPE**。Qwen3 的 1D RoPE 编码的是 token 在序列里的次序，而 14x14 棋盘
   没有次序 —— 按行优先展开的话，(0,13) 与 (1,0) 位置只差 1，棋盘上却隔着
   整整一行。这里把 `head_dim` 劈成两半：前一半按行旋转、后一半按列旋转，
   于是「相对位置」恢复成棋盘上的二维位移。棋子/标量 token 不在棋盘上，
   给它们独立的可学习位置向量，RoPE 对它们取恒等（角度为 0）。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def rope2d_tables(n_cells: int, board: int, head_dim: int, n_extra: int,
                  theta: float = 1e6) -> tuple[torch.Tensor, torch.Tensor]:
    """2D RoPE 的 cos/sin 表，形状都是 ``(N, head_dim)``，N = n_cells + n_extra。

    `head_dim` 劈成两半：前一半按 row 旋转、后一半按 col 旋转。每一半内部
    再按 RoPE 的老规矩两两配对，所以每半需要 `head_dim//4` 个频率。

    **棋盘之外的 token（棋子、标量）角度取 0**，即 cos=1、sin=0，RoPE 退化成
    恒等 —— 它们的位置信息由各自的可学习嵌入提供，硬塞一个棋盘坐标反而是假的。

    返回的是**确定性**的表：只依赖入参，所以每个工作进程各自建一份必然相同。
    调用方要用 `register_buffer(..., persistent=False)` 挂上去 ——
    `pool.py` 的 `_layout()` 只遍历 `named_parameters()`，而 `load_weights()`
    对 state_dict 里的未知键直接抛错，持久 buffer 两头都会出事。
    """
    assert head_dim % 4 == 0, f"head_dim 要能被 4 整除（劈两半、每半再两两配对），得到 {head_dim}"
    half = head_dim // 2
    n_freq = half // 2
    # 每半各自的频率，与 1D RoPE 同式
    inv = 1.0 / (theta ** (torch.arange(0, n_freq, dtype=torch.float32) / n_freq))

    rows = torch.arange(n_cells, dtype=torch.float32) // board
    cols = torch.arange(n_cells, dtype=torch.float32) % board
    if n_extra:
        zeros = torch.zeros(n_extra, dtype=torch.float32)
        rows = torch.cat([rows, zeros])
        cols = torch.cat([cols, zeros])

    ar = torch.outer(rows, inv)          # (N, n_freq)
    ac = torch.outer(cols, inv)
    # 每个角度重复两次，配合 rotate_half 的 (x1,x2) 配对方式
    ang = torch.cat([ar, ar, ac, ac], dim=-1)   # (N, head_dim)
    return ang.cos(), ang.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """把 head_dim 的两半各自做 (x1, x2) -> (-x2, x1)。

    **必须分半做**：整条 head_dim 一起转的话，行的后半会和列的前半配成一对，
    两个坐标轴就串了 —— 而且串了之后网络照样能训，只是位置编码是乱的，
    从 loss 上完全看不出来。
    """
    d = x.shape[-1] // 2
    a, b = x[..., :d], x[..., d:]

    def half(t: torch.Tensor) -> torch.Tensor:
        t1, t2 = t.chunk(2, dim=-1)
        return torch.cat((-t2, t1), dim=-1)

    return torch.cat((half(a), half(b)), dim=-1)


def apply_rope2d(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, H, N, head_dim)；cos/sin: (N, head_dim)。"""
    c = cos.to(x.dtype).unsqueeze(0).unsqueeze(0)
    s = sin.to(x.dtype).unsqueeze(0).unsqueeze(0)
    return x * c + _rotate_half(x) * s


class QwenAttention(nn.Module):
    """GQA + QK-Norm + 2D RoPE 的**双向**注意力。"""

    def __init__(self, cfg, make_linear, match_dtype, force_bf16: bool = False):
        super().__init__()
        self._match = match_dtype
        d = cfg.dim
        self.heads = cfg.heads
        self.kv_heads = cfg.kv_heads
        self.head_dim = cfg.attn_head_dim
        assert self.heads % self.kv_heads == 0, "Q 头数要能被 KV 头数整除"

        q_out = self.heads * self.head_dim
        kv_out = self.kv_heads * self.head_dim
        self.q_proj = make_linear(cfg, d, q_out, force_bf16=force_bf16)
        self.k_proj = make_linear(cfg, d, kv_out, force_bf16=force_bf16)
        self.v_proj = make_linear(cfg, d, kv_out, force_bf16=force_bf16)
        self.o_proj = make_linear(cfg, q_out, d, force_bf16=force_bf16)
        # QK-Norm：作用在 head_dim 上，不是 hidden 上
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        xm = self._match(x, self.q_proj)
        q = self.q_proj(xm).view(b, n, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(xm).view(b, n, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(xm).view(b, n, self.kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rope2d(self.q_norm(q), cos, sin)
        k = apply_rope2d(self.k_norm(k), cos, sin)

        # is_causal=False：棋盘上没有「未来」，240 个 token 互相可见
        o = F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=True)
        o = o.transpose(1, 2).reshape(b, n, self.heads * self.head_dim)
        return self.o_proj(self._match(o, self.o_proj))


class QwenLayer(nn.Module):
    """一层：pre-norm 注意力 + pre-norm SwiGLU，都带残差。

    **MLP 必须叫 `mlp`** —— `train.py` 的 `_compile_hot_modules` 写死了
    `for blk in self.model.blocks: blk.mlp.compile(...)`，改名它会静默不编译，
    表现只是「训练慢了一截」，没有任何报错。
    """

    def __init__(self, cfg, make_linear, match_dtype, swiglu_cls, force_bf16: bool = False):
        super().__init__()
        self.norm_attn = nn.RMSNorm(cfg.dim)
        self.attn = QwenAttention(cfg, make_linear, match_dtype, force_bf16=force_bf16)
        self.norm_mlp = nn.RMSNorm(cfg.dim)
        self.mlp = swiglu_cls(cfg, force_bf16=force_bf16)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm_attn(x), cos, sin)
        return x + self.mlp(self.norm_mlp(x))
