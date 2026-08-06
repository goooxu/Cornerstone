// 256 位棋盘。
//
// 逻辑棋盘是 14x14，但物理上按 16 列一行摆放（STRIDE=16），共 16 行 = 256 位 = 4 个 uint64。
// 取 16 而不是 14 的理由：行方向的邻接就退化成「移 16 位」，列方向就是「移 1 位」，
// 八个方向的邻接全部变成一次移位。而因为第 14/15 列与第 14/15 行永远不属于 VALID，
// 任何移位造成的跨行绕回都会落在 VALID 之外，只要移位后统一 & VALID 就自动清理干净，
// 不需要额外的防绕回列掩码。
//
// 这里刻意只用可移植的位运算（无 AVX/NEON intrinsic）—— 开发机是 aarch64，
// 编译器会把这些映射到合适的标量/向量指令。

#pragma once

#include <cstdint>
#include <cstddef>

namespace cornerstone {

inline constexpr int BOARD_N   = 14;                   // 逻辑边长
inline constexpr int STRIDE    = 16;                   // 物理行宽
inline constexpr int NUM_CELLS = BOARD_N * BOARD_N;    // 196
inline constexpr int NUM_BITS  = STRIDE * STRIDE;      // 256

// 逻辑格 (r, c) 对应的物理位号
constexpr int bit_of(int r, int c) noexcept { return r * STRIDE + c; }
constexpr int row_of(int bit) noexcept { return bit / STRIDE; }
constexpr int col_of(int bit) noexcept { return bit % STRIDE; }

// 逻辑格序号 (0..195) <-> 物理位号 (0..255)
constexpr int cell_to_bit(int cell) noexcept { return bit_of(cell / BOARD_N, cell % BOARD_N); }
constexpr int bit_to_cell(int bit) noexcept { return row_of(bit) * BOARD_N + col_of(bit); }

struct BB {
    uint64_t w[4];

    constexpr BB() noexcept : w{0, 0, 0, 0} {}
    constexpr BB(uint64_t a, uint64_t b, uint64_t c, uint64_t d) noexcept : w{a, b, c, d} {}

    static constexpr BB zero() noexcept { return BB(); }

    static constexpr BB single(int bit) noexcept {
        BB r;
        r.w[bit >> 6] = uint64_t(1) << (bit & 63);
        return r;
    }

    constexpr bool test(int bit) const noexcept {
        return (w[bit >> 6] >> (bit & 63)) & 1u;
    }
    constexpr void set(int bit) noexcept { w[bit >> 6] |= uint64_t(1) << (bit & 63); }
    constexpr void clear(int bit) noexcept { w[bit >> 6] &= ~(uint64_t(1) << (bit & 63)); }

    constexpr bool empty() const noexcept { return (w[0] | w[1] | w[2] | w[3]) == 0; }
    constexpr bool any() const noexcept { return !empty(); }

    int popcount() const noexcept {
        return __builtin_popcountll(w[0]) + __builtin_popcountll(w[1]) +
               __builtin_popcountll(w[2]) + __builtin_popcountll(w[3]);
    }

    constexpr BB operator|(const BB& o) const noexcept {
        return BB(w[0] | o.w[0], w[1] | o.w[1], w[2] | o.w[2], w[3] | o.w[3]);
    }
    constexpr BB operator&(const BB& o) const noexcept {
        return BB(w[0] & o.w[0], w[1] & o.w[1], w[2] & o.w[2], w[3] & o.w[3]);
    }
    constexpr BB operator^(const BB& o) const noexcept {
        return BB(w[0] ^ o.w[0], w[1] ^ o.w[1], w[2] ^ o.w[2], w[3] ^ o.w[3]);
    }
    constexpr BB operator~() const noexcept { return BB(~w[0], ~w[1], ~w[2], ~w[3]); }

    constexpr BB& operator|=(const BB& o) noexcept { *this = *this | o; return *this; }
    constexpr BB& operator&=(const BB& o) noexcept { *this = *this & o; return *this; }

    constexpr bool operator==(const BB& o) const noexcept {
        return w[0] == o.w[0] && w[1] == o.w[1] && w[2] == o.w[2] && w[3] == o.w[3];
    }

    // 是否与 o 有交集
    constexpr bool intersects(const BB& o) const noexcept {
        return ((w[0] & o.w[0]) | (w[1] & o.w[1]) | (w[2] & o.w[2]) | (w[3] & o.w[3])) != 0;
    }
    // 是否是 o 的子集
    constexpr bool subset_of(const BB& o) const noexcept {
        return ((w[0] & ~o.w[0]) | (w[1] & ~o.w[1]) |
                (w[2] & ~o.w[2]) | (w[3] & ~o.w[3])) == 0;
    }
};

// 左移（朝高位方向，即行号/列号增大）。要求 0 < k < 64。
constexpr BB shl(const BB& b, int k) noexcept {
    return BB(b.w[0] << k,
              (b.w[1] << k) | (b.w[0] >> (64 - k)),
              (b.w[2] << k) | (b.w[1] >> (64 - k)),
              (b.w[3] << k) | (b.w[2] >> (64 - k)));
}

// 右移（朝低位方向）。要求 0 < k < 64。
constexpr BB shr(const BB& b, int k) noexcept {
    return BB((b.w[0] >> k) | (b.w[1] << (64 - k)),
              (b.w[1] >> k) | (b.w[2] << (64 - k)),
              (b.w[2] >> k) | (b.w[3] << (64 - k)),
              b.w[3] >> k);
}

// 14x14 有效区域掩码
inline const BB& valid_mask() noexcept {
    static const BB m = [] {
        BB v;
        for (int r = 0; r < BOARD_N; ++r)
            for (int c = 0; c < BOARD_N; ++c) v.set(bit_of(r, c));
        return v;
    }();
    return m;
}

// 四邻（上下左右）膨胀，结果已裁剪到有效区域
inline BB edge_dilate(const BB& b) noexcept {
    BB r = shr(b, STRIDE) | shl(b, STRIDE) | shr(b, 1) | shl(b, 1);
    return r & valid_mask();
}

// 四角（对角）膨胀，结果已裁剪到有效区域
inline BB diag_dilate(const BB& b) noexcept {
    BB r = shr(b, STRIDE + 1) | shr(b, STRIDE - 1) | shl(b, STRIDE - 1) | shl(b, STRIDE + 1);
    return r & valid_mask();
}

// 遍历所有置位（回调收到物理位号）
template <typename F>
inline void for_each_bit(const BB& b, F&& f) {
    for (int i = 0; i < 4; ++i) {
        uint64_t x = b.w[i];
        while (x) {
            int t = __builtin_ctzll(x);
            f(i * 64 + t);
            x &= x - 1;
        }
    }
}

}  // namespace cornerstone
