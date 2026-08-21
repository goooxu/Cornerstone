// Blokus Duo 局面。
//
// 规则要点（本项目采用的版本）：
//  - 首手必须覆盖己方起始格；此后每手必须与己方棋子角相邻、且不得与己方棋子边相邻
//  - 与对方棋子怎么贴都可以
//  - 一方无合法着法时停手，另一方继续；双方都无合法着法则终局
//  - 终局后占格数多者胜，相同为和局（没有「全部落完 +15」之类的加分项）
//
// Board 刻意做成可平凡拷贝的（历史用定长数组而非 vector），MCTS 里按值拷贝局面很频繁。

#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "cornerstone/bitboard.hpp"
#include "cornerstone/pieces.hpp"

namespace cornerstone {

// 双方各 21 枚，一局最多 42 手
inline constexpr int MAX_PLIES = 2 * NUM_PIECES;

// 平面布局（顺序即兼容性契约 —— 老模型靠「切前 9 个」继续工作）：
//   0..7  八个位板平面：己方占格 / 对方占格 / 双方可落区 / 双方角点 / 双方起始格
//   8     恒为 1，给网络一个偏置与边界参考
//   9     己方可达度：该格被己方多少个合法着法覆盖，min(n,64)/64
//   10    对方可达度：同上
//
// **常数平面必须钉死在下标 8**，不能写成 NUM_PLANES-1 —— 扩容时它会跟着漂到
// 下标 10，于是老 checkpoint 切出来的第 9 个平面从「恒 1」变成「恒 0」。
// 那不会报任何错：模型照样加载、照样推理，只是棋力莫名其妙掉一截。
inline constexpr int N_BB_PLANES = 8;    // 前八个位板平面
inline constexpr int PLANE_CONST = 8;    // 恒 1 的那个平面
inline constexpr int PLANE_MOB_ME = 9;   // 己方可达度
inline constexpr int PLANE_MOB_OP = 10;  // 对方可达度
inline constexpr int NUM_PLANES  = 11;
// 老模型只吃前 9 个平面（ModelConfig.in_planes 默认 9）。
inline constexpr int NUM_PLANES_LEGACY = 9;
inline constexpr int NUM_SCALARS = 2 * NUM_PIECES + 2;   // 44
inline constexpr int PLANE_SIZE  = NUM_CELLS;

class Board {
public:
    Board() { reset(); }
    void reset();

    int  current_player() const { return cur_; }
    bool terminal() const { return terminal_; }
    int  ply() const { return ply_; }
    int  score(int p) const { return score_[p]; }

    // 从 p 的角度看的结果：+1 胜 / 0 和 / -1 负。仅在终局有意义。
    int result_for(int p) const {
        if (score_[p] > score_[1 - p]) return 1;
        if (score_[p] < score_[1 - p]) return -1;
        return 0;
    }

    bool piece_remaining(int p, int piece) const { return (remaining_[p] >> piece) & 1u; }
    uint32_t remaining_mask(int p) const { return remaining_[p]; }
    const BB& occupancy(int p) const { return occ_[p]; }

    // 当前行棋方的全部合法着法
    void legal_moves(std::vector<int32_t>& out) const;
    std::vector<int32_t> legal_moves() const {
        std::vector<int32_t> v;
        legal_moves(v);
        return v;
    }
    int  legal_count() const;
    bool has_any_move(int p) const;
    bool is_legal(int action) const;
    // 写入长度为 NUM_ACTIONS 的 0/1 掩码
    void legal_mask(uint8_t* out) const;

    // 可达度平面：p 方的每个格被多少个合法着法覆盖，归一化后写进 dst[NUM_CELLS]。
    void mobility_plane(int p, float* dst) const;

    // 落子。非法着法抛异常。落子后自动处理停手与终局判定。
    void play(int action);

    // 以当前行棋方视角写出特征。
    //   planes:  NUM_PLANES * NUM_CELLS 个 float
    //   scalars: NUM_SCALARS 个 float
    void features(float* planes, float* scalars) const;

    // 调试用的可读棋盘
    std::string to_string() const;

    int history_size() const { return ply_; }
    int history_at(int i) const { return history_[i]; }
    std::vector<int32_t> history() const { return {history_.begin(), history_.begin() + ply_}; }

    // 计算 p 的可落格与角点前沿（对外暴露主要是给基线启发式与前端高亮用）
    void allowed_and_anchors(int p, BB& allowed, BB& anchors) const;

private:
    BB       occ_[2];
    uint32_t remaining_[2];
    int32_t  score_[2];
    int32_t  cur_;
    int32_t  ply_;
    bool     terminal_;
    bool     placed_first_[2];
    std::array<int32_t, MAX_PLIES> history_;

    // StopAtFirst 为真时只判断「是否存在合法着法」，找到一个就返回
    template <bool StopAtFirst>
    int generate(int p, std::vector<int32_t>* out) const;

    void advance_turn();
};

}  // namespace cornerstone
