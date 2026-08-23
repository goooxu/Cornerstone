#include "cornerstone/board.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace cornerstone {
namespace {

// 着法生成里用来去重的时间戳表。一个摆放可能从多个角点前沿格被枚举到，
// 必须去重。用 epoch 戳而不是每次清零，避免每次生成都刷 17836 个字节。
// 放 thread_local 而不是放进 Board，是为了让 Board 保持可平凡拷贝。
struct GenScratch {
    std::vector<uint16_t> stamp;
    uint16_t epoch = 0;
    GenScratch() : stamp(NUM_ACTIONS, 0) {}
    void bump() {
        if (++epoch == 0) {
            std::fill(stamp.begin(), stamp.end(), uint16_t(0));
            epoch = 1;
        }
    }
};

GenScratch& scratch() {
    static thread_local GenScratch s;
    return s;
}

}  // namespace

void Board::reset() {
    occ_[0] = occ_[1] = BB();
    remaining_[0] = remaining_[1] = (uint32_t(1) << NUM_PIECES) - 1;
    score_[0] = score_[1] = 0;
    cur_ = 0;
    ply_ = 0;
    terminal_ = false;
    placed_first_[0] = placed_first_[1] = false;
    history_.fill(-1);
}

void Board::allowed_and_anchors(int p, BB& allowed, BB& anchors) const {
    const BB occupied = occ_[0] | occ_[1];
    allowed = ~occupied & valid_mask();
    if (!placed_first_[p]) {
        // 首手：唯一的「落点锚」就是己方起始格，摆放必须盖住它
        anchors = BB::single(bit_of(START_R[p], START_C[p])) & allowed;
    } else {
        allowed &= ~edge_dilate(occ_[p]);      // 不得与己方棋子边相邻
        anchors = diag_dilate(occ_[p]) & allowed;  // 必须与己方棋子角相邻
    }
}

// 枚举方式：遍历角点前沿格 f，让棋子的第 k 格落在 f 上。
// 这样「角相邻」由构造保证，只需再验证所有格都落在 allowed 内。
// 前沿格通常只有几十个，比盲扫 17836 个 (朝向, 锚点) 组合快一个量级。
template <bool StopAtFirst>
int Board::generate(int p, std::vector<int32_t>* out) const {
    if (terminal_) return 0;

    BB allowed, anchors;
    allowed_and_anchors(p, allowed, anchors);
    if (anchors.empty()) return 0;

    const auto& oris = orientations();
    const auto& per_piece = piece_orientations();
    GenScratch* sc = nullptr;
    if constexpr (!StopAtFirst) {
        sc = &scratch();
        sc->bump();
    }

    int count = 0;
    for (int word = 0; word < 4; ++word) {
        uint64_t bits = anchors.w[word];
        while (bits) {
            const int fbit = word * 64 + __builtin_ctzll(bits);
            bits &= bits - 1;
            const int fr = row_of(fbit), fc = col_of(fbit);

            for (int pc = 0; pc < NUM_PIECES; ++pc) {
                if (!((remaining_[p] >> pc) & 1u)) continue;
                for (uint8_t o : per_piece[pc]) {
                    const Orientation& od = oris[o];
                    for (int k = 0; k < od.size; ++k) {
                        const int ar = fr - od.dr[k];
                        const int ac = fc - od.dc[k];
                        if (ar < 0 || ac < 0) continue;
                        if (ar + od.h > BOARD_N || ac + od.w > BOARD_N) continue;

                        bool ok = true;
                        for (int i = 0; i < od.size; ++i) {
                            if (!allowed.test(bit_of(ar + od.dr[i], ac + od.dc[i]))) {
                                ok = false;
                                break;
                            }
                        }
                        if (!ok) continue;

                        if constexpr (StopAtFirst) {
                            return 1;
                        } else {
                            const int a = encode_action(o, ar * BOARD_N + ac);
                            if (sc->stamp[a] == sc->epoch) continue;
                            sc->stamp[a] = sc->epoch;
                            if (out) out->push_back(a);
                            ++count;
                        }
                    }
                }
            }
        }
    }
    return count;
}

void Board::legal_moves(std::vector<int32_t>& out) const {
    out.clear();
    generate<false>(cur_, &out);
}

int Board::legal_count() const { return generate<false>(cur_, nullptr); }

bool Board::has_any_move(int p) const { return generate<true>(p, nullptr) > 0; }

bool Board::is_legal(int action) const {
    if (terminal_) return false;
    if (action < 0 || action >= NUM_ACTIONS) return false;
    if (!action_in_bounds()[action]) return false;

    const int piece = orientations()[action_ori(action)].piece;
    if (!((remaining_[cur_] >> piece) & 1u)) return false;

    BB allowed, anchors;
    allowed_and_anchors(cur_, allowed, anchors);
    const BB& m = action_masks()[action];
    return m.subset_of(allowed) && m.intersects(anchors);
}

void Board::legal_mask(uint8_t* out) const {
    std::memset(out, 0, NUM_ACTIONS);
    std::vector<int32_t> mv;
    generate<false>(cur_, &mv);
    for (int32_t a : mv) out[a] = 1;
}

void Board::play(int action) {
    if (terminal_) throw std::logic_error("终局之后不能再落子");
    if (!is_legal(action)) throw std::invalid_argument("非法着法: " + std::to_string(action));

    const int p = cur_;
    const int piece = orientations()[action_ori(action)].piece;

    occ_[p] |= action_masks()[action];
    remaining_[p] &= ~(uint32_t(1) << piece);
    score_[p] += piece_sizes()[piece];
    placed_first_[p] = true;

    history_[ply_] = action;
    ++ply_;

    advance_turn();
}

void Board::advance_turn() {
    const int other = 1 - cur_;
    if (has_any_move(other)) {
        cur_ = other;
        return;
    }
    // 对方停手。己方还有着法就继续走，否则双方都动不了 -> 终局
    if (has_any_move(cur_)) return;
    terminal_ = true;
}

void Board::mobility_plane(int p, float* dst) const {
    std::memset(dst, 0, sizeof(float) * PLANE_SIZE);
    if (terminal_) return;                      // 终局：两边都无从落子

    std::vector<int32_t> mv;
    generate<false>(p, &mv);                    // 已支持任意玩家，见 has_any_move
    if (mv.empty()) return;                     // 该方已停手，全 0 是正确的

    int count[NUM_CELLS] = {0};
    for (int32_t a : mv) {
        const BB& m = action_masks()[size_t(a)];
        for (int cell = 0; cell < NUM_CELLS; ++cell)
            if (m.test(cell_to_bit(cell))) ++count[cell];
    }
    constexpr float SCALE = 64.0f;
    for (int cell = 0; cell < NUM_CELLS; ++cell)
        dst[cell] = std::min(float(count[cell]), SCALE) / SCALE;
}

void Board::features(float* planes, float* scalars, bool with_mobility) const {
    const int me = cur_, op = 1 - cur_;

    BB my_allowed, my_anchors, op_allowed, op_anchors;
    allowed_and_anchors(me, my_allowed, my_anchors);
    allowed_and_anchors(op, op_allowed, op_anchors);

    const BB my_start = BB::single(bit_of(START_R[me], START_C[me]));
    const BB op_start = BB::single(bit_of(START_R[op], START_C[op]));

    // **这里一律用具名常量，不用 NUM_PLANES-1** —— 后者在扩平面时会跟着漂：
    // 数组维度变大而初始化值不够（空指针），常数平面也会从下标 8 挪走，
    // 而后者不报任何错，只是让老 checkpoint 的第 9 个平面从恒 1 变成恒 0。
    const BB* src[N_BB_PLANES] = {
        &occ_[me], &occ_[op], &my_allowed, &my_anchors,
        &op_allowed, &op_anchors, &my_start, &op_start,
    };

    std::memset(planes, 0, sizeof(float) * NUM_PLANES * PLANE_SIZE);
    for (int pl = 0; pl < N_BB_PLANES; ++pl) {
        float* dst = planes + pl * PLANE_SIZE;
        for (int cell = 0; cell < NUM_CELLS; ++cell)
            dst[cell] = src[pl]->test(cell_to_bit(cell)) ? 1.0f : 0.0f;
    }
    // 偏置/边界参考平面，恒为 1
    std::fill(planes + PLANE_CONST * PLANE_SIZE,
              planes + (PLANE_CONST + 1) * PLANE_SIZE, 1.0f);

    // 可达度场：每个格被该方多少个合法着法覆盖。
    //
    // 这是 Blokus 的核心量 —— 只在这个场上做贪心的 greedy-mobility 就能赢过
    // 多数人类。网络本来要自己从「21 枚在手棋子 x 局部形状匹配」里推出它，
    // 那是很深的组合推理；而走法生成器本来就在算这些着法，白拿。
    //
    // 归一化到 [0,1]：开局单格可被上百手覆盖，除以 64 后截断，
    // 让常见范围落在 0~1 而不是让少数极大值把其余压成 0。
    // 默认不算 —— 见头文件。memset 已经把这两个平面清零了。
    if (with_mobility) {
        mobility_plane(me, planes + PLANE_MOB_ME * PLANE_SIZE);
        mobility_plane(op, planes + PLANE_MOB_OP * PLANE_SIZE);
    }

    for (int i = 0; i < NUM_PIECES; ++i) {
        scalars[i] = (remaining_[me] >> i) & 1u ? 1.0f : 0.0f;
        scalars[NUM_PIECES + i] = (remaining_[op] >> i) & 1u ? 1.0f : 0.0f;
    }
    // 这两个刻意用「己方/对方占格数」而不是手数 ply。
    // 原因：rot180 与反对角变换会把己方起始格映射到另一个座位的起始格，
    // 而 ply 的奇偶性和座位相关（开局无停手时先手走偶数手），
    // 放 ply 会让这两个增广产生分布外的样本。占格数是座位无关的，
    // 且 own+opp 已经隐含了对局进度，信息量不比 ply 少。
    scalars[2 * NUM_PIECES]     = float(score_[me]) / float(TOTAL_SQUARES);
    scalars[2 * NUM_PIECES + 1] = float(score_[op]) / float(TOTAL_SQUARES);
}

std::string Board::to_string() const {
    std::string s;
    s.reserve(NUM_CELLS * 2 + 64);
    for (int r = 0; r < BOARD_N; ++r) {
        for (int c = 0; c < BOARD_N; ++c) {
            const int b = bit_of(r, c);
            char ch = '.';
            if (occ_[0].test(b)) ch = 'A';
            else if (occ_[1].test(b)) ch = 'B';
            else if ((r == START_R[0] && c == START_C[0]) || (r == START_R[1] && c == START_C[1])) ch = '*';
            s.push_back(ch);
            s.push_back(' ');
        }
        s.push_back('\n');
    }
    s += "ply=" + std::to_string(ply_) + " cur=" + std::to_string(cur_) +
         " score=" + std::to_string(score_[0]) + ":" + std::to_string(score_[1]) +
         (terminal_ ? " [终局]" : "") + "\n";
    return s;
}

}  // namespace cornerstone
