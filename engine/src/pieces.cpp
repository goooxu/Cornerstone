#include "cornerstone/pieces.hpp"

#include <algorithm>
#include <cassert>
#include <map>
#include <stdexcept>

namespace cornerstone {
namespace {

struct Cell {
    int r, c;
    bool operator<(const Cell& o) const { return r != o.r ? r < o.r : c < o.c; }
    bool operator==(const Cell& o) const { return r == o.r && c == o.c; }
};

// 21 枚基础形状。格数合计 1+2+3+3+4*5+5*12 = 89。
struct BasePiece {
    const char* name;
    int n;
    int8_t rc[MAX_PIECE_SIZE][2];
};

constexpr BasePiece kBase[NUM_PIECES] = {
    {"I1", 1, {{0, 0}}},
    {"I2", 2, {{0, 0}, {0, 1}}},
    {"I3", 3, {{0, 0}, {0, 1}, {0, 2}}},
    {"V3", 3, {{0, 0}, {0, 1}, {1, 0}}},
    {"I4", 4, {{0, 0}, {0, 1}, {0, 2}, {0, 3}}},
    {"L4", 4, {{0, 0}, {0, 1}, {0, 2}, {1, 0}}},
    {"O4", 4, {{0, 0}, {0, 1}, {1, 0}, {1, 1}}},
    {"S4", 4, {{0, 1}, {0, 2}, {1, 0}, {1, 1}}},
    {"T4", 4, {{0, 0}, {0, 1}, {0, 2}, {1, 1}}},
    {"F5", 5, {{0, 1}, {0, 2}, {1, 0}, {1, 1}, {2, 1}}},
    {"I5", 5, {{0, 0}, {0, 1}, {0, 2}, {0, 3}, {0, 4}}},
    {"L5", 5, {{0, 0}, {0, 1}, {0, 2}, {0, 3}, {1, 0}}},
    {"N5", 5, {{0, 1}, {0, 2}, {0, 3}, {1, 0}, {1, 1}}},
    {"P5", 5, {{0, 0}, {0, 1}, {1, 0}, {1, 1}, {2, 0}}},
    {"T5", 5, {{0, 0}, {0, 1}, {0, 2}, {1, 1}, {2, 1}}},
    {"U5", 5, {{0, 0}, {0, 2}, {1, 0}, {1, 1}, {1, 2}}},
    {"V5", 5, {{0, 0}, {1, 0}, {2, 0}, {2, 1}, {2, 2}}},
    {"W5", 5, {{0, 0}, {1, 0}, {1, 1}, {2, 1}, {2, 2}}},
    {"X5", 5, {{0, 1}, {1, 0}, {1, 1}, {1, 2}, {2, 1}}},
    {"Y5", 5, {{0, 1}, {1, 0}, {1, 1}, {2, 1}, {3, 1}}},
    {"Z5", 5, {{0, 0}, {0, 1}, {1, 1}, {2, 1}, {2, 2}}},
};

// 归一化：平移到包围盒左上角贴 (0,0)，再按行序排序，得到形状的规范表示
void normalize(std::vector<Cell>& cs) {
    int mr = cs[0].r, mc = cs[0].c;
    for (const auto& c : cs) {
        mr = std::min(mr, c.r);
        mc = std::min(mc, c.c);
    }
    for (auto& c : cs) {
        c.r -= mr;
        c.c -= mc;
    }
    std::sort(cs.begin(), cs.end());
}

// 规范形状的哈希键。包围盒不超过 5x5，故每格可压进一个字节。
uint64_t shape_key(const std::vector<Cell>& cs) {
    uint64_t k = cs.size();
    for (size_t i = 0; i < cs.size(); ++i)
        k |= uint64_t(cs[i].r * 8 + cs[i].c) << (8 * (i + 1));
    return k;
}

// t 的低 2 位是旋转次数，第 3 位是是否先做镜像
std::vector<Cell> transform(const std::vector<Cell>& in, int t) {
    std::vector<Cell> out = in;
    if (t & 4)
        for (auto& c : out) c.c = -c.c;
    for (int i = 0; i < (t & 3); ++i)
        for (auto& c : out) {
            int r = c.r;
            c.r = c.c;
            c.c = -r;
        }
    normalize(out);
    return out;
}

struct Tables {
    std::array<Orientation, NUM_ORI> ori{};
    std::array<uint8_t, NUM_PIECES> sizes{};
    std::array<std::string, NUM_PIECES> names{};
    std::array<std::vector<uint8_t>, NUM_PIECES> per_piece{};
    std::map<uint64_t, int> shape_to_ori;

    std::vector<uint8_t> in_bounds;   // NUM_ACTIONS
    std::vector<BB> masks;            // NUM_ACTIONS
    std::array<std::array<int16_t, NUM_CELLS>, NUM_SYM> scell{};
    std::array<std::vector<int32_t>, NUM_SYM> saction{};

    Tables() {
        build_orientations();
        build_actions();
        build_symmetry();
    }

    void build_orientations() {
        int next = 0;
        for (int p = 0; p < NUM_PIECES; ++p) {
            const BasePiece& bp = kBase[p];
            sizes[p] = uint8_t(bp.n);
            names[p] = bp.name;

            std::vector<Cell> base;
            for (int i = 0; i < bp.n; ++i) base.push_back({bp.rc[i][0], bp.rc[i][1]});
            normalize(base);

            // 变换顺序固定，先出现的先占用 id —— 动作编码的稳定性依赖于此
            for (int t = 0; t < 8; ++t) {
                std::vector<Cell> cs = transform(base, t);
                uint64_t key = shape_key(cs);
                if (shape_to_ori.count(key)) continue;

                if (next >= NUM_ORI) throw std::logic_error("朝向数超出 NUM_ORI");
                Orientation& o = ori[next];
                o.piece = uint8_t(p);
                o.size  = uint8_t(cs.size());
                o.h = 0;
                o.w = 0;
                for (size_t i = 0; i < cs.size(); ++i) {
                    o.dr[i] = int8_t(cs[i].r);
                    o.dc[i] = int8_t(cs[i].c);
                    o.h = uint8_t(std::max<int>(o.h, cs[i].r + 1));
                    o.w = uint8_t(std::max<int>(o.w, cs[i].c + 1));
                }
                shape_to_ori[key] = next;
                per_piece[p].push_back(uint8_t(next));
                ++next;
            }
        }
        // 21 枚棋子去重后恰好 91 个朝向。对不上说明基础形状写错了。
        if (next != NUM_ORI) throw std::logic_error("朝向总数不是 91");

        int total = 0;
        for (int p = 0; p < NUM_PIECES; ++p) total += sizes[p];
        if (total != TOTAL_SQUARES) throw std::logic_error("单方总格数不是 89");
    }

    void build_actions() {
        in_bounds.assign(NUM_ACTIONS, 0);
        masks.assign(NUM_ACTIONS, BB());
        for (int o = 0; o < NUM_ORI; ++o) {
            const Orientation& od = ori[o];
            for (int cell = 0; cell < NUM_CELLS; ++cell) {
                int ar = cell / BOARD_N, ac = cell % BOARD_N;
                if (ar + od.h > BOARD_N || ac + od.w > BOARD_N) continue;
                BB m;
                for (int i = 0; i < od.size; ++i)
                    m.set(bit_of(ar + od.dr[i], ac + od.dc[i]));
                int a = encode_action(o, cell);
                in_bounds[a] = 1;
                masks[a] = m;
            }
        }
    }

    void build_symmetry() {
        for (int s = 0; s < NUM_SYM; ++s) {
            for (int cell = 0; cell < NUM_CELLS; ++cell) {
                int r = cell / BOARD_N, c = cell % BOARD_N, nr, nc;
                switch (s) {
                    case SYM_ID:        nr = r;                nc = c;                break;
                    case SYM_TRANSPOSE: nr = c;                nc = r;                break;
                    case SYM_ROT180:    nr = BOARD_N - 1 - r;  nc = BOARD_N - 1 - c;  break;
                    default:            nr = BOARD_N - 1 - c;  nc = BOARD_N - 1 - r;  break;
                }
                scell[s][cell] = int16_t(nr * BOARD_N + nc);
            }

            saction[s].assign(NUM_ACTIONS, -1);
            for (int a = 0; a < NUM_ACTIONS; ++a) {
                if (!in_bounds[a]) continue;
                const Orientation& od = ori[action_ori(a)];
                int ar = action_anchor(a) / BOARD_N, ac = action_anchor(a) % BOARD_N;

                std::vector<Cell> cs;
                cs.reserve(od.size);
                for (int i = 0; i < od.size; ++i) {
                    int cell = (ar + od.dr[i]) * BOARD_N + (ac + od.dc[i]);
                    int t = scell[s][cell];
                    cs.push_back({t / BOARD_N, t % BOARD_N});
                }
                // 变换后的绝对锚点 = 新包围盒左上角
                int nr = cs[0].r, nc = cs[0].c;
                for (const auto& x : cs) {
                    nr = std::min(nr, x.r);
                    nc = std::min(nc, x.c);
                }
                std::vector<Cell> shape = cs;
                normalize(shape);
                auto it = shape_to_ori.find(shape_key(shape));
                if (it == shape_to_ori.end()) throw std::logic_error("对称变换后的形状不在朝向表里");
                saction[s][a] = encode_action(it->second, nr * BOARD_N + nc);
            }
        }
    }
};

const Tables& tables() {
    static const Tables t;   // 函数局部静态变量，C++11 起初始化是线程安全的
    return t;
}

}  // namespace

const std::array<Orientation, NUM_ORI>& orientations() { return tables().ori; }
const std::array<uint8_t, NUM_PIECES>& piece_sizes() { return tables().sizes; }
const std::array<std::string, NUM_PIECES>& piece_names() { return tables().names; }
const std::array<std::vector<uint8_t>, NUM_PIECES>& piece_orientations() { return tables().per_piece; }
const std::vector<uint8_t>& action_in_bounds() { return tables().in_bounds; }
const std::vector<BB>& action_masks() { return tables().masks; }
const std::array<std::array<int16_t, NUM_CELLS>, NUM_SYM>& sym_cell() { return tables().scell; }
const std::array<std::vector<int32_t>, NUM_SYM>& sym_action() { return tables().saction; }

}  // namespace cornerstone
