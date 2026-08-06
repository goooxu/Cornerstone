#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cornerstone/agents.hpp"
#include "cornerstone/board.hpp"
#include "cornerstone/dataset.hpp"
#include "cornerstone/mcts.hpp"
#include "cornerstone/pieces.hpp"
#include "cornerstone/playout.hpp"
#include "cornerstone/reference.hpp"

namespace py = pybind11;
using namespace cornerstone;

namespace {

py::array_t<int32_t> to_i32(const std::vector<int32_t>& v) {
    py::array_t<int32_t> a(py::ssize_t(v.size()));
    if (!v.empty()) std::memcpy(a.mutable_data(), v.data(), v.size() * sizeof(int32_t));
    return a;
}

py::tuple board_features(const Board& b) {
    py::array_t<float> planes({NUM_PLANES, BOARD_N, BOARD_N});
    py::array_t<float> scalars(NUM_SCALARS);
    b.features(planes.mutable_data(), scalars.mutable_data());
    return py::make_tuple(planes, scalars);
}

// 批量取特征：自博弈里一次要喂几千个局面给 GPU，逐个过 Python 边界太慢
py::tuple features_batch(const std::vector<Board>& boards) {
    const py::ssize_t n = py::ssize_t(boards.size());
    py::array_t<float> planes({n, py::ssize_t(NUM_PLANES), py::ssize_t(BOARD_N), py::ssize_t(BOARD_N)});
    py::array_t<float> scalars({n, py::ssize_t(NUM_SCALARS)});
    float* pp = planes.mutable_data();
    float* sp = scalars.mutable_data();
    {
        py::gil_scoped_release release;
        for (py::ssize_t i = 0; i < n; ++i)
            boards[size_t(i)].features(pp + i * NUM_PLANES * PLANE_SIZE, sp + i * NUM_SCALARS);
    }
    return py::make_tuple(planes, scalars);
}

py::array_t<uint8_t> board_legal_mask(const Board& b) {
    py::array_t<uint8_t> m(NUM_ACTIONS);
    b.legal_mask(m.mutable_data());
    return m;
}

py::array_t<int32_t> sym_action_table() {
    const auto& t = sym_action();
    py::array_t<int32_t> a({py::ssize_t(NUM_SYM), py::ssize_t(NUM_ACTIONS)});
    for (int s = 0; s < NUM_SYM; ++s)
        std::memcpy(a.mutable_data() + s * NUM_ACTIONS, t[s].data(), NUM_ACTIONS * sizeof(int32_t));
    return a;
}

py::array_t<int16_t> sym_cell_table() {
    const auto& t = sym_cell();
    py::array_t<int16_t> a({py::ssize_t(NUM_SYM), py::ssize_t(NUM_CELLS)});
    for (int s = 0; s < NUM_SYM; ++s)
        std::memcpy(a.mutable_data() + s * NUM_CELLS, t[s].data(), NUM_CELLS * sizeof(int16_t));
    return a;
}

py::array_t<uint8_t> in_bounds_table() {
    const auto& t = action_in_bounds();
    py::array_t<uint8_t> a(NUM_ACTIONS);
    std::memcpy(a.mutable_data(), t.data(), NUM_ACTIONS);
    return a;
}

py::dict decode_action(int action) {
    if (action < 0 || action >= NUM_ACTIONS) throw py::index_error("动作编号越界");
    const int o = action_ori(action), anchor = action_anchor(action);
    const Orientation& od = orientations()[o];
    py::list cells;
    const int ar = anchor / BOARD_N, ac = anchor % BOARD_N;
    for (int i = 0; i < od.size; ++i)
        cells.append(py::make_tuple(ar + od.dr[i], ac + od.dc[i]));

    py::dict d;
    d["action"] = action;
    d["orientation"] = o;
    d["piece"] = int(od.piece);
    d["piece_name"] = piece_names()[od.piece];
    d["size"] = int(od.size);
    d["anchor"] = py::make_tuple(ar, ac);
    d["in_bounds"] = bool(action_in_bounds()[action]);
    d["cells"] = cells;
    return d;
}

}  // namespace

PYBIND11_MODULE(_engine, m) {
    m.doc() = "cornerstone Blokus Duo 引擎（C++ 核心）";

    m.attr("BOARD_N")       = BOARD_N;
    m.attr("NUM_CELLS")     = NUM_CELLS;
    m.attr("NUM_PIECES")    = NUM_PIECES;
    m.attr("NUM_ORI")       = NUM_ORI;
    m.attr("NUM_ACTIONS")   = NUM_ACTIONS;
    m.attr("NUM_PLANES")    = NUM_PLANES;
    m.attr("NUM_SCALARS")   = NUM_SCALARS;
    m.attr("NUM_SYM")       = int(NUM_SYM);
    m.attr("MAX_PLIES")     = MAX_PLIES;
    m.attr("TOTAL_SQUARES") = TOTAL_SQUARES;
    m.attr("START_CELLS")   = py::make_tuple(py::make_tuple(START_R[0], START_C[0]),
                                             py::make_tuple(START_R[1], START_C[1]));

    py::class_<Board>(m, "Board")
        .def(py::init<>())
        .def(py::init<const Board&>())
        .def("reset", &Board::reset)
        .def("copy", [](const Board& b) { return Board(b); })
        .def("__copy__", [](const Board& b) { return Board(b); })
        .def("__deepcopy__", [](const Board& b, py::dict) { return Board(b); })
        .def_property_readonly("current_player", &Board::current_player)
        .def_property_readonly("terminal", &Board::terminal)
        .def_property_readonly("ply", &Board::ply)
        .def("score", &Board::score, py::arg("player"))
        .def("scores", [](const Board& b) { return py::make_tuple(b.score(0), b.score(1)); })
        .def("result_for", &Board::result_for, py::arg("player"))
        .def("piece_remaining", &Board::piece_remaining, py::arg("player"), py::arg("piece"))
        .def("remaining_mask", &Board::remaining_mask, py::arg("player"))
        .def("legal_moves", [](const Board& b) { return to_i32(b.legal_moves()); })
        .def("legal_count", &Board::legal_count)
        .def("legal_mask", &board_legal_mask)
        .def("has_any_move", &Board::has_any_move, py::arg("player"))
        .def("is_legal", &Board::is_legal, py::arg("action"))
        .def("play", &Board::play, py::arg("action"))
        .def("features", &board_features)
        .def("history", [](const Board& b) { return to_i32(b.history()); })
        .def("occupancy_cells",
             [](const Board& b, int p) {
                 py::list out;
                 for (int cell = 0; cell < NUM_CELLS; ++cell)
                     if (b.occupancy(p).test(cell_to_bit(cell)))
                         out.append(py::make_tuple(cell / BOARD_N, cell % BOARD_N));
                 return out;
             },
             py::arg("player"))
        .def("anchor_cells",
             [](const Board& b, int p) {
                 BB allowed, anchors;
                 b.allowed_and_anchors(p, allowed, anchors);
                 py::list out;
                 for (int cell = 0; cell < NUM_CELLS; ++cell)
                     if (anchors.test(cell_to_bit(cell)))
                         out.append(py::make_tuple(cell / BOARD_N, cell % BOARD_N));
                 return out;
             },
             py::arg("player"))
        .def("__str__", &Board::to_string)
        .def("__repr__", [](const Board& b) {
            return "<Board ply=" + std::to_string(b.ply()) +
                   " cur=" + std::to_string(b.current_player()) +
                   " score=" + std::to_string(b.score(0)) + ":" + std::to_string(b.score(1)) +
                   (b.terminal() ? " terminal" : "") + ">";
        });

    m.def("features_batch", &features_batch, py::arg("boards"));

    m.def("random_playouts",
          [](int64_t n_games, uint64_t seed, int threads) {
              PlayoutStats s;
              {
                  py::gil_scoped_release release;   // 全程在 C++ 里跑，放开 GIL 才能真正用满多核
                  s = random_playouts(n_games, seed, threads);
              }
              py::dict d;
              d["games"] = s.games;
              d["plies"] = s.plies;
              d["movegens"] = s.movegens;
              d["wins0"] = s.wins0;
              d["wins1"] = s.wins1;
              d["draws"] = s.draws;
              d["score0"] = s.score0;
              d["score1"] = s.score1;
              return d;
          },
          py::arg("n_games"), py::arg("seed") = 0, py::arg("threads") = 1);

    m.def("rollout_value",
          [](const Board& b, int n, uint64_t seed) {
              py::gil_scoped_release release;
              return rollout_value(b, n, seed);
          },
          py::arg("board"), py::arg("n"), py::arg("seed") = 0);
    m.def("legal_moves_reference",
          [](const Board& b) { return to_i32(legal_moves_reference(b)); },
          py::arg("board"), "朴素参考实现，仅用于测试交叉比对");

    // ---- 规则基线智能体 ----
    // 必须先于 EvalConfig 注册 —— EvalConfig 的默认参数里带一个 AgentConfig 实例，
    // pybind11 在 def() 时就要把它转成 Python 对象
    py::enum_<AgentKind>(m, "AgentKind")
        .value("Random", AgentKind::Random)
        .value("GreedyArea", AgentKind::GreedyArea)
        .value("GreedyMobility", AgentKind::GreedyMobility)
        .value("FlatMCTS", AgentKind::FlatMCTS);

    {
        const AgentConfig d;   // 默认权重来自 tools/tune_mobility.py 的扫描结果
        py::class_<AgentConfig>(m, "AgentConfig")
            .def(py::init([](AgentKind kind, int rollouts, double w_size, double w_own_anchors,
                             double w_opp_anchors, double temperature) {
                     AgentConfig c;
                     c.kind = kind;
                     c.rollouts = rollouts;
                     c.w_size = w_size;
                     c.w_own_anchors = w_own_anchors;
                     c.w_opp_anchors = w_opp_anchors;
                     c.temperature = temperature;
                     return c;
                 }),
                 py::arg("kind") = AgentKind::Random, py::arg("rollouts") = d.rollouts,
                 py::arg("w_size") = d.w_size, py::arg("w_own_anchors") = d.w_own_anchors,
                 py::arg("w_opp_anchors") = d.w_opp_anchors,
                 py::arg("temperature") = d.temperature)
            .def_readwrite("kind", &AgentConfig::kind)
            .def_readwrite("rollouts", &AgentConfig::rollouts)
            .def_readwrite("w_size", &AgentConfig::w_size)
            .def_readwrite("w_own_anchors", &AgentConfig::w_own_anchors)
            .def_readwrite("w_opp_anchors", &AgentConfig::w_opp_anchors)
            .def_readwrite("temperature", &AgentConfig::temperature);
    }

    // ---- Gumbel AlphaZero 自博弈 ----
    m.attr("MAX_TOPK") = MAX_TOPK;

    py::class_<MctsConfig>(m, "MctsConfig")
        .def(py::init([](int simulations, int max_considered, double c_visit, double c_scale,
                         int temperature_plies, int top_k, double value_from_score) {
                 MctsConfig c;
                 c.simulations = simulations;
                 c.max_considered = max_considered;
                 c.c_visit = c_visit;
                 c.c_scale = c_scale;
                 c.temperature_plies = temperature_plies;
                 c.top_k = top_k;
                 c.value_from_score = value_from_score;
                 return c;
             }),
             py::arg("simulations") = 128, py::arg("max_considered") = 16,
             py::arg("c_visit") = 50.0, py::arg("c_scale") = 1.0,
             py::arg("temperature_plies") = 12, py::arg("top_k") = MAX_TOPK,
             py::arg("value_from_score") = 0.0)
        .def_readwrite("simulations", &MctsConfig::simulations)
        .def_readwrite("max_considered", &MctsConfig::max_considered)
        .def_readwrite("c_visit", &MctsConfig::c_visit)
        .def_readwrite("c_scale", &MctsConfig::c_scale)
        .def_readwrite("temperature_plies", &MctsConfig::temperature_plies)
        .def_readwrite("top_k", &MctsConfig::top_k)
        .def_readwrite("value_from_score", &MctsConfig::value_from_score);

    py::class_<EvalConfig>(m, "EvalConfig")
        .def(py::init([](bool enabled, const AgentConfig& opponent, bool net_opponent,
                         int opening_plies) {
                 EvalConfig c;
                 c.enabled = enabled;
                 c.opponent = opponent;
                 c.net_opponent = net_opponent;
                 c.opening_plies = opening_plies;
                 return c;
             }),
             py::arg("enabled") = false, py::arg("opponent") = AgentConfig{},
             py::arg("net_opponent") = false, py::arg("opening_plies") = 4)
        .def_readwrite("enabled", &EvalConfig::enabled)
        .def_readwrite("opponent", &EvalConfig::opponent)
        .def_readwrite("net_opponent", &EvalConfig::net_opponent)
        .def_readwrite("opening_plies", &EvalConfig::opening_plies);

    py::class_<SelfPlayEngine>(m, "SelfPlayEngine")
        .def(py::init<int, const MctsConfig&, uint64_t, const EvalConfig&, int>(),
             py::arg("num_games"), py::arg("config"), py::arg("seed") = 0,
             py::arg("eval") = EvalConfig{}, py::arg("threads") = 1)
        .def_property_readonly("num_games", &SelfPlayEngine::num_games)
        .def_property_readonly("max_batch", &SelfPlayEngine::max_batch)
        .def_property_readonly("finished_games", &SelfPlayEngine::finished_games)
        // 缓冲区由调用方预分配复用，避免每轮都申请几 MB 的 numpy 数组
        .def("prepare",
             [](SelfPlayEngine& e, py::array_t<float, py::array::c_style> planes,
                py::array_t<float, py::array::c_style> scalars, py::object which_obj) {
                 const py::ssize_t cap = e.max_batch();
                 if (planes.size() < cap * NUM_PLANES * PLANE_SIZE)
                     throw py::value_error("planes 缓冲区太小");
                 if (scalars.size() < cap * NUM_SCALARS)
                     throw py::value_error("scalars 缓冲区太小");
                 int8_t* w = nullptr;
                 py::array_t<int8_t, py::array::c_style> which;
                 if (!which_obj.is_none()) {
                     which = which_obj.cast<py::array_t<int8_t, py::array::c_style>>();
                     if (which.size() < cap) throw py::value_error("which_net 缓冲区太小");
                     w = which.mutable_data();
                 }
                 float* p = planes.mutable_data();
                 float* s = scalars.mutable_data();
                 py::gil_scoped_release release;
                 return e.prepare(p, s, w);
             },
             py::arg("planes"), py::arg("scalars"), py::arg("which_net") = py::none())
        .def("feed",
             [](SelfPlayEngine& e, py::array_t<float, py::array::c_style | py::array::forcecast> logits,
                py::array_t<float, py::array::c_style | py::array::forcecast> wdl) {
                 if (logits.ndim() != 2 || logits.shape(1) != NUM_ACTIONS)
                     throw py::value_error("logits 形状应为 [n, NUM_ACTIONS]");
                 if (wdl.ndim() != 2 || wdl.shape(1) != 3)
                     throw py::value_error("wdl 形状应为 [n, 3]");
                 if (logits.shape(0) != wdl.shape(0))
                     throw py::value_error("logits 与 wdl 的批大小不一致");
                 const float* lg = logits.data();
                 const float* wd = wdl.data();
                 py::gil_scoped_release release;
                 e.feed(lg, wd);
             },
             py::arg("logits"), py::arg("wdl"))
        .def("set_position",
             [](SelfPlayEngine& e, const std::vector<int32_t>& actions) {
                 e.set_position(actions);
             },
             py::arg("actions"),
             "把所有局重置到给定着法序列对应的局面（Web 试玩的单局面搜索用）")
        .def("root_info",
             [](const SelfPlayEngine& e, int game) {
                 const auto info = e.root_info(game);
                 py::dict d;
                 d["ready"] = info.ready;
                 d["value"] = info.value;
                 d["actions"] = to_i32(info.actions);
                 d["visits"] = to_i32(info.visits);
                 py::array_t<float> probs(py::ssize_t(info.probs.size()));
                 py::array_t<float> priors(py::ssize_t(info.priors.size()));
                 if (!info.probs.empty()) {
                     std::memcpy(probs.mutable_data(), info.probs.data(),
                                 info.probs.size() * sizeof(float));
                     std::memcpy(priors.mutable_data(), info.priors.data(),
                                 info.priors.size() * sizeof(float));
                 }
                 d["probs"] = probs;
                 d["priors"] = priors;
                 return d;
             },
             py::arg("game") = 0)
        .def("advance", [](SelfPlayEngine& e) {
            std::vector<GameRecord> games;
            {
                py::gil_scoped_release release;
                games = e.advance();
            }
            py::list out;
            for (const GameRecord& g : games) {
                const py::ssize_t t = py::ssize_t(g.moves.size());
                py::array_t<int32_t> actions(t), n_legal(t), top_actions({t, py::ssize_t(MAX_TOPK)});
                py::array_t<int8_t> players(t);
                py::array_t<uint8_t> n_top(t);
                py::array_t<float> rest(t), root_values(t), top_probs({t, py::ssize_t(MAX_TOPK)});

                for (py::ssize_t i = 0; i < t; ++i) {
                    const MoveTarget& mt = g.moves[size_t(i)];
                    actions.mutable_data()[i] = mt.action;
                    players.mutable_data()[i] = mt.player;
                    n_legal.mutable_data()[i] = mt.n_legal;
                    n_top.mutable_data()[i] = mt.n_top;
                    rest.mutable_data()[i] = mt.rest_prob;
                    root_values.mutable_data()[i] = mt.root_value;
                    std::memcpy(top_actions.mutable_data() + i * MAX_TOPK, mt.top_action,
                                sizeof(int32_t) * MAX_TOPK);
                    std::memcpy(top_probs.mutable_data() + i * MAX_TOPK, mt.top_prob,
                                sizeof(float) * MAX_TOPK);
                }
                py::dict d;
                d["result0"] = int(g.result0);
                d["score0"] = int(g.score0);
                d["score1"] = int(g.score1);
                d["net_player"] = int(g.net_player);
                d["selfplay"] = g.selfplay;
                d["actions"] = actions;
                d["players"] = players;
                d["n_legal"] = n_legal;
                d["n_top"] = n_top;
                d["rest_prob"] = rest;
                d["root_values"] = root_values;
                d["top_actions"] = top_actions;
                d["top_probs"] = top_probs;
                out.append(d);
            }
            return out;
        });

    m.def("select_move",
          [](const Board& b, const AgentConfig& cfg, uint64_t seed) {
              uint64_t s = seed;
              return select_move(b, cfg, s);
          },
          py::arg("board"), py::arg("config"), py::arg("seed") = 0);

    m.def("play_match",
          [](const AgentConfig& a, const AgentConfig& b, int64_t games, uint64_t seed,
             int threads, int opening_plies) {
              MatchResult r;
              {
                  py::gil_scoped_release release;
                  r = play_match(a, b, games, seed, threads, opening_plies);
              }
              py::dict d;
              d["games"] = r.games;
              d["wins_a"] = r.wins_a;
              d["wins_b"] = r.wins_b;
              d["draws"] = r.draws;
              d["score_a"] = r.score_a;
              d["score_b"] = r.score_b;
              d["plies"] = r.plies;
              d["a_as_first"] = r.a_as_first;
              d["a_wins_as_first"] = r.a_wins_as_first;
              d["a_wins_as_second"] = r.a_wins_as_second;
              return d;
          },
          py::arg("a"), py::arg("b"), py::arg("games"), py::arg("seed") = 0,
          py::arg("threads") = 1, py::arg("opening_plies") = 0);

    m.def("build_batch",
          [](py::array_t<int32_t, py::array::c_style> actions,
             py::array_t<int32_t, py::array::c_style> game_offsets,
             py::array_t<int32_t, py::array::c_style> want_ply,
             py::array_t<int32_t, py::array::c_style> want_offsets,
             py::object syms_obj,
             py::array_t<float, py::array::c_style> planes,
             py::array_t<float, py::array::c_style> scalars,
             py::array_t<uint8_t, py::array::c_style> legal, int threads) {
              const int n_games = int(game_offsets.size()) - 1;
              if (n_games < 0) throw py::value_error("game_offsets 至少要有 1 个元素");
              if (want_offsets.size() != game_offsets.size())
                  throw py::value_error("want_offsets 与 game_offsets 长度必须相同");
              const py::ssize_t n = want_ply.size();
              if (planes.size() < n * NUM_PLANES * PLANE_SIZE) throw py::value_error("planes 太小");
              if (scalars.size() < n * NUM_SCALARS) throw py::value_error("scalars 太小");
              if (legal.size() < n * NUM_ACTIONS) throw py::value_error("legal 太小");

              const int8_t* syms = nullptr;
              py::array_t<int8_t, py::array::c_style> syms_arr;
              if (!syms_obj.is_none()) {
                  syms_arr = syms_obj.cast<py::array_t<int8_t, py::array::c_style>>();
                  if (syms_arr.size() != n) throw py::value_error("syms 长度必须等于样本数");
                  syms = syms_arr.data();
              }

              const int32_t* a = actions.data();
              const int32_t* go = game_offsets.data();
              const int32_t* wp = want_ply.data();
              const int32_t* wo = want_offsets.data();
              float* p = planes.mutable_data();
              float* s = scalars.mutable_data();
              uint8_t* l = legal.mutable_data();

              py::gil_scoped_release release;
              build_batch(a, go, n_games, wp, wo, syms, p, s, l, threads);
          },
          py::arg("actions"), py::arg("game_offsets"), py::arg("want_ply"),
          py::arg("want_offsets"), py::arg("syms"), py::arg("planes"), py::arg("scalars"),
          py::arg("legal"), py::arg("threads") = 1);

    m.def("decode_action", &decode_action, py::arg("action"));
    m.def("encode_action", [](int ori, int anchor_cell) { return encode_action(ori, anchor_cell); },
          py::arg("orientation"), py::arg("anchor_cell"));
    m.def("sym_action_table", &sym_action_table);
    m.def("sym_cell_table", &sym_cell_table);
    m.def("action_in_bounds", &in_bounds_table);
    m.def("piece_names", []() { return std::vector<std::string>(piece_names().begin(), piece_names().end()); });
    m.def("piece_sizes", []() { return std::vector<int>(piece_sizes().begin(), piece_sizes().end()); });
    m.def("piece_orientations", [](int piece) {
        const auto& v = piece_orientations()[size_t(piece)];
        return std::vector<int>(v.begin(), v.end());
    }, py::arg("piece"));
    m.def("orientation_info", [](int o) {
        const Orientation& od = orientations()[size_t(o)];
        py::list cells;
        for (int i = 0; i < od.size; ++i) cells.append(py::make_tuple(int(od.dr[i]), int(od.dc[i])));
        py::dict d;
        d["piece"] = int(od.piece);
        d["piece_name"] = piece_names()[od.piece];
        d["size"] = int(od.size);
        d["height"] = int(od.h);
        d["width"] = int(od.w);
        d["cells"] = cells;
        return d;
    }, py::arg("orientation"));
}
