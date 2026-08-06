#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cornerstone/board.hpp"
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
