"""棋子朝向表与动作编码。"""

import numpy as np
import pytest

import cornerstone as cs

# 21 枚棋子各自的不同朝向数（4 旋转 x 2 镜像去重后），合计 91
EXPECTED_ORI_COUNTS = [1, 2, 2, 4, 2, 8, 1, 4, 4, 8, 2, 8, 8, 8, 4, 4, 4, 4, 1, 8, 4]
EXPECTED_SIZES = [1, 2, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]


def test_constants():
    assert cs.NUM_PIECES == 21
    assert cs.NUM_ORI == 91
    assert cs.NUM_CELLS == 196
    assert cs.NUM_ACTIONS == 91 * 196 == 17836
    assert cs.TOTAL_SQUARES == 89
    assert cs.START_CELLS == ((4, 4), (9, 9))


def test_piece_sizes():
    assert list(cs.piece_sizes()) == EXPECTED_SIZES
    assert sum(cs.piece_sizes()) == cs.TOTAL_SQUARES


def test_orientation_counts():
    counts = [len(cs.piece_orientations(p)) for p in range(cs.NUM_PIECES)]
    assert counts == EXPECTED_ORI_COUNTS
    assert sum(counts) == cs.NUM_ORI


def test_orientation_ids_partition_all():
    seen = sorted(o for p in range(cs.NUM_PIECES) for o in cs.piece_orientations(p))
    assert seen == list(range(cs.NUM_ORI))


@pytest.mark.parametrize("o", range(cs.NUM_ORI))
def test_orientation_wellformed(o):
    info = cs.orientation_info(o)
    cells = info["cells"]

    assert len(cells) == info["size"] == cs.piece_sizes()[info["piece"]]
    assert len(set(cells)) == len(cells), "同一朝向里出现重复格"

    # 归一化：包围盒左上角贴 (0,0)
    assert min(r for r, _ in cells) == 0
    assert min(c for _, c in cells) == 0
    assert max(r for r, _ in cells) + 1 == info["height"]
    assert max(c for _, c in cells) + 1 == info["width"]

    # 连通性（四邻）
    todo, seen = [cells[0]], {cells[0]}
    cellset = set(cells)
    while todo:
        r, c = todo.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (nr, nc) in cellset and (nr, nc) not in seen:
                seen.add((nr, nc))
                todo.append((nr, nc))
    assert seen == cellset, "朝向的格集合不连通"


def test_all_orientation_shapes_distinct():
    shapes = {frozenset(cs.orientation_info(o)["cells"]) for o in range(cs.NUM_ORI)}
    assert len(shapes) == cs.NUM_ORI, "存在重复朝向，去重逻辑有问题"


def test_action_roundtrip():
    for o in range(cs.NUM_ORI):
        for cell in range(cs.NUM_CELLS):
            a = cs.encode_action(o, cell)
            d = cs.decode_action(a)
            assert d["orientation"] == o
            assert d["anchor"] == (cell // cs.BOARD_N, cell % cs.BOARD_N)


def test_in_bounds_table_matches_geometry():
    ib = cs.action_in_bounds()
    assert ib.shape == (cs.NUM_ACTIONS,)
    for a in range(cs.NUM_ACTIONS):
        d = cs.decode_action(a)
        inside = all(0 <= r < cs.BOARD_N and 0 <= c < cs.BOARD_N for r, c in d["cells"])
        assert bool(ib[a]) == inside == d["in_bounds"]
    # 每个朝向的合法锚点数 = (14-h+1)*(14-w+1)
    total = 0
    for o in range(cs.NUM_ORI):
        info = cs.orientation_info(o)
        total += (cs.BOARD_N - info["height"] + 1) * (cs.BOARD_N - info["width"] + 1)
    assert int(ib.sum()) == total


def test_action_cells_match_orientation_offsets():
    rng = np.random.default_rng(7)
    for a in rng.integers(0, cs.NUM_ACTIONS, size=2000):
        d = cs.decode_action(int(a))
        info = cs.orientation_info(d["orientation"])
        ar, ac = d["anchor"]
        assert d["cells"] == [(ar + dr, ac + dc) for dr, dc in info["cells"]]
