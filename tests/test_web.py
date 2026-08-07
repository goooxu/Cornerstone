"""Web 试玩后端的选择逻辑。

只测不需要 GPU 的那部分：后端 ID 解析、可选项发现、LRU 淘汰。
这几条都有「静默走错分支」的特点 —— 解析错了不会报错，只会让人对着
一个**不是自己选的**模型下棋，而棋力差异肉眼分辨不出来。
"""

import importlib.util
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_server():
    spec = importlib.util.spec_from_file_location(
        "web_server", os.path.join(REPO, "web", "server.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["web_server"] = mod
    spec.loader.exec_module(mod)
    return mod


server = _load_server()


@pytest.fixture
def runs(tmp_path, monkeypatch):
    """造一个假的 runs/ 目录树。"""
    def make(run: str, steps: list[int], latest: str | None = None):
        d = tmp_path / run / "ckpt"
        d.mkdir(parents=True, exist_ok=True)
        for s in steps:
            (d / f"step{s:08d}.pt").write_bytes(b"x")
        if latest is not None:
            (d / "latest").write_text(latest)
        return d

    monkeypatch.setattr(server, "RUNS_DIR", str(tmp_path))
    return make


# ------------------------------------------------------------------ 规则基线

def test_rule_backends_resolve():
    for name in server.RULE_BACKENDS:
        kind, target = server.resolve_backend(f"rule:{name}")
        assert kind == "rule" and target == name


def test_requested_three_baselines_are_offered():
    # 用户点名要这三个
    assert set(server.RULE_BACKENDS) == {"greedy-area", "corner-min", "greedy-mobility"}


def test_rule_backends_exist_in_arena():
    from cornerstone.arena import BASELINES
    for name in server.RULE_BACKENDS:
        assert name in BASELINES, f"{name} 不在 arena 的基线表里"


def test_unknown_backend_rejected():
    for bad in ("rule:不存在", "net:", "greedy-area", "", "rule:flat-mcts-4k"):
        with pytest.raises(ValueError):
            server.resolve_backend(bad)


# ---------------------------------------------------------------- 网络后端

def test_specific_checkpoint_resolves(runs):
    runs("ab-fp8", [1000, 2000])
    kind, path = server.resolve_backend("net:ab-fp8/step00001000.pt")
    assert kind == "net" and os.path.basename(path) == "step00001000.pt"


def test_latest_follows_pointer_file(runs):
    runs("ab-fp8", [1000, 2000], latest="step00002000.pt")
    _, path = server.resolve_backend("net:ab-fp8/latest")
    assert os.path.basename(path) == "step00002000.pt"


def test_latest_survives_pruned_pointer(runs):
    """latest 指向的那份被轮换删了，要回退到实际存在的最大步数。

    训练侧 _prune_checkpoints 会删旧 checkpoint，而 latest 文件是单独写的，
    两者之间存在窗口。不处理的话试玩会直接 500。
    """
    d = runs("ab-fp8", [1000, 2000], latest="step00002000.pt")
    (d / "step00002000.pt").unlink()
    _, path = server.resolve_backend("net:ab-fp8/latest")
    assert os.path.basename(path) == "step00001000.pt"


def test_latest_without_pointer_file(runs):
    runs("ab-bf16", [500, 30000, 7000])          # 故意不按顺序创建
    _, path = server.resolve_backend("net:ab-bf16/latest")
    assert os.path.basename(path) == "step00030000.pt", "要按步数排，不能按文件名字典序"


def test_latest_resolves_afresh_each_call(runs):
    """「跟随训练」必须每次重新解析，否则选中那一刻就被钉死了。"""
    d = runs("ab-fp8", [1000], latest="step00001000.pt")
    _, first = server.resolve_backend("net:ab-fp8/latest")
    (d / "step00009000.pt").write_bytes(b"x")
    (d / "latest").write_text("step00009000.pt")
    _, second = server.resolve_backend("net:ab-fp8/latest")
    assert os.path.basename(first) == "step00001000.pt"
    assert os.path.basename(second) == "step00009000.pt"


def test_deleted_checkpoint_raises_readable_error(runs):
    runs("ab-fp8", [1000])
    with pytest.raises(ValueError, match="已不存在"):
        server.resolve_backend("net:ab-fp8/step00007777.pt")


def test_missing_run_raises(runs):
    runs("ab-fp8", [1000])
    with pytest.raises(ValueError, match="找不到训练跑"):
        server.resolve_backend("net:没这个跑/latest")


def test_path_traversal_is_contained(runs):
    """后端 ID 来自请求体，不能让它跳出 ckpt 目录。"""
    runs("ab-fp8", [1000])
    with pytest.raises(ValueError):
        server.resolve_backend("net:ab-fp8/../../../etc/passwd")


# ------------------------------------------------------------------- 发现

def test_discover_lists_rules_and_checkpoints(runs):
    runs("ab-fp8", [1000, 2000], latest="step00002000.pt")
    runs("ab-bf16", [3000], latest="step00003000.pt")
    got = server.discover_backends()
    ids = [b["id"] for b in got]

    for name in server.RULE_BACKENDS:
        assert f"rule:{name}" in ids
    assert "net:ab-fp8/latest" in ids
    assert "net:ab-bf16/latest" in ids
    assert "net:ab-fp8/step00001000.pt" in ids
    # 每一项都能解析回去 —— 列出来却点不动是最难查的那种坏
    for b in got:
        server.resolve_backend(b["id"])


def test_discover_skips_empty_ckpt_dir(runs):
    runs("空跑", [])
    ids = [b["id"] for b in server.discover_backends()]
    assert not any(i.startswith("net:空跑") for i in ids)


def test_discover_orders_newest_first(runs):
    runs("ab-fp8", [1000, 5000, 3000], latest="step00005000.pt")
    steps = [b["id"] for b in server.discover_backends()
             if b["id"].startswith("net:ab-fp8/step")]
    assert steps == ["net:ab-fp8/step00005000.pt",
                     "net:ab-fp8/step00003000.pt",
                     "net:ab-fp8/step00001000.pt"]


# --------------------------------------------------------------- 模型缓存

class _FakeNet:
    """替身，避免测试里真的 torch.load 一个 checkpoint。"""
    loaded: list[str] = []

    def __init__(self, path, device):
        self.path = path
        self.label = f"fake({os.path.basename(path)})"
        _FakeNet.loaded.append(path)


def test_pool_caches_and_evicts_lru(runs, monkeypatch):
    runs("r", [1, 2, 3, 4], latest="step00000004.pt")
    monkeypatch.setattr(server, "NetBrain", _FakeNet)
    _FakeNet.loaded = []

    pool = server.BrainPool("cpu", max_models=2)
    a = pool.get("net:r/step00000001.pt")
    pool.get("net:r/step00000002.pt")
    assert pool.get("net:r/step00000001.pt") is a, "缓存命中应返回同一个对象"
    assert len(_FakeNet.loaded) == 2

    pool.get("net:r/step00000003.pt")              # 超出上限，淘汰最久未用的那个
    assert len(pool.nets) == 2
    pool.get("net:r/step00000002.pt")
    assert len(_FakeNet.loaded) == 4, "被淘汰的那个应重新加载"


def test_pool_reuses_rule_brains(runs):
    pool = server.BrainPool("cpu")
    a = pool.get("rule:greedy-mobility")
    assert pool.get("rule:greedy-mobility") is a
    assert a.has_net is False
    assert a.analyse([], 64) is None, "规则基线没有分析面板"


# ----------------------------------------------------------------- 座位

def test_human_player_index():
    assert server.Session(players=[None, "rule:greedy-area"]).human_player == 0
    assert server.Session(players=["rule:greedy-area", None]).human_player == 1


def test_human_player_is_minus_one_when_both_ai():
    """AI 对战没有「人类座位」。返回 -1 而不是 0 —— 返回 0 的话
    前端会认为先手是人，点击就能替它落子，而 result 也会算错方向。"""
    s = server.Session(players=["rule:greedy-area", "rule:corner-min"])
    assert s.human_player == -1


def test_default_session_is_human_vs_ai():
    s = server.Session()
    assert s.players[0] is None and s.players[1] == server.DEFAULT_BACKEND
    assert s.human_player == 0


def test_pool_latest_tracks_new_checkpoint(runs, monkeypatch):
    """选了「最新」之后，训练又落了一份 —— 池子要跟着换，而不是抱着旧的。"""
    d = runs("r", [10], latest="step00000010.pt")
    monkeypatch.setattr(server, "NetBrain", _FakeNet)
    _FakeNet.loaded = []

    pool = server.BrainPool("cpu", max_models=3)
    first = pool.get("net:r/latest")
    (d / "step00000020.pt").write_bytes(b"x")
    (d / "latest").write_text("step00000020.pt")
    second = pool.get("net:r/latest")

    assert first is not second
    assert os.path.basename(second.path) == "step00000020.pt"


# ------------------------------------------------------- 端到端（只用规则基线）
#
# 规则基线不碰 GPU、每步微秒级，所以能在单测里真打完整局。
# 网络后端的那条路由逻辑完全一样，差别只在 BrainPool 给回哪种 Brain。

pytest.importorskip("fastapi")
TestClient = pytest.importorskip("fastapi.testclient").TestClient


@pytest.fixture
def client():
    pool = server.BrainPool("cpu")
    app = server.build_app(pool, "rule:greedy-mobility")
    server.SESSIONS.clear()
    with TestClient(app) as c:
        yield c


def test_ai_vs_ai_plays_a_full_game(client):
    r = client.post("/api/new", json={
        "players": ["rule:greedy-area", "rule:corner-min"], "difficulty": "普通"})
    assert r.status_code == 200, r.text
    sid = r.json()["sid"]
    st = r.json()["state"]
    assert st["human_player"] == -1
    assert r.json()["labels"] == ["规则基线 greedy-area", "规则基线 corner-min"]

    plies = 0
    while not st["terminal"]:
        rr = client.post("/api/ai", json={"sid": sid})
        assert rr.status_code == 200, rr.text
        st = rr.json()
        plies += 1
        assert plies <= 42, "一局不可能超过 42 手"

    a, b = st["scores"]
    assert a + b <= 178                       # 双方各 89 格
    assert plies == len(st["history"])
    assert st["result"] == (0 if a == b else (1 if a > b else -1)), "result 应相对先手"


def test_ai_move_rejected_on_human_turn(client):
    r = client.post("/api/new", json={"players": [None, "rule:greedy-area"]})
    sid = r.json()["sid"]
    rr = client.post("/api/ai", json={"sid": sid})     # 先手是人，还没走
    assert rr.status_code == 400
    assert "人类" in rr.json()["detail"]


def test_undo_in_ai_vs_ai_pops_one_ply(client):
    """人机模式下悔棋要退到「轮到人类」；AI 对战没有人类，
    照那个条件找下去会一路 pop 到空棋盘 —— 看起来像整局被撤了。"""
    r = client.post("/api/new", json={"players": ["rule:greedy-area", "rule:corner-min"]})
    sid = r.json()["sid"]
    for _ in range(4):
        client.post("/api/ai", json={"sid": sid})
    before = len(client.get(f"/api/state?sid={sid}").json()["history"])
    assert before == 4

    after = len(client.post("/api/undo", json={"sid": sid}).json()["history"])
    assert after == before - 1, "AI 对战悔棋应只退一手"


def test_undo_in_human_game_returns_to_human(client):
    r = client.post("/api/new", json={"players": [None, "rule:greedy-area"]})
    sid = r.json()["sid"]
    st = r.json()["state"]
    client.post("/api/move", json={"sid": sid, "action": st["legal_actions"][0]})
    client.post("/api/ai", json={"sid": sid})
    st = client.post("/api/undo", json={"sid": sid}).json()
    assert st["current_player"] == st["human_player"]
    assert st["history"] == []


def test_switch_seats_midgame_keeps_board(client):
    r = client.post("/api/new", json={"players": ["rule:greedy-area", "rule:corner-min"]})
    sid = r.json()["sid"]
    for _ in range(3):
        st = client.post("/api/ai", json={"sid": sid}).json()
    hist, scores = list(st["history"]), list(st["scores"])

    rr = client.post("/api/backend", json={
        "sid": sid, "players": ["rule:greedy-mobility", "rule:corner-min"]})
    assert rr.status_code == 200
    st2 = rr.json()["state"]
    assert st2["history"] == hist and st2["scores"] == scores, "换座位不该动棋盘"
    assert st2["players"][0] == "rule:greedy-mobility"


def test_bad_players_rejected(client):
    for bad in ([], ["rule:x"], ["rule:x", "rule:y"], [None, "net:没这个跑/latest"]):
        r = client.post("/api/new", json={"players": bad})
        assert r.status_code == 400, f"{bad} 应被拒绝，实得 {r.status_code}"


def test_human_vs_human_is_allowed(client):
    """两个座位都是人 —— 同屏两人对下。引擎不关心，不该报错。"""
    r = client.post("/api/new", json={"players": [None, None]})
    assert r.status_code == 200
    assert r.json()["state"]["human_player"] == 0
    assert r.json()["labels"] == ["人类", "人类"]


def test_analysis_without_any_net_is_none(client):
    r = client.post("/api/new", json={"players": ["rule:greedy-area", "rule:corner-min"]})
    sid = r.json()["sid"]
    got = client.get(f"/api/analysis?sid={sid}").json()
    assert got["analysis"] is None and got["has_net"] is False


def test_legacy_new_game_params_still_work(client):
    """旧的 human_player + backend 形式要继续可用。"""
    r = client.post("/api/new", json={"human_player": 1, "backend": "rule:greedy-area"})
    assert r.status_code == 200
    st = r.json()["state"]
    assert st["human_player"] == 1
    assert st["players"] == ["rule:greedy-area", None]
