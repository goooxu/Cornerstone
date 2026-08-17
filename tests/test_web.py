"""Web 试玩后端的选择逻辑。

只测不需要 GPU 的那部分：后端 ID 解析、可选项发现、LRU 淘汰。
这几条都有「静默走错分支」的特点 —— 解析错了不会报错，只会让人对着
一个**不是自己选的**模型下棋，而棋力差异肉眼分辨不出来。
"""

import importlib.util
import os
import re
import sys
import time

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
    """造一个假的 runs/ 目录树。

    网络后端一律来自 **`model/`（发布包）**，不是 `ckpt/`（训练档）——
    web 只做推理，而收割之后 ckpt/ 里只剩为续训留的一两份。
    所以这里连 `ckpt/` 都一并造出来、且**故意放进不同的步数**：
    一旦哪天发现逻辑又跑回去扫 ckpt/，下面的断言会立刻对不上。
    """
    def make(run: str, steps: list[int], decoys: list[int] | None = None):
        d = tmp_path / run / "model"
        d.mkdir(parents=True, exist_ok=True)
        for s in steps:
            (d / f"step{s:08d}.pt").write_bytes(b"x")
        c = tmp_path / run / "ckpt"
        c.mkdir(parents=True, exist_ok=True)
        for s in (decoys if decoys is not None else [999999]):
            (c / f"step{s:08d}.pt").write_bytes(b"x")
        (c / "latest").write_text("step00999999.pt")
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
    runs("v2-fp8", [1000, 2000])
    kind, path = server.resolve_backend("net:v2-fp8/step00001000.pt")
    assert kind == "net" and os.path.basename(path) == "step00001000.pt"


def test_latest_is_the_highest_step_release(runs):
    """`latest` 取发布包里步数最大的那份。

    发布包目录**没有 latest 指针文件** —— 那是训练档的东西（续训入口）。
    这里按步数排，不能按文件名字典序。
    """
    runs("v2-bf16", [500, 30000, 7000])          # 故意不按顺序创建
    _, path = server.resolve_backend("net:v2-bf16/latest")
    assert os.path.basename(path) == "step00030000.pt"
    assert os.sep + "model" + os.sep in path, "解析到了训练档目录"


def test_latest_ignores_the_training_checkpoint_dir(runs):
    """`ckpt/` 里那份步数更大的诱饵不能被选中。

    收割之后 `ckpt/` 里留的是终点档，步数往往**比中间的发布包都大**；
    逻辑要是跑回去扫它，界面上会出现一个没人导出过的条目，点下去就炸。
    """
    runs("v2-fp8", [1000, 2000], decoys=[999999])
    _, path = server.resolve_backend("net:v2-fp8/latest")
    assert os.path.basename(path) == "step00002000.pt"


def test_latest_resolves_afresh_each_call(runs):
    """「跟随训练」必须每次重新解析，否则选中那一刻就被钉死了。"""
    d = runs("v2-fp8", [1000])
    _, first = server.resolve_backend("net:v2-fp8/latest")
    (d / "step00009000.pt").write_bytes(b"x")
    _, second = server.resolve_backend("net:v2-fp8/latest")
    assert os.path.basename(first) == "step00001000.pt"
    assert os.path.basename(second) == "step00009000.pt"


def test_deleted_checkpoint_raises_readable_error(runs):
    runs("v2-fp8", [1000])
    with pytest.raises(ValueError, match="已不存在"):
        server.resolve_backend("net:v2-fp8/step00007777.pt")


def test_missing_run_raises(runs):
    runs("v2-fp8", [1000])
    with pytest.raises(ValueError, match="发布包"):
        server.resolve_backend("net:没这个跑/latest")


def test_path_traversal_is_contained(runs):
    """后端 ID 来自请求体，不能让它跳出发布包目录。"""
    runs("v2-fp8", [1000])
    with pytest.raises(ValueError):
        server.resolve_backend("net:v2-fp8/../../../etc/passwd")


# ------------------------------------------------------------------- 发现

def test_discover_lists_rules_and_checkpoints(runs):
    runs("v2-fp8", [1000, 2000])
    runs("v2-bf16", [3000])
    got = server.discover_backends()
    ids = [b["id"] for b in got]

    for name in server.RULE_BACKENDS:
        assert f"rule:{name}" in ids
    assert "net:v2-fp8/latest" in ids
    assert "net:v2-bf16/latest" in ids
    assert "net:v2-fp8/step00001000.pt" in ids
    # 每一项都能解析回去 —— 列出来却点不动是最难查的那种坏
    for b in got:
        server.resolve_backend(b["id"])


def test_discover_skips_empty_ckpt_dir(runs):
    runs("空跑", [])
    ids = [b["id"] for b in server.discover_backends()]
    assert not any(i.startswith("net:空跑") for i in ids)


def test_discover_orders_newest_first(runs):
    runs("v2-fp8", [1000, 5000, 3000])
    steps = [b["id"] for b in server.discover_backends()
             if b["id"].startswith("net:v2-fp8/step")]
    assert steps == ["net:v2-fp8/step00005000.pt",
                     "net:v2-fp8/step00003000.pt",
                     "net:v2-fp8/step00001000.pt"]


# --------------------------------------------------------------- 模型缓存

class _FakeNet:
    """替身，避免测试里真的 torch.load 一个 checkpoint。"""
    loaded: list[str] = []

    def __init__(self, path, device):
        self.path = path
        self.label = f"fake({os.path.basename(path)})"
        _FakeNet.loaded.append(path)


def test_pool_caches_and_evicts_lru(runs, monkeypatch):
    runs("r", [1, 2, 3, 4])
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


def test_sims_is_inert_for_rule_baselines():
    """模拟数对规则基线完全无效 —— 它不搜索。

    界面上因此把该座位的模拟数下拉置灰。这条测的是被置灰的那个前提本身 ——
    哪天 RuleBrain 真开始用 sims 了，就该把置灰去掉。
    """
    import cornerstone as cs
    brain = server.RuleBrain("greedy-mobility")
    board = cs.Board()
    for sims in (0, 1, 16, 800, 10**6):
        action, info = brain.choose(board, [], sims)
        assert board.is_legal(action)
        assert info is None
        assert brain.analyse([], sims) is None


def test_sim_choices_are_sane():
    # 界面直接列这些数字，要递增、要在允许范围内、默认值要在其中
    assert server.SIM_CHOICES == sorted(server.SIM_CHOICES)
    assert server.DEFAULT_SIMS in server.SIM_CHOICES
    assert all(0 <= n <= server.MAX_SIMS for n in server.SIM_CHOICES)
    # 界面上只留这三档
    assert server.SIM_CHOICES == [64, 256, 800]
    assert server.PURE_POLICY == 0


def test_pure_policy_is_not_offered_in_the_ui():
    """纯策略不进界面，但 API 仍然接受。

    它会给出**相反**的排名：同一批 checkpoint，纯策略口径下「训练越久越差」，
    带搜索重打则终点最强，直接头对头从 0.527 翻成 0.485（docs/08）。
    摆在试玩界面里等于请人用一把已知会读反的尺子比较模型，而界面上看不出异常。

    仍然保留 API 与 `_choose_by_policy` 那条代码路径 —— net_arena 和 diag
    工具要用它做研究口径，那里有文档说明。
    """
    assert server.PURE_POLICY not in server.SIM_CHOICES, "界面不该提供纯策略"
    assert min(server.SIM_CHOICES) > 0
    # 代码路径还在
    import inspect
    assert "_choose_by_policy" in inspect.getsource(server.NetBrain)


def test_pure_policy_is_accepted(client):
    """0 = 纯策略，是一档合法选项。"""
    r = client.post("/api/new", json={"players": [None, "rule:greedy-area"], "sims": [0, 64]})
    assert r.status_code == 200, r.text
    assert r.json()["state"]["sims"] == [0, 64]


def test_negative_sims_still_rejected(client):
    r = client.post("/api/new", json={"players": [None, "rule:greedy-area"], "sims": [-1, 64]})
    assert r.status_code == 400


def test_pure_policy_reads_priors_not_improved_policy():
    """纯策略必须取 root.prior 的 argmax。

    不能靠「把模拟数调很小」来近似 —— 恰恰相反，少量模拟是随机性**最大**
    的情形：改进策略里 sigma = (c_visit + max_n) * c_scale ≈ 51，
    一次随机采样到的 q 会盖过整个 log 先验（实测 sims=1 局局不同、
    sims=64 完全确定）。root.prior 则是展开时写一次就不再变的网络输出。
    """
    import inspect
    src = inspect.getsource(server.NetBrain._choose_by_policy)
    assert 'info["priors"]' in src, "要读 priors，不是 probs"
    assert "argmax" in src


def test_disabled_sims_shows_not_applicable():
    """规则基线那侧的模拟数要显示「—」。

    置灰但仍显示 64，会被读成「这局它搜了 64 次」—— 而它根本不搜索。
    """
    js = _code("app.js")
    assert "const NA = ''" in js
    assert "sel.value = NA" in js, "对手是规则基线时要把该侧显示成「—」"
    assert "dataset.last" in js, "切回网络时要恢复用户手选的值"
    assert re.search(r"v === NA \? S\.meta\.default_sims", js), \
        "「—」是给人看的，提交给后端仍要是合法整数"


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
    d = runs("r", [10])
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
        "players": ["rule:greedy-area", "rule:corner-min"], "sims": [64, 64]})
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


def test_undo_never_pops_into_the_injected_opening(client):
    """注入的开局是地板 —— 它不是谁走的手，退进去也没法重来同一个开局。"""
    r = client.post("/api/new", json={
        "players": ["rule:greedy-area", "rule:corner-min"], "opening_seed": 7})
    sid = r.json()["sid"]
    client.post("/api/ai", json={"sid": sid})
    for _ in range(6):                              # 使劲退
        got = client.post("/api/undo", json={"sid": sid}).json()["history"]
    assert len(got) == server.OPENING_PLIES, "悔棋不该退掉注入的开局"


def test_config_is_not_locked_by_the_injected_opening(client):
    """开局是注入的、不是走出来的，所以对局还没「开始」，配置仍可改。

    判据要是写成 `board.ply > 0`，注入之后对局一出生就 ply=2，
    配置当场锁死 —— 界面上表现为「刚点开始就再也改不了双方」。
    """
    r = client.post("/api/new", json={
        "players": ["rule:greedy-area", "rule:corner-min"],
        "sims": [16, 512], "opening_seed": 7})
    assert r.json()["state"]["ply"] == server.OPENING_PLIES
    sid = r.json()["sid"]
    rr = client.post("/api/backend", json={"sid": sid, "sims": [800, 32]})
    assert rr.status_code == 200, rr.text
    # 真走了一手之后才该锁上
    client.post("/api/ai", json={"sid": sid})
    rr = client.post("/api/backend", json={"sid": sid, "sims": [64, 64]})
    assert rr.status_code == 400


def test_undo_in_human_game_returns_to_human(client):
    r = client.post("/api/new", json={"players": [None, "rule:greedy-area"]})
    sid = r.json()["sid"]
    st = r.json()["state"]
    client.post("/api/move", json={"sid": sid, "action": st["legal_actions"][0]})
    client.post("/api/ai", json={"sid": sid})
    st = client.post("/api/undo", json={"sid": sid}).json()
    assert st["current_player"] == st["human_player"]
    assert st["history"] == []


def test_config_locked_once_the_game_starts(client):
    """对局一开跑就不能再改双方或模拟数。

    中途换引擎会让「这一局是谁对谁」没法陈述 —— 棋盘上一半的手是 A 走的、
    一半是 B 走的，最后那个比分不属于任何一对组合。
    拦截必须在服务端：界面把下拉置灰只是提示，请求照样能直接发过来。
    """
    r = client.post("/api/new", json={"players": ["rule:greedy-area", "rule:corner-min"]})
    sid = r.json()["sid"]

    # 还没落子，改得动
    ok = client.post("/api/backend", json={
        "sid": sid, "players": ["rule:greedy-mobility", "rule:corner-min"], "sims": [128, 128]})
    assert ok.status_code == 200, ok.text
    assert ok.json()["state"]["players"][0] == "rule:greedy-mobility"

    st = client.post("/api/ai", json={"sid": sid}).json()   # 落一手，锁上
    assert st["ply"] == 1
    for body in ({"players": ["rule:corner-min", "rule:corner-min"]},
                 {"sims": [16, 16]},
                 {"players": [None, "rule:corner-min"], "sims": [800, 800]}):
        rr = client.post("/api/backend", json={"sid": sid, **body})
        assert rr.status_code == 400, f"{body} 应被拒绝，实得 {rr.status_code}"
        assert "对局进行中" in rr.json()["detail"]

    # 配置没被改动
    now = client.get(f"/api/state?sid={sid}").json()
    assert now["players"] == ["rule:greedy-mobility", "rule:corner-min"]
    assert now["sims"] == [128, 128]


def test_config_unlocks_after_the_game_ends(client):
    """终局之后可以改 —— 那时改不会让任何一局的归属变模糊。"""
    r = client.post("/api/new", json={"players": ["rule:greedy-area", "rule:corner-min"]})
    sid = r.json()["sid"]
    st = r.json()["state"]
    n = 0
    while not st["terminal"] and n < 42:
        st = client.post("/api/ai", json={"sid": sid}).json()
        n += 1
    assert st["terminal"]

    rr = client.post("/api/backend", json={
        "sid": sid, "players": [None, "rule:greedy-mobility"], "sims": [256, 256]})
    assert rr.status_code == 200, rr.text
    assert rr.json()["state"]["human_player"] == 0


def test_frontend_locks_config_while_playing():
    """开局后配置要锁死，且看得出来锁了。"""
    js = _code("app.js")
    assert "seatSel(i).disabled = locked" in js
    assert "$('series-count').disabled = locked" in js, "连续对战局数也要锁"
    assert "lock-hint" in js


def test_per_seat_sims_are_independent(client):
    """两个座位各自的模拟数互不影响 —— 这正是「同一个网络多搜一倍值多少」
    这类对比要用的东西，混成一个值就测不了了。"""
    r = client.post("/api/new", json={
        "players": ["rule:greedy-area", "rule:corner-min"], "sims": [16, 512]})
    assert r.status_code == 200, r.text
    assert r.json()["state"]["sims"] == [16, 512]

    sid = r.json()["sid"]
    rr = client.post("/api/backend", json={"sid": sid, "sims": [800, 32]})
    assert rr.status_code == 200
    assert rr.json()["state"]["sims"] == [800, 32]


def test_sims_default_when_omitted(client):
    r = client.post("/api/new", json={"players": [None, "rule:greedy-area"]})
    assert r.json()["state"]["sims"] == [server.DEFAULT_SIMS, server.DEFAULT_SIMS]


def test_bad_sims_rejected(client):
    """上限不是审美问题：单局面搜索批大小恒为 1，模拟数线性折算成等待时间。

    两道关卡，返回码不同但都是拒绝：类型不对的由 Pydantic 挡在 422，
    类型对但越界的由 validate_sims 挡在 400。
    """
    # 注意 0 不在此列 —— 它是「纯策略」这一档，合法
    for bad in ([64], [64, 64, 64], [-1, 64], [64, -1],
                [64, server.MAX_SIMS + 1], ["快一点", 64]):
        r = client.post("/api/new", json={"players": [None, "rule:greedy-area"],
                                          "sims": bad})
        assert r.status_code in (400, 422), f"{bad} 应被拒绝，实得 {r.status_code}"


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


def test_rule_move_carries_no_analysis(client):
    r = client.post("/api/new", json={"players": ["rule:greedy-area", "rule:corner-min"]})
    sid = r.json()["sid"]
    got = client.post("/api/ai", json={"sid": sid}).json()
    assert got.get("analysis") is None


def test_on_demand_analysis_endpoint_is_gone(client):
    """「分析当前局面」已移除，对应的路由也要一并撤掉，别留着没人调的入口。"""
    r = client.post("/api/new", json={"players": [None, "rule:greedy-area"]})
    sid = r.json()["sid"]
    assert client.get(f"/api/analysis?sid={sid}").status_code == 404


def test_analysis_payload_has_no_heatmap():
    """热力图已移除。它是按「每个着法的概率加到它覆盖的每个格子上」算的，
    合法着法几百个时这是纯浪费 —— 前端不再画，服务端就不该再算、也不该再传。"""
    import cornerstone as cs
    info = {"actions": [0], "probs": [1.0], "visits": [1], "value": 0.0}
    got = server.analysis_payload(info, cs.Board())
    assert "heatmap" not in got
    assert set(got) == {"value", "win_rate", "top_moves"}


# ------------------------------------------------- 前端静态检查（没有 JS 运行时）
#
# 容器和开发机都没有 node，前端跑不了单测。下面几条用纯文本检查兜住
# 「一改就整页卡死」的那几类低级错误。

def _front(name: str) -> str:
    return open(os.path.join(REPO, "web", "static", name), encoding="utf-8").read()


def _code(name: str) -> str:
    """去掉注释后的源码。

    下面几条是文本匹配，注释里举的**反例**会被当成真代码匹配到 ——
    比如注释里写「不要写成 $('x').onclick = ...」，检查就会指着注释报错。
    """
    src = _front(name)
    src = re.sub(r"/\*[\s\S]*?\*/", "", src)
    return re.sub(r"^\s*//.*$", "", src, flags=re.M)


def _js_identifiers(src: str):
    """粗略地找出 app.js 里「用到但没声明」的标识符。

    没有 JS 运行时，这类错误只能等浏览器抛 ReferenceError 才发现 ——
    而它一抛就是整块逻辑停摆。真事：合并状态栏时把 `const noHuman = ...`
    删了，但后面还在用，于是一刷新就「初始局面加载失败：noHuman is not defined」。
    """
    s = re.sub(r"/\*[\s\S]*?\*/", "", src)
    s = re.sub(r"//.*$", "", s, flags=re.M)
    for q in ("`", "'", '"'):
        s = re.sub(q + r"(?:[^" + q + r"\\]|\\.)*" + q, q * 2, s)

    declared = set()
    for pat in (r"function\s+([\w$]+)", r"(?:const|let|var)\s+([\w$]+)",
                r",\s*([\w$]+)\s*=", r"catch\s*\(\s*([\w$]+)",
                r"for\s*\(\s*(?:const|let|var)\s+([\w$]+)"):
        declared |= set(re.findall(pat, s))
    for grp in re.findall(r"(?:const|let|var)\s*[\{\[]([^\}\]]*)[\}\]]", s):
        declared |= set(re.findall(r"[\w$]+", grp))
    for params in (re.findall(r"function\s*[\w$]*\s*\(([^)]*)\)", s)
                   + re.findall(r"\(([^)]*)\)\s*=>", s)):
        declared |= set(re.findall(r"[\w$]+", params))
    declared |= set(re.findall(r"([\w$]+)\s*=>", s))

    body = re.sub(r"\.\s*[\w$]+", "", s)      # 属性访问不算引用标识符
    body = re.sub(r"[\w$]+\s*:", "", body)     # 对象字面量的键
    used = set(re.findall(r"\b([a-zA-Z_$][\w$]*)\b", body))

    keywords = set("""await async break case catch class const continue default delete do else export
        extends finally for function if import in instanceof let new of return static super switch this
        throw try typeof var void while with yield true false null undefined""".split())
    globals_ = set("""window document console Math JSON Object Array String Number Boolean Promise Set
        Map setTimeout setInterval clearInterval clearTimeout fetch URL Blob Infinity NaN parseInt
        parseFloat requestAnimationFrame localStorage isNaN Error""".split())
    return sorted(used - declared - keywords - globals_)


def test_no_undefined_identifiers_in_frontend():
    """用到但没声明的标识符 = 浏览器一抛 ReferenceError，整块逻辑停摆。"""
    missing = _js_identifiers(_front("app.js"))
    assert not missing, f"这些标识符没找到声明：{missing}"


def test_the_undefined_check_actually_catches_things():
    """检查本身要有效，否则等于没测。"""
    assert "noHuman" in _js_identifiers("function f(st) { if (noHuman) return; }")


def test_every_element_id_used_by_js_exists_in_html():
    """`$('xxx')` 取不到就是 null，接着取属性即抛异常、整页停摆。

    删控件时最容易漏掉对应的 JS 引用，而症状离原因很远。
    """
    js, html = _code("app.js"), _front("index.html")
    have = set(re.findall(r'id="([^"]+)"', html))
    # 动态拼的 id（seat0/seat1、sims0/sims1）单独列出
    want = set(re.findall(r"\$\('([^']+)'\)", js)) | {"seat0", "seat1", "sims0", "sims1"}
    missing = sorted(want - have)
    assert not missing, f"app.js 引用了 index.html 里不存在的 id：{missing}"




def test_buttons_are_all_two_state():
    """按钮各有两态，状态由 phase 一处推出：
    开始/结束、暂停/恢复、连续对战/停止。

    以前是「新对局 / 悔棋 / 走一步 / 自动对战」四个各管一摊，
    组合起来有说不清的中间态（自动对战开着但轮到人该怎么办）。
    """
    html = _front("index.html")
    ids = re.findall(r'<button[^>]*id="([^"]+)"', html)
    assert ids == ["btn-start", "btn-pause"], f"实得 {ids}"
    for tag in re.findall(r"<button[^>]*>", html):
        assert "title=" in tag and "aria-label" in tag, f"纯图标按钮缺 title/aria-label：{tag}"
    # 纯图标：按钮里不该再有文字标签
    for tag in re.findall(r"<button[\s\S]*?</button>", html):
        txt = re.sub(r"<[^>]+>", "", tag).strip()
        assert txt == "", f"图标按钮里不该有文字：{txt!r}"
    # 每个按钮只放一个 svg：两个 svg 叠着靠 CSS 挑一个显示的话，
    # 样式一旦没生效（比如浏览器用了旧 CSS）就会两个一起冒出来
    for tag in re.findall(r"<button[\s\S]*?</button>", html):
        n = tag.count("<svg")
        assert n == 1, f"按钮里应只有一个 svg，实得 {n} 个：{tag[:80]}"
    js = _code("app.js")
    assert "const ICONS" in js and "function setIcon" in js, "图形应由 JS 按状态替换"


def test_phase_machine_drives_the_ui():
    """界面状态全部由 idle/playing/paused 这一个来源推出。"""
    js = _front("app.js")
    for name in ("startGame", "endGame", "togglePause", "pump", "syncControls"):
        assert f"function {name}" in js or f"async function {name}" in js, f"缺 {name}"
    assert "S.phase === 'idle' ? startGame() : endGame()" in js
    # 暂停时人和 AI 都不能落子
    assert "S.phase !== 'playing'" in js, "点击落子要先判断对局是否进行中"
    assert re.search(r"while \(S\.phase === 'playing'", js), "AI 驱动循环要能被暂停打断"


def test_js_errors_are_surfaced_on_the_page():
    """前端异常必须在页面上看得见。

    没有构建步骤也没有 JS 运行时可做单测，一个未捕获的异常表现出来
    就是「某个控件忽然没反应」—— 症状离原因极远，不打开 F12 根本
    不知道发生了什么。实际吃过好几次亏。
    """
    js, html, css = _front("app.js"), _front("index.html"), _front("style.css")
    assert 'id="js-error"' in html and ".js-error" in css
    assert "window.addEventListener('error'" in js
    assert "unhandledrejection" in js
    # 初始局面加载失败不能连累配置界面
    assert re.search(r"catch \(e\) \{\s*fatal\('初始局面加载失败", js), \
        "启动时建会话失败要单独兜住，别让配置界面一起废掉"


def test_bindings_survive_a_missing_element():
    """一个元素对不上，不能把整个模块带停。

    真事：浏览器拿着旧的 app.js 配新的 index.html，旧脚本去绑一个
    已经不存在的按钮，抛 `Cannot set properties of null`，
    于是**后面填充下拉框的启动代码根本不执行** ——
    表现是「选不了对战双方」，离原因隔了十万八千里。
    """
    js = _code("app.js")
    assert re.search(r"function on\(id, event, handler\)", js), "绑定要走统一的兜底入口"
    assert not re.search(r"\$\('[\w-]+'\)\.on(?:click|change)\s*=", js), \
        "不要直接 $('x').onclick = ...，缺元素会把整个模块带停"


def test_static_assets_are_versioned():
    """页面与脚本分开缓存，版本错配会让旧脚本去绑不存在的元素、
    旧样式让两个图标一起显示。

    光靠 no-cache 不够 —— 那只在浏览器**真的发请求**时才起作用，
    它认为手上那份还新鲜就可以连问都不问。所以资源 URL 要带版本号：
    内容一变 URL 就变，手上那份根本对不上。
    """
    srv = open(os.path.join(REPO, "web", "server.py"), encoding="utf-8").read()
    assert "def asset_version" in srv
    assert 'f"/static/{name}?v={v}"' in srv, "index.html 里的资源链接要加版本号"
    assert "revalidate_static" in srv and "no-cache" in srv


def test_asset_version_changes_with_content(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "STATIC", str(tmp_path))
    for n in ("app.js", "style.css", "index.html"):
        (tmp_path / n).write_text("v1")
    first = server.asset_version()
    os.utime(tmp_path / "app.js", (12345, 12345))
    assert server.asset_version() != first, "改了文件版本号就该变"








def test_piece_name_labels_are_gone():
    """棋子下面那两三个字母（I1/L4/P5）没有信息量，已删除。"""
    js = _front("app.js")
    assert "p.name" not in js, "棋子托盘不该再渲染棋子名"
    css = _front("style.css")
    assert ".tray .piece span" not in css, "对应的样式也该一并清掉"


def test_both_seats_get_a_readonly_tray_when_nobody_is_seated():
    """两个座位都是 AI 时，两侧各摆一块只读面板，用来看双方的棋子消耗。

    可交互的判据必须是「轮到这个座位、且这个座位是人」——
    只判「有人在座」的话，AI 思考期间人这侧仍会展开朝向圈，
    暗示可以落子，其实点了没用。
    """
    js = _front("app.js")
    assert "trayPlan" in js and "renderTrays" in js
    assert "readonly" in js, "只读托盘要有区分用的 class"
    assert re.search(r"interactive:\s*S\.phase === 'playing'\s*&&\s*st\.players\[seat\]\s*===\s*null"
                     r"\s*&&\s*!st\.terminal\s*&&\s*st\.current_player\s*===\s*seat", js), \
        "可交互 = 对局进行中 且 轮到该座位 且 该座位是人"
    assert re.search(r"human_player\s*>=\s*0[\s\S]{0,200}\[0,\s*1\]", js), \
        "没有人类座位时要摆两块面板"


# ----------------------------------------------------- 前端的哨兵值误用（静态检查）

def test_frontend_never_indexes_seat_arrays_by_human_player():
    """`human_player` 在 AI 对战下是 -1，拿它索引按座位分的数组会得到 undefined。

    这条真炸过：选了两个 AI 之后 `remaining[-1]` 是 undefined，
    再取 `[0]` 就抛「Cannot read properties of undefined」，整个界面卡住。
    前端没有 JS 运行时可测（容器里没有 node），所以用静态检查兜这一条。

    `COLORS[st.human_player]` 是允许的 —— 它们都在
    `current_player === human_player` 的判断里面，-1 时根本进不去。
    危险的是按座位分的数据数组，取到 undefined 之后还会继续往下取。
    """
    js = open(os.path.join(REPO, "web", "static", "app.js"), encoding="utf-8").read()
    bad = re.findall(r"\b(remaining|players|labels|anchors)\s*\[[^\]]*human_player[^\]]*\]", js)
    assert not bad, (f"这些按座位分的数组用 human_player 做了下标：{bad}；"
                     "AI 对战时它是 -1，应改用 viewSeat()")


def test_legacy_new_game_params_still_work(client):
    """旧的 human_player + backend 形式要继续可用。"""
    r = client.post("/api/new", json={"human_player": 1, "backend": "rule:greedy-area"})
    assert r.status_code == 200
    st = r.json()["state"]
    assert st["human_player"] == 1
    assert st["players"] == ["rule:greedy-area", None]


# ------------------------------------------------------------------- 连续对战

def test_series_counts_offered():
    """连续对战只是「开始对局」的一个选项，1 局就是普通单局。"""
    assert server.SERIES_COUNTS == [1, 100, 200, 400]
    assert server.SERIES_COUNTS[0] == 1


def test_no_background_match_endpoints():
    """连续对战在前端逐局跑，服务端不该再有后台批量对局那套。"""
    srv = open(os.path.join(REPO, "web", "server.py"), encoding="utf-8").read()
    for gone in ("MatchRunner", "/api/match", "MatchState"):
        assert gone not in srv, f"{gone} 应已移除"


def test_series_plays_games_one_after_another():
    """一局终局就开下一局，棋盘照常逐手显示 —— 不是后台批量。"""
    js = _code("app.js")
    assert "function recordResult" in js and "function renderSeries" in js
    assert re.search(r"if \(S\.state\.terminal\)[\s\S]{0,200}await newSession\(\)", js), \
        "终局后要自动开下一局"
    assert "S.series.played >= S.series.total" in js, "打满局数才收工"


def test_series_alternates_sides():
    """连续对战必须逐局交换先后手。

    不交换的话得分率量的是「甲执先 vs 乙执后」，而本项目先手优势极大
    （网络自博弈的先手胜率能到 0.97），那个数字和相对棋力基本无关。
    换边之后，战绩就必须按**引擎**记而不是按座位记 —— 座位每局都在换。
    """
    js = _code("app.js")
    assert "function orderedForThisGame" in js
    assert re.search(r"swap\)\s*\?\s*\{ players: \[p\[1\], p\[0\]\], sims: \[m\[1\], m\[0\]\] \}", js), \
        "换边时 players 和 sims 要一起换 —— 引擎带着自己的模拟数走"
    # 只断言「会翻」，**不绑在哪儿翻** —— 时机曾经写在 recordResult 里，
    # 而那正是「一局打完颜色对调」的成因（颜色从 swap 现算）。
    # 时机由 test_sides_do_not_swap_colors_at_game_end 单独守。
    assert re.search(r"swap = !\s*S?\.?series\.swap|swap = !s\.swap", js), "要逐局翻边"
    assert "function seatOfA" in js and "const a = seatOfA()" in js, "战绩要按引擎记"
    assert "orderedForThisGame()" in js and "o.players" in js, "开局要用换过边的顺序"


def test_series_reports_per_side_records():
    """执先胜/执后胜要**显示出来**，不只是记在内存里。

    换边之后这两个数才看得出先手优势有多大，也用来验证换边确实在起作用
    （两边执先次数应该各占一半）。
    """
    js = _code("app.js")
    assert "firstWins" in js and "secondWins" in js
    assert re.search(r"执先胜[\s\S]{0,120}s\.firstWins\[0\]", js), "执先胜要进表格"
    assert re.search(r"执后胜[\s\S]{0,120}s\.secondWins\[0\]", js), "执后胜要进表格"


def test_model_label_includes_simulation_count():
    """一方是模型时，光写 step 数不够，还要写这局搜了多少次。"""
    js = _code("app.js")
    assert "function simsText" in js and "function describe" in js
    assert re.search(r"backendId\.startsWith\('net:'\) \? label \+ ' · ' \+ simsText", js)
    assert "seatDesc(0)" in js and "seatDesc(1)" in js


def test_model_label_includes_precision():
    """标签要带精度。bf16 / fp8 / fp4 三组的 checkpoint 混在一个下拉里，
    光看 step 分不出是哪一条。

    精度取自 checkpoint 自带的 model_config，不是从跑名猜的 ——
    跑名可以随便起，模型配置不会骗人。
    """
    srv = open(os.path.join(REPO, "web", "server.py"), encoding="utf-8").read()
    assert 'getattr(model.cfg, "precision", "bf16")' in srv
    assert "· {self.precision}" in srv


def test_status_bar_is_one_line_and_doubles_as_the_end_banner():
    """棋盘下面只有一条：左边当前状态、右边上一手，终局时整条变结果横幅。

    三块东西各占一行（终局横幅 + 状态 + 上一手）白白吃竖向空间，
    而且终局横幅一出一进还会顶动棋盘。合成一条之后高度恒定。
    """
    js, html, css = _code("app.js"), _front("index.html"), _front("style.css")
    assert 'id="status-main"' in html and 'id="status-side"' in html
    for gone in ("endbanner", "lastmove"):
        assert gone not in html and gone not in js and gone not in css, f"{gone} 应已并入状态条"
    assert "drawEndBanner" not in js, "画布上那套浮层应已移除"
    assert "function renderStatus" in js
    assert ".status.over" in css, "终局态要有独立样式"


def test_rule_baseline_never_shows_a_simulation_count():
    """规则基线不搜索，界面上就不该给它写模拟数。

    真出现过：「上一手：规则基线 greedy-mobility · 64 次模拟 · 0s」——
    因为那一处自己拼了 simsText，没走 describe() 的判断。
    凡是要显示一方名字的地方，都必须经过 describe()。
    """
    js = _code("app.js")
    assert re.search(r"describe\(who, r\.ai_label, sims\)", js), \
        "「上一手」要走 describe()，别自己拼模拟数"
    # 把 simsText 的定义和 describe 的函数体挖掉，剩下的地方都不该再拼模拟数
    rest = re.sub(r"function simsText[\s\S]*?\n\}?\n", "", js, count=1)
    rest = re.sub(r"function describe\([\s\S]*?\n\}\n", "", rest, count=1)
    hits = [m.strip() for m in re.findall(r"[^\n]*simsText\([^\n]*", rest)]
    assert not hits, f"这些地方绕过了 describe()：{hits}"


def test_status_bar_height_is_fixed():
    """终局态字更大，高度必须钉死，否则棋盘会跟着上下跳。"""
    css = _front("style.css")
    assert re.search(r"\.status \{[^}]*min-height: 52px", css, re.S)


def test_status_shows_which_game_when_playing_a_series():
    """连打时只写「胜」不知道进行到第几局了。"""
    js = _code("app.js")
    assert re.search(r"第 \$\{s\.played\} / \$\{s\.total\} 局", js)


def test_engine_games_get_a_paired_random_opening():
    """引擎对战默认注入随机开局两手，且同一个种子给出逐手相同的开局。

    为什么必须有：自博弈的开局会塌到只剩一种首手（docs/08：第 8 万步之后
    512 局里每局同一手）。不注入的话两个网络打 400 局，量到的是「这条特定
    开局线上谁强」，纯策略档更是 400 局全同。三把尺子的 Elo 刻度就是按
    `play_pair(opening_plies=2)` 量的，web 不对齐就不是同一个口径。

    成对靠**同一对的两局传同一个种子**：开局逐手相同、先后手已翻。
    """
    two = ["rule:greedy-area", "rule:corner-min"]
    a = server.new_game_for_test(players=two, opening_seed=1234)
    b = server.new_game_for_test(players=two, opening_seed=1234)
    c = server.new_game_for_test(players=two, opening_seed=5678)
    assert len(a) == server.OPENING_PLIES, "引擎对战该走满开局手数"
    assert a == b, "同种子必须给出逐手相同的开局，否则成不了对"
    assert a != c, "不同种子该给出不同开局"


def test_single_game_gets_no_random_opening():
    """不传种子 = 不注入。单局的用途是**看棋**，而「网络自己开什么」恰恰是
    想看的东西（自博弈开局塌到只剩一种首手，见 docs/08）—— 随机掉前两手
    反而把它盖住了。单局也谈不上「双方各执先一次」，成对的理由只在多局成立。

    前端据此决定：`series.total <= 1` 时 opening_seed 传 null。
    """
    two = ["rule:greedy-area", "rule:corner-min"]
    assert server.new_game_for_test(players=two, opening_seed=None) == []


def test_client_only_seeds_openings_for_multi_game_series():
    """把「单局不随机」这条钉在前端那一行上。"""
    js = _code("app.js")
    m = re.search(r"function openingSeedForThisGame\(\)\s*\{(.*?)\n\}", js, re.S)
    assert m, "找不到 openingSeedForThisGame"
    assert "s.total <= 1" in m.group(1) and "return null" in m.group(1), \
        "单局必须传 null（不注入）"


def test_human_games_get_no_random_opening():
    """人机对局不注入 —— 随机掉人的前两手没有意义。"""
    got = server.new_game_for_test(players=[None, "rule:greedy-area"], opening_seed=1234)
    assert got == [], "有人参与的局不该被注入开局"


def test_opening_plies_matches_the_arena_protocol():
    """web 的开局手数必须和 arena 的默认值一致，否则两边刻度对不上。"""
    import inspect
    from cornerstone import arena
    sig = inspect.signature(arena.play_pair)
    assert "opening_plies" in sig.parameters
    from cornerstone import evaluate
    assert inspect.signature(evaluate.evaluate_vs_baseline).parameters[
        "opening_plies"].default == server.OPENING_PLIES


def test_sides_do_not_swap_colors_at_game_end():
    """翻边必须在**开下一局**时做，不能在终局那一刻做。

    颜色是从 swap 现算的（seatColor -> engineOfSeat）。在 recordResult 里翻的话，
    刚下完、还摆在屏幕上的这一局会当场被按下一局的座位重画 —— 表现就是
    「一局打完，双方棋子颜色对调」。实际发生过。

    顺带守住成对开局：newSession 里必须**先翻 swap 再取种子**，
    否则 `!swap` 判「新一对开始」会错位，成对口径就破了。
    """
    js = _code("app.js")
    rec = js[js.index("function recordResult"):]
    rec = rec[:rec.index("\nfunction ")]
    assert "swap = !" not in rec, "终局时翻边会让刚下完那局的颜色对调"

    ns = js[js.index("async function newSession"):]
    ns = ns[:ns.index("\n}")]
    assert "swap = !" in ns, "换边要在开下一局时做"
    assert ns.index("swap = !") < ns.index("openingSeedForThisGame"), \
        "必须先翻 swap 再取开局种子，否则成对判据错位"


def test_placement_first_interaction_has_no_overlay():
    """选子改成「先点棋盘格、右侧列候选」，**不能再有浮层**。

    原先是「先挑棋子、再展开朝向圈选摆法」。圆盘是个直径近 300px 的覆盖物，
    而托盘每排只有 56px —— 展开必然压住上下两排。由此派生出三个真实故障：
    飞向按钮时圈被邻居抢走、吃事件后指针被困住、opacity:0 的隐形按钮铺满托盘
    导致某些格根本点不开。改了三次都没治本，因为根子是「浮层盖住邻居」。

    现在一律平铺：点格 -> 右侧列出会盖住它的棋子 -> 点开某枚列具体摆法。
    实测点开局起始格有 414 个候选（首手必须盖住它），所以**必须分两层**：
    按棋子分组后第一层最多 21 个、第二层中位数只有 1~4。
    """
    js, css = _code("app.js"), _front("style.css")
    for gone in ("orientRing", "ringPiece", "scheduleRing", "holdRing"):
        assert gone not in js, f"{gone} 应随圆盘一并移除"
    assert ".ring" not in css and "ori-btn" not in css, "圆盘样式应已删除"

    assert "function candidatesAt" in js, "缺少候选计算"
    assert "S.pickCell = cell" in js, "点棋盘要选位置"


def test_only_anchor_cells_are_clickable():
    """只有**当前方的角点**能点 —— 那是 Blokus 的落子规则：新棋子必须斜接
    自己已有的棋子。开局角点只有一个（起始格）。

    角点由**服务端**给（`state.anchors`，来自引擎的 `Board.anchor_cells()`）。
    前端不重算 —— 合法性判断只有 C++ 引擎那一份，训练、评测、试玩共用，
    在 JS 里抄一份迟早两边对不上（见 app.js 开头那条硬规则）。
    """
    js = _code("app.js")
    fn = js[js.index("function playableCells"):]
    fn = fn[:fn.index("\n}")]
    assert "st.anchors" in fn, "可点格要用服务端给的角点"
    # 不能在前端重算角点
    for reinvent in ("对角", "diagonal", "dr * dc", "Math.abs(dr) === 1 && Math.abs(dc) === 1"):
        assert reinvent not in fn, f"疑似在前端重算规则：{reinvent}"
    assert "isPlayable(cell)" in js, "点非角点不该进入选位置状态"
    # 放不下东西的角点要滤掉：实测第 20 手时 12 个角点里有 4 个没有候选
    assert "candidatesAt(r, c).size > 0" in fn, "没有候选的角点不该可点"


def test_state_payload_carries_anchors():
    """服务端必须把角点传给前端，否则前端只能自己重算规则。"""
    import cornerstone as cs
    b = cs.Board()
    assert b.anchor_cells(0) == [(4, 4)], "开局角点应只有起始格"
    assert '"anchors"' in open(os.path.join(REPO, "web", "server.py"),
                               encoding="utf-8").read()


def test_css_braces_and_comments_are_balanced():
    """CSS 的大括号与注释必须配平。

    **grep 看不出这个错。** 曾经用 `s.index("/* 朝向圈")` 定位要替换的段落，
    结果匹配到的是 `.tray .piece` 规则里的**行内注释**「/* 朝向圈以它为定位原点 */」，
    于是新内容插进了那条规则中间、把它的 `}` 挤掉。后面所有规则一并失效，
    候选面板的 flex 布局没生效、变成一列 —— 而 `grep '.cand'` 照样能匹配到，
    因为它分不清文本在不在注释里。
    """
    css = _front("style.css")
    assert css.count("{") == css.count("}"), \
        f"大括号不配平：{{ {css.count('{')} 个、}} {css.count('}')} 个"
    assert css.count("/*") == css.count("*/"), "注释不配平"
    # **嵌套深度不能超过 1** —— 这条才是真正抓得住那个故障的判据。
    # 本项目的 CSS 没有嵌套规则也没有 @media，一条规则若没闭合就开下一条，
    # 深度会变成 2。光看括号总数是抓不到的：被挤掉的 `}` 往往被后面某条
    # 规则的 `}` 凑平，总数照样配平。
    flat = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    depth = worst = 0
    for ch in flat:
        if ch == "{":
            depth += 1
            worst = max(worst, depth)
        elif ch == "}":
            depth -= 1
    assert worst <= 1, f"出现了未闭合就开下一条的规则（最大嵌套深度 {worst}）"
    assert depth == 0, "大括号未回到 0"


def test_candidate_levels_collapse_when_few():
    """层次是 棋子 -> 朝向 -> 位置，但两处**自动折叠**。

    实测（角点口径）：
      开局    棋子 21、朝向中位 4、位置中位 5   -> 棋子x朝向 91 项
      第 6 手 棋子 18、朝向中位 2、位置中位 1   -> 44 项
      第 20 手 棋子 7、朝向中位 1、位置中位 1   -> **8 项**

    所以：项数少时把「选棋子」和「选朝向」合成一步（残局中位 8 项，
    而残局正是最不想多点一下的时候）；「位置」那一层中位数就是 1，
    根本不该占一次点击，改由鼠标在棋盘上滑动决定。
    """
    js = _code("app.js")
    assert "MERGE_LIMIT" in js and "pairCount" in js, "要按项数决定合不合并"
    assert "function pickPlacement" in js, "位置要由鼠标决定"
    # 同一朝向只有一处可放时直接落子，不让人再滑一次
    assert re.search(r"actions\.length === 1.*play\(actions\[0\]\)", js), \
        "只有一处可放时应直接落子"
    # 鼠标移动要能切换位置
    mm = js[js.index("CV.addEventListener('mousemove'"):]
    mm = mm[:mm.index("});")]
    assert "pickPlacement" in mm, "鼠标在棋盘上移动时要跟着切位置"


def test_candidate_preview_is_immediate():
    """悬停候选 -> 棋盘上立刻出现预览，**不许有任何延迟**。

    圆盘那一版曾经为了防「飞向按钮途中被邻居抢走」加过 150ms 的悬停意图延迟。
    现在没有浮层、没有可抢的邻居，那个理由不存在了；加延迟只会让预览发黏。
    """
    js = _code("app.js")
    fn = js[js.index("function renderCandidates"):]
    fn = fn[:fn.index("\nfunction ")]
    assert "onmouseenter" in fn and "draw();" in fn, "悬停要同步设预览并重画"
    assert "setTimeout" not in fn, "悬停路径里不许有 setTimeout"
    # 预览必须是同步赋值 + 立刻重画，不能排队
    assert re.search(r"onmouseenter = \(\) => \{ S\.preview = [^;]+; draw\(\); \}", fn), \
        "悬停应立刻设预览并重画"


def test_playable_cells_cache_is_keyed_on_the_position():
    """可点格缓存必须带上手数。

    只用 sid 做键的话，一局之内每走一手合法集合就变、而 sid 不变，
    缓存会发馊 —— 棋盘上画出的是上一手的可点格，而且不报任何错。
    """
    js = _code("app.js")
    fn = js[js.index("function playableCells"):]
    fn = fn[:fn.index("\n}")]
    assert "key" in fn and "return" in fn, \
        "playableCells 应有缓存 —— draw() 被 mousemove 频繁调用，而它要对每个角点算候选"
    assert "st.ply" in fn, "缓存键要带手数，否则走一手之后就馊了"


def test_background_is_css_only():
    """背景用 CSS 画，不引外部图片 —— 没有构建步骤，也不该多一个要部署的文件。"""
    css, html = _front("style.css"), _front("index.html")
    assert "radial-gradient" in css and "background-attachment: fixed" in css
    assert "url(" not in css, "不引任何图片，包括内联 data URI —— 图案和棋子太像"
    assert "<img" not in html
    # 面板要半透明，否则整页被不透明卡片盖住，等于没有背景
    assert "backdrop-filter" in css, "面板要半透明，否则背景透不出来"
    assert re.search(r"background: rgba\([\d, .]+\)", css)


def test_series_table_truncates_long_engine_names():
    """引擎名很长（「CornerNet 14.2M · FP8 · step 140652 · 256 次模拟」），
    不截断会把整个界面撑宽。

    表格单元格要省略号截断，必须 table-layout: fixed + max-width: 0 —— 
    只写 text-overflow 是不生效的，单元格会按内容撑开。
    """
    css, js = _front("style.css"), _code("app.js")
    assert "table-layout: fixed" in css
    assert re.search(r"\.mstat th, \.mstat td \{[^}]*max-width: 0[^}]*text-overflow: ellipsis",
                     css, re.S), "截断三件套要齐"
    # 完整内容要留在 title 里，截断不能丢信息
    assert re.search(r"title=\"' \+ esc\(s\.names\[0\]\)", js)
    assert "function esc" in js, "往 HTML 属性里塞文本要转义"


def test_board_column_width_is_pinned():
    """棋盘那一栏宽度要钉死，否则里面最宽的内容会把整个界面拉宽。"""
    css = _front("style.css")
    assert re.search(r"\.board-col \{[^}]*width: 722px", css), \
        "canvas 700 + padding 20 + border 2"


def test_one_place_decides_how_a_seat_is_described():
    """同一个对手在徽标、比分格、战绩表里必须写法一致。

    真出现过不一致：徽标用服务端返回的 label（带解析后的真实 step），
    比分格和战绩表各自去下拉框取选项文字
    （「最新（跟随训练，当前 step …）」）—— 三处三个样。
    """
    js = _code("app.js")
    assert "function describeSeat" in js
    assert "seatLabelForSeat" not in js, "旧的下拉框取名路径应已移除"
    # 三处显示都得走 describeSeat
    assert js.count("describeSeat(st, ") >= 4
    assert re.search(r"\$\('ai-name'\)\.innerHTML =[\s\S]{0,140}describeSeat", js)
    assert re.search(r"S\.series\.names\[0\] = describeSeat", js)
    assert re.search(r"const desc = describeSeat\(st, seat\)", js)


def test_background_glows_are_inside_the_viewport():
    """色晕的圆心必须落在视口之内。

    第一版放在 6%/-14%、106%/4%、48%/120%，全在屏幕外，
    进到画面里的只剩最淡的尾巴 —— 等于没画。
    """
    css = _front("style.css")
    body = css[css.index("body {"):css.index("}", css.index("body {"))]
    centers = re.findall(r"radial-gradient\([^)]*?at\s+(-?\d+)%\s+(-?\d+)%", body)
    assert len(centers) >= 3, f"应有三团色晕，实得 {centers}"
    for x, y in centers:
        assert 0 <= int(x) <= 100, f"圆心横坐标 {x}% 在视口外"
        assert 0 <= int(y) <= 100, f"圆心纵坐标 {y}% 在视口外"


def test_tray_is_seven_by_three():
    """21 枚正好 7 列 3 排，比 6 列紧凑，也不留半排空格。"""
    css = _front("style.css")
    assert "repeat(7, 1fr)" in css


def test_board_only_ever_paints_the_two_engine_colors():
    """棋盘上不该冒出第三种颜色。

    top 着法列表原先有个悬停高亮，把落点涂成靛蓝 #818cf8aa —— 一个谁都不是
    的颜色；而且它「先 draw() 再往画布上糊一层」，任何重绘都会抹掉它，
    连打时列表每手重建还会在光标下反复自触发。这个功能已整个删掉。
    """
    js = _code("app.js")
    assert "#818cf8" not in js, "棋盘上不该有第三种颜色"
    for gone in ("function highlight", "S.hoverMove", "setLineDash"):
        assert gone not in js, f"{gone} 应已随悬停高亮一起移除"
    # 原先这里还拦 `onmouseenter` 整个词，当作「高亮删干净了」的代理。
    # **范围太宽**：棋子托盘的朝向圈用悬停意图控制展开（见 style.css 的 .ring），
    # 那是另一件事，正当用途。改成只盯 top 着法列表这一段。
    tm = js[js.index("function renderAnalysis()"):]
    tm = tm[:tm.index("\nfunction ")]
    for gone in ("onmouseenter", "onmouseleave", "onmouseover"):
        assert gone not in tm, f"top 着法列表不该再挂 {gone}"
    # 画棋盘时允许的颜色：双方色走 seatColor()，其余只有这几个
    body = js[js.index("function draw()"):js.index("function verdict(")]
    hexes = set(re.findall(r"#[0-9a-fA-F]{3,8}", body))
    allowed = {"#0e1118", "#2a3143", "#f8717166", "#f87171", "#ffffff88"}
    assert hexes <= allowed, f"draw() 里出现了意料之外的颜色：{hexes - allowed}"


def test_analysis_payload_drops_unused_cells():
    """前端不再用 top 着法的落点坐标了，服务端就不该再算、再传。"""
    import cornerstone as cs
    info = {"actions": [0], "probs": [1.0], "visits": [1], "value": 0.0}
    got = server.analysis_payload(info, cs.Board())
    assert set(got["top_moves"][0]) == {"action", "piece", "prob", "visits"}


def test_colors_are_bound_to_engines_not_seats():
    """颜色绑对战双方，不绑先后手。

    连续对战逐局换边，若按座位上色，同一个引擎会一局一个颜色，
    根本看不出谁是谁。
    """
    js = _code("app.js")
    assert "function engineOfSeat" in js and "function seatColor" in js
    assert re.search(r"return \(S\.series && S\.series\.swap\) \? 1 - seat : seat;", js)
    # 画棋盘时不能再直接按座位索引颜色
    for bad in ("COLORS[v]", "COLORS[p]", "COLORS[st.current_player]", "COLORS[seat]", "COLORS[me]"):
        assert bad not in js, f"{bad} 应改用 seatColor()"


def test_no_first_second_wording_outside_seat_pickers():
    """除了选择对战双方那两行，界面上不该再出现「先手/后手」——
    自动换边之后它已经不指代任何一方了，改用颜色。
    """
    html = _front("index.html")
    # 只允许出现在座位选择的两个 label 上
    assert html.count("先手") == 1 and html.count("后手") == 1
    for tag in re.findall(r"<label[^>]*>[^<]*", html):
        pass
    js = _code("app.js")
    # 去掉行尾注释后，面向界面的字符串里不该有这两个词
    code = re.sub(r"//.*$", "", js, flags=re.M)
    for m in re.finditer(r"'[^']*'|`[^`]*`", code):
        assert "先手" not in m.group(0) and "后手" not in m.group(0), \
            f"界面字符串里仍有先手/后手：{m.group(0)[:60]}"


def test_series_pauses_between_games():
    """连打时每局之间停一下，否则棋盘会「啪」地跳到下一局的空盘。"""
    js = _code("app.js")
    assert "BETWEEN_GAMES_MS = 2000" in js
    assert re.search(r"await sleep\(BETWEEN_GAMES_MS\);\s*\n\s*if \(S\.phase !== 'playing'\) break;", js), \
        "停顿期间被暂停/结束要能退出"


def test_series_card_sits_under_the_board():
    """战绩放棋盘下面：连打时眼睛在棋盘上，统计跟着一起看才顺。"""
    html = _front("index.html")
    board = html.index('id="board"')
    card = html.index('id="series-card"')
    side = html.index('class="side-col"')
    assert board < card < side, "战绩卡应在棋盘那一栏里、侧栏之前"


def test_series_counts_each_game_once():
    """applyState 会被调用很多次，战绩必须按局去重，不能一局记多次。"""
    js = _code("app.js")
    assert "s.lastSid === S.sid" in js, "要按 sid 去重"
    assert "s.lastSid = S.sid" in js


def test_series_row_hidden_when_not_applicable():
    """条件不满足就把整行藏掉，而不是摆个灰控件再配一句「为什么不能用」。

    后者只是把界面弄复杂，并没有多给出信息。
    """
    js, html, css = _code("app.js"), _front("index.html"), _front("style.css")
    assert 'id="series-row"' in html
    assert "$('series-row').classList.toggle('gone', !bothAi)" in js
    assert ".row.gone" in css, "整行隐藏要有对应样式"
    assert "series-hint" not in js and "series-hint" not in html, "解释性提示应已删除"
