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
    # 界面上只留这四档
    assert server.SIM_CHOICES == [0, 64, 256, 800]
    assert server.PURE_POLICY == 0


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
    assert "s.swap = !s.swap" in js, "每局结束要翻转"
    assert "function seatOfA" in js and "const a = seatOfA()" in js, "战绩要按引擎记"
    assert "orderedForThisGame()" in js and "o.players" in js, "开局要用换过边的顺序"


def test_series_reports_per_side_records():
    """执先胜/执后胜要分开报 —— 换边之后这两个数才看得出先手优势有多大。"""
    js = _code("app.js")
    assert "firstWins" in js and "secondWins" in js


def test_model_label_includes_simulation_count():
    """一方是模型时，光写 step 数不够，还要写这局搜了多少次。"""
    js = _code("app.js")
    assert "function simsText" in js and "function describe" in js
    assert re.search(r"backendId\.startsWith\('net:'\) \? label \+ ' · ' \+ simsText", js)
    assert "seatDesc(0)" in js and "seatDesc(1)" in js


def test_model_label_includes_precision():
    """标签要带精度。ab-fp8 / ab-bf16 的 checkpoint 混在一个下拉里，
    光看 step 分不出是哪一条。

    精度取自 checkpoint 自带的 model_config，不是从跑名猜的 ——
    跑名可以随便起，模型配置不会骗人。
    """
    srv = open(os.path.join(REPO, "web", "server.py"), encoding="utf-8").read()
    assert 'self.fp8 = bool(getattr(model.cfg, "fp8", False))' in srv
    assert 'self.precision = "FP8" if self.fp8 else "BF16"' in srv
    assert "· {self.precision}" in srv


def test_game_over_banner_does_not_cover_the_board():
    """终局提示要醒目，但**不能盖住盘面** —— 一局刚结束正是要看盘的时候。

    所以它是棋盘上方的独立 DOM 元素，不是画在画布上的浮层。
    """
    js, html, css = _code("app.js"), _front("index.html"), _front("style.css")
    assert 'id="endbanner"' in html and ".endbanner" in css
    assert "function renderEndBanner" in js
    assert "drawEndBanner" not in js, "画布上那套浮层应已移除"
    # 横幅必须在棋盘容器之前（上方），不在它里面
    assert html.index('id="endbanner"') < html.index('id="board-wrap"')


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
