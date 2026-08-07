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
