# 被文档引用的原始评测数据

这几份是 `docs/` 与 `reports/` 里点名引用的原始结果。

2026-08-25 收尾时 `runs/` 整个清空了（训练档与 replay 快照动辄 8 GB），
但**这些文件是文档结论的凭据、加起来只有 80 KB**，所以搬进版本库保留。
引用它们的地方原先写的是 `runs/<名字>.json`，现在在 `docs/data/<名字>.json`。

| 文件 | 内容 | 引用处 |
|---|---|---|
| `diag_search_ladder.json` | 36 对 × 400 局的搜索阶梯 —— 证明「见顶回退」是纯策略量尺的假象 | `docs/08`、`reports/BF16训练报告.md` |
| `rr30.json` | 435 对循环赛，架构对比的原始数据 | `reports/BF16训练报告.md` §7.3 |
| `cross_ab.json` | 28 对交叉臂对打，FP8/BF16 的 A/B 凭据 | `reports/FP8训练报告.md` §5 |
| `diag_diversity.json` | 512 局自博弈的开局多样性统计 | `docs/08` |
| `diag_churn.json` | 相邻档 checkpoint 之间的策略 churn | `docs/08` |

arena 的对局记录另存为 [`../附录-arena原始结果.md`](../附录-arena原始结果.md)（44 场）。
