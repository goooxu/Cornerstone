# 附录：arena 原始对局记录

`docs/09`、`docs/04`、`docs/06` 里的每一个 Elo 数字都来自这里。

原始的 `runs/arena_*.json` 与日志在 2026-08-25 收尾时随训练数据一起清掉了 ——
那些跑动辄 8 GB，而结论已经落进文档。**但对局记录本身只有几十 KB，
删掉的话文档里的数字就再也无法复核**，所以在删之前压成了这一份存进版本库。

`rate` 是**前者**的得分率，`Elo` 由 `400*log10(w/(1-w))` 换算。
每对 1200 局的标准误是 0.0144（约 10 Elo），800 局是 0.0177（约 12 Elo）。

**两条读数提醒**（都是实测踩出来的，见 `docs/09` 第四节）：

1. **场内的 ± 是低估的。** 同一个模型在不同场次之间跳过 9.3 Elo，
   而它那一场自己报的 ± 只有 8.5。
2. **不要跨场相减。** 要比十几 Elo 的差别，两个模型必须在同一场里、
   而且最好共享训练历史。

## `arena_arch.json`

每对 1200 局，64 次模拟

| 参赛者 | Elo | ± | 得分率 | 局数 |
|---|---:|---:|---:|---:|
| `poly-111k.pt` | +0.0 | 0.0 | 0.741 | 2400 |
| `poly-40k.pt` | -127.0 | 12.2 | 0.496 | 2400 |
| `qwen-40k.pt` | -247.7 | 11.7 | 0.263 | 2400 |

```
  poly-111k.pt           vs poly-40k.pt              1200 局  rate 0.671   +123.7 Elo
  poly-111k.pt           vs qwen-40k.pt              1200 局  rate 0.811   +252.8 Elo
  poly-40k.pt            vs qwen-40k.pt              1200 局  rate 0.663   +117.5 Elo
```

## `arena_attn.json`

每对 1200 局，64 次模拟

| 参赛者 | Elo | ± | 得分率 | 局数 |
|---|---:|---:|---:|---:|
| `attn4.pt` | +830.5 | 54.7 | 0.769 | 2400 |
| `attn16.pt` | +798.3 | 55.2 | 0.722 | 2400 |
| `greedy-mobility` | +0.0 | 0.0 | 0.009 | 2400 |

```
  greedy-mobility        vs attn4.pt                 1200 局  rate 0.005   -934.7 Elo
  greedy-mobility        vs attn16.pt                1200 局  rate 0.013   -753.3 Elo
  attn4.pt               vs attn16.pt                1200 局  rate 0.543    +29.9 Elo
```

## `arena_attn_lr.json`

每对 1200 局，64 次模拟

| 参赛者 | Elo | ± | 得分率 | 局数 |
|---|---:|---:|---:|---:|
| `attn4.pt` | +0.0 | 0.0 | 0.537 | 2400 |
| `attn16.pt` | -22.3 | 10.9 | 0.489 | 2400 |
| `attn16-lr13.pt` | -29.6 | 10.6 | 0.474 | 2400 |

```
  attn4.pt               vs attn16.pt                1200 局  rate 0.534    +23.5 Elo
  attn4.pt               vs attn16-lr13.pt           1200 局  rate 0.541    +28.4 Elo
  attn16.pt              vs attn16-lr13.pt           1200 局  rate 0.512     +8.4 Elo
```

## `arena_budget-40k.json`

每对 1200 局，64 次模拟

| 参赛者 | Elo | ± | 得分率 | 局数 |
|---|---:|---:|---:|---:|
| `poly-111k.pt` | +0.0 | 0.0 | 0.671 | 1200 |
| `poly-40k.pt` | -123.6 | 14.3 | 0.329 | 1200 |

```
  poly-111k.pt           vs poly-40k.pt              1200 局  rate 0.671   +123.7 Elo
```

## `arena_curve.json`

每对 1200 局，64 次模拟

| 参赛者 | Elo | ± | 得分率 | 局数 |
|---|---:|---:|---:|---:|
| `d4-111k.pt` | +63.4 | 9.9 | 0.541 | 3600 |
| `d4-80k.pt` | +52.9 | 10.5 | 0.521 | 3600 |
| `d4-60k.pt` | +51.5 | 9.6 | 0.518 | 3600 |
| `base.pt` | +0.0 | 0.0 | 0.420 | 3600 |

```
  base.pt                vs d4-60k.pt                1200 局  rate 0.440    -41.6 Elo
  base.pt                vs d4-80k.pt                1200 局  rate 0.417    -58.5 Elo
  base.pt                vs d4-111k.pt               1200 局  rate 0.403    -68.0 Elo
  d4-60k.pt              vs d4-80k.pt                1200 局  rate 0.513     +9.0 Elo
  d4-60k.pt              vs d4-111k.pt               1200 局  rate 0.482    -12.5 Elo
  d4-80k.pt              vs d4-111k.pt               1200 局  rate 0.492     -5.5 Elo
```

## `arena_curve2.json`

每对 1200 局，64 次模拟，锚点 `base.pt`

| 参赛者 | Elo | ± | 得分率 | 局数 |
|---|---:|---:|---:|---:|
| `d4-a144.pt` | +87.2 | 9.1 | 0.556 | 3600 |
| `d4-111k.pt` | +72.2 | 10.4 | 0.528 | 3600 |
| `d4-a111.pt` | +70.3 | 9.8 | 0.524 | 3600 |
| `base.pt` | +0.0 | 0.0 | 0.392 | 3600 |

```
  base.pt                vs d4-a111.pt               1200 局  rate 0.399    -71.3 Elo
  base.pt                vs d4-a144.pt               1200 局  rate 0.372    -90.6 Elo
  base.pt                vs d4-111k.pt               1200 局  rate 0.403    -68.0 Elo
  d4-a111.pt             vs d4-a144.pt               1200 局  rate 0.478    -15.1 Elo
  d4-a111.pt             vs d4-111k.pt               1200 局  rate 0.493     -4.6 Elo
  d4-a144.pt             vs d4-111k.pt               1200 局  rate 0.520    +13.6 Elo
```

## `arena_curve3.json`

每对 1200 局，64 次模拟，锚点 `base.pt`

| 参赛者 | Elo | ± | 得分率 | 局数 |
|---|---:|---:|---:|---:|
| `a211.pt` | +85.3 | 8.6 | 0.540 | 6000 |
| `d4-a144.pt` | +84.6 | 8.3 | 0.539 | 6000 |
| `a189.pt` | +71.4 | 7.8 | 0.516 | 6000 |
| `d4-a111.pt` | +67.8 | 7.6 | 0.510 | 6000 |
| `d4-111k.pt` | +62.9 | 8.5 | 0.501 | 6000 |
| `base.pt` | +0.0 | 0.0 | 0.395 | 6000 |

```
  base.pt                vs d4-a111.pt               1200 局  rate 0.399    -71.3 Elo
  base.pt                vs d4-a144.pt               1200 局  rate 0.372    -90.6 Elo
  base.pt                vs a189.pt                  1200 局  rate 0.405    -67.1 Elo
  base.pt                vs a211.pt                  1200 局  rate 0.390    -77.4 Elo
  base.pt                vs d4-111k.pt               1200 局  rate 0.406    -65.9 Elo
  d4-a111.pt             vs d4-a144.pt               1200 局  rate 0.482    -12.7 Elo
  d4-a111.pt             vs a189.pt                  1200 局  rate 0.492     -5.8 Elo
  d4-a111.pt             vs a211.pt                  1200 局  rate 0.451    -34.0 Elo
  d4-a111.pt             vs d4-111k.pt               1200 局  rate 0.523    +16.2 Elo
  d4-a144.pt             vs a189.pt                  1200 局  rate 0.513     +9.0 Elo
  d4-a144.pt             vs a211.pt                  1200 局  rate 0.505     +3.2 Elo
  d4-a144.pt             vs d4-111k.pt               1200 局  rate 0.530    +20.6 Elo
  a189.pt                vs a211.pt                  1200 局  rate 0.484    -11.3 Elo
  a189.pt                vs d4-111k.pt               1200 局  rate 0.505     +3.8 Elo
  a211.pt                vs d4-111k.pt               1200 局  rate 0.529    +20.0 Elo
```

## `arena_data.json`

每对 1200 局，64 次模拟

| 参赛者 | Elo | ± | 得分率 | 局数 |
|---|---:|---:|---:|---:|
| `data4.pt` | +61.6 | 11.6 | 0.599 | 2400 |
| `base.pt` | +0.0 | 0.0 | 0.468 | 2400 |
| `data2.pt` | -15.8 | 10.9 | 0.434 | 2400 |

```
  base.pt                vs data2.pt                 1200 局  rate 0.528    +19.4 Elo
  base.pt                vs data4.pt                 1200 局  rate 0.407    -65.3 Elo
  data2.pt               vs data4.pt                 1200 局  rate 0.395    -73.8 Elo
```

## `arena_data4.json`

每对 1200 局，64 次模拟

| 参赛者 | Elo | ± | 得分率 | 局数 |
|---|---:|---:|---:|---:|
| `data4.pt` | +956.5 | 74.2 | 0.796 | 2400 |
| `base.pt` | +889.3 | 74.1 | 0.699 | 2400 |
| `greedy-mobility` | +0.0 | 0.0 | 0.005 | 2400 |

```
  greedy-mobility        vs base.pt                  1200 局  rate 0.005   -934.7 Elo
  greedy-mobility        vs data4.pt                 1200 局  rate 0.005   -934.7 Elo
  base.pt                vs data4.pt                 1200 局  rate 0.403    -68.0 Elo
```

## `arena_data4b.json`

每对 1200 局，64 次模拟

| 参赛者 | Elo | ± | 得分率 | 局数 |
|---|---:|---:|---:|---:|
| `data4.pt` | +70.7 | 14.9 | 0.600 | 1200 |
| `base.pt` | +0.0 | 0.0 | 0.400 | 1200 |

```
  base.pt                vs data4.pt                 1200 局  rate 0.400    -70.7 Elo
```

## `arena_fp4-width.json`

每对 1200 局，64 次模拟

| 参赛者 | Elo | ± | 得分率 | 局数 |
|---|---:|---:|---:|---:|
| `d384-lr13.pt` | +21.9 | 11.0 | 0.517 | 2400 |
| `d512-lr10.pt` | +19.8 | 11.2 | 0.513 | 2400 |
| `d256-lr20.pt` | +0.0 | 0.0 | 0.470 | 2400 |

```
  d256-lr20.pt           vs d384-lr13.pt             1200 局  rate 0.475    -17.7 Elo
  d256-lr20.pt           vs d512-lr10.pt             1200 局  rate 0.465    -24.1 Elo
  d384-lr13.pt           vs d512-lr10.pt             1200 局  rate 0.509     +6.4 Elo
```

## `arena_mobp.json`

每对 1200 局，64 次模拟

| 参赛者 | Elo | ± | 得分率 | 局数 |
|---|---:|---:|---:|---:|
| `mobp.pt` | +884.5 | 67.0 | 0.752 | 2400 |
| `base.pt` | +877.0 | 66.5 | 0.742 | 2400 |
| `greedy-mobility` | +0.0 | 0.0 | 0.006 | 2400 |

```
  greedy-mobility        vs base.pt                  1200 局  rate 0.005   -934.7 Elo
  greedy-mobility        vs mobp.pt                  1200 局  rate 0.007   -858.7 Elo
  base.pt                vs mobp.pt                  1200 局  rate 0.488     -8.4 Elo
```

## `arena_own.json`

每对 1200 局，64 次模拟

| 参赛者 | Elo | ± | 得分率 | 局数 |
|---|---:|---:|---:|---:|
| `base.pt` | +911.8 | 68.0 | 0.774 | 2400 |
| `owner.pt` | +875.6 | 68.8 | 0.721 | 2400 |
| `greedy-mobility` | +0.0 | 0.0 | 0.005 | 2400 |

```
  greedy-mobility        vs base.pt                  1200 局  rate 0.005   -934.7 Elo
  greedy-mobility        vs owner.pt                 1200 局  rate 0.006   -880.6 Elo
  base.pt                vs owner.pt                 1200 局  rate 0.552    +36.0 Elo
```

## `arena_own_w.json`

每对 1200 局，64 次模拟

| 参赛者 | Elo | ± | 得分率 | 局数 |
|---|---:|---:|---:|---:|
| `base.pt` | +0.0 | 0.0 | 0.564 | 2400 |
| `owner-w10.pt` | -43.1 | 10.6 | 0.471 | 2400 |
| `owner.pt` | -45.6 | 11.0 | 0.466 | 2400 |

```
  base.pt                vs owner.pt                 1200 局  rate 0.572    +50.1 Elo
  base.pt                vs owner-w10.pt             1200 局  rate 0.555    +38.7 Elo
  owner.pt               vs owner-w10.pt             1200 局  rate 0.503     +2.0 Elo
```

## `arena_polh.json`

每对 1200 局，64 次模拟

| 参赛者 | Elo | ± | 得分率 | 局数 |
|---|---:|---:|---:|---:|
| `base.pt` | +882.1 | 66.9 | 0.749 | 2400 |
| `polh.pt` | +879.3 | 67.1 | 0.745 | 2400 |
| `greedy-mobility` | +0.0 | 0.0 | 0.006 | 2400 |

```
  greedy-mobility        vs base.pt                  1200 局  rate 0.005   -934.7 Elo
  greedy-mobility        vs polh.pt                  1200 局  rate 0.007   -858.7 Elo
  base.pt                vs polh.pt                  1200 局  rate 0.503     +2.0 Elo
```

## `arena_prec5.json`

每对 1200 局，64 次模拟

| 参赛者 | Elo | ± | 得分率 | 局数 |
|---|---:|---:|---:|---:|
| `FP8-QAT.pt` | +13.7 | 8.6 | 0.521 | 4801 |
| `FP4-PTQ.pt` | +1.7 | 8.9 | 0.500 | 4800 |
| `FP8-PTQ.pt` | +0.5 | 8.1 | 0.498 | 4800 |
| `BF16.pt` | +0.0 | 0.0 | 0.497 | 4800 |
| `FP4-QAT.pt` | -6.9 | 9.8 | 0.484 | 4801 |

```
  BF16.pt                vs FP8-QAT.pt               1200 局  rate 0.491     -6.4 Elo
  BF16.pt                vs FP4-QAT.pt               1200 局  rate 0.504     +2.6 Elo
  BF16.pt                vs FP8-PTQ.pt               1200 局  rate 0.496     -2.6 Elo
  BF16.pt                vs FP4-PTQ.pt               1200 局  rate 0.496     -2.6 Elo
  FP8-QAT.pt             vs FP4-QAT.pt               1201 局  rate 0.528    +19.7 Elo
  FP8-QAT.pt             vs FP8-PTQ.pt               1200 局  rate 0.518    +12.7 Elo
  FP8-QAT.pt             vs FP4-PTQ.pt               1200 局  rate 0.530    +20.9 Elo
  FP4-QAT.pt             vs FP8-PTQ.pt               1200 局  rate 0.488     -8.1 Elo
  FP4-QAT.pt             vs FP4-PTQ.pt               1200 局  rate 0.481    -13.3 Elo
  FP8-PTQ.pt             vs FP4-PTQ.pt               1200 局  rate 0.494     -4.3 Elo
```

## `arena_prec5_rule.json`

每对 1200 局，64 次模拟

| 参赛者 | Elo | ± | 得分率 | 局数 |
|---|---:|---:|---:|---:|
| `FP8-QAT.pt` | +869.2 | 39.2 | 0.619 | 6001 |
| `BF16.pt` | +858.3 | 39.7 | 0.604 | 6000 |
| `FP4-QAT.pt` | +851.3 | 39.6 | 0.593 | 6000 |
| `FP8-PTQ.pt` | +849.7 | 39.3 | 0.591 | 6000 |
| `FP4-PTQ.pt` | +846.5 | 39.2 | 0.586 | 6001 |
| `greedy-mobility` | +0.0 | 0.0 | 0.007 | 6002 |

```
  greedy-mobility        vs BF16.pt                  1200 局  rate 0.005   -934.7 Elo
  greedy-mobility        vs FP8-QAT.pt               1201 局  rate 0.006   -880.7 Elo
  greedy-mobility        vs FP4-QAT.pt               1200 局  rate 0.012   -771.2 Elo
  greedy-mobility        vs FP8-PTQ.pt               1200 局  rate 0.004   -951.4 Elo
  greedy-mobility        vs FP4-PTQ.pt               1201 局  rate 0.007   -848.8 Elo
  BF16.pt                vs FP8-QAT.pt               1200 局  rate 0.491     -6.4 Elo
  BF16.pt                vs FP4-QAT.pt               1200 局  rate 0.501     +0.9 Elo
  BF16.pt                vs FP8-PTQ.pt               1200 局  rate 0.514     +9.6 Elo
  BF16.pt                vs FP4-PTQ.pt               1200 局  rate 0.516    +11.3 Elo
  FP8-QAT.pt             vs FP4-QAT.pt               1200 局  rate 0.535    +24.7 Elo
  FP8-QAT.pt             vs FP8-PTQ.pt               1200 局  rate 0.533    +22.9 Elo
  FP8-QAT.pt             vs FP4-PTQ.pt               1200 局  rate 0.525    +17.1 Elo
  FP4-QAT.pt             vs FP8-PTQ.pt               1200 局  rate 0.502     +1.4 Elo
  FP4-QAT.pt             vs FP4-PTQ.pt               1200 局  rate 0.512     +8.7 Elo
  FP8-PTQ.pt             vs FP4-PTQ.pt               1200 局  rate 0.507     +5.2 Elo
```

## `arena_qwen-fp4.json`

每对 1200 局，64 次模拟

| 参赛者 | Elo | ± | 得分率 | 局数 |
|---|---:|---:|---:|---:|
| `poly-111k.pt` | +0.0 | 0.0 | 0.649 | 1200 |
| `qwen-fp4-40k.pt` | -106.8 | 14.0 | 0.351 | 1200 |

```
  poly-111k.pt           vs qwen-fp4-40k.pt          1200 局  rate 0.649   +106.9 Elo
```

## `arena_qwen60.json`

每对 1200 局，64 次模拟

| 参赛者 | Elo | ± | 得分率 | 局数 |
|---|---:|---:|---:|---:|
| `poly-111k.pt` | +0.0 | 0.0 | 0.649 | 2400 |
| `qwen-fp4-60k.pt` | -105.8 | 11.6 | 0.428 | 2400 |
| `qwen-fp4-40k.pt` | -108.1 | 12.3 | 0.423 | 2400 |

```
  poly-111k.pt           vs qwen-fp4-40k.pt          1200 局  rate 0.649   +106.9 Elo
  poly-111k.pt           vs qwen-fp4-60k.pt          1200 局  rate 0.650   +107.2 Elo
  qwen-fp4-40k.pt        vs qwen-fp4-60k.pt          1200 局  rate 0.495     -3.5 Elo
```

## `arena_v2_bf16.json`

每对 400 局，0 次模拟，锚点 `random`

| 参赛者 | Elo | ± | 得分率 | 局数 |
|---|---:|---:|---:|---:|
| `net@120227` | +1522.6 | 32.6 | 0.707 | 7600 |
| `net@100227` | +1519.6 | 33.2 | 0.704 | 7600 |
| `net@110227` | +1512.7 | 32.9 | 0.697 | 7600 |
| `net@130227` | +1511.6 | 32.0 | 0.696 | 7600 |
| `net@140227` | +1511.4 | 32.6 | 0.695 | 7600 |
| `net@80227` | +1509.3 | 32.9 | 0.693 | 7600 |
| `net@90227` | +1505.1 | 32.4 | 0.689 | 7600 |
| `net@150227` | +1503.7 | 32.9 | 0.687 | 7600 |
| `net@70227` | +1495.2 | 32.5 | 0.679 | 7600 |
| `net@50227` | +1469.0 | 32.5 | 0.651 | 7600 |
| `net@60227` | +1467.9 | 33.1 | 0.650 | 7600 |
| `net@40227` | +1430.3 | 32.4 | 0.611 | 7600 |
| `net@30227` | +1383.6 | 32.7 | 0.563 | 7601 |
| `net@20227` | +1214.2 | 32.4 | 0.415 | 7602 |
| `net@10227` | +980.3 | 32.9 | 0.283 | 7606 |
| `greedy-mobility` | +824.6 | 30.7 | 0.220 | 7608 |
| `flat-mcts-1k` | +658.6 | 29.4 | 0.163 | 7600 |
| `flat-mcts-256` | +463.3 | 30.3 | 0.103 | 7601 |
| `greedy-area` | +398.0 | 29.1 | 0.086 | 7600 |
| `random` | +0.0 | 0.0 | 0.009 | 7600 |

```
  random                 vs greedy-area               400 局  rate 0.090   -401.9 Elo
  random                 vs flat-mcts-256             400 局  rate 0.055   -494.0 Elo
  random                 vs flat-mcts-1k              400 局  rate 0.004   -969.7 Elo
  random                 vs greedy-mobility           400 局  rate 0.000     +nan Elo
  random                 vs net@10227                 400 局  rate 0.022   -655.2 Elo
  random                 vs net@20227                 400 局  rate 0.000     +nan Elo
  random                 vs net@30227                 400 局  rate 0.000     +nan Elo
  random                 vs net@40227                 400 局  rate 0.000     +nan Elo
  random                 vs net@50227                 400 局  rate 0.000     +nan Elo
  random                 vs net@60227                 400 局  rate 0.000     +nan Elo
  random                 vs net@70227                 400 局  rate 0.000     +nan Elo
  random                 vs net@80227                 400 局  rate 0.000     +nan Elo
  random                 vs net@90227                 400 局  rate 0.000     +nan Elo
  random                 vs net@100227                400 局  rate 0.000     +nan Elo
  random                 vs net@110227                400 局  rate 0.000     +nan Elo
  random                 vs net@120227                400 局  rate 0.000     +nan Elo
  random                 vs net@130227                400 局  rate 0.000     +nan Elo
  random                 vs net@140227                400 局  rate 0.000     +nan Elo
  random                 vs net@150227                400 局  rate 0.000     +nan Elo
  greedy-area            vs flat-mcts-256             400 局  rate 0.434    -46.3 Elo
  greedy-area            vs flat-mcts-1k              400 局  rate 0.158   -291.3 Elo
  greedy-area            vs greedy-mobility           400 局  rate 0.090   -401.9 Elo
  greedy-area            vs net@10227                 400 局  rate 0.030   -603.9 Elo
  greedy-area            vs net@20227                 400 局  rate 0.003  -1040.4 Elo
  greedy-area            vs net@30227                 400 局  rate 0.000     +nan Elo
  greedy-area            vs net@40227                 400 局  rate 0.001  -1161.0 Elo
  greedy-area            vs net@50227                 400 局  rate 0.000     +nan Elo
  greedy-area            vs net@60227                 400 局  rate 0.000     +nan Elo
  greedy-area            vs net@70227                 400 局  rate 0.000     +nan Elo
  greedy-area            vs net@80227                 400 局  rate 0.000     +nan Elo
  greedy-area            vs net@90227                 400 局  rate 0.000     +nan Elo
  greedy-area            vs net@100227                400 局  rate 0.000     +nan Elo
  greedy-area            vs net@110227                400 局  rate 0.000     +nan Elo
  greedy-area            vs net@120227                400 局  rate 0.000     +nan Elo
  greedy-area            vs net@130227                400 局  rate 0.000     +nan Elo
  greedy-area            vs net@140227                400 局  rate 0.000     +nan Elo
  greedy-area            vs net@150227                400 局  rate 0.000     +nan Elo
  flat-mcts-256          vs flat-mcts-1k              400 局  rate 0.239   -201.4 Elo
  flat-mcts-256          vs greedy-mobility           400 局  rate 0.102   -376.9 Elo
  flat-mcts-256          vs net@10227                 401 局  rate 0.111   -361.5 Elo
  flat-mcts-256          vs net@20227                 400 局  rate 0.000     +nan Elo
  flat-mcts-256          vs net@30227                 400 局  rate 0.000     +nan Elo
  flat-mcts-256          vs net@40227                 400 局  rate 0.000     +nan Elo
  flat-mcts-256          vs net@50227                 400 局  rate 0.000     +nan Elo
  flat-mcts-256          vs net@60227                 400 局  rate 0.000     +nan Elo
  flat-mcts-256          vs net@70227                 400 局  rate 0.000     +nan Elo
  flat-mcts-256          vs net@80227                 400 局  rate 0.000     +nan Elo
  flat-mcts-256          vs net@90227                 400 局  rate 0.000     +nan Elo
  flat-mcts-256          vs net@100227                400 局  rate 0.000     +nan Elo
  flat-mcts-256          vs net@110227                400 局  rate 0.000     +nan Elo
  flat-mcts-256          vs net@120227                400 局  rate 0.000     +nan Elo
  flat-mcts-256          vs net@130227                400 局  rate 0.000     +nan Elo
  flat-mcts-256          vs net@140227                400 局  rate 0.000     +nan Elo
  flat-mcts-256          vs net@150227                400 局  rate 0.000     +nan Elo
  flat-mcts-1k           vs greedy-mobility           400 局  rate 0.305   -143.1 Elo
  flat-mcts-1k           vs net@10227                 400 局  rate 0.159   -289.7 Elo
  flat-mcts-1k           vs net@20227                 400 局  rate 0.026   -627.7 Elo
  flat-mcts-1k           vs net@30227                 400 局  rate 0.000     +nan Elo
  flat-mcts-1k           vs net@40227                 400 局  rate 0.003  -1040.4 Elo
  flat-mcts-1k           vs net@50227                 400 局  rate 0.000     +nan Elo
  flat-mcts-1k           vs net@60227                 400 局  rate 0.004   -969.7 Elo
  flat-mcts-1k           vs net@70227                 400 局  rate 0.000     +nan Elo
  flat-mcts-1k           vs net@80227                 400 局  rate 0.000     +nan Elo
  flat-mcts-1k           vs net@90227                 400 局  rate 0.000     +nan Elo
  flat-mcts-1k           vs net@100227                400 局  rate 0.000     +nan Elo
  flat-mcts-1k           vs net@110227                400 局  rate 0.003  -1040.4 Elo
  flat-mcts-1k           vs net@120227                400 局  rate 0.000     +nan Elo
  flat-mcts-1k           vs net@130227                400 局  rate 0.000     +nan Elo
  flat-mcts-1k           vs net@140227                400 局  rate 0.000     +nan Elo
  flat-mcts-1k           vs net@150227                400 局  rate 0.000     +nan Elo
  greedy-mobility        vs net@10227                 405 局  rate 0.352   -106.1 Elo
  greedy-mobility        vs net@20227                 402 局  rate 0.128   -333.2 Elo
  greedy-mobility        vs net@30227                 401 局  rate 0.035   -576.6 Elo
  greedy-mobility        vs net@40227                 400 局  rate 0.022   -655.2 Elo
  greedy-mobility        vs net@50227                 400 局  rate 0.016   -712.8 Elo
  greedy-mobility        vs net@60227                 400 局  rate 0.021   -665.3 Elo
  greedy-mobility        vs net@70227                 400 局  rate 0.015   -726.9 Elo
  greedy-mobility        vs net@80227                 400 局  rate 0.010   -798.3 Elo
  greedy-mobility        vs net@90227                 400 局  rate 0.010   -798.3 Elo
  greedy-mobility        vs net@100227                400 局  rate 0.005   -919.5 Elo
  greedy-mobility        vs net@110227                400 局  rate 0.020   -676.1 Elo
  greedy-mobility        vs net@120227                400 局  rate 0.005   -919.5 Elo
  greedy-mobility        vs net@130227                400 局  rate 0.006   -880.6 Elo
  greedy-mobility        vs net@140227                400 局  rate 0.026   -627.7 Elo
  greedy-mobility        vs net@150227                400 局  rate 0.007   -848.7 Elo
  net@10227              vs net@20227                 400 局  rate 0.250   -190.8 Elo
  net@10227              vs net@30227                 400 局  rate 0.124   -340.0 Elo
  net@10227              vs net@40227                 400 局  rate 0.074   -439.6 Elo
  net@10227              vs net@50227                 400 局  rate 0.076   -433.3 Elo
  net@10227              vs net@60227                 400 局  rate 0.061   -474.2 Elo
  net@10227              vs net@70227                 400 局  rate 0.043   -541.1 Elo
  net@10227              vs net@80227                 400 局  rate 0.052   -502.6 Elo
  net@10227              vs net@90227                 400 局  rate 0.050   -511.5 Elo
  net@10227              vs net@100227                400 局  rate 0.045   -530.7 Elo
  net@10227              vs net@110227                400 局  rate 0.043   -541.1 Elo
  net@10227              vs net@120227                400 局  rate 0.031   -596.5 Elo
  net@10227              vs net@130227                400 局  rate 0.066   -459.6 Elo
  net@10227              vs net@140227                400 局  rate 0.065   -463.2 Elo
  net@10227              vs net@150227                400 局  rate 0.060   -478.0 Elo
  net@20227              vs net@30227                 400 局  rate 0.249   -192.0 Elo
  net@20227              vs net@40227                 400 局  rate 0.235   -205.0 Elo
  net@20227              vs net@50227                 400 局  rate 0.211   -228.9 Elo
  net@20227              vs net@60227                 400 局  rate 0.160   -288.1 Elo
  net@20227              vs net@70227                 400 局  rate 0.194   -247.7 Elo
  net@20227              vs net@80227                 400 局  rate 0.155   -294.6 Elo
  net@20227              vs net@90227                 400 局  rate 0.165   -281.7 Elo
  net@20227              vs net@100227                400 局  rate 0.160   -288.1 Elo
  net@20227              vs net@110227                400 局  rate 0.181   -261.9 Elo
  net@20227              vs net@120227                400 局  rate 0.165   -281.7 Elo
  net@20227              vs net@130227                400 局  rate 0.115   -354.5 Elo
  net@20227              vs net@140227                400 局  rate 0.146   -306.5 Elo
  net@20227              vs net@150227                400 局  rate 0.163   -284.9 Elo
  net@30227              vs net@40227                 400 局  rate 0.434    -46.3 Elo
  net@30227              vs net@50227                 400 局  rate 0.386    -80.4 Elo
  net@30227              vs net@60227                 400 局  rate 0.374    -89.7 Elo
  net@30227              vs net@70227                 400 局  rate 0.330   -123.0 Elo
  net@30227              vs net@80227                 400 局  rate 0.319   -131.9 Elo
  net@30227              vs net@90227                 400 局  rate 0.295   -151.3 Elo
  net@30227              vs net@100227                400 局  rate 0.340   -115.2 Elo
  net@30227              vs net@110227                400 局  rate 0.282   -161.9 Elo
  net@30227              vs net@120227                400 局  rate 0.312   -137.0 Elo
  net@30227              vs net@130227                400 局  rate 0.350   -107.5 Elo
  net@30227              vs net@140227                400 局  rate 0.352   -105.6 Elo
  net@30227              vs net@150227                400 局  rate 0.335   -119.1 Elo
  net@40227              vs net@50227                 400 局  rate 0.440    -41.9 Elo
  net@40227              vs net@60227                 400 局  rate 0.449    -35.7 Elo
  net@40227              vs net@70227                 400 局  rate 0.396    -73.2 Elo
  net@40227              vs net@80227                 400 局  rate 0.374    -89.7 Elo
  net@40227              vs net@90227                 400 局  rate 0.407    -65.0 Elo
  net@40227              vs net@100227                400 局  rate 0.386    -80.4 Elo
  net@40227              vs net@110227                400 局  rate 0.394    -75.0 Elo
  net@40227              vs net@120227                400 局  rate 0.361    -99.0 Elo
  net@40227              vs net@130227                400 局  rate 0.399    -71.3 Elo
  net@40227              vs net@140227                400 局  rate 0.356   -102.8 Elo
  net@40227              vs net@150227                400 局  rate 0.411    -62.3 Elo
  net@50227              vs net@60227                 400 局  rate 0.521    +14.8 Elo
  net@50227              vs net@70227                 400 局  rate 0.502     +1.7 Elo
  net@50227              vs net@80227                 400 局  rate 0.405    -66.8 Elo
  net@50227              vs net@90227                 400 局  rate 0.420    -56.1 Elo
  net@50227              vs net@100227                400 局  rate 0.399    -71.3 Elo
  net@50227              vs net@110227                400 局  rate 0.459    -28.7 Elo
  net@50227              vs net@120227                400 局  rate 0.435    -45.4 Elo
  net@50227              vs net@130227                400 局  rate 0.434    -46.3 Elo
  net@50227              vs net@140227                400 局  rate 0.466    -23.5 Elo
  net@50227              vs net@150227                400 局  rate 0.460    -27.9 Elo
  net@60227              vs net@70227                 400 局  rate 0.458    -29.6 Elo
  net@60227              vs net@80227                 400 局  rate 0.456    -30.5 Elo
  net@60227              vs net@90227                 400 局  rate 0.465    -24.4 Elo
  net@60227              vs net@100227                400 局  rate 0.378    -86.9 Elo
  net@60227              vs net@110227                400 局  rate 0.438    -43.7 Elo
  net@60227              vs net@120227                400 局  rate 0.450    -34.9 Elo
  net@60227              vs net@130227                400 局  rate 0.422    -54.3 Elo
  net@60227              vs net@140227                400 局  rate 0.461    -27.0 Elo
  net@60227              vs net@150227                400 局  rate 0.412    -61.4 Elo
  net@70227              vs net@80227                 400 局  rate 0.507     +5.2 Elo
  net@70227              vs net@90227                 400 局  rate 0.495     -3.5 Elo
  net@70227              vs net@100227                400 局  rate 0.465    -24.4 Elo
  net@70227              vs net@110227                400 局  rate 0.479    -14.8 Elo
  net@70227              vs net@120227                400 局  rate 0.460    -27.9 Elo
  net@70227              vs net@130227                400 局  rate 0.470    -20.9 Elo
  net@70227              vs net@140227                400 局  rate 0.466    -23.5 Elo
  net@70227              vs net@150227                400 局  rate 0.487     -8.7 Elo
  net@80227              vs net@90227                 400 局  rate 0.547    +33.1 Elo
  net@80227              vs net@100227                400 局  rate 0.466    -23.5 Elo
  net@80227              vs net@110227                400 局  rate 0.499     -0.9 Elo
  net@80227              vs net@120227                400 局  rate 0.501     +0.9 Elo
  net@80227              vs net@130227                400 局  rate 0.500     +0.0 Elo
  net@80227              vs net@140227                400 局  rate 0.440    -41.9 Elo
  net@80227              vs net@150227                400 局  rate 0.495     -3.5 Elo
  net@90227              vs net@100227                400 局  rate 0.501     +0.9 Elo
  net@90227              vs net@110227                400 局  rate 0.459    -28.7 Elo
  net@90227              vs net@120227                400 局  rate 0.461    -27.0 Elo
  net@90227              vs net@130227                400 局  rate 0.487     -8.7 Elo
  net@90227              vs net@140227                400 局  rate 0.504     +2.6 Elo
  net@90227              vs net@150227                400 局  rate 0.529    +20.0 Elo
  net@100227             vs net@110227                400 局  rate 0.510     +6.9 Elo
  net@100227             vs net@120227                400 局  rate 0.487     -8.7 Elo
  net@100227             vs net@130227                400 局  rate 0.496     -2.6 Elo
  net@100227             vs net@140227                400 局  rate 0.526    +18.3 Elo
  net@100227             vs net@150227                400 局  rate 0.496     -2.6 Elo
  net@110227             vs net@120227                400 局  rate 0.487     -8.7 Elo
  net@110227             vs net@130227                400 局  rate 0.491     -6.1 Elo
  net@110227             vs net@140227                400 局  rate 0.509     +6.1 Elo
  net@110227             vs net@150227                400 局  rate 0.514     +9.6 Elo
  net@120227             vs net@130227                400 局  rate 0.539    +27.0 Elo
  net@120227             vs net@140227                400 局  rate 0.521    +14.8 Elo
  net@120227             vs net@150227                400 局  rate 0.527    +19.1 Elo
  net@130227             vs net@140227                400 局  rate 0.495     -3.5 Elo
  net@130227             vs net@150227                400 局  rate 0.496     -2.6 Elo
  net@140227             vs net@150227                400 局  rate 0.546    +32.2 Elo
```

## `arena_v2_fp8.json`

每对 400 局，0 次模拟，锚点 `random`

| 参赛者 | Elo | ± | 得分率 | 局数 |
|---|---:|---:|---:|---:|
| `net@110234` | +1609.8 | 34.9 | 0.712 | 7600 |
| `net@100234` | +1602.3 | 34.4 | 0.704 | 7600 |
| `net@80234` | +1601.3 | 34.4 | 0.703 | 7600 |
| `net@130234` | +1599.7 | 33.9 | 0.702 | 7600 |
| `net@90234` | +1599.5 | 34.7 | 0.702 | 7600 |
| `net@120234` | +1598.2 | 33.2 | 0.700 | 7600 |
| `net@140234` | +1590.3 | 33.6 | 0.692 | 7600 |
| `net@150234` | +1587.6 | 35.0 | 0.689 | 7600 |
| `net@70234` | +1585.6 | 35.3 | 0.688 | 7600 |
| `net@60234` | +1578.3 | 34.2 | 0.680 | 7600 |
| `net@50234` | +1548.1 | 33.9 | 0.649 | 7600 |
| `net@40234` | +1505.1 | 34.0 | 0.605 | 7600 |
| `net@30234` | +1426.4 | 33.3 | 0.529 | 7601 |
| `net@20234` | +1233.0 | 33.0 | 0.380 | 7603 |
| `net@10234` | +1073.5 | 36.3 | 0.296 | 7608 |
| `greedy-mobility` | +863.9 | 34.3 | 0.217 | 7612 |
| `flat-mcts-1k` | +681.1 | 32.0 | 0.159 | 7600 |
| `flat-mcts-256` | +472.2 | 31.9 | 0.100 | 7600 |
| `greedy-area` | +414.3 | 30.3 | 0.085 | 7600 |
| `random` | +0.0 | 0.0 | 0.008 | 7600 |

```
  random                 vs greedy-area               400 局  rate 0.090   -401.9 Elo
  random                 vs flat-mcts-256             400 局  rate 0.055   -494.0 Elo
  random                 vs flat-mcts-1k              400 局  rate 0.004   -969.7 Elo
  random                 vs greedy-mobility           400 局  rate 0.000     +nan Elo
  random                 vs net@10234                 400 局  rate 0.005   -919.5 Elo
  random                 vs net@20234                 400 局  rate 0.000     +nan Elo
  random                 vs net@30234                 400 局  rate 0.000     +nan Elo
  random                 vs net@40234                 400 局  rate 0.000     +nan Elo
  random                 vs net@50234                 400 局  rate 0.000     +nan Elo
  random                 vs net@60234                 400 局  rate 0.000     +nan Elo
  random                 vs net@70234                 400 局  rate 0.000     +nan Elo
  random                 vs net@80234                 400 局  rate 0.000     +nan Elo
  random                 vs net@90234                 400 局  rate 0.000     +nan Elo
  random                 vs net@100234                400 局  rate 0.000     +nan Elo
  random                 vs net@110234                400 局  rate 0.000     +nan Elo
  random                 vs net@120234                400 局  rate 0.000     +nan Elo
  random                 vs net@130234                400 局  rate 0.000     +nan Elo
  random                 vs net@140234                400 局  rate 0.000     +nan Elo
  random                 vs net@150234                400 局  rate 0.000     +nan Elo
  greedy-area            vs flat-mcts-256             400 局  rate 0.434    -46.3 Elo
  greedy-area            vs flat-mcts-1k              400 局  rate 0.158   -291.3 Elo
  greedy-area            vs greedy-mobility           400 局  rate 0.090   -401.9 Elo
  greedy-area            vs net@10234                 400 局  rate 0.013   -759.1 Elo
  greedy-area            vs net@20234                 400 局  rate 0.005   -919.5 Elo
  greedy-area            vs net@30234                 400 局  rate 0.000     +nan Elo
  greedy-area            vs net@40234                 400 局  rate 0.000     +nan Elo
  greedy-area            vs net@50234                 400 局  rate 0.000     +nan Elo
  greedy-area            vs net@60234                 400 局  rate 0.000     +nan Elo
  greedy-area            vs net@70234                 400 局  rate 0.000     +nan Elo
  greedy-area            vs net@80234                 400 局  rate 0.000     +nan Elo
  greedy-area            vs net@90234                 400 局  rate 0.000     +nan Elo
  greedy-area            vs net@100234                400 局  rate 0.000     +nan Elo
  greedy-area            vs net@110234                400 局  rate 0.000     +nan Elo
  greedy-area            vs net@120234                400 局  rate 0.000     +nan Elo
  greedy-area            vs net@130234                400 局  rate 0.000     +nan Elo
  greedy-area            vs net@140234                400 局  rate 0.000     +nan Elo
  greedy-area            vs net@150234                400 局  rate 0.000     +nan Elo
  flat-mcts-256          vs flat-mcts-1k              400 局  rate 0.239   -201.4 Elo
  flat-mcts-256          vs greedy-mobility           400 局  rate 0.102   -376.9 Elo
  flat-mcts-256          vs net@10234                 400 局  rate 0.037   -563.7 Elo
  flat-mcts-256          vs net@20234                 400 局  rate 0.005   -919.5 Elo
  flat-mcts-256          vs net@30234                 400 局  rate 0.000     +nan Elo
  flat-mcts-256          vs net@40234                 400 局  rate 0.000     +nan Elo
  flat-mcts-256          vs net@50234                 400 局  rate 0.000     +nan Elo
  flat-mcts-256          vs net@60234                 400 局  rate 0.000     +nan Elo
  flat-mcts-256          vs net@70234                 400 局  rate 0.000     +nan Elo
  flat-mcts-256          vs net@80234                 400 局  rate 0.000     +nan Elo
  flat-mcts-256          vs net@90234                 400 局  rate 0.000     +nan Elo
  flat-mcts-256          vs net@100234                400 局  rate 0.000     +nan Elo
  flat-mcts-256          vs net@110234                400 局  rate 0.000     +nan Elo
  flat-mcts-256          vs net@120234                400 局  rate 0.000     +nan Elo
  flat-mcts-256          vs net@130234                400 局  rate 0.000     +nan Elo
  flat-mcts-256          vs net@140234                400 局  rate 0.000     +nan Elo
  flat-mcts-256          vs net@150234                400 局  rate 0.000     +nan Elo
  flat-mcts-1k           vs greedy-mobility           400 局  rate 0.305   -143.1 Elo
  flat-mcts-1k           vs net@10234                 400 局  rate 0.091   -399.3 Elo
  flat-mcts-1k           vs net@20234                 400 局  rate 0.031   -596.5 Elo
  flat-mcts-1k           vs net@30234                 400 局  rate 0.000     +nan Elo
  flat-mcts-1k           vs net@40234                 400 局  rate 0.001  -1161.0 Elo
  flat-mcts-1k           vs net@50234                 400 局  rate 0.000     +nan Elo
  flat-mcts-1k           vs net@60234                 400 局  rate 0.000     +nan Elo
  flat-mcts-1k           vs net@70234                 400 局  rate 0.000     +nan Elo
  flat-mcts-1k           vs net@80234                 400 局  rate 0.001  -1161.0 Elo
  flat-mcts-1k           vs net@90234                 400 局  rate 0.000     +nan Elo
  flat-mcts-1k           vs net@100234                400 局  rate 0.000     +nan Elo
  flat-mcts-1k           vs net@110234                400 局  rate 0.000     +nan Elo
  flat-mcts-1k           vs net@120234                400 局  rate 0.000     +nan Elo
  flat-mcts-1k           vs net@130234                400 局  rate 0.000     +nan Elo
  flat-mcts-1k           vs net@140234                400 局  rate 0.000     +nan Elo
  flat-mcts-1k           vs net@150234                400 局  rate 0.000     +nan Elo
  greedy-mobility        vs net@10234                 408 局  rate 0.315   -135.0 Elo
  greedy-mobility        vs net@20234                 403 局  rate 0.144   -309.8 Elo
  greedy-mobility        vs net@30234                 401 局  rate 0.021   -665.8 Elo
  greedy-mobility        vs net@40234                 400 局  rate 0.015   -726.9 Elo
  greedy-mobility        vs net@50234                 400 局  rate 0.020   -676.1 Elo
  greedy-mobility        vs net@60234                 400 局  rate 0.015   -726.9 Elo
  greedy-mobility        vs net@70234                 400 局  rate 0.016   -712.8 Elo
  greedy-mobility        vs net@80234                 400 局  rate 0.009   -821.7 Elo
  greedy-mobility        vs net@90234                 400 局  rate 0.013   -759.1 Elo
  greedy-mobility        vs net@100234                400 局  rate 0.010   -798.3 Elo
  greedy-mobility        vs net@110234                400 局  rate 0.011   -777.6 Elo
  greedy-mobility        vs net@120234                400 局  rate 0.007   -848.7 Elo
  greedy-mobility        vs net@130234                400 局  rate 0.001  -1161.0 Elo
  greedy-mobility        vs net@140234                400 局  rate 0.007   -848.7 Elo
  greedy-mobility        vs net@150234                400 局  rate 0.011   -777.6 Elo
  net@10234              vs net@20234                 400 局  rate 0.354   -104.7 Elo
  net@10234              vs net@30234                 400 局  rate 0.125   -338.0 Elo
  net@10234              vs net@40234                 400 局  rate 0.064   -466.8 Elo
  net@10234              vs net@50234                 400 局  rate 0.069   -452.7 Elo
  net@10234              vs net@60234                 400 局  rate 0.048   -520.9 Elo
  net@10234              vs net@70234                 400 局  rate 0.050   -511.5 Elo
  net@10234              vs net@80234                 400 局  rate 0.059   -481.9 Elo
  net@10234              vs net@90234                 400 局  rate 0.029   -611.5 Elo
  net@10234              vs net@100234                400 局  rate 0.049   -516.1 Elo
  net@10234              vs net@110234                400 局  rate 0.033   -589.5 Elo
  net@10234              vs net@120234                400 局  rate 0.043   -541.1 Elo
  net@10234              vs net@130234                400 局  rate 0.062   -470.4 Elo
  net@10234              vs net@140234                400 局  rate 0.052   -502.6 Elo
  net@10234              vs net@150234                400 局  rate 0.051   -507.0 Elo
  net@20234              vs net@30234                 400 局  rate 0.228   -212.4 Elo
  net@20234              vs net@40234                 400 局  rate 0.196   -244.9 Elo
  net@20234              vs net@50234                 400 局  rate 0.170   -275.5 Elo
  net@20234              vs net@60234                 400 局  rate 0.124   -340.0 Elo
  net@20234              vs net@70234                 400 局  rate 0.134   -324.5 Elo
  net@20234              vs net@80234                 400 局  rate 0.110   -363.2 Elo
  net@20234              vs net@90234                 400 局  rate 0.113   -358.8 Elo
  net@20234              vs net@100234                400 局  rate 0.114   -356.6 Elo
  net@20234              vs net@110234                400 局  rate 0.100   -381.7 Elo
  net@20234              vs net@120234                400 局  rate 0.133   -326.4 Elo
  net@20234              vs net@130234                400 局  rate 0.111   -361.0 Elo
  net@20234              vs net@140234                400 局  rate 0.104   -374.6 Elo
  net@20234              vs net@150234                400 局  rate 0.111   -361.0 Elo
  net@30234              vs net@40234                 400 局  rate 0.417    -57.9 Elo
  net@30234              vs net@50234                 400 局  rate 0.330   -123.0 Elo
  net@30234              vs net@60234                 400 局  rate 0.276   -167.3 Elo
  net@30234              vs net@70234                 400 局  rate 0.270   -172.8 Elo
  net@30234              vs net@80234                 400 局  rate 0.274   -169.5 Elo
  net@30234              vs net@90234                 400 局  rate 0.230   -209.9 Elo
  net@30234              vs net@100234                400 局  rate 0.241   -199.1 Elo
  net@30234              vs net@110234                400 局  rate 0.260   -181.7 Elo
  net@30234              vs net@120234                400 局  rate 0.273   -170.6 Elo
  net@30234              vs net@130234                400 局  rate 0.254   -187.4 Elo
  net@30234              vs net@140234                400 局  rate 0.316   -133.9 Elo
  net@30234              vs net@150234                400 局  rate 0.279   -165.1 Elo
  net@40234              vs net@50234                 400 局  rate 0.429    -49.8 Elo
  net@40234              vs net@60234                 400 局  rate 0.378    -86.9 Elo
  net@40234              vs net@70234                 400 局  rate 0.380    -85.0 Elo
  net@40234              vs net@80234                 400 局  rate 0.351   -106.6 Elo
  net@40234              vs net@90234                 400 局  rate 0.360   -100.0 Elo
  net@40234              vs net@100234                400 局  rate 0.389    -78.6 Elo
  net@40234              vs net@110234                400 局  rate 0.361    -99.0 Elo
  net@40234              vs net@120234                400 局  rate 0.365    -96.2 Elo
  net@40234              vs net@130234                400 局  rate 0.367    -94.3 Elo
  net@40234              vs net@140234                400 局  rate 0.399    -71.3 Elo
  net@40234              vs net@150234                400 局  rate 0.414    -60.5 Elo
  net@50234              vs net@60234                 400 局  rate 0.440    -41.9 Elo
  net@50234              vs net@70234                 400 局  rate 0.456    -30.5 Elo
  net@50234              vs net@80234                 400 局  rate 0.424    -53.4 Elo
  net@50234              vs net@90234                 400 局  rate 0.412    -61.4 Elo
  net@50234              vs net@100234                400 局  rate 0.431    -48.1 Elo
  net@50234              vs net@110234                400 局  rate 0.445    -38.4 Elo
  net@50234              vs net@120234                400 局  rate 0.422    -54.3 Elo
  net@50234              vs net@130234                400 局  rate 0.426    -51.6 Elo
  net@50234              vs net@140234                400 局  rate 0.470    -20.9 Elo
  net@50234              vs net@150234                400 局  rate 0.421    -55.2 Elo
  net@60234              vs net@70234                 400 局  rate 0.455    -31.4 Elo
  net@60234              vs net@80234                 400 局  rate 0.456    -30.5 Elo
  net@60234              vs net@90234                 400 局  rate 0.495     -3.5 Elo
  net@60234              vs net@100234                400 局  rate 0.454    -32.2 Elo
  net@60234              vs net@110234                400 局  rate 0.431    -48.1 Elo
  net@60234              vs net@120234                400 局  rate 0.459    -28.7 Elo
  net@60234              vs net@130234                400 局  rate 0.472    -19.1 Elo
  net@60234              vs net@140234                400 局  rate 0.486     -9.6 Elo
  net@60234              vs net@150234                400 局  rate 0.492     -5.2 Elo
  net@70234              vs net@80234                 400 局  rate 0.471    -20.0 Elo
  net@70234              vs net@90234                 400 局  rate 0.472    -19.1 Elo
  net@70234              vs net@100234                400 局  rate 0.510     +6.9 Elo
  net@70234              vs net@110234                400 局  rate 0.459    -28.7 Elo
  net@70234              vs net@120234                400 局  rate 0.466    -23.5 Elo
  net@70234              vs net@130234                400 局  rate 0.451    -34.0 Elo
  net@70234              vs net@140234                400 局  rate 0.477    -15.6 Elo
  net@70234              vs net@150234                400 局  rate 0.516    +11.3 Elo
  net@80234              vs net@90234                 400 局  rate 0.500     +0.0 Elo
  net@80234              vs net@100234                400 局  rate 0.468    -22.6 Elo
  net@80234              vs net@110234                400 局  rate 0.484    -11.3 Elo
  net@80234              vs net@120234                400 局  rate 0.522    +15.6 Elo
  net@80234              vs net@130234                400 局  rate 0.525    +17.4 Elo
  net@80234              vs net@140234                400 局  rate 0.519    +13.0 Elo
  net@80234              vs net@150234                400 局  rate 0.502     +1.7 Elo
  net@90234              vs net@100234                400 局  rate 0.490     -6.9 Elo
  net@90234              vs net@110234                400 局  rate 0.489     -7.8 Elo
  net@90234              vs net@120234                400 局  rate 0.472    -19.1 Elo
  net@90234              vs net@130234                400 局  rate 0.504     +2.6 Elo
  net@90234              vs net@140234                400 局  rate 0.489     -7.8 Elo
  net@90234              vs net@150234                400 局  rate 0.511     +7.8 Elo
  net@100234             vs net@110234                400 局  rate 0.486     -9.6 Elo
  net@100234             vs net@120234                400 局  rate 0.529    +20.0 Elo
  net@100234             vs net@130234                400 局  rate 0.527    +19.1 Elo
  net@100234             vs net@140234                400 局  rate 0.499     -0.9 Elo
  net@100234             vs net@150234                400 局  rate 0.497     -1.7 Elo
  net@110234             vs net@120234                400 局  rate 0.546    +32.2 Elo
  net@110234             vs net@130234                400 局  rate 0.507     +5.2 Elo
  net@110234             vs net@140234                400 局  rate 0.551    +35.7 Elo
  net@110234             vs net@150234                400 局  rate 0.481    -13.0 Elo
  net@120234             vs net@130234                400 局  rate 0.481    -13.0 Elo
  net@120234             vs net@140234                400 局  rate 0.515    +10.4 Elo
  net@120234             vs net@150234                400 局  rate 0.547    +33.1 Elo
  net@130234             vs net@140234                400 局  rate 0.512     +8.7 Elo
  net@130234             vs net@150234                400 局  rate 0.512     +8.7 Elo
  net@140234             vs net@150234                400 局  rate 0.551    +35.7 Elo
```

## `arena_v3-gate.json`

每对 400 局，0 次模拟，锚点 `random`

| 参赛者 | Elo | ± | 得分率 | 局数 |
|---|---:|---:|---:|---:|
| `net@60227` | +1475.6 | 30.2 | 0.770 | 7600 |
| `net@70227` | +1472.1 | 30.6 | 0.766 | 7601 |
| `net@50227` | +1460.9 | 29.8 | 0.756 | 7601 |
| `net@80227` | +1426.1 | 30.0 | 0.722 | 7601 |
| `net@90227` | +1396.2 | 30.7 | 0.691 | 7601 |
| `net@40227` | +1383.6 | 29.6 | 0.678 | 7601 |
| `net@100227` | +1376.2 | 30.0 | 0.671 | 7600 |
| `net@110227` | +1339.9 | 30.4 | 0.633 | 7600 |
| `net@120227` | +1334.5 | 29.8 | 0.627 | 7600 |
| `net@130227` | +1306.0 | 29.9 | 0.597 | 7600 |
| `net@30227` | +1300.1 | 30.6 | 0.591 | 7601 |
| `net@140227` | +1291.4 | 29.5 | 0.582 | 7600 |
| `net@150227` | +1277.6 | 30.6 | 0.568 | 7603 |
| `net@20227` | +1126.0 | 30.8 | 0.426 | 7603 |
| `greedy-mobility` | +918.9 | 30.1 | 0.283 | 7612 |
| `net@10227` | +875.1 | 30.1 | 0.260 | 7600 |
| `flat-mcts-1k` | +688.4 | 30.2 | 0.178 | 7600 |
| `flat-mcts-256` | +475.1 | 30.0 | 0.105 | 7600 |
| `greedy-area` | +414.4 | 29.1 | 0.088 | 7600 |
| `random` | +0.0 | 0.0 | 0.008 | 7600 |

```
  random                 vs greedy-area               400 局  rate 0.090   -401.9 Elo
  random                 vs flat-mcts-256             400 局  rate 0.055   -494.0 Elo
  random                 vs flat-mcts-1k              400 局  rate 0.004   -969.7 Elo
  random                 vs greedy-mobility           400 局  rate 0.000     +nan Elo
  random                 vs net@10227                 400 局  rate 0.010   -798.3 Elo
  random                 vs net@20227                 400 局  rate 0.000     +nan Elo
  random                 vs net@30227                 400 局  rate 0.000     +nan Elo
  random                 vs net@40227                 400 局  rate 0.000     +nan Elo
  random                 vs net@50227                 400 局  rate 0.000     +nan Elo
  random                 vs net@60227                 400 局  rate 0.000     +nan Elo
  random                 vs net@70227                 400 局  rate 0.000     +nan Elo
  random                 vs net@80227                 400 局  rate 0.000     +nan Elo
  random                 vs net@90227                 400 局  rate 0.000     +nan Elo
  random                 vs net@100227                400 局  rate 0.000     +nan Elo
  random                 vs net@110227                400 局  rate 0.000     +nan Elo
  random                 vs net@120227                400 局  rate 0.000     +nan Elo
  random                 vs net@130227                400 局  rate 0.000     +nan Elo
  random                 vs net@140227                400 局  rate 0.000     +nan Elo
  random                 vs net@150227                400 局  rate 0.000     +nan Elo
  greedy-area            vs flat-mcts-256             400 局  rate 0.434    -46.3 Elo
  greedy-area            vs flat-mcts-1k              400 局  rate 0.158   -291.3 Elo
  greedy-area            vs greedy-mobility           400 局  rate 0.090   -401.9 Elo
  greedy-area            vs net@10227                 400 局  rate 0.043   -541.1 Elo
  greedy-area            vs net@20227                 400 局  rate 0.011   -777.6 Elo
  greedy-area            vs net@30227                 400 局  rate 0.000     +nan Elo
  greedy-area            vs net@40227                 400 局  rate 0.000     +nan Elo
  greedy-area            vs net@50227                 400 局  rate 0.000     +nan Elo
  greedy-area            vs net@60227                 400 局  rate 0.000     +nan Elo
  greedy-area            vs net@70227                 400 局  rate 0.000     +nan Elo
  greedy-area            vs net@80227                 400 局  rate 0.005   -919.5 Elo
  greedy-area            vs net@90227                 400 局  rate 0.000     +nan Elo
  greedy-area            vs net@100227                400 局  rate 0.003  -1040.4 Elo
  greedy-area            vs net@110227                400 局  rate 0.003  -1040.4 Elo
  greedy-area            vs net@120227                400 局  rate 0.000     +nan Elo
  greedy-area            vs net@130227                400 局  rate 0.009   -821.7 Elo
  greedy-area            vs net@140227                400 局  rate 0.005   -919.5 Elo
  greedy-area            vs net@150227                400 局  rate 0.003  -1040.4 Elo
  flat-mcts-256          vs flat-mcts-1k              400 局  rate 0.239   -201.4 Elo
  flat-mcts-256          vs greedy-mobility           400 局  rate 0.102   -376.9 Elo
  flat-mcts-256          vs net@10227                 400 局  rate 0.126   -336.1 Elo
  flat-mcts-256          vs net@20227                 400 局  rate 0.006   -880.6 Elo
  flat-mcts-256          vs net@30227                 400 局  rate 0.000     +nan Elo
  flat-mcts-256          vs net@40227                 400 局  rate 0.000     +nan Elo
  flat-mcts-256          vs net@50227                 400 局  rate 0.000     +nan Elo
  flat-mcts-256          vs net@60227                 400 局  rate 0.000     +nan Elo
  flat-mcts-256          vs net@70227                 400 局  rate 0.000     +nan Elo
  flat-mcts-256          vs net@80227                 400 局  rate 0.000     +nan Elo
  flat-mcts-256          vs net@90227                 400 局  rate 0.000     +nan Elo
  flat-mcts-256          vs net@100227                400 局  rate 0.004   -969.7 Elo
  flat-mcts-256          vs net@110227                400 局  rate 0.004   -969.7 Elo
  flat-mcts-256          vs net@120227                400 局  rate 0.000     +nan Elo
  flat-mcts-256          vs net@130227                400 局  rate 0.000     +nan Elo
  flat-mcts-256          vs net@140227                400 局  rate 0.000     +nan Elo
  flat-mcts-256          vs net@150227                400 局  rate 0.006   -880.6 Elo
  flat-mcts-1k           vs greedy-mobility           400 局  rate 0.305   -143.1 Elo
  flat-mcts-1k           vs net@10227                 400 局  rate 0.255   -186.2 Elo
  flat-mcts-1k           vs net@20227                 400 局  rate 0.064   -466.8 Elo
  flat-mcts-1k           vs net@30227                 400 局  rate 0.021   -665.3 Elo
  flat-mcts-1k           vs net@40227                 400 局  rate 0.006   -880.6 Elo
  flat-mcts-1k           vs net@50227                 400 局  rate 0.003  -1040.4 Elo
  flat-mcts-1k           vs net@60227                 400 局  rate 0.005   -919.5 Elo
  flat-mcts-1k           vs net@70227                 400 局  rate 0.005   -919.5 Elo
  flat-mcts-1k           vs net@80227                 400 局  rate 0.013   -759.1 Elo
  flat-mcts-1k           vs net@90227                 400 局  rate 0.003  -1040.4 Elo
  flat-mcts-1k           vs net@100227                400 局  rate 0.004   -969.7 Elo
  flat-mcts-1k           vs net@110227                400 局  rate 0.019   -687.5 Elo
  flat-mcts-1k           vs net@120227                400 局  rate 0.011   -777.6 Elo
  flat-mcts-1k           vs net@130227                400 局  rate 0.010   -798.3 Elo
  flat-mcts-1k           vs net@140227                400 局  rate 0.026   -627.7 Elo
  flat-mcts-1k           vs net@150227                400 局  rate 0.025   -636.4 Elo
  greedy-mobility        vs net@10227                 400 局  rate 0.580    +56.1 Elo
  greedy-mobility        vs net@20227                 403 局  rate 0.256   -185.7 Elo
  greedy-mobility        vs net@30227                 401 局  rate 0.106   -370.4 Elo
  greedy-mobility        vs net@40227                 401 局  rate 0.085   -413.3 Elo
  greedy-mobility        vs net@50227                 401 局  rate 0.045   -531.2 Elo
  greedy-mobility        vs net@60227                 400 局  rate 0.036   -569.9 Elo
  greedy-mobility        vs net@70227                 401 局  rate 0.039   -558.3 Elo
  greedy-mobility        vs net@80227                 401 局  rate 0.070   -449.8 Elo
  greedy-mobility        vs net@90227                 401 局  rate 0.064   -467.2 Elo
  greedy-mobility        vs net@100227                400 局  rate 0.095   -391.6 Elo
  greedy-mobility        vs net@110227                400 局  rate 0.087   -407.3 Elo
  greedy-mobility        vs net@120227                400 局  rate 0.100   -381.7 Elo
  greedy-mobility        vs net@130227                400 局  rate 0.105   -372.3 Elo
  greedy-mobility        vs net@140227                400 局  rate 0.117   -350.3 Elo
  greedy-mobility        vs net@150227                403 局  rate 0.103   -376.0 Elo
  net@10227              vs net@20227                 400 局  rate 0.236   -203.8 Elo
  net@10227              vs net@30227                 400 局  rate 0.061   -474.2 Elo
  net@10227              vs net@40227                 400 局  rate 0.025   -636.4 Elo
  net@10227              vs net@50227                 400 局  rate 0.024   -645.6 Elo
  net@10227              vs net@60227                 400 局  rate 0.005   -919.5 Elo
  net@10227              vs net@70227                 400 局  rate 0.035   -576.2 Elo
  net@10227              vs net@80227                 400 局  rate 0.033   -589.5 Elo
  net@10227              vs net@90227                 400 局  rate 0.044   -535.8 Elo
  net@10227              vs net@100227                400 局  rate 0.044   -535.8 Elo
  net@10227              vs net@110227                400 局  rate 0.070   -449.4 Elo
  net@10227              vs net@120227                400 局  rate 0.070   -449.4 Elo
  net@10227              vs net@130227                400 局  rate 0.084   -415.6 Elo
  net@10227              vs net@140227                400 局  rate 0.117   -350.3 Elo
  net@10227              vs net@150227                400 局  rate 0.113   -358.8 Elo
  net@20227              vs net@30227                 400 局  rate 0.279   -165.1 Elo
  net@20227              vs net@40227                 400 局  rate 0.133   -326.4 Elo
  net@20227              vs net@50227                 400 局  rate 0.117   -350.3 Elo
  net@20227              vs net@60227                 400 局  rate 0.101   -379.3 Elo
  net@20227              vs net@70227                 400 局  rate 0.140   -315.3 Elo
  net@20227              vs net@80227                 400 局  rate 0.119   -348.2 Elo
  net@20227              vs net@90227                 400 局  rate 0.209   -231.5 Elo
  net@20227              vs net@100227                400 局  rate 0.219   -221.1 Elo
  net@20227              vs net@110227                400 局  rate 0.254   -187.4 Elo
  net@20227              vs net@120227                400 局  rate 0.244   -196.7 Elo
  net@20227              vs net@130227                400 局  rate 0.282   -161.9 Elo
  net@20227              vs net@140227                400 局  rate 0.253   -188.5 Elo
  net@20227              vs net@150227                400 局  rate 0.309   -140.0 Elo
  net@30227              vs net@40227                 400 局  rate 0.375    -88.7 Elo
  net@30227              vs net@50227                 400 局  rate 0.296   -150.3 Elo
  net@30227              vs net@60227                 400 局  rate 0.244   -196.7 Elo
  net@30227              vs net@70227                 400 局  rate 0.259   -182.8 Elo
  net@30227              vs net@80227                 400 局  rate 0.291   -154.5 Elo
  net@30227              vs net@90227                 400 局  rate 0.369    -93.4 Elo
  net@30227              vs net@100227                400 局  rate 0.374    -89.7 Elo
  net@30227              vs net@110227                400 局  rate 0.469    -21.7 Elo
  net@30227              vs net@120227                400 局  rate 0.481    -13.0 Elo
  net@30227              vs net@130227                400 局  rate 0.471    -20.0 Elo
  net@30227              vs net@140227                400 局  rate 0.535    +24.4 Elo
  net@30227              vs net@150227                400 局  rate 0.532    +22.6 Elo
  net@40227              vs net@50227                 400 局  rate 0.386    -80.4 Elo
  net@40227              vs net@60227                 400 局  rate 0.346   -110.4 Elo
  net@40227              vs net@70227                 400 局  rate 0.369    -93.4 Elo
  net@40227              vs net@80227                 400 局  rate 0.429    -49.8 Elo
  net@40227              vs net@90227                 400 局  rate 0.472    -19.1 Elo
  net@40227              vs net@100227                400 局  rate 0.477    -15.6 Elo
  net@40227              vs net@110227                400 局  rate 0.551    +35.7 Elo
  net@40227              vs net@120227                400 局  rate 0.547    +33.1 Elo
  net@40227              vs net@130227                400 局  rate 0.610    +77.7 Elo
  net@40227              vs net@140227                400 局  rate 0.652   +109.5 Elo
  net@40227              vs net@150227                400 局  rate 0.669   +122.0 Elo
  net@50227              vs net@60227                 400 局  rate 0.475    -17.4 Elo
  net@50227              vs net@70227                 400 局  rate 0.504     +2.6 Elo
  net@50227              vs net@80227                 400 局  rate 0.554    +37.5 Elo
  net@50227              vs net@90227                 400 局  rate 0.580    +56.1 Elo
  net@50227              vs net@100227                400 局  rate 0.632    +94.3 Elo
  net@50227              vs net@110227                400 局  rate 0.631    +93.4 Elo
  net@50227              vs net@120227                400 局  rate 0.703   +149.3 Elo
  net@50227              vs net@130227                400 局  rate 0.715   +159.8 Elo
  net@50227              vs net@140227                400 局  rate 0.716   +160.9 Elo
  net@50227              vs net@150227                400 局  rate 0.723   +166.2 Elo
  net@60227              vs net@70227                 400 局  rate 0.497     -1.7 Elo
  net@60227              vs net@80227                 400 局  rate 0.556    +39.3 Elo
  net@60227              vs net@90227                 400 局  rate 0.576    +53.4 Elo
  net@60227              vs net@100227                400 局  rate 0.647   +105.6 Elo
  net@60227              vs net@110227                400 局  rate 0.684   +133.9 Elo
  net@60227              vs net@120227                400 局  rate 0.675   +127.0 Elo
  net@60227              vs net@130227                400 局  rate 0.719   +163.0 Elo
  net@60227              vs net@140227                400 局  rate 0.723   +166.2 Elo
  net@60227              vs net@150227                400 局  rate 0.760   +200.2 Elo
  net@70227              vs net@80227                 400 局  rate 0.588    +61.4 Elo
  net@70227              vs net@90227                 400 局  rate 0.598    +68.6 Elo
  net@70227              vs net@100227                400 局  rate 0.652   +109.5 Elo
  net@70227              vs net@110227                400 局  rate 0.710   +155.5 Elo
  net@70227              vs net@120227                400 局  rate 0.694   +142.1 Elo
  net@70227              vs net@130227                400 局  rate 0.716   +160.9 Elo
  net@70227              vs net@140227                400 局  rate 0.709   +154.5 Elo
  net@70227              vs net@150227                400 局  rate 0.744   +185.1 Elo
  net@80227              vs net@90227                 400 局  rate 0.550    +34.9 Elo
  net@80227              vs net@100227                400 局  rate 0.527    +19.1 Elo
  net@80227              vs net@110227                400 局  rate 0.585    +59.6 Elo
  net@80227              vs net@120227                400 局  rate 0.595    +66.8 Elo
  net@80227              vs net@130227                400 局  rate 0.685   +135.0 Elo
  net@80227              vs net@140227                400 局  rate 0.699   +146.2 Elo
  net@80227              vs net@150227                400 局  rate 0.726   +169.5 Elo
  net@90227              vs net@100227                400 局  rate 0.562    +43.7 Elo
  net@90227              vs net@110227                400 局  rate 0.554    +37.5 Elo
  net@90227              vs net@120227                400 局  rate 0.583    +57.9 Elo
  net@90227              vs net@130227                400 局  rate 0.621    +86.0 Elo
  net@90227              vs net@140227                400 局  rate 0.651   +108.5 Elo
  net@90227              vs net@150227                400 局  rate 0.626    +89.7 Elo
  net@100227             vs net@110227                400 局  rate 0.581    +57.0 Elo
  net@100227             vs net@120227                400 局  rate 0.569    +48.1 Elo
  net@100227             vs net@130227                400 局  rate 0.620    +85.0 Elo
  net@100227             vs net@140227                400 局  rate 0.603    +72.2 Elo
  net@100227             vs net@150227                400 局  rate 0.610    +77.7 Elo
  net@110227             vs net@120227                400 局  rate 0.514     +9.6 Elo
  net@110227             vs net@130227                400 局  rate 0.532    +22.6 Elo
  net@110227             vs net@140227                400 局  rate 0.556    +39.3 Elo
  net@110227             vs net@150227                400 局  rate 0.619    +84.1 Elo
  net@120227             vs net@130227                400 局  rate 0.554    +37.5 Elo
  net@120227             vs net@140227                400 局  rate 0.550    +34.9 Elo
  net@120227             vs net@150227                400 局  rate 0.594    +65.9 Elo
  net@130227             vs net@140227                400 局  rate 0.545    +31.4 Elo
  net@130227             vs net@150227                400 局  rate 0.535    +24.4 Elo
  net@140227             vs net@150227                400 局  rate 0.516    +11.3 Elo
```

## `arena_v3-rndopen.json`

每对 400 局，0 次模拟，锚点 `random`

| 参赛者 | Elo | ± | 得分率 | 局数 |
|---|---:|---:|---:|---:|
| `net@150024` | +1760.1 | 34.3 | 0.732 | 7600 |
| `net@140024` | +1757.6 | 33.2 | 0.729 | 7600 |
| `net@130024` | +1747.9 | 34.9 | 0.720 | 7600 |
| `net@120024` | +1743.2 | 34.7 | 0.715 | 7600 |
| `net@110024` | +1730.9 | 34.6 | 0.703 | 7600 |
| `net@90024` | +1724.5 | 34.7 | 0.696 | 7600 |
| `net@100024` | +1723.5 | 34.6 | 0.695 | 7600 |
| `net@80024` | +1711.1 | 33.7 | 0.683 | 7600 |
| `net@70024` | +1704.0 | 34.2 | 0.675 | 7600 |
| `net@60024` | +1683.0 | 34.5 | 0.654 | 7600 |
| `net@50024` | +1661.0 | 33.9 | 0.632 | 7601 |
| `net@40024` | +1624.8 | 33.6 | 0.595 | 7600 |
| `net@30024` | +1549.2 | 35.1 | 0.523 | 7601 |
| `net@20024` | +1409.8 | 33.6 | 0.413 | 7601 |
| `net@10024` | +1151.9 | 34.6 | 0.288 | 7601 |
| `greedy-mobility` | +864.0 | 32.5 | 0.203 | 7602 |
| `flat-mcts-1k` | +685.3 | 30.1 | 0.155 | 7600 |
| `flat-mcts-256` | +473.6 | 28.4 | 0.098 | 7600 |
| `greedy-area` | +418.8 | 28.2 | 0.084 | 7600 |
| `random` | +0.0 | 0.0 | 0.008 | 7600 |

```
  random                 vs greedy-area               400 局  rate 0.090   -401.9 Elo
  random                 vs flat-mcts-256             400 局  rate 0.055   -494.0 Elo
  random                 vs flat-mcts-1k              400 局  rate 0.004   -969.7 Elo
  random                 vs greedy-mobility           400 局  rate 0.000     +nan Elo
  random                 vs net@10024                 400 局  rate 0.000     +nan Elo
  random                 vs net@20024                 400 局  rate 0.000     +nan Elo
  random                 vs net@30024                 400 局  rate 0.000     +nan Elo
  random                 vs net@40024                 400 局  rate 0.000     +nan Elo
  random                 vs net@50024                 400 局  rate 0.000     +nan Elo
  random                 vs net@60024                 400 局  rate 0.000     +nan Elo
  random                 vs net@70024                 400 局  rate 0.000     +nan Elo
  random                 vs net@80024                 400 局  rate 0.000     +nan Elo
  random                 vs net@90024                 400 局  rate 0.000     +nan Elo
  random                 vs net@100024                400 局  rate 0.000     +nan Elo
  random                 vs net@110024                400 局  rate 0.000     +nan Elo
  random                 vs net@120024                400 局  rate 0.000     +nan Elo
  random                 vs net@130024                400 局  rate 0.000     +nan Elo
  random                 vs net@140024                400 局  rate 0.000     +nan Elo
  random                 vs net@150024                400 局  rate 0.000     +nan Elo
  greedy-area            vs flat-mcts-256             400 局  rate 0.434    -46.3 Elo
  greedy-area            vs flat-mcts-1k              400 局  rate 0.158   -291.3 Elo
  greedy-area            vs greedy-mobility           400 局  rate 0.090   -401.9 Elo
  greedy-area            vs net@10024                 400 局  rate 0.004   -969.7 Elo
  greedy-area            vs net@20024                 400 局  rate 0.000     +nan Elo
  greedy-area            vs net@30024                 400 局  rate 0.000     +nan Elo
  greedy-area            vs net@40024                 400 局  rate 0.000     +nan Elo
  greedy-area            vs net@50024                 400 局  rate 0.000     +nan Elo
  greedy-area            vs net@60024                 400 局  rate 0.000     +nan Elo
  greedy-area            vs net@70024                 400 局  rate 0.000     +nan Elo
  greedy-area            vs net@80024                 400 局  rate 0.000     +nan Elo
  greedy-area            vs net@90024                 400 局  rate 0.000     +nan Elo
  greedy-area            vs net@100024                400 局  rate 0.000     +nan Elo
  greedy-area            vs net@110024                400 局  rate 0.000     +nan Elo
  greedy-area            vs net@120024                400 局  rate 0.000     +nan Elo
  greedy-area            vs net@130024                400 局  rate 0.000     +nan Elo
  greedy-area            vs net@140024                400 局  rate 0.000     +nan Elo
  greedy-area            vs net@150024                400 局  rate 0.000     +nan Elo
  flat-mcts-256          vs flat-mcts-1k              400 局  rate 0.239   -201.4 Elo
  flat-mcts-256          vs greedy-mobility           400 局  rate 0.102   -376.9 Elo
  flat-mcts-256          vs net@10024                 400 局  rate 0.005   -919.5 Elo
  flat-mcts-256          vs net@20024                 400 局  rate 0.000     +nan Elo
  flat-mcts-256          vs net@30024                 400 局  rate 0.000     +nan Elo
  flat-mcts-256          vs net@40024                 400 局  rate 0.000     +nan Elo
  flat-mcts-256          vs net@50024                 400 局  rate 0.000     +nan Elo
  flat-mcts-256          vs net@60024                 400 局  rate 0.000     +nan Elo
  flat-mcts-256          vs net@70024                 400 局  rate 0.000     +nan Elo
  flat-mcts-256          vs net@80024                 400 局  rate 0.000     +nan Elo
  flat-mcts-256          vs net@90024                 400 局  rate 0.000     +nan Elo
  flat-mcts-256          vs net@100024                400 局  rate 0.000     +nan Elo
  flat-mcts-256          vs net@110024                400 局  rate 0.000     +nan Elo
  flat-mcts-256          vs net@120024                400 局  rate 0.000     +nan Elo
  flat-mcts-256          vs net@130024                400 局  rate 0.000     +nan Elo
  flat-mcts-256          vs net@140024                400 局  rate 0.000     +nan Elo
  flat-mcts-256          vs net@150024                400 局  rate 0.000     +nan Elo
  flat-mcts-1k           vs greedy-mobility           400 局  rate 0.305   -143.1 Elo
  flat-mcts-1k           vs net@10024                 400 局  rate 0.022   -655.2 Elo
  flat-mcts-1k           vs net@20024                 400 局  rate 0.011   -777.6 Elo
  flat-mcts-1k           vs net@30024                 400 局  rate 0.000     +nan Elo
  flat-mcts-1k           vs net@40024                 400 局  rate 0.000     +nan Elo
  flat-mcts-1k           vs net@50024                 400 局  rate 0.000     +nan Elo
  flat-mcts-1k           vs net@60024                 400 局  rate 0.000     +nan Elo
  flat-mcts-1k           vs net@70024                 400 局  rate 0.003  -1040.4 Elo
  flat-mcts-1k           vs net@80024                 400 局  rate 0.000     +nan Elo
  flat-mcts-1k           vs net@90024                 400 局  rate 0.000     +nan Elo
  flat-mcts-1k           vs net@100024                400 局  rate 0.000     +nan Elo
  flat-mcts-1k           vs net@110024                400 局  rate 0.000     +nan Elo
  flat-mcts-1k           vs net@120024                400 局  rate 0.000     +nan Elo
  flat-mcts-1k           vs net@130024                400 局  rate 0.000     +nan Elo
  flat-mcts-1k           vs net@140024                400 局  rate 0.000     +nan Elo
  flat-mcts-1k           vs net@150024                400 局  rate 0.000     +nan Elo
  greedy-mobility        vs net@10024                 401 局  rate 0.204   -236.0 Elo
  greedy-mobility        vs net@20024                 401 局  rate 0.076   -433.8 Elo
  greedy-mobility        vs net@30024                 400 局  rate 0.024   -645.6 Elo
  greedy-mobility        vs net@40024                 400 局  rate 0.001  -1161.0 Elo
  greedy-mobility        vs net@50024                 400 局  rate 0.014   -742.3 Elo
  greedy-mobility        vs net@60024                 400 局  rate 0.007   -848.7 Elo
  greedy-mobility        vs net@70024                 400 局  rate 0.009   -821.7 Elo
  greedy-mobility        vs net@80024                 400 局  rate 0.007   -848.7 Elo
  greedy-mobility        vs net@90024                 400 局  rate 0.006   -880.6 Elo
  greedy-mobility        vs net@100024                400 局  rate 0.003  -1040.4 Elo
  greedy-mobility        vs net@110024                400 局  rate 0.003  -1040.4 Elo
  greedy-mobility        vs net@120024                400 局  rate 0.001  -1161.0 Elo
  greedy-mobility        vs net@130024                400 局  rate 0.000     +nan Elo
  greedy-mobility        vs net@140024                400 局  rate 0.000     +nan Elo
  greedy-mobility        vs net@150024                400 局  rate 0.003  -1040.4 Elo
  net@10024              vs net@20024                 400 局  rate 0.181   -261.9 Elo
  net@10024              vs net@30024                 400 局  rate 0.102   -376.9 Elo
  net@10024              vs net@40024                 400 局  rate 0.034   -582.7 Elo
  net@10024              vs net@50024                 400 局  rate 0.044   -535.8 Elo
  net@10024              vs net@60024                 400 局  rate 0.061   -474.2 Elo
  net@10024              vs net@70024                 400 局  rate 0.033   -589.5 Elo
  net@10024              vs net@80024                 400 局  rate 0.037   -563.7 Elo
  net@10024              vs net@90024                 400 局  rate 0.037   -563.7 Elo
  net@10024              vs net@100024                400 局  rate 0.043   -541.1 Elo
  net@10024              vs net@110024                400 局  rate 0.024   -645.6 Elo
  net@10024              vs net@120024                400 局  rate 0.025   -636.4 Elo
  net@10024              vs net@130024                400 局  rate 0.024   -645.6 Elo
  net@10024              vs net@140024                400 局  rate 0.033   -589.5 Elo
  net@10024              vs net@150024                400 局  rate 0.030   -603.9 Elo
  net@20024              vs net@30024                 400 局  rate 0.316   -133.9 Elo
  net@20024              vs net@40024                 400 局  rate 0.253   -188.5 Elo
  net@20024              vs net@50024                 400 局  rate 0.184   -259.0 Elo
  net@20024              vs net@60024                 400 局  rate 0.159   -289.7 Elo
  net@20024              vs net@70024                 400 局  rate 0.147   -304.8 Elo
  net@20024              vs net@80024                 400 局  rate 0.165   -281.7 Elo
  net@20024              vs net@90024                 400 局  rate 0.145   -308.2 Elo
  net@20024              vs net@100024                400 局  rate 0.117   -350.3 Elo
  net@20024              vs net@110024                400 局  rate 0.120   -346.1 Elo
  net@20024              vs net@120024                400 局  rate 0.138   -319.0 Elo
  net@20024              vs net@130024                400 局  rate 0.136   -320.8 Elo
  net@20024              vs net@140024                400 局  rate 0.135   -322.7 Elo
  net@20024              vs net@150024                400 局  rate 0.109   -365.4 Elo
  net@30024              vs net@40024                 400 局  rate 0.396    -73.2 Elo
  net@30024              vs net@50024                 401 局  rate 0.355   -103.5 Elo
  net@30024              vs net@60024                 400 局  rate 0.329   -124.0 Elo
  net@30024              vs net@70024                 400 局  rate 0.271   -171.7 Elo
  net@30024              vs net@80024                 400 局  rate 0.311   -138.0 Elo
  net@30024              vs net@90024                 400 局  rate 0.250   -190.8 Elo
  net@30024              vs net@100024                400 局  rate 0.253   -188.5 Elo
  net@30024              vs net@110024                400 局  rate 0.256   -185.1 Elo
  net@30024              vs net@120024                400 局  rate 0.268   -175.0 Elo
  net@30024              vs net@130024                400 局  rate 0.233   -207.5 Elo
  net@30024              vs net@140024                400 局  rate 0.240   -200.2 Elo
  net@30024              vs net@150024                400 局  rate 0.223   -217.3 Elo
  net@40024              vs net@50024                 400 局  rate 0.460    -27.9 Elo
  net@40024              vs net@60024                 400 局  rate 0.403    -68.6 Elo
  net@40024              vs net@70024                 400 局  rate 0.390    -77.7 Elo
  net@40024              vs net@80024                 400 局  rate 0.372    -90.6 Elo
  net@40024              vs net@90024                 400 局  rate 0.310   -139.0 Elo
  net@40024              vs net@100024                400 局  rate 0.371    -91.5 Elo
  net@40024              vs net@110024                400 局  rate 0.366    -95.3 Elo
  net@40024              vs net@120024                400 局  rate 0.333   -121.1 Elo
  net@40024              vs net@130024                400 局  rate 0.341   -114.3 Elo
  net@40024              vs net@140024                400 局  rate 0.341   -114.3 Elo
  net@40024              vs net@150024                400 局  rate 0.306   -142.1 Elo
  net@50024              vs net@60024                 400 局  rate 0.461    -27.0 Elo
  net@50024              vs net@70024                 400 局  rate 0.436    -44.5 Elo
  net@50024              vs net@80024                 400 局  rate 0.419    -57.0 Elo
  net@50024              vs net@90024                 400 局  rate 0.455    -31.4 Elo
  net@50024              vs net@100024                400 局  rate 0.427    -50.7 Elo
  net@50024              vs net@110024                400 局  rate 0.401    -69.5 Elo
  net@50024              vs net@120024                400 局  rate 0.360   -100.0 Elo
  net@50024              vs net@130024                400 局  rate 0.364    -97.1 Elo
  net@50024              vs net@140024                400 局  rate 0.384    -82.3 Elo
  net@50024              vs net@150024                400 局  rate 0.349   -108.5 Elo
  net@60024              vs net@70024                 400 局  rate 0.464    -25.2 Elo
  net@60024              vs net@80024                 400 局  rate 0.482    -12.2 Elo
  net@60024              vs net@90024                 400 局  rate 0.449    -35.7 Elo
  net@60024              vs net@100024                400 局  rate 0.455    -31.4 Elo
  net@60024              vs net@110024                400 局  rate 0.424    -53.4 Elo
  net@60024              vs net@120024                400 局  rate 0.412    -61.4 Elo
  net@60024              vs net@130024                400 局  rate 0.415    -59.6 Elo
  net@60024              vs net@140024                400 局  rate 0.374    -89.7 Elo
  net@60024              vs net@150024                400 局  rate 0.371    -91.5 Elo
  net@70024              vs net@80024                 400 局  rate 0.494     -4.3 Elo
  net@70024              vs net@90024                 400 局  rate 0.456    -30.5 Elo
  net@70024              vs net@100024                400 局  rate 0.477    -15.6 Elo
  net@70024              vs net@110024                400 局  rate 0.461    -27.0 Elo
  net@70024              vs net@120024                400 局  rate 0.477    -15.6 Elo
  net@70024              vs net@130024                400 局  rate 0.424    -53.4 Elo
  net@70024              vs net@140024                400 局  rate 0.383    -83.2 Elo
  net@70024              vs net@150024                400 局  rate 0.412    -61.4 Elo
  net@80024              vs net@90024                 400 局  rate 0.487     -8.7 Elo
  net@80024              vs net@100024                400 局  rate 0.499     -0.9 Elo
  net@80024              vs net@110024                400 局  rate 0.458    -29.6 Elo
  net@80024              vs net@120024                400 局  rate 0.438    -43.7 Elo
  net@80024              vs net@130024                400 局  rate 0.460    -27.9 Elo
  net@80024              vs net@140024                400 局  rate 0.451    -34.0 Elo
  net@80024              vs net@150024                400 局  rate 0.465    -24.4 Elo
  net@90024              vs net@100024                400 局  rate 0.494     -4.3 Elo
  net@90024              vs net@110024                400 局  rate 0.468    -22.6 Elo
  net@90024              vs net@120024                400 局  rate 0.455    -31.4 Elo
  net@90024              vs net@130024                400 局  rate 0.470    -20.9 Elo
  net@90024              vs net@140024                400 局  rate 0.464    -25.2 Elo
  net@90024              vs net@150024                400 局  rate 0.472    -19.1 Elo
  net@100024             vs net@110024                400 局  rate 0.500     +0.0 Elo
  net@100024             vs net@120024                400 局  rate 0.475    -17.4 Elo
  net@100024             vs net@130024                400 局  rate 0.489     -7.8 Elo
  net@100024             vs net@140024                400 局  rate 0.427    -50.7 Elo
  net@100024             vs net@150024                400 局  rate 0.455    -31.4 Elo
  net@110024             vs net@120024                400 局  rate 0.486     -9.6 Elo
  net@110024             vs net@130024                400 局  rate 0.470    -20.9 Elo
  net@110024             vs net@140024                400 局  rate 0.417    -57.9 Elo
  net@110024             vs net@150024                400 局  rate 0.455    -31.4 Elo
  net@120024             vs net@130024                400 局  rate 0.487     -8.7 Elo
  net@120024             vs net@140024                400 局  rate 0.484    -11.3 Elo
  net@120024             vs net@150024                400 局  rate 0.479    -14.8 Elo
  net@130024             vs net@140024                400 局  rate 0.501     +0.9 Elo
  net@130024             vs net@150024                400 局  rate 0.482    -12.2 Elo
  net@140024             vs net@150024                400 局  rate 0.489     -7.8 Elo
```

## `arena_v4-bf16-anneal-late.json`

每对 400 局，64 次模拟

| 参赛者 | Elo | ± | 得分率 | 局数 |
|---|---:|---:|---:|---:|
| `net@111389` | +0.0 | 0.0 | 0.547 | 2400 |
| `net@122189` | -4.6 | 12.6 | 0.540 | 2400 |
| `net@178189` | -22.2 | 13.4 | 0.510 | 2400 |
| `net@211389` | -35.8 | 12.9 | 0.487 | 2400 |
| `net@189389` | -37.8 | 13.7 | 0.484 | 2400 |
| `net@220189` | -45.9 | 13.8 | 0.470 | 2400 |
| `net@200189` | -51.4 | 13.2 | 0.461 | 2400 |

```
  net@111389             vs net@122189                400 局  rate 0.516    +11.3 Elo
  net@111389             vs net@178189                400 局  rate 0.521    +14.8 Elo
  net@111389             vs net@189389                400 局  rate 0.550    +34.9 Elo
  net@111389             vs net@200189                400 局  rate 0.569    +48.1 Elo
  net@111389             vs net@211389                400 局  rate 0.556    +39.3 Elo
  net@111389             vs net@220189                400 局  rate 0.571    +49.8 Elo
  net@122189             vs net@178189                400 局  rate 0.524    +16.5 Elo
  net@122189             vs net@189389                400 局  rate 0.561    +42.8 Elo
  net@122189             vs net@200189                400 局  rate 0.568    +47.2 Elo
  net@122189             vs net@211389                400 局  rate 0.559    +41.0 Elo
  net@122189             vs net@220189                400 局  rate 0.542    +29.6 Elo
  net@178189             vs net@189389                400 局  rate 0.517    +12.2 Elo
  net@178189             vs net@200189                400 局  rate 0.542    +29.6 Elo
  net@178189             vs net@211389                400 局  rate 0.505     +3.5 Elo
  net@178189             vs net@220189                400 局  rate 0.541    +28.7 Elo
  net@189389             vs net@200189                400 局  rate 0.521    +14.8 Elo
  net@189389             vs net@211389                400 局  rate 0.501     +0.9 Elo
  net@189389             vs net@220189                400 局  rate 0.510     +6.9 Elo
  net@200189             vs net@211389                400 局  rate 0.461    -27.0 Elo
  net@200189             vs net@220189                400 局  rate 0.506     +4.3 Elo
  net@211389             vs net@220189                400 局  rate 0.506     +4.3 Elo
```

## `arena_v4-bf16-anneal.json`

每对 400 局，64 次模拟，锚点 `greedy-mobility`

| 参赛者 | Elo | ± | 得分率 | 局数 |
|---|---:|---:|---:|---:|
| `net@122189` | +850.0 | 51.0 | 0.589 | 2800 |
| `net@133389` | +848.2 | 51.3 | 0.586 | 2800 |
| `net@111389` | +842.6 | 50.8 | 0.578 | 2800 |
| `net@144189` | +839.5 | 49.9 | 0.573 | 2800 |
| `net@155389` | +835.9 | 51.1 | 0.568 | 2800 |
| `net@167389` | +835.2 | 50.8 | 0.567 | 2800 |
| `net@220189` | +810.8 | 51.3 | 0.532 | 2800 |
| `greedy-mobility` | +0.0 | 0.0 | 0.007 | 2800 |

```
  greedy-mobility        vs net@111389                400 局  rate 0.005   -919.5 Elo
  greedy-mobility        vs net@122189                400 局  rate 0.015   -726.9 Elo
  greedy-mobility        vs net@133389                400 局  rate 0.003  -1040.4 Elo
  greedy-mobility        vs net@144189                400 局  rate 0.010   -798.3 Elo
  greedy-mobility        vs net@155389                400 局  rate 0.003  -1040.4 Elo
  greedy-mobility        vs net@167389                400 局  rate 0.000     +nan Elo
  greedy-mobility        vs net@220189                400 局  rate 0.013   -759.1 Elo
  net@111389             vs net@122189                400 局  rate 0.490     -6.9 Elo
  net@111389             vs net@133389                400 局  rate 0.494     -4.3 Elo
  net@111389             vs net@144189                400 局  rate 0.519    +13.0 Elo
  net@111389             vs net@155389                400 局  rate 0.496     -2.6 Elo
  net@111389             vs net@167389                400 局  rate 0.512     +8.7 Elo
  net@111389             vs net@220189                400 局  rate 0.539    +27.0 Elo
  net@122189             vs net@133389                400 局  rate 0.509     +6.1 Elo
  net@122189             vs net@144189                400 局  rate 0.519    +13.0 Elo
  net@122189             vs net@155389                400 局  rate 0.512     +8.7 Elo
  net@122189             vs net@167389                400 局  rate 0.544    +30.5 Elo
  net@122189             vs net@220189                400 局  rate 0.541    +28.7 Elo
  net@133389             vs net@144189                400 局  rate 0.490     -6.9 Elo
  net@133389             vs net@155389                400 局  rate 0.507     +5.2 Elo
  net@133389             vs net@167389                400 局  rate 0.527    +19.1 Elo
  net@133389             vs net@220189                400 局  rate 0.583    +57.9 Elo
  net@144189             vs net@155389                400 局  rate 0.506     +4.3 Elo
  net@144189             vs net@167389                400 局  rate 0.530    +20.9 Elo
  net@144189             vs net@220189                400 局  rate 0.515    +10.4 Elo
  net@155389             vs net@167389                400 局  rate 0.491     -6.1 Elo
  net@155389             vs net@220189                400 局  rate 0.511     +7.8 Elo
  net@167389             vs net@220189                400 局  rate 0.575    +52.5 Elo
```

## `arena_v4-bf16-early.json`

每对 400 局，64 次模拟

| 参赛者 | Elo | ± | 得分率 | 局数 |
|---|---:|---:|---:|---:|
| `fork-a111.pt` | +17.2 | 13.3 | 0.540 | 2400 |
| `main-a111.pt` | +14.8 | 13.6 | 0.536 | 2400 |
| `main-a144.pt` | +0.2 | 14.5 | 0.512 | 2400 |
| `fork-a100.pt` | +0.0 | 0.0 | 0.511 | 2400 |
| `fork-a89.pt` | -13.4 | 13.8 | 0.489 | 2400 |
| `fork-a78.pt` | -31.6 | 13.6 | 0.458 | 2400 |
| `fork-a67.pt` | -34.2 | 13.2 | 0.454 | 2400 |

```
  fork-a100.pt           vs fork-a111.pt              400 局  rate 0.455    -31.4 Elo
  fork-a100.pt           vs fork-a67.pt               400 局  rate 0.554    +37.5 Elo
  fork-a100.pt           vs fork-a78.pt               400 局  rate 0.551    +35.7 Elo
  fork-a100.pt           vs fork-a89.pt               400 局  rate 0.517    +12.2 Elo
  fork-a100.pt           vs main-a111.pt              400 局  rate 0.477    -15.6 Elo
  fork-a100.pt           vs main-a144.pt              400 局  rate 0.512     +8.7 Elo
  fork-a111.pt           vs fork-a67.pt               400 局  rate 0.550    +34.9 Elo
  fork-a111.pt           vs fork-a78.pt               400 局  rate 0.560    +41.9 Elo
  fork-a111.pt           vs fork-a89.pt               400 局  rate 0.564    +44.5 Elo
  fork-a111.pt           vs main-a111.pt              400 局  rate 0.510     +6.9 Elo
  fork-a111.pt           vs main-a144.pt              400 局  rate 0.511     +7.8 Elo
  fork-a67.pt            vs fork-a78.pt               400 局  rate 0.511     +7.8 Elo
  fork-a67.pt            vs fork-a89.pt               400 局  rate 0.469    -21.7 Elo
  fork-a67.pt            vs main-a111.pt              400 局  rate 0.405    -66.8 Elo
  fork-a67.pt            vs main-a144.pt              400 局  rate 0.443    -40.1 Elo
  fork-a78.pt            vs fork-a89.pt               400 局  rate 0.451    -34.0 Elo
  fork-a78.pt            vs main-a111.pt              400 局  rate 0.450    -34.9 Elo
  fork-a78.pt            vs main-a144.pt              400 局  rate 0.471    -20.0 Elo
  fork-a89.pt            vs main-a111.pt              400 局  rate 0.460    -27.9 Elo
  fork-a89.pt            vs main-a144.pt              400 局  rate 0.474    -18.3 Elo
  main-a111.pt           vs main-a144.pt              400 局  rate 0.519    +13.0 Elo
```

## `arena_v4-bf16-long.json`

每对 400 局，64 次模拟，锚点 `greedy-mobility`

| 参赛者 | Elo | ± | 得分率 | 局数 |
|---|---:|---:|---:|---:|
| `net@111389` | +791.4 | 49.0 | 0.611 | 2800 |
| `net@130189` | +784.4 | 48.0 | 0.601 | 2800 |
| `net@170189` | +763.3 | 48.1 | 0.571 | 2800 |
| `net@220189` | +761.4 | 47.6 | 0.568 | 2800 |
| `net@110189` | +752.2 | 48.6 | 0.554 | 2800 |
| `net@150189` | +747.1 | 46.9 | 0.547 | 2800 |
| `net@190189` | +740.3 | 48.3 | 0.537 | 2800 |
| `greedy-mobility` | +0.0 | 0.0 | 0.011 | 2800 |

```
  greedy-mobility        vs net@110189                400 局  rate 0.025   -636.4 Elo
  greedy-mobility        vs net@130189                400 局  rate 0.007   -848.7 Elo
  greedy-mobility        vs net@150189                400 局  rate 0.010   -798.3 Elo
  greedy-mobility        vs net@170189                400 局  rate 0.006   -880.6 Elo
  greedy-mobility        vs net@190189                400 局  rate 0.016   -712.8 Elo
  greedy-mobility        vs net@220189                400 局  rate 0.003  -1040.4 Elo
  greedy-mobility        vs net@111389                400 局  rate 0.010   -798.3 Elo
  net@110189             vs net@130189                400 局  rate 0.456    -30.5 Elo
  net@110189             vs net@150189                400 局  rate 0.490     -6.9 Elo
  net@110189             vs net@170189                400 局  rate 0.521    +14.8 Elo
  net@110189             vs net@190189                400 局  rate 0.512     +8.7 Elo
  net@110189             vs net@220189                400 局  rate 0.468    -22.6 Elo
  net@110189             vs net@111389                400 局  rate 0.459    -28.7 Elo
  net@130189             vs net@150189                400 局  rate 0.560    +41.9 Elo
  net@130189             vs net@170189                400 局  rate 0.517    +12.2 Elo
  net@130189             vs net@190189                400 局  rate 0.576    +53.4 Elo
  net@130189             vs net@220189                400 局  rate 0.561    +42.8 Elo
  net@130189             vs net@111389                400 局  rate 0.456    -30.5 Elo
  net@150189             vs net@170189                400 局  rate 0.477    -15.6 Elo
  net@150189             vs net@190189                400 局  rate 0.507     +5.2 Elo
  net@150189             vs net@220189                400 局  rate 0.466    -23.5 Elo
  net@150189             vs net@111389                400 局  rate 0.438    -43.7 Elo
  net@170189             vs net@190189                400 局  rate 0.552    +36.6 Elo
  net@170189             vs net@220189                400 局  rate 0.494     -4.3 Elo
  net@170189             vs net@111389                400 局  rate 0.470    -20.9 Elo
  net@190189             vs net@220189                400 局  rate 0.476    -16.5 Elo
  net@190189             vs net@111389                400 局  rate 0.449    -35.7 Elo
  net@220189             vs net@111389                400 局  rate 0.441    -41.0 Elo
```

## `arena_v4-bf16.json`

每对 400 局，64 次模拟，锚点 `greedy-mobility`

| 参赛者 | Elo | ± | 得分率 | 局数 |
|---|---:|---:|---:|---:|
| `net@111389` | +776.9 | 35.0 | 0.651 | 3200 |
| `net@110189` | +762.4 | 34.9 | 0.631 | 3200 |
| `net@100589` | +748.4 | 35.7 | 0.612 | 3200 |
| `net@90189` | +726.7 | 34.4 | 0.581 | 3200 |
| `net@80189` | +694.5 | 35.3 | 0.535 | 3200 |
| `net@70189` | +687.3 | 34.7 | 0.524 | 3200 |
| `net@60189` | +671.9 | 35.8 | 0.502 | 3200 |
| `net@40189` | +632.7 | 34.5 | 0.448 | 3200 |
| `greedy-mobility` | +0.0 | 0.0 | 0.016 | 3200 |

```
  greedy-mobility        vs net@40189                 400 局  rate 0.021   -665.3 Elo
  greedy-mobility        vs net@60189                 400 局  rate 0.029   -611.5 Elo
  greedy-mobility        vs net@70189                 400 局  rate 0.020   -676.1 Elo
  greedy-mobility        vs net@80189                 400 局  rate 0.015   -726.9 Elo
  greedy-mobility        vs net@90189                 400 局  rate 0.013   -759.1 Elo
  greedy-mobility        vs net@100589                400 局  rate 0.010   -798.3 Elo
  greedy-mobility        vs net@110189                400 局  rate 0.013   -759.1 Elo
  greedy-mobility        vs net@111389                400 局  rate 0.005   -919.5 Elo
  net@40189              vs net@60189                 400 局  rate 0.434    -46.3 Elo
  net@40189              vs net@70189                 400 局  rate 0.466    -23.5 Elo
  net@40189              vs net@80189                 400 局  rate 0.395    -74.1 Elo
  net@40189              vs net@90189                 400 局  rate 0.349   -108.5 Elo
  net@40189              vs net@100589                400 局  rate 0.335   -119.1 Elo
  net@40189              vs net@110189                400 局  rate 0.321   -129.9 Elo
  net@40189              vs net@111389                400 局  rate 0.305   -143.1 Elo
  net@60189              vs net@70189                 400 局  rate 0.440    -41.9 Elo
  net@60189              vs net@80189                 400 局  rate 0.470    -20.9 Elo
  net@60189              vs net@90189                 400 局  rate 0.446    -37.5 Elo
  net@60189              vs net@100589                400 局  rate 0.388    -79.5 Elo
  net@60189              vs net@110189                400 局  rate 0.385    -81.4 Elo
  net@60189              vs net@111389                400 局  rate 0.354   -104.7 Elo
  net@70189              vs net@80189                 400 局  rate 0.477    -15.6 Elo
  net@70189              vs net@90189                 400 局  rate 0.453    -33.1 Elo
  net@70189              vs net@100589                400 局  rate 0.421    -55.2 Elo
  net@70189              vs net@110189                400 局  rate 0.403    -68.6 Elo
  net@70189              vs net@111389                400 局  rate 0.367    -94.3 Elo
  net@80189              vs net@90189                 400 局  rate 0.454    -32.2 Elo
  net@80189              vs net@100589                400 局  rate 0.406    -65.9 Elo
  net@80189              vs net@110189                400 局  rate 0.386    -80.4 Elo
  net@80189              vs net@111389                400 局  rate 0.389    -78.6 Elo
  net@90189              vs net@100589                400 局  rate 0.470    -20.9 Elo
  net@90189              vs net@110189                400 局  rate 0.455    -31.4 Elo
  net@90189              vs net@111389                400 局  rate 0.434    -46.3 Elo
  net@100589             vs net@110189                400 局  rate 0.482    -12.2 Elo
  net@100589             vs net@111389                400 局  rate 0.440    -41.9 Elo
  net@110189             vs net@111389                400 局  rate 0.495     -3.5 Elo
```

## `arena_v4-champs.json`

每对 2000 局，64 次模拟

| 参赛者 | Elo | ± | 得分率 | 局数 |
|---|---:|---:|---:|---:|
| `fp8-144k.pt` | +14.2 | 7.9 | 0.525 | 6000 |
| `bf16-111k.pt` | +0.0 | 0.0 | 0.498 | 6000 |
| `fp4-100k.pt` | -3.5 | 7.8 | 0.492 | 6000 |
| `fp4-144k.pt` | -7.1 | 7.2 | 0.485 | 6000 |

```
  bf16-111k.pt           vs fp4-100k.pt              2000 局  rate 0.496     -3.0 Elo
  bf16-111k.pt           vs fp4-144k.pt              2000 局  rate 0.518    +12.3 Elo
  bf16-111k.pt           vs fp8-144k.pt              2000 局  rate 0.481    -13.0 Elo
  fp4-100k.pt            vs fp4-144k.pt              2000 局  rate 0.494     -4.2 Elo
  fp4-100k.pt            vs fp8-144k.pt              2000 局  rate 0.476    -16.3 Elo
  fp4-144k.pt            vs fp8-144k.pt              2000 局  rate 0.466    -23.8 Elo
```

## `arena_v4-cross-all.json`

每对 800 局，64 次模拟

| 参赛者 | Elo | ± | 得分率 | 局数 |
|---|---:|---:|---:|---:|
| `fp8-144k.pt` | +13.0 | 9.8 | 0.526 | 4000 |
| `bf16-111k.pt` | +0.0 | 0.0 | 0.504 | 4000 |
| `bf16-133k.pt` | -2.8 | 9.7 | 0.499 | 4000 |
| `fp8-122k.pt` | -3.5 | 9.8 | 0.498 | 4000 |
| `fp4-144k.pt` | -10.0 | 9.9 | 0.487 | 4000 |
| `fp4-111k.pt` | -10.1 | 9.8 | 0.486 | 4000 |

```
  bf16-111k.pt           vs bf16-133k.pt              800 局  rate 0.511     +7.4 Elo
  bf16-111k.pt           vs fp4-111k.pt               800 局  rate 0.509     +6.5 Elo
  bf16-111k.pt           vs fp4-144k.pt               800 局  rate 0.509     +6.5 Elo
  bf16-111k.pt           vs fp8-122k.pt               800 局  rate 0.514     +9.6 Elo
  bf16-111k.pt           vs fp8-144k.pt               800 局  rate 0.476    -16.5 Elo
  bf16-133k.pt           vs fp4-111k.pt               800 局  rate 0.521    +14.3 Elo
  bf16-133k.pt           vs fp4-144k.pt               800 局  rate 0.516    +10.9 Elo
  bf16-133k.pt           vs fp8-122k.pt               800 局  rate 0.509     +6.1 Elo
  bf16-133k.pt           vs fp8-144k.pt               800 局  rate 0.461    -27.4 Elo
  fp4-111k.pt            vs fp4-144k.pt               800 局  rate 0.488     -8.3 Elo
  fp4-111k.pt            vs fp8-122k.pt               800 局  rate 0.492     -5.2 Elo
  fp4-111k.pt            vs fp8-144k.pt               800 局  rate 0.481    -13.0 Elo
  fp4-144k.pt            vs fp8-122k.pt               800 局  rate 0.486     -9.6 Elo
  fp4-144k.pt            vs fp8-144k.pt               800 局  rate 0.460    -27.9 Elo
  fp8-122k.pt            vs fp8-144k.pt               800 局  rate 0.490     -6.9 Elo
```

## `arena_v4-cross-bf16-fp8.json`

每对 800 局，64 次模拟

| 参赛者 | Elo | ± | 得分率 | 局数 |
|---|---:|---:|---:|---:|
| `fp8-144k.pt` | +16.9 | 11.3 | 0.523 | 2400 |
| `bf16-133k.pt` | +3.8 | 11.8 | 0.498 | 2400 |
| `bf16-111k.pt` | +0.0 | 0.0 | 0.491 | 2400 |
| `fp8-122k.pt` | -1.2 | 12.6 | 0.488 | 2400 |

```
  bf16-111k.pt           vs bf16-133k.pt              800 局  rate 0.492     -5.6 Elo
  bf16-111k.pt           vs fp8-122k.pt               800 局  rate 0.494     -4.3 Elo
  bf16-111k.pt           vs fp8-144k.pt               800 局  rate 0.486     -9.6 Elo
  bf16-133k.pt           vs fp8-122k.pt               800 局  rate 0.511     +7.8 Elo
  bf16-133k.pt           vs fp8-144k.pt               800 局  rate 0.474    -17.8 Elo
  fp8-122k.pt            vs fp8-144k.pt               800 局  rate 0.470    -20.9 Elo
```

## `arena_v4-cross2.json`

每对 1200 局，64 次模拟

| 参赛者 | Elo | ± | 得分率 | 局数 |
|---|---:|---:|---:|---:|
| `fp8-144k.pt` | +20.3 | 11.1 | 0.539 | 3600 |
| `bf16-111k.pt` | +0.0 | 0.0 | 0.500 | 3600 |
| `fp4-144k.pt` | -5.8 | 11.1 | 0.489 | 3600 |
| `fp4-111k.pt` | -15.4 | 10.6 | 0.471 | 3600 |

```
  bf16-111k.pt           vs fp4-111k.pt              1200 局  rate 0.530    +21.2 Elo
  bf16-111k.pt           vs fp4-144k.pt              1200 局  rate 0.506     +4.1 Elo
  bf16-111k.pt           vs fp8-144k.pt              1200 局  rate 0.465    -24.4 Elo
  fp4-111k.pt            vs fp4-144k.pt              1200 局  rate 0.493     -4.6 Elo
  fp4-111k.pt            vs fp8-144k.pt              1200 局  rate 0.450    -34.9 Elo
  fp4-144k.pt            vs fp8-144k.pt              1200 局  rate 0.467    -22.9 Elo
```

## `arena_v4-cross3.json`

每对 1200 局，64 次模拟

| 参赛者 | Elo | ± | 得分率 | 局数 |
|---|---:|---:|---:|---:|
| `fp8-144k.pt` | +7.8 | 10.0 | 0.520 | 3600 |
| `bf16-111k.pt` | +0.0 | 0.0 | 0.505 | 3600 |
| `bf16-133k.pt` | -2.2 | 9.8 | 0.500 | 3600 |
| `fp4-144k.pt` | -15.4 | 10.0 | 0.475 | 3600 |

```
  bf16-111k.pt           vs bf16-133k.pt             1200 局  rate 0.501     +0.9 Elo
  bf16-111k.pt           vs fp4-144k.pt              1200 局  rate 0.518    +12.7 Elo
  bf16-111k.pt           vs fp8-144k.pt              1200 局  rate 0.495     -3.8 Elo
  bf16-133k.pt           vs fp4-144k.pt              1200 局  rate 0.529    +20.3 Elo
  bf16-133k.pt           vs fp8-144k.pt              1200 局  rate 0.473    -18.5 Elo
  fp4-144k.pt            vs fp8-144k.pt              1200 局  rate 0.473    -18.8 Elo
```

## `arena_v4-cross4.json`

每对 1200 局，64 次模拟

| 参赛者 | Elo | ± | 得分率 | 局数 |
|---|---:|---:|---:|---:|
| `fp8-144k.pt` | +9.4 | 10.4 | 0.532 | 3600 |
| `bf16-111k.pt` | +0.0 | 0.0 | 0.514 | 3600 |
| `fp4-111k.pt` | -16.5 | 10.7 | 0.483 | 3600 |
| `fp8-111k.pt` | -23.0 | 10.6 | 0.470 | 3600 |

```
  bf16-111k.pt           vs fp4-111k.pt              1200 局  rate 0.519    +13.3 Elo
  bf16-111k.pt           vs fp8-111k.pt              1200 局  rate 0.533    +23.2 Elo
  bf16-111k.pt           vs fp8-144k.pt              1200 局  rate 0.491     -6.4 Elo
  fp4-111k.pt            vs fp8-111k.pt              1200 局  rate 0.497     -2.3 Elo
  fp4-111k.pt            vs fp8-144k.pt              1200 局  rate 0.471    -20.3 Elo
  fp8-111k.pt            vs fp8-144k.pt              1200 局  rate 0.441    -41.3 Elo
```

## `arena_v4-fp4-anneal.json`

每对 400 局，64 次模拟

| 参赛者 | Elo | ± | 得分率 | 局数 |
|---|---:|---:|---:|---:|
| `net@144187` | +5.7 | 11.4 | 0.533 | 3600 |
| `net@133387` | +4.5 | 10.9 | 0.531 | 3600 |
| `net@122187` | +1.0 | 10.8 | 0.526 | 3600 |
| `net@111387` | +0.0 | 0.0 | 0.524 | 3600 |
| `net@155387` | -17.5 | 10.8 | 0.496 | 3600 |
| `net@178187` | -23.7 | 11.0 | 0.486 | 3600 |
| `net@200187` | -28.3 | 10.9 | 0.479 | 3600 |
| `net@167387` | -29.4 | 10.7 | 0.477 | 3600 |
| `net@211387` | -30.2 | 9.9 | 0.476 | 3600 |
| `net@189387` | -31.4 | 10.8 | 0.474 | 3600 |

```
  net@111387             vs net@122187                400 局  rate 0.482    -12.2 Elo
  net@111387             vs net@133387                400 局  rate 0.491     -6.1 Elo
  net@111387             vs net@144187                400 局  rate 0.504     +2.6 Elo
  net@111387             vs net@155387                400 局  rate 0.497     -1.7 Elo
  net@111387             vs net@167387                400 局  rate 0.532    +22.6 Elo
  net@111387             vs net@178187                400 局  rate 0.555    +38.4 Elo
  net@111387             vs net@189387                400 局  rate 0.566    +46.3 Elo
  net@111387             vs net@200187                400 局  rate 0.546    +32.2 Elo
  net@111387             vs net@211387                400 局  rate 0.540    +27.9 Elo
  net@122187             vs net@133387                400 局  rate 0.510     +6.9 Elo
  net@122187             vs net@144187                400 局  rate 0.501     +0.9 Elo
  net@122187             vs net@155387                400 局  rate 0.532    +22.6 Elo
  net@122187             vs net@167387                400 局  rate 0.536    +25.2 Elo
  net@122187             vs net@178187                400 局  rate 0.505     +3.5 Elo
  net@122187             vs net@189387                400 局  rate 0.547    +33.1 Elo
  net@122187             vs net@200187                400 局  rate 0.551    +35.7 Elo
  net@122187             vs net@211387                400 局  rate 0.529    +20.0 Elo
  net@133387             vs net@144187                400 局  rate 0.490     -6.9 Elo
  net@133387             vs net@155387                400 局  rate 0.540    +27.9 Elo
  net@133387             vs net@167387                400 局  rate 0.536    +25.2 Elo
  net@133387             vs net@178187                400 局  rate 0.573    +50.7 Elo
  net@133387             vs net@189387                400 局  rate 0.547    +33.1 Elo
  net@133387             vs net@200187                400 局  rate 0.545    +31.4 Elo
  net@133387             vs net@211387                400 局  rate 0.550    +34.9 Elo
  net@144187             vs net@155387                400 局  rate 0.557    +40.1 Elo
  net@144187             vs net@167387                400 局  rate 0.565    +45.4 Elo
  net@144187             vs net@178187                400 局  rate 0.544    +30.5 Elo
  net@144187             vs net@189387                400 局  rate 0.545    +31.4 Elo
  net@144187             vs net@200187                400 局  rate 0.546    +32.2 Elo
  net@144187             vs net@211387                400 局  rate 0.534    +23.5 Elo
  net@155387             vs net@167387                400 局  rate 0.525    +17.4 Elo
  net@155387             vs net@178187                400 局  rate 0.529    +20.0 Elo
  net@155387             vs net@189387                400 局  rate 0.512     +8.7 Elo
  net@155387             vs net@200187                400 局  rate 0.512     +8.7 Elo
  net@155387             vs net@211387                400 局  rate 0.511     +7.8 Elo
  net@167387             vs net@178187                400 局  rate 0.495     -3.5 Elo
  net@167387             vs net@189387                400 局  rate 0.471    -20.0 Elo
  net@167387             vs net@200187                400 局  rate 0.500     +0.0 Elo
  net@167387             vs net@211387                400 局  rate 0.521    +14.8 Elo
  net@178187             vs net@189387                400 局  rate 0.531    +21.7 Elo
  net@178187             vs net@200187                400 局  rate 0.512     +8.7 Elo
  net@178187             vs net@211387                400 局  rate 0.530    +20.9 Elo
  net@189387             vs net@200187                400 局  rate 0.481    -13.0 Elo
  net@189387             vs net@211387                400 局  rate 0.502     +1.7 Elo
  net@200187             vs net@211387                400 局  rate 0.502     +1.7 Elo
```

## `arena_v4-fp4-early.json`

每对 400 局，64 次模拟

| 参赛者 | Elo | ± | 得分率 | 局数 |
|---|---:|---:|---:|---:|
| `fork-a100.pt` | +0.0 | 0.0 | 0.516 | 2400 |
| `fork-a78.pt` | -3.2 | 13.7 | 0.510 | 2400 |
| `fork-a89.pt` | -4.3 | 12.9 | 0.509 | 2400 |
| `fork-a111.pt` | -8.7 | 12.9 | 0.501 | 2400 |
| `main-a111.pt` | -11.0 | 14.0 | 0.497 | 2400 |
| `main-a144.pt` | -12.1 | 13.0 | 0.495 | 2400 |
| `fork-a67.pt` | -26.5 | 13.9 | 0.471 | 2400 |

```
  fork-a100.pt           vs fork-a111.pt              400 局  rate 0.531    +21.7 Elo
  fork-a100.pt           vs fork-a67.pt               400 局  rate 0.516    +11.3 Elo
  fork-a100.pt           vs fork-a78.pt               400 局  rate 0.490     -6.9 Elo
  fork-a100.pt           vs fork-a89.pt               400 局  rate 0.527    +19.1 Elo
  fork-a100.pt           vs main-a111.pt              400 局  rate 0.504     +2.6 Elo
  fork-a100.pt           vs main-a144.pt              400 局  rate 0.526    +18.3 Elo
  fork-a111.pt           vs fork-a67.pt               400 局  rate 0.535    +24.4 Elo
  fork-a111.pt           vs fork-a78.pt               400 局  rate 0.477    -15.6 Elo
  fork-a111.pt           vs fork-a89.pt               400 局  rate 0.492     -5.2 Elo
  fork-a111.pt           vs main-a111.pt              400 局  rate 0.524    +16.5 Elo
  fork-a111.pt           vs main-a144.pt              400 局  rate 0.510     +6.9 Elo
  fork-a67.pt            vs fork-a78.pt               400 局  rate 0.470    -20.9 Elo
  fork-a67.pt            vs fork-a89.pt               400 局  rate 0.468    -22.6 Elo
  fork-a67.pt            vs main-a111.pt              400 局  rate 0.469    -21.7 Elo
  fork-a67.pt            vs main-a144.pt              400 局  rate 0.472    -19.1 Elo
  fork-a78.pt            vs fork-a89.pt               400 局  rate 0.511     +7.8 Elo
  fork-a78.pt            vs main-a111.pt              400 局  rate 0.475    -17.4 Elo
  fork-a78.pt            vs main-a144.pt              400 局  rate 0.514     +9.6 Elo
  fork-a89.pt            vs main-a111.pt              400 局  rate 0.532    +22.6 Elo
  fork-a89.pt            vs main-a144.pt              400 局  rate 0.517    +12.2 Elo
  main-a111.pt           vs main-a144.pt              400 局  rate 0.487     -8.7 Elo
```

## `arena_v4-fp4.json`

每对 400 局，64 次模拟，锚点 `greedy-mobility`

| 参赛者 | Elo | ± | 得分率 | 局数 |
|---|---:|---:|---:|---:|
| `net@100587` | +773.3 | 34.6 | 0.613 | 3200 |
| `net@80187` | +772.3 | 34.6 | 0.611 | 3200 |
| `net@110187` | +771.0 | 34.8 | 0.609 | 3200 |
| `net@111387` | +768.0 | 34.8 | 0.605 | 3201 |
| `net@90187` | +737.7 | 34.7 | 0.562 | 3201 |
| `net@60187` | +700.9 | 35.8 | 0.509 | 3200 |
| `net@70187` | +697.3 | 34.1 | 0.504 | 3200 |
| `net@40187` | +675.8 | 35.3 | 0.474 | 3200 |
| `greedy-mobility` | +0.0 | 0.0 | 0.013 | 3200 |

```
  greedy-mobility        vs net@40187                 400 局  rate 0.022   -655.2 Elo
  greedy-mobility        vs net@60187                 400 局  rate 0.028   -619.4 Elo
  greedy-mobility        vs net@70187                 400 局  rate 0.019   -687.5 Elo
  greedy-mobility        vs net@80187                 400 局  rate 0.005   -919.5 Elo
  greedy-mobility        vs net@90187                 400 局  rate 0.013   -759.1 Elo
  greedy-mobility        vs net@100587                400 局  rate 0.003  -1040.4 Elo
  greedy-mobility        vs net@110187                400 局  rate 0.005   -919.5 Elo
  greedy-mobility        vs net@111387                400 局  rate 0.013   -759.1 Elo
  net@40187              vs net@60187                 400 局  rate 0.470    -20.9 Elo
  net@40187              vs net@70187                 400 局  rate 0.476    -16.5 Elo
  net@40187              vs net@80187                 400 局  rate 0.374    -89.7 Elo
  net@40187              vs net@90187                 400 局  rate 0.385    -81.4 Elo
  net@40187              vs net@100587                400 局  rate 0.369    -93.4 Elo
  net@40187              vs net@110187                400 局  rate 0.362    -98.1 Elo
  net@40187              vs net@111387                400 局  rate 0.375    -88.7 Elo
  net@60187              vs net@70187                 400 局  rate 0.514     +9.6 Elo
  net@60187              vs net@80187                 400 局  rate 0.411    -62.3 Elo
  net@60187              vs net@90187                 400 局  rate 0.445    -38.4 Elo
  net@60187              vs net@100587                400 局  rate 0.411    -62.3 Elo
  net@60187              vs net@110187                400 局  rate 0.389    -78.6 Elo
  net@60187              vs net@111387                400 局  rate 0.400    -70.4 Elo
  net@70187              vs net@80187                 400 局  rate 0.409    -64.1 Elo
  net@70187              vs net@90187                 400 局  rate 0.427    -50.7 Elo
  net@70187              vs net@100587                400 局  rate 0.404    -67.7 Elo
  net@70187              vs net@110187                400 局  rate 0.400    -70.4 Elo
  net@70187              vs net@111387                400 局  rate 0.400    -70.4 Elo
  net@80187              vs net@90187                 400 局  rate 0.555    +38.4 Elo
  net@80187              vs net@100587                400 局  rate 0.500     +0.0 Elo
  net@80187              vs net@110187                400 局  rate 0.496     -2.6 Elo
  net@80187              vs net@111387                400 局  rate 0.537    +26.1 Elo
  net@90187              vs net@100587                400 局  rate 0.432    -47.2 Elo
  net@90187              vs net@110187                400 局  rate 0.440    -41.9 Elo
  net@90187              vs net@111387                401 局  rate 0.446    -37.4 Elo
  net@100587             vs net@110187                400 局  rate 0.521    +14.8 Elo
  net@100587             vs net@111387                400 局  rate 0.499     -0.9 Elo
  net@110187             vs net@111387                400 局  rate 0.489     -7.8 Elo
```

## `arena_v4-fp8-anneal.json`

每对 400 局，64 次模拟

| 参赛者 | Elo | ± | 得分率 | 局数 |
|---|---:|---:|---:|---:|
| `net@111391` | +0.0 | 0.0 | — | — |
| `net@122191` | +0.0 | 0.0 | — | — |
| `net@144191` | +0.0 | 0.0 | — | — |
| `net@167391` | +0.0 | 0.0 | — | — |
| `net@189391` | +0.0 | 0.0 | — | — |
| `net@211391` | +0.0 | 0.0 | — | — |

```
  net@111391             vs net@122191                400 局  rate 0.480    -13.9 Elo
  net@111391             vs net@144191                400 局  rate 0.468    -22.3 Elo
  net@111391             vs net@167391                400 局  rate 0.479    -14.6 Elo
  net@111391             vs net@189391                400 局  rate 0.510     +6.9 Elo
  net@111391             vs net@211391                400 局  rate 0.486     -9.7 Elo
  net@122191             vs net@144191                400 局  rate 0.448    -36.3 Elo
  net@122191             vs net@167391                400 局  rate 0.524    +16.7 Elo
  net@122191             vs net@189391                400 局  rate 0.544    +30.7 Elo
  net@122191             vs net@211391                400 局  rate 0.510     +6.9 Elo
  net@144191             vs net@167391                400 局  rate 0.517    +11.8 Elo
  net@144191             vs net@189391                400 局  rate 0.530    +20.9 Elo
  net@144191             vs net@211391                400 局  rate 0.546    +32.1 Elo
```

## `arena_v4-fp8-early.json`

每对 400 局，64 次模拟

| 参赛者 | Elo | ± | 得分率 | 局数 |
|---|---:|---:|---:|---:|
| `main-a144.pt` | +44.4 | 12.8 | 0.563 | 2400 |
| `fork-a111.pt` | +27.3 | 13.5 | 0.534 | 2400 |
| `main-a111.pt` | +19.9 | 12.4 | 0.522 | 2400 |
| `fork-a100.pt` | +0.0 | 0.0 | 0.489 | 2400 |
| `fork-a89.pt` | -1.5 | 13.9 | 0.486 | 2400 |
| `fork-a78.pt` | -10.2 | 12.3 | 0.472 | 2400 |
| `fork-a67.pt` | -32.9 | 13.8 | 0.434 | 2400 |

```
  fork-a100.pt           vs fork-a111.pt              400 局  rate 0.448    -36.6 Elo
  fork-a100.pt           vs fork-a67.pt               400 局  rate 0.542    +29.6 Elo
  fork-a100.pt           vs fork-a78.pt               400 局  rate 0.530    +20.9 Elo
  fork-a100.pt           vs fork-a89.pt               400 局  rate 0.521    +14.8 Elo
  fork-a100.pt           vs main-a111.pt              400 局  rate 0.497     -1.7 Elo
  fork-a100.pt           vs main-a144.pt              400 局  rate 0.394    -75.0 Elo
  fork-a111.pt           vs fork-a67.pt               400 局  rate 0.611    +78.6 Elo
  fork-a111.pt           vs fork-a78.pt               400 局  rate 0.544    +30.5 Elo
  fork-a111.pt           vs fork-a89.pt               400 局  rate 0.524    +16.5 Elo
  fork-a111.pt           vs main-a111.pt              400 局  rate 0.501     +0.9 Elo
  fork-a111.pt           vs main-a144.pt              400 局  rate 0.474    -18.3 Elo
  fork-a67.pt            vs fork-a78.pt               400 局  rate 0.471    -20.0 Elo
  fork-a67.pt            vs fork-a89.pt               400 局  rate 0.451    -34.0 Elo
  fork-a67.pt            vs main-a111.pt              400 局  rate 0.432    -47.2 Elo
  fork-a67.pt            vs main-a144.pt              400 局  rate 0.403    -68.6 Elo
  fork-a78.pt            vs fork-a89.pt               400 局  rate 0.510     +6.9 Elo
  fork-a78.pt            vs main-a111.pt              400 局  rate 0.451    -34.0 Elo
  fork-a78.pt            vs main-a144.pt              400 局  rate 0.414    -60.5 Elo
  fork-a89.pt            vs main-a111.pt              400 局  rate 0.481    -13.0 Elo
  fork-a89.pt            vs main-a144.pt              400 局  rate 0.443    -40.1 Elo
  main-a111.pt           vs main-a144.pt              400 局  rate 0.496     -2.6 Elo
```

## `arena_v4-fp8.json`

每对 400 局，64 次模拟，锚点 `greedy-mobility`

| 参赛者 | Elo | ± | 得分率 | 局数 |
|---|---:|---:|---:|---:|
| `net@111391` | +745.3 | 32.1 | 0.640 | 3200 |
| `net@110191` | +735.3 | 31.8 | 0.626 | 3200 |
| `net@100591` | +714.8 | 32.7 | 0.597 | 3201 |
| `net@70191` | +705.1 | 31.9 | 0.583 | 3200 |
| `net@80191` | +701.0 | 30.7 | 0.578 | 3200 |
| `net@60191` | +648.0 | 31.2 | 0.502 | 3200 |
| `net@90191` | +642.5 | 31.9 | 0.494 | 3200 |
| `net@40191` | +619.5 | 31.1 | 0.462 | 3200 |
| `greedy-mobility` | +0.0 | 0.0 | 0.018 | 3201 |

```
  greedy-mobility        vs net@40191                 400 局  rate 0.022   -655.2 Elo
  greedy-mobility        vs net@60191                 400 局  rate 0.028   -619.4 Elo
  greedy-mobility        vs net@70191                 400 局  rate 0.014   -742.3 Elo
  greedy-mobility        vs net@80191                 400 局  rate 0.010   -798.3 Elo
  greedy-mobility        vs net@90191                 400 局  rate 0.030   -603.9 Elo
  greedy-mobility        vs net@100591                401 局  rate 0.025   -636.9 Elo
  greedy-mobility        vs net@110191                400 局  rate 0.011   -777.6 Elo
  greedy-mobility        vs net@111391                400 局  rate 0.004   -969.7 Elo
  net@40191              vs net@60191                 400 局  rate 0.480    -13.9 Elo
  net@40191              vs net@70191                 400 局  rate 0.388    -79.5 Elo
  net@40191              vs net@80191                 400 局  rate 0.371    -91.5 Elo
  net@40191              vs net@90191                 400 局  rate 0.494     -4.3 Elo
  net@40191              vs net@100591                400 局  rate 0.359   -100.9 Elo
  net@40191              vs net@110191                400 局  rate 0.325   -127.0 Elo
  net@40191              vs net@111391                400 局  rate 0.300   -147.2 Elo
  net@60191              vs net@70191                 400 局  rate 0.421    -55.2 Elo
  net@60191              vs net@80191                 400 局  rate 0.435    -45.4 Elo
  net@60191              vs net@90191                 400 局  rate 0.486     -9.6 Elo
  net@60191              vs net@100591                400 局  rate 0.398    -72.2 Elo
  net@60191              vs net@110191                400 局  rate 0.394    -75.0 Elo
  net@60191              vs net@111391                400 局  rate 0.388    -79.5 Elo
  net@70191              vs net@80191                 400 局  rate 0.536    +25.2 Elo
  net@70191              vs net@90191                 400 局  rate 0.569    +48.1 Elo
  net@70191              vs net@100591                400 局  rate 0.490     -6.9 Elo
  net@70191              vs net@110191                400 局  rate 0.453    -33.1 Elo
  net@70191              vs net@111391                400 局  rate 0.443    -40.1 Elo
  net@80191              vs net@90191                 400 局  rate 0.579    +55.2 Elo
  net@80191              vs net@100591                400 局  rate 0.510     +6.9 Elo
  net@80191              vs net@110191                400 局  rate 0.438    -43.7 Elo
  net@80191              vs net@111391                400 局  rate 0.446    -37.5 Elo
  net@90191              vs net@100591                400 局  rate 0.361    -99.0 Elo
  net@90191              vs net@110191                400 局  rate 0.405    -66.8 Elo
  net@90191              vs net@111391                400 局  rate 0.343   -113.3 Elo
  net@100591             vs net@110191                400 局  rate 0.459    -28.7 Elo
  net@100591             vs net@111391                400 局  rate 0.461    -27.0 Elo
  net@110191             vs net@111391                400 局  rate 0.494     -4.3 Elo
```

## `arena_v5-gen.json`

每对 800 局，64 次模拟

| 参赛者 | Elo | ± | 得分率 | 局数 |
|---|---:|---:|---:|---:|
| `d256-fp8.pt` | +19.2 | 10.8 | 0.602 | 4000 |
| `d256-bf16.pt` | +0.0 | 0.0 | 0.571 | 4000 |
| `d256-fp4.pt` | -7.4 | 9.9 | 0.558 | 4000 |
| `d384-fp8.pt` | -41.5 | 9.5 | 0.501 | 4000 |
| `d384-bf16.pt` | -64.0 | 10.4 | 0.464 | 4000 |
| `d384-fp4.pt` | -163.9 | 10.7 | 0.303 | 4000 |

```
  d256-bf16.pt           vs d256-fp4.pt               800 局  rate 0.531    +21.7 Elo
  d256-bf16.pt           vs d256-fp8.pt               800 局  rate 0.463    -26.1 Elo
  d256-bf16.pt           vs d384-bf16.pt              800 局  rate 0.589    +62.3 Elo
  d256-bf16.pt           vs d384-fp4.pt               800 局  rate 0.711   +156.1 Elo
  d256-bf16.pt           vs d384-fp8.pt               800 局  rate 0.561    +42.3 Elo
  d256-fp4.pt            vs d256-fp8.pt               800 局  rate 0.483    -11.7 Elo
  d256-fp4.pt            vs d384-bf16.pt              800 局  rate 0.581    +57.0 Elo
  d256-fp4.pt            vs d384-fp4.pt               800 局  rate 0.719   +163.5 Elo
  d256-fp4.pt            vs d384-fp8.pt               800 局  rate 0.540    +27.9 Elo
  d256-fp8.pt            vs d384-bf16.pt              800 局  rate 0.628    +91.1 Elo
  d256-fp8.pt            vs d384-fp4.pt               800 局  rate 0.748   +189.1 Elo
  d256-fp8.pt            vs d384-fp8.pt               800 局  rate 0.581    +57.0 Elo
  d384-bf16.pt           vs d384-fp4.pt               800 局  rate 0.644   +103.3 Elo
  d384-bf16.pt           vs d384-fp8.pt               800 局  rate 0.472    -19.6 Elo
  d384-fp4.pt            vs d384-fp8.pt               800 局  rate 0.339   -115.7 Elo
```

## `arena_v5-lr-fp4.json`

每对 1200 局，64 次模拟

| 参赛者 | Elo | ± | 得分率 | 局数 |
|---|---:|---:|---:|---:|
| `d384-lr13.pt` | +21.8 | 11.2 | 0.635 | 2400 |
| `d256-lr20.pt` | +0.0 | 0.0 | 0.591 | 2400 |
| `d384-lr20.pt` | -158.0 | 11.5 | 0.274 | 2400 |

```
  d256-lr20.pt           vs d384-lr13.pt             1200 局  rate 0.475    -17.7 Elo
  d256-lr20.pt           vs d384-lr20.pt             1200 局  rate 0.707   +153.1 Elo
  d384-lr13.pt           vs d384-lr20.pt             1200 局  rate 0.744   +185.5 Elo
```

## `arena_v5-lr.json`

每对 1200 局，64 次模拟

| 参赛者 | Elo | ± | 得分率 | 局数 |
|---|---:|---:|---:|---:|
| `d256-lr20.pt` | +0.0 | 0.0 | 0.568 | 2400 |
| `d384-lr13.pt` | -24.2 | 10.2 | 0.516 | 2400 |
| `d384-lr20.pt` | -70.6 | 10.7 | 0.417 | 2400 |

```
  d256-lr20.pt           vs d384-lr13.pt             1200 局  rate 0.541    +28.4 Elo
  d256-lr20.pt           vs d384-lr20.pt             1200 局  rate 0.594    +66.2 Elo
  d384-lr13.pt           vs d384-lr20.pt             1200 局  rate 0.573    +50.7 Elo
```

## `arena_v5-width.json`

每对 1200 局，64 次模拟

| 参赛者 | Elo | ± | 得分率 | 局数 |
|---|---:|---:|---:|---:|
| `d256-111k.pt` | +0.0 | 0.0 | 0.609 | 1200 |
| `d384-111k.pt` | -77.0 | 13.5 | 0.391 | 1200 |

```
  d256-111k.pt           vs d384-111k.pt             1200 局  rate 0.609    +77.1 Elo
```

