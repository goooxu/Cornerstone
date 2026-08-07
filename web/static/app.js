// cornerstone 试玩前端。
//
// 一条硬规则：**合法性判断一律不在这里重写**。后端返回 legal_actions，
// 前端只做「这个动作在不在集合里」的查表。规则实现只有 C++ 引擎那一份，
// 训练、评测、试玩共用，不会出现两边判定不一致。

const $ = (id) => document.getElementById(id);

const S = {
  meta: null,
  sid: null,
  state: null,
  legal: new Set(),
  piece: null,          // 选中的棋子 id
  ori: null,            // 选中的朝向 id
  hover: null,          // [r, c]
  analysis: null,
  busy: false,
  backends: [],
  autoplay: false,      // AI 对战连打中
};

const COLORS = ['#2dd4bf', '#fb923c'];

// ---------------------------------------------------------------- 网络请求

async function api(path, opts) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).detail || msg; } catch (e) { /* 忽略 */ }
    throw new Error(msg);
  }
  return res.json();
}

const post = (path, body) => api(path, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body),
});

// ---------------------------------------------------------------- 棋盘绘制

const CV = $('board');
const CTX = CV.getContext('2d');
let CELL = 46, PAD = 14;

function layout() {
  const n = S.meta.board_n;
  CELL = Math.floor((CV.width - PAD * 2) / n);
}

function cellAt(ev) {
  const rect = CV.getBoundingClientRect();
  const x = (ev.clientX - rect.left) * (CV.width / rect.width) - PAD;
  const y = (ev.clientY - rect.top) * (CV.height / rect.height) - PAD;
  const c = Math.floor(x / CELL), r = Math.floor(y / CELL);
  const n = S.meta.board_n;
  return (r >= 0 && r < n && c >= 0 && c < n) ? [r, c] : null;
}

function actionFor(r, c) {
  if (S.ori === null) return null;
  return S.ori * S.meta.num_cells + r * S.meta.board_n + c;
}

// 当前朝向所有合法的锚点（左上角）
function legalAnchors() {
  const out = [];
  if (S.ori === null) return out;
  const base = S.ori * S.meta.num_cells;
  for (let i = 0; i < S.meta.num_cells; i++) {
    if (S.legal.has(base + i)) out.push([Math.floor(i / S.meta.board_n), i % S.meta.board_n]);
  }
  return out;
}

function oriCells(oriId) {
  for (const p of S.meta.pieces) {
    for (const o of p.orientations) if (o.id === oriId) return o.cells;
  }
  return [];
}

function draw() {
  const n = S.meta.board_n;
  CTX.clearRect(0, 0, CV.width, CV.height);
  CTX.fillStyle = '#0e1118';
  CTX.fillRect(0, 0, CV.width, CV.height);

  const st = S.state;

  // 热力图（AI 想下在哪）
  if ($('show-heat').checked && S.analysis && S.analysis.heatmap) {
    for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) {
      const v = S.analysis.heatmap[r][c];
      if (v > 0.01) {
        CTX.fillStyle = `rgba(129,140,248,${(0.12 + 0.55 * v).toFixed(3)})`;
        CTX.fillRect(PAD + c * CELL, PAD + r * CELL, CELL, CELL);
      }
    }
  }

  // 网格
  CTX.strokeStyle = '#2a3143';
  CTX.lineWidth = 1;
  for (let i = 0; i <= n; i++) {
    CTX.beginPath();
    CTX.moveTo(PAD + i * CELL, PAD); CTX.lineTo(PAD + i * CELL, PAD + n * CELL);
    CTX.moveTo(PAD, PAD + i * CELL); CTX.lineTo(PAD + n * CELL, PAD + i * CELL);
    CTX.stroke();
  }

  // 起始格
  S.meta.start_cells.forEach(([r, c], p) => {
    CTX.strokeStyle = COLORS[p]; CTX.lineWidth = 2;
    CTX.beginPath();
    CTX.arc(PAD + (c + 0.5) * CELL, PAD + (r + 0.5) * CELL, CELL * 0.28, 0, Math.PI * 2);
    CTX.stroke();
  });

  // 已落子
  if (st) {
    for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) {
      const v = st.grid[r][c];
      if (v < 0) continue;
      CTX.fillStyle = COLORS[v];
      roundRect(PAD + c * CELL + 1.5, PAD + r * CELL + 1.5, CELL - 3, CELL - 3, 4);
      CTX.fill();
    }
  }

  // 选中朝向的合法锚点
  if (st && !st.terminal && st.current_player === st.human_player) {
    CTX.fillStyle = COLORS[st.human_player] + '66';
    for (const [r, c] of legalAnchors()) {
      CTX.beginPath();
      CTX.arc(PAD + (c + 0.5) * CELL, PAD + (r + 0.5) * CELL, 3.5, 0, Math.PI * 2);
      CTX.fill();
    }
  }

  // 悬停预览
  if (S.hover && S.ori !== null && st && !st.terminal && st.current_player === st.human_player) {
    const [hr, hc] = S.hover;
    const a = actionFor(hr, hc);
    const ok = S.legal.has(a);
    CTX.fillStyle = ok ? COLORS[st.human_player] + 'bb' : '#f8717166';
    CTX.strokeStyle = ok ? '#ffffff88' : '#f87171';
    CTX.lineWidth = 1.5;
    for (const [dr, dc] of oriCells(S.ori)) {
      const r = hr + dr, c = hc + dc;
      if (r < 0 || r >= n || c < 0 || c >= n) continue;
      roundRect(PAD + c * CELL + 1.5, PAD + r * CELL + 1.5, CELL - 3, CELL - 3, 4);
      CTX.fill(); CTX.stroke();
    }
  }
}

function roundRect(x, y, w, h, r) {
  CTX.beginPath();
  CTX.moveTo(x + r, y);
  CTX.arcTo(x + w, y, x + w, y + h, r);
  CTX.arcTo(x + w, y + h, x, y + h, r);
  CTX.arcTo(x, y + h, x, y, r);
  CTX.arcTo(x, y, x + w, y, r);
  CTX.closePath();
}

// ---------------------------------------------------------------- 棋子托盘

function miniCanvas(cells, size, color) {
  const h = Math.max(...cells.map(c => c[0])) + 1;
  const w = Math.max(...cells.map(c => c[1])) + 1;
  const cv = document.createElement('canvas');
  const s = size;
  cv.width = w * s; cv.height = h * s;
  const g = cv.getContext('2d');
  g.fillStyle = color;
  for (const [r, c] of cells) g.fillRect(c * s + 0.5, r * s + 0.5, s - 1, s - 1);
  return cv;
}

// 棋子托盘展示哪个座位的棋子。
// human_player 在 AI 对战下是 -1，**不能拿它去索引数组** ——
// remaining[-1] 是 undefined，再取 [0] 就是
// 「Cannot read properties of undefined」。没有人类座位时展示当前行棋方的棋子。
function viewSeat() {
  if (!S.state) return 0;
  return S.state.human_player >= 0 ? S.state.human_player : S.state.current_player;
}

function renderTray() {
  const tray = $('tray');
  tray.innerHTML = '';
  const me = viewSeat();
  const remaining = S.state ? S.state.remaining[me] : S.meta.pieces.map(() => true);

  for (const p of S.meta.pieces) {
    const div = document.createElement('div');
    div.className = 'piece' + (remaining[p.id] ? '' : ' used') + (S.piece === p.id ? ' sel' : '');
    div.appendChild(miniCanvas(p.orientations[0].cells, 8, COLORS[me]));
    const label = document.createElement('span');
    label.textContent = p.name;
    div.appendChild(label);
    if (remaining[p.id]) {
      div.onclick = () => { S.piece = p.id; S.ori = p.orientations[0].id; renderTray(); renderOrients(); draw(); };
    }
    tray.appendChild(div);
  }
}

function renderOrients() {
  const box = $('orients');
  box.innerHTML = '';
  if (S.piece === null) return;
  const me = viewSeat();
  const p = S.meta.pieces[S.piece];
  for (const o of p.orientations) {
    const d = document.createElement('div');
    d.className = 'ori' + (S.ori === o.id ? ' sel' : '');
    d.appendChild(miniCanvas(o.cells, 11, COLORS[me]));
    d.onclick = () => { S.ori = o.id; renderOrients(); draw(); };
    box.appendChild(d);
  }
}

// ---------------------------------------------------------------- 状态渲染

function applyState(st, analysis) {
  S.state = st;
  S.legal = new Set(st.legal_actions);
  if (analysis !== undefined) S.analysis = analysis;

  $('score0').textContent = st.scores[0];
  $('score1').textContent = st.scores[1];
  $('ply').textContent = st.ply;

  const status = $('status');
  status.className = 'status';
  const noHuman = st.human_player < 0;
  if (st.terminal) {
    const [a, b] = st.scores;
    if (noHuman) {
      // 胜负规则：占格多者胜，相同为和局
      const verdict = a > b ? '先手胜' : a < b ? '后手胜' : '和局';
      status.textContent = `终局：占格 ${a} : ${b} —— ${verdict}`;
    } else {
      const verdict = st.result > 0 ? '你赢了' : st.result < 0 ? '你输了' : '和局';
      status.textContent = `终局：占格 ${a} : ${b} —— ${verdict}`;
      if (st.result > 0) status.classList.add('win');
      if (st.result < 0) status.classList.add('lose');
    }
  } else if (noHuman) {
    const who = st.current_player === 0 ? '先手' : '后手';
    status.textContent = `AI 对战 · 轮到${who}（第 ${st.ply} 手）`;
  } else if (st.current_player === st.human_player) {
    const first = st.ply < 2 ? '（首手必须盖住你的起点）' : '';
    status.textContent = `轮到你走，共 ${st.legal_actions.length} 种合法着法 ${first}`;
  } else {
    status.textContent = 'AI 思考中…';
  }

  // 选中的棋子已经用掉了就取消选中（同样不能用 -1 去索引）
  if (S.piece !== null && !st.remaining[viewSeat()][S.piece]) { S.piece = null; S.ori = null; }

  renderTray(); renderOrients(); renderAnalysis(); draw();
  $('btn-ai').disabled = st.terminal || st.current_player === st.human_player;
  $('btn-undo').disabled = st.ply === 0;
}

function renderAnalysis() {
  const a = S.analysis;
  const list = $('topmoves');
  if (!a) {
    $('winfill').style.width = '50%';
    $('wintext').textContent = '—';
    list.innerHTML = '<li class="empty">点「分析当前局面」或让 AI 走一步</li>';
    return;
  }
  // value 是「当前行棋方」视角，转成 AI 视角展示
  const pct = Math.round(a.win_rate * 100);
  $('winfill').style.width = pct + '%';
  $('wintext').textContent = `行棋方胜势 ${pct}%`;

  list.innerHTML = '';
  for (const m of a.top_moves) {
    const li = document.createElement('li');
    li.innerHTML = `${m.piece} <b>${(m.prob * 100).toFixed(1)}%</b> <small>(${m.visits} 次访问)</small>`;
    li.onmouseenter = () => { highlight(m.cells); };
    li.onmouseleave = () => { draw(); };
    list.appendChild(li);
  }
}

function highlight(cells) {
  draw();
  CTX.fillStyle = '#818cf8aa';
  for (const [r, c] of cells) {
    roundRect(PAD + c * CELL + 1.5, PAD + r * CELL + 1.5, CELL - 3, CELL - 3, 4);
    CTX.fill();
  }
}

// ---------------------------------------------------------------- 交互

async function guard(fn) {
  if (S.busy) return;
  S.busy = true;
  try { await fn(); }
  catch (e) { $('status').textContent = '出错：' + e.message; }
  finally { S.busy = false; }
}

// ------------------------------------------------------------ 座位与后端选择

// 两个座位各选一个后端，'' 表示人来下。于是「人机」和「AI 对战」不是两种模式，
// 只是两种填法 —— 少一个模式开关，就少一类「模式和实际配置对不上」的 bug。
const HUMAN = '';

function seatSel(i) { return $('seat' + i); }
function seatValues() { return [seatSel(0).value, seatSel(1).value]; }

// 传给后端时空串要变回 null
function seatPlayers() { return seatValues().map(v => (v === HUMAN ? null : v)); }

// 选项分组重建。保留当前选中项 —— 刷新的目的是让新 checkpoint 出现，
// 不是把用户正在用的那个换掉。
async function loadBackends(keep) {
  const r = await api('/api/backends');
  S.backends = r.backends;

  const groups = new Map();
  for (const b of r.backends) {
    if (!groups.has(b.group)) groups.set(b.group, []);
    groups.get(b.group).push(b);
  }

  for (let i = 0; i < 2; i++) {
    const sel = seatSel(i);
    const want = (keep && keep[i] !== undefined) ? keep[i] : sel.value;
    sel.innerHTML = '';
    const human = document.createElement('option');
    human.value = HUMAN; human.textContent = '我来下';
    sel.appendChild(human);
    for (const [name, items] of groups) {
      const og = document.createElement('optgroup');
      og.label = name;
      for (const b of items) {
        const o = document.createElement('option');
        o.value = b.id; o.textContent = b.label;
        og.appendChild(o);
      }
      sel.appendChild(og);
    }
    // 原先选中的 checkpoint 可能已被训练侧轮换删掉；option 不存在时
    // 赋值会得到空串，正好落到「我来下」，不会卡在一个不存在的后端上
    sel.value = want === undefined ? '' : want;
  }
}

function backendInfo(id) {
  return S.backends.find(b => b.id === id) || null;
}

function seatLabel(v) {
  if (v === HUMAN) return '人类';
  const info = backendInfo(v);
  return info ? info.label : v;
}

// 难度就是搜索的模拟数，只对网络后端有意义。
// 两个座位都没有网络时（人 vs 规则、规则 vs 规则）就没什么可调的。
function syncBackendUi() {
  const vals = seatValues();
  const isNet = vals.map(v => v !== HUMAN && (backendInfo(v) || {}).kind === 'net');
  const anyNet = isNet[0] || isNet[1];
  const bothAi = vals.every(v => v !== HUMAN);
  $('difficulty').disabled = !anyNet;
  $('btn-analyse').disabled = !anyNet;
  $('btn-autoplay').disabled = !bothAi;
  $('ai-name').textContent = seatLabel(vals[0]) + '  vs  ' + seatLabel(vals[1]);

  // 明确写出难度**作用在哪一方**。它只是 MCTS 的模拟数，规则基线不搜索，
  // 所以两边都是规则基线时它完全不起作用 —— 光把下拉置灰不够，
  // 不说原因的话只会让人猜「那它到底影响谁」。
  const seats = [];
  if (isNet[0]) seats.push('先手');
  if (isNet[1]) seats.push('后手');
  const parts = [];
  if (seats.length === 0) {
    parts.push('难度不起作用：两边都不用网络，规则基线不搜索');
  } else if (seats.length === 2) {
    parts.push('难度 = 每步的 MCTS 模拟数，双方同用');
  } else {
    parts.push(`难度 = 每步的 MCTS 模拟数，只作用于${seats[0]}（另一方是规则基线，不搜索）`);
  }
  if (bothAi) parts.push('点「自动对战」连着走到终局');
  $('backend-hint').textContent = parts.join('；');
}

// 换座位不重置棋盘：同一个局面换个引擎接着下，正是试玩要干的事
async function changeSeats() {
  await guard(async () => {
    syncBackendUi();
    if (!S.sid) return;
    const r = await post('/api/backend', {
      sid: S.sid,
      players: seatPlayers(),
      difficulty: $('difficulty').value,
    });
    S.analysis = null;
    applyState(r.state, null);
    await maybeAutoRespond();
  });
}

// 轮到 AI 且开着自动应手就替它走。AI 对战时不在这里连打 ——
// 那是「自动对战」按钮的事，否则一按新对局就会失控地跑到终局。
async function maybeAutoRespond() {
  if (!$('auto-ai').checked) return;
  if (S.state.human_player < 0) return;
  while (!S.state.terminal && S.state.current_player !== S.state.human_player) {
    await aiMove();
  }
}

async function newGame() {
  await guard(async () => {
    S.autoplay = false;
    S.analysis = null; S.piece = null; S.ori = null;
    const r = await post('/api/new', {
      players: seatPlayers(),
      difficulty: $('difficulty').value,
    });
    S.sid = r.sid;
    applyState(r.state, null);
    await maybeAutoRespond();
  });
}

// ---------------------------------------------------------------- AI 对战

function setAutoplayUi(on) {
  $('btn-autoplay').textContent = on ? '■ 停止' : '▶ 自动对战';
  $('btn-autoplay').classList.toggle('primary', on);
}

// 逐步走而不是让后端一次跑完：每步都刷新棋盘，随时能停。
// 极难档一步要十几秒，一次性跑完整局的话页面会干等几分钟。
async function autoplay() {
  if (S.autoplay) { S.autoplay = false; setAutoplayUi(false); return; }
  if (S.state && S.state.human_player >= 0) return;   // 有人在座，不能自动
  S.autoplay = true;
  setAutoplayUi(true);
  try {
    while (S.autoplay && S.state && !S.state.terminal) {
      await aiMove();
    }
  } catch (e) {
    $('status').textContent = '出错：' + e.message;
  } finally {
    S.autoplay = false;
    setAutoplayUi(false);
  }
}

async function aiMove() {
  const r = await post('/api/ai', { sid: S.sid });
  if (r.labels) $('ai-name').textContent = r.labels[0] + '  vs  ' + r.labels[1];
  if (r.ai_seconds !== undefined) {
    const who = r.ai_player === 0 ? '先手' : '后手';
    $('lastmove').textContent = `上一手：${who} · ${r.ai_label} · ${r.ai_seconds}s`;
  }
  applyState(r, r.analysis || null);
}

async function play(action) {
  await guard(async () => {
    const st = await post('/api/move', { sid: S.sid, action });
    S.piece = null; S.ori = null;
    applyState(st, null);
    if (!st.terminal && st.current_player !== st.human_player && $('auto-ai').checked) {
      await aiMove();
      // AI 走完后对方可能仍无法落子（停手），需要继续
      while (!S.state.terminal && S.state.current_player !== S.state.human_player) await aiMove();
    }
  });
}

CV.addEventListener('mousemove', (ev) => {
  const cell = cellAt(ev);
  const changed = JSON.stringify(cell) !== JSON.stringify(S.hover);
  S.hover = cell;
  if (changed) draw();
});
CV.addEventListener('mouseleave', () => { S.hover = null; draw(); });
CV.addEventListener('click', (ev) => {
  const st = S.state;
  if (!st || st.terminal || st.current_player !== st.human_player || S.ori === null) return;
  const cell = cellAt(ev);
  if (!cell) return;
  const a = actionFor(cell[0], cell[1]);
  if (S.legal.has(a)) play(a);
});

$('btn-new').onclick = newGame;
$('btn-ai').onclick = () => guard(aiMove);
$('seat0').onchange = changeSeats;
$('seat1').onchange = changeSeats;
$('difficulty').onchange = changeSeats;
$('btn-autoplay').onclick = autoplay;
$('btn-swap').onclick = () => guard(async () => {
  const [a, b] = seatValues();
  await loadBackends([b, a]);
  syncBackendUi();
  // 换边改的是「谁执先」，那是开局属性，必须重开一局
  $('status').textContent = '已换边，点「新对局」生效';
});
$('btn-refresh').onclick = () => guard(async () => {
  await loadBackends(seatValues());
  syncBackendUi();
  // 训练在跑，"最新" 指向的 step 会变，重扫之后标签才是新的
  $('status').textContent = '已重新扫描：' + S.backends.length + ' 个可选对手';
});
$('btn-undo').onclick = () => guard(async () => {
  S.analysis = null;
  applyState(await post('/api/undo', { sid: S.sid }), null);
});
$('btn-analyse').onclick = () => guard(async () => {
  $('status').textContent = '分析中…';
  const r = await api('/api/analysis?sid=' + S.sid);
  S.analysis = r.analysis;
  applyState(S.state, r.analysis);
});
$('btn-export').onclick = () => {
  const blob = new Blob([JSON.stringify({
    history: S.state.history, scores: S.state.scores, result: S.state.result,
  }, null, 2)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'cornerstone-game.json';
  a.click();
};
$('show-heat').onchange = draw;

document.addEventListener('keydown', (ev) => {
  if (S.piece === null) return;
  const oris = S.meta.pieces[S.piece].orientations;
  const idx = oris.findIndex(o => o.id === S.ori);
  if (ev.key === 'r' || ev.key === 'R') {
    S.ori = oris[(idx + 1) % oris.length].id; renderOrients(); draw();
  } else if (ev.key === 'f' || ev.key === 'F') {
    S.ori = oris[(idx + Math.ceil(oris.length / 2)) % oris.length].id; renderOrients(); draw();
  } else if (ev.key === 'Escape') {
    S.piece = null; S.ori = null; renderTray(); renderOrients(); draw();
  }
});

// ---------------------------------------------------------------- 启动

(async () => {
  S.meta = await api('/api/meta');
  layout();
  const sel = $('difficulty');
  for (const d of S.meta.difficulties) {
    const o = document.createElement('option');
    o.value = d; o.textContent = d;
    if (d === '普通') o.selected = true;
    sel.appendChild(o);
  }
  // 默认人执先、AI 执后，和改版前一致
  await loadBackends([HUMAN, S.meta.default_backend]);
  syncBackendUi();
  setAutoplayUi(false);
  renderTray();
  await newGame();
})();
