// cornerstone 试玩前端。
//
// 一条硬规则：**合法性判断一律不在这里重写**。后端返回 legal_actions，
// 前端只做「这个动作在不在集合里」的查表。规则实现只有 C++ 引擎那一份，
// 训练、评测、试玩共用，不会出现两边判定不一致。

const $ = (id) => document.getElementById(id);

// 把 JS 错误摆到页面上。
//
// 这里没有构建步骤、也没有 JS 运行时可做单测，所以一个未捕获的异常
// 表现出来就是「某个控件忽然没反应」——症状离原因极远，只能靠
// 打开 F12 才知道发生了什么。实际吃过好几次亏，所以让它自己说话。
function fatal(msg) {
  const bar = document.getElementById('js-error');
  if (bar) { bar.textContent = '前端出错：' + msg; bar.classList.remove('gone'); }
}
window.addEventListener('error', (e) => fatal(e.message));
window.addEventListener('unhandledrejection', (e) => fatal(
  (e.reason && (e.reason.message || e.reason)) || '未知的 Promise 拒绝'));

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
  phase: 'idle',        // idle | playing | paused
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
  const humanToMove = S.phase === 'playing' && st && !st.terminal
    && st.players[st.current_player] === null;
  if (humanToMove) {
    CTX.fillStyle = COLORS[st.current_player] + '66';
    for (const [r, c] of legalAnchors()) {
      CTX.beginPath();
      CTX.arc(PAD + (c + 0.5) * CELL, PAD + (r + 0.5) * CELL, 3.5, 0, Math.PI * 2);
      CTX.fill();
    }
  }

  // 悬停预览
  if (humanToMove && S.hover && S.ori !== null) {
    const [hr, hc] = S.hover;
    const a = actionFor(hr, hc);
    const ok = S.legal.has(a);
    CTX.fillStyle = ok ? COLORS[st.current_player] + 'bb' : '#f8717166';
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

// 哪些座位要摆棋子面板，以及各自可不可交互。
//
// 有人在座：只摆人那一侧。两边都是 AI：**两侧各摆一块**，都只读 ——
// 看双方还剩哪些棋子是 Blokus 里很实的信息（棋子越大越难安置，
// 谁手里压着大件谁后面越吃紧），只是没人要落子，所以不给交互。
//
// 「可交互」的判据是**轮到这个座位、而且这个座位是人**，不是「有人在座」。
// 用后者的话，AI 思考期间人这侧仍然展开朝向圈，暗示可以落子，其实点了没用。
function trayPlan() {
  const st = S.state;
  if (!st) return [];
  const seats = st.human_player >= 0
    ? st.players.map((p, i) => (p === null ? i : -1)).filter(i => i >= 0)
    : [0, 1];
  return seats.map(seat => ({
    seat,
    interactive: S.phase === 'playing' && st.players[seat] === null
      && !st.terminal && st.current_player === seat,
  }));
}

function renderTrays() {
  const wrap = $('pieces-wrap');
  wrap.innerHTML = '';
  const st = S.state;
  if (!st) return;

  for (const { seat, interactive } of trayPlan()) {
    const card = document.createElement('div');
    card.className = 'card';

    const h = document.createElement('h2');
    h.innerHTML = `<span class="seat p${seat}"></span>${seat === 0 ? '先手' : '后手'}棋子`
      + `<small>剩 ${st.remaining[seat].filter(Boolean).length} 枚 · 已占 ${st.scores[seat]} 格`
      + (interactive ? ' · 悬停选形态' : ' · 只读') + '</small>';
    card.appendChild(h);

    const tray = document.createElement('div');
    tray.className = 'tray' + (interactive ? '' : ' readonly');
    for (const p of S.meta.pieces) {
      const div = document.createElement('div');
      const used = !st.remaining[seat][p.id];
      div.className = 'piece' + (used ? ' used' : '')
        + (interactive && S.piece === p.id ? ' sel' : '');
      div.appendChild(miniCanvas(p.orientations[0].cells, 9, COLORS[seat]));
      if (interactive && !used) div.appendChild(orientRing(p, seat));
      tray.appendChild(div);
    }
    card.appendChild(tray);
    wrap.appendChild(card);
  }
}

// 把 k 个朝向均匀摆在一个圆周上，从正上方开始顺时针。
// 半径随 k 增大，否则 8 个朝向（4 旋转 x 2 镜像的满配）会挤在一起。
function orientRing(p, me) {
  const ring = document.createElement('div');
  ring.className = 'ring';
  const oris = p.orientations;
  const k = oris.length;
  const radius = k <= 2 ? 42 : k <= 4 ? 52 : 64;
  // 圆形暗底的大小跟着半径走，由 CSS 用 calc 加上按钮尺寸
  ring.style.setProperty('--r', radius + 'px');

  oris.forEach((o, i) => {
    const ang = -Math.PI / 2 + (i * 2 * Math.PI) / k;
    const btn = document.createElement('div');
    btn.className = 'ori-btn' + (S.piece === p.id && S.ori === o.id ? ' sel' : '');
    btn.style.left = `calc(50% + ${(radius * Math.cos(ang)).toFixed(1)}px)`;
    btn.style.top = `calc(50% + ${(radius * Math.sin(ang)).toFixed(1)}px)`;
    btn.appendChild(miniCanvas(o.cells, 9, COLORS[me]));
    btn.onclick = (ev) => {
      ev.stopPropagation();
      S.piece = p.id; S.ori = o.id;
      renderTrays(); draw();
    };
    ring.appendChild(btn);
  });
  return ring;
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
  } else if (S.phase === 'idle') {
    status.textContent = st.ply === 0
      ? '选好双方与模拟数，点「开始对局」'
      : `已结束对局（停在第 ${st.ply} 手）；点「开始对局」重来一局`;
  } else if (S.phase === 'paused') {
    status.textContent = `已暂停（第 ${st.ply} 手）—— 点「恢复」继续`;
  } else if (st.players[st.current_player] === null) {
    const who = st.current_player === 0 ? '先手' : '后手';
    const first = st.ply < 2 ? '（首手必须盖住起点）' : '';
    status.textContent = noHuman ? `轮到${who}` :
      `轮到你走（${who}），共 ${st.legal_actions.length} 种合法着法 ${first}`;
  } else {
    const who = st.current_player === 0 ? '先手' : '后手';
    status.textContent = `${who} AI 思考中…`;
  }

  // 没人在座就不存在「选中的棋子」；有人在座时，选中的棋子被用掉了要清掉。
  // 这里按人的座位索引，noHuman 时直接清空，绝不拿 -1 去索引。
  if (noHuman) {
    S.piece = null; S.ori = null;
  } else {
    const seat = st.human_player;             // 走到这里一定 >= 0
    if (S.piece !== null && !st.remaining[seat][S.piece]) { S.piece = null; S.ori = null; }
  }

  syncControls();
  renderTrays(); renderAnalysis(); draw();
}

function renderAnalysis() {
  const a = S.analysis;
  const list = $('topmoves');
  if (!a) {
    $('winfill').style.width = '50%';
    $('wintext').textContent = '—';
    list.innerHTML = '<li class="empty">让 AI 走一步后显示</li>';
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
function simsSel(i) { return $('sims' + i); }
function seatValues() { return [seatSel(0).value, seatSel(1).value]; }
function seatSims() { return [0, 1].map(i => parseInt(simsSel(i).value, 10)); }

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

// ------------------------------------------------------------------ 对局状态机
//
//   idle    还没开始（或已结束）。配置可改，谁都不能落子
//   playing 进行中。配置锁死，轮到谁谁落子，AI 由 pump() 驱动
//   paused  已暂停。配置仍锁着，人和 AI 都不能落子
//
// 只有这三态，界面上的每个可用/禁用状态都由它推出来 ——
// 以前是「自动应手」「自动对战」「锁定」几个布尔各管一摊，
// 组合起来有说不清的中间态（比如自动对战开着但轮到人）。

// 图标形状表。一个按钮任何时刻只画其中一个。
const ICONS = {
  play: '<path d="M7 4l13 8-13 8z" fill="currentColor" stroke="currentColor" stroke-width="1.5"/>',
  stop: '<rect x="6" y="6" width="12" height="12" rx="1.5" fill="currentColor" stroke="currentColor" stroke-width="1.5"/>',
  pause: '<path d="M9 5v14M15 5v14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>',
};

function setIcon(id, name) {
  const svg = $(id);
  if (svg && svg.dataset.icon !== name) {
    svg.innerHTML = ICONS[name];
    svg.dataset.icon = name;
  }
}

function setPhase(p) {
  S.phase = p;
  syncControls();
}

function syncControls() {
  const playing = S.phase !== 'idle';
  const paused = S.phase === 'paused';

  // 纯图标按钮，状态体现在图标、颜色和 title 上。
  // 图形直接换 svg 的内容，而不是塞两个 svg 用 CSS 挑一个显示 ——
  // 后者在样式没生效时会两个一起冒出来。
  const start = $('btn-start');
  start.classList.toggle('running', playing);
  start.title = playing ? '结束对局' : '开始对局';
  start.setAttribute('aria-label', start.title);
  setIcon('ico-start', playing ? 'stop' : 'play');

  const pause = $('btn-pause');
  pause.classList.toggle('gone', !playing);
  pause.classList.toggle('paused', paused);
  pause.title = paused ? '恢复对局' : '暂停对局';
  pause.setAttribute('aria-label', pause.title);
  setIcon('ico-pause', paused ? 'play' : 'pause');

  // 配置只在 idle 可改。这里不需要服务端再拦一道 ——
  // 双方与模拟数只在 /api/new 时提交，对局中根本没有改它的通道。
  for (let i = 0; i < 2; i++) {
    seatSel(i).disabled = playing;
    simsSel(i).disabled = playing || simsSel(i).dataset.ruleDisabled === '1';
  }
  $('lock-hint').textContent = paused ? '已暂停 —— 双方都不能落子'
    : playing ? '对局进行中，配置已锁定' : '';
}

// 模拟数是**每个座位各自的**，所以置灰也按座位来：
// 规则基线不搜索，它旁边那个模拟数下拉就没有意义。
function syncBackendUi() {
  const vals = seatValues();
  const isNet = vals.map(v => v !== HUMAN && (backendInfo(v) || {}).kind === 'net');
  const bothAi = vals.every(v => v !== HUMAN);
  // 两种置灰的原因要分开记：规则基线本身用不上模拟数（这里），
  // 以及对局进行中全部锁死（syncLock）。混在一起的话，解锁时会把
  // 本该一直灰着的规则基线那一侧一起点亮。
  for (let i = 0; i < 2; i++) {
    simsSel(i).dataset.ruleDisabled = isNet[i] ? '0' : '1';
    simsSel(i).disabled = !isNet[i] || S.phase !== 'idle';
  }
  $('ai-name').textContent = seatLabel(vals[0]) + '  vs  ' + seatLabel(vals[1]);
  void bothAi;

  $('backend-hint').textContent = (!isNet[0] && !isNet[1])
    ? '两边都不用网络，模拟数不起作用（规则基线不搜索）'
    : '右侧数字 = 该座位每步的 MCTS 模拟数，越大越强也越慢';
}

// 配置改动只留在本地，等「开始对局」时一次性提交给 /api/new。
// 不再中途调 /api/backend —— 对局中根本没有改配置的通道，比事后拦更干净。
function onConfigChange() {
  syncBackendUi();
}

// ---------------------------------------------------------------- 开始 / 结束

async function startGame() {
  await guard(async () => {
    S.analysis = null; S.piece = null; S.ori = null;
    const r = await post('/api/new', { players: seatPlayers(), sims: seatSims() });
    S.sid = r.sid;
    setPhase('playing');
    applyState(r.state, null);
  });
  await pump();
}

// 结束只是停手：棋盘留在原样供查看，配置解锁。
// 服务端那边的会话不用管，下次开始会新建一个。
function endGame() {
  setPhase('idle');
  syncBackendUi();          // 解锁后要重算规则基线那一侧的置灰
  if (S.state) applyState(S.state, undefined);
}

function togglePause() {
  if (S.phase === 'playing') {
    setPhase('paused');
    if (S.state) applyState(S.state, undefined);
  } else if (S.phase === 'paused') {
    setPhase('playing');
    if (S.state) applyState(S.state, undefined);
    pump();
  }
}

// 驱动 AI 落子：轮到的座位若是 AI 就替它走，一直走到轮到人、
// 被暂停、被结束、或终局为止。人机与 AI 对战共用这一条路径。
//
// **暂停只能在两次落子之间生效** —— 一次搜索已经交给引擎了，中途打不断。
// 所以按下暂停后，可能还要等当前这一步搜完（800 次模拟约十几秒）。
async function pump() {
  await guard(async () => {
    while (S.phase === 'playing' && S.state && !S.state.terminal
           && S.state.players[S.state.current_player] !== null) {
      await aiMove();
    }
  });
  if (S.state && S.state.terminal && S.phase !== 'idle') endGame();
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
  });
  // pump 里那个 while 顺带覆盖了「AI 走完对方仍无法落子（停手）」的情况
  await pump();
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
  if (S.phase !== 'playing' || !st || st.terminal || S.ori === null) return;
  if (st.players[st.current_player] !== null) return;      // 轮到 AI，人不能替它下
  const cell = cellAt(ev);
  if (!cell) return;
  const a = actionFor(cell[0], cell[1]);
  if (S.legal.has(a)) play(a);
});

// 绑定一律走这里，不直接 $('x').onclick = ...
//
// 直接赋值的话，只要有一个元素对不上（最常见的原因是浏览器拿着旧的
// app.js 配新的 index.html），就会抛 `Cannot set properties of null`，
// **而这一抛整个模块就停了** —— 后面填充下拉框的启动代码根本不会执行，
// 表现却是「某个控件用不了」，离原因非常远。
// 这里改成：缺了就报出来并跳过，别的绑定照常。
function on(id, event, handler) {
  const el = $(id);
  if (!el) {
    fatal(`界面元素 ${id} 不存在 —— 多半是页面与脚本版本不一致，强制刷新一下（Ctrl+F5）`);
    return;
  }
  el[event] = handler;
}

on('btn-start', 'onclick', () => (S.phase === 'idle' ? startGame() : endGame()));
on('btn-pause', 'onclick', togglePause);
on('seat0', 'onchange', onConfigChange);
on('seat1', 'onchange', onConfigChange);
on('sims0', 'onchange', onConfigChange);
on('sims1', 'onchange', onConfigChange);

document.addEventListener('keydown', (ev) => {
  // 没人在座就没有「选棋子」这回事，快捷键一并停掉
  if (S.state && S.state.human_player < 0) return;
  if (S.piece === null) return;
  const oris = S.meta.pieces[S.piece].orientations;
  const idx = oris.findIndex(o => o.id === S.ori);
  if (ev.key === 'r' || ev.key === 'R') {
    S.ori = oris[(idx + 1) % oris.length].id; renderTrays(); draw();
  } else if (ev.key === 'f' || ev.key === 'F') {
    S.ori = oris[(idx + Math.ceil(oris.length / 2)) % oris.length].id; renderTrays(); draw();
  } else if (ev.key === 'Escape') {
    S.piece = null; S.ori = null; renderTrays(); draw();
  }
});

// ---------------------------------------------------------------- 启动

(async () => {
  S.meta = await api('/api/meta');
  layout();
  for (let i = 0; i < 2; i++) {
    const sel = simsSel(i);
    for (const n of S.meta.sim_choices) {
      const o = document.createElement('option');
      o.value = String(n); o.textContent = String(n);
      if (n === S.meta.default_sims) o.selected = true;
      sel.appendChild(o);
    }
  }
  // 默认人执先、AI 执后，和改版前一致
  await loadBackends([HUMAN, S.meta.default_backend]);
  syncBackendUi();
  setPhase('idle');

  // 开一个会话只为渲染出空棋盘；真正开局要点「开始对局」。
  // **这一步失败不能连累配置界面** —— 上面 loadBackends 已经把双方
  // 选项填好了，就算这里炸了也要让人能选、能点开始。
  try {
    const r0 = await post('/api/new', { players: seatPlayers(), sims: seatSims() });
    S.sid = r0.sid;
    applyState(r0.state, null);
  } catch (e) {
    fatal('初始局面加载失败：' + e.message + '（选好双方后点「开始对局」仍可继续）');
    syncControls();
  }
})();
