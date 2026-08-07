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
    interactive: st.players[seat] === null && !st.terminal && st.current_player === seat,
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
      div.appendChild(miniCanvas(p.orientations[0].cells, 8, COLORS[seat]));
      const label = document.createElement('span');
      label.textContent = p.name;
      div.appendChild(label);
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
  } else if (noHuman) {
    const who = st.current_player === 0 ? '先手' : '后手';
    status.textContent = `AI 对战 · 轮到${who}（第 ${st.ply} 手）`;
  } else if (st.current_player === st.human_player) {
    const first = st.ply < 2 ? '（首手必须盖住你的起点）' : '';
    status.textContent = `轮到你走，共 ${st.legal_actions.length} 种合法着法 ${first}`;
  } else {
    status.textContent = 'AI 思考中…';
  }

  // 两个座位都是 AI 时整块棋子面板都收起来 —— 没人要落子，
  // 选棋子和朝向都没有意义，留着只会占地方并且看着像能点。
  // 没人在座就不存在「选中的棋子」；有人在座时，选中的棋子被用掉了要清掉。
  // 这里按人的座位索引，noHuman 时直接清空，绝不拿 -1 去索引。
  if (noHuman) {
    S.piece = null; S.ori = null;
  } else {
    const seat = st.human_player;             // 走到这里一定 >= 0
    if (S.piece !== null && !st.remaining[seat][S.piece]) { S.piece = null; S.ori = null; }
  }

  syncLock(st);
  renderTrays(); renderAnalysis(); draw();
  $('btn-ai').disabled = st.terminal || st.current_player === st.human_player;
  $('btn-undo').disabled = st.ply === 0;
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

// 对局一开跑，双方与模拟数就锁死，直到终局或开新局。
//
// 中途换引擎会让「这一局是谁对谁」变得没法陈述 —— 棋盘上一半的手是
// A 走的、一半是 B 走的，最后那个比分就不属于任何一对组合。
// 服务端也会拒（不能只靠界面置灰），这里只是让不可点这件事看得见。
function isLocked(st) {
  return !!st && st.ply > 0 && !st.terminal;
}

function syncLock(st) {
  const locked = isLocked(st);
  for (let i = 0; i < 2; i++) {
    seatSel(i).disabled = locked;
    simsSel(i).disabled = locked || simsSel(i).dataset.ruleDisabled === '1';
  }
  $('lock-hint').textContent = locked ? '对局进行中，配置已锁定 —— 点「新对局」可重新设置' : '';
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
    simsSel(i).disabled = !isNet[i] || isLocked(S.state);
  }
  $('btn-autoplay').disabled = !bothAi;
  $('ai-name').textContent = seatLabel(vals[0]) + '  vs  ' + seatLabel(vals[1]);

  const parts = [];
  if (!isNet[0] && !isNet[1]) {
    parts.push('两边都不用网络，模拟数不起作用（规则基线不搜索）');
  } else {
    parts.push('右侧数字 = 该座位每步的 MCTS 模拟数，越大越强也越慢');
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
      sims: seatSims(),
    });
    S.analysis = null;
    applyState(r.state, null);
    await maybeAutoRespond();
  });
}

// 人机对局里轮到 AI 就替它走。以前这里有个「AI 自动应手」开关，
// 现在是固定行为 —— 关掉它只会让人每走一手都要多点一次「让 AI 走一步」。
//
// AI 对战时**不在这里连打**：那是「自动对战」按钮的事，
// 否则一按新对局就会失控地一路跑到终局。
async function maybeAutoRespond() {
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
      sims: seatSims(),
    });
    S.sid = r.sid;
    applyState(r.state, null);
    await maybeAutoRespond();
  });
}

// ---------------------------------------------------------------- AI 对战

function setAutoplayUi(on) {
  const b = $('btn-autoplay');
  b.classList.toggle('running', on);      // CSS 据此在播放/停止两个图标间切换
  b.classList.toggle('primary', on);
  b.title = on ? '停止自动对战' : '自动对战';
  b.setAttribute('aria-label', b.title);
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
    // maybeAutoRespond 里那个 while 已经覆盖了「AI 走完对方仍无法落子（停手）」的情况
    await maybeAutoRespond();
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
$('sims0').onchange = changeSeats;
$('sims1').onchange = changeSeats;
$('btn-autoplay').onclick = autoplay;
$('btn-undo').onclick = () => guard(async () => {
  S.analysis = null;
  applyState(await post('/api/undo', { sid: S.sid }), null);
});

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
  setAutoplayUi(false);
  await newGame();
})();
