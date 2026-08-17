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
  // 交互是**先点位置、再选摆法**：点棋盘一个格，右侧列出会盖住它的合法着法。
  // 原先是「先挑棋子、再找位置」—— 21 枚棋子、91 种朝向，等于让人先在脑子里
  // 做一遍搜索；而且展开朝向要用浮层，浮层必然盖住邻居，改了三次都不对。
  pickCell: null,       // 点中的棋盘格 [r,c]；非空时右侧显示候选而不是托盘
  pickPiece: null,      // 展开的棋子 id（null = 还在第一层）
  pickOri: null,        // 选中的朝向 id；同一朝向有多个位置时由鼠标在棋盘上滑动决定
  oriActions: null,     // 该朝向的全部可放位置
  preview: null,        // 悬停候选时在棋盘上预览的着法
  hover: null,          // [r, c]
  analysis: null,
  busy: false,
  backends: [],
  phase: 'idle',        // idle | playing | paused
  labels: null,         // 服务端给的双方 label（按座位），显示一律以它为准
  lastMove: '',         // 上一手的描述，显示在状态条右半边
  series: null,         // 连续对战的累计战绩
};

// 颜色绑的是**对战双方**，不是先后手。
// 连续对战逐局换边，若按座位上色，同一个引擎会一局一个颜色，根本看不出谁是谁。
// COLORS[0] 恒为「甲」（座位选择里第一个下拉的引擎），COLORS[1] 恒为「乙」。
const COLORS = ['#2dd4bf', '#fb923c'];

// 某个座位这一局坐的是甲还是乙
function engineOfSeat(seat) {
  return (S.series && S.series.swap) ? 1 - seat : seat;
}
function seatColor(seat) { return COLORS[engineOfSeat(seat)]; }
// 一个带颜色的小圆点，用来代替「先手/后手」这种字样
function dotHtml(seat) {
  return `<span class="seat" style="background:${seatColor(seat)}"></span>`;
}

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
    CTX.strokeStyle = seatColor(p); CTX.lineWidth = 2;
    CTX.beginPath();
    CTX.arc(PAD + (c + 0.5) * CELL, PAD + (r + 0.5) * CELL, CELL * 0.28, 0, Math.PI * 2);
    CTX.stroke();
  });

  // 已落子
  if (st) {
    for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) {
      const v = st.grid[r][c];
      if (v < 0) continue;
      CTX.fillStyle = seatColor(v);
      roundRect(PAD + c * CELL + 1.5, PAD + r * CELL + 1.5, CELL - 3, CELL - 3, 4);
      CTX.fill();
    }
  }

  const humanToMove = S.phase === 'playing' && st && !st.terminal
    && st.players[st.current_player] === null;

  // 还没选格：把「点了有候选」的格标出来，告诉人可以点哪儿
  if (humanToMove && !S.pickCell) {
    CTX.fillStyle = seatColor(st.current_player) + '55';
    for (const [r, c] of playableCells()) {
      CTX.beginPath();
      CTX.arc(PAD + (c + 0.5) * CELL, PAD + (r + 0.5) * CELL, 3.5, 0, Math.PI * 2);
      CTX.fill();
    }
  }

  // 选中的那个格描个框
  if (humanToMove && S.pickCell) {
    const [pr, pc] = S.pickCell;
    CTX.strokeStyle = seatColor(st.current_player);
    CTX.lineWidth = 2.5;
    roundRect(PAD + pc * CELL + 1.5, PAD + pr * CELL + 1.5, CELL - 3, CELL - 3, 4);
    CTX.stroke();
  }

  // 悬停某个候选时，把它整块画在棋盘上
  if (humanToMove && S.preview !== null) {
    CTX.fillStyle = seatColor(st.current_player) + 'bb';
    CTX.strokeStyle = '#ffffff88';
    CTX.lineWidth = 1.5;
    for (const [r, c] of actionCells(S.preview)) {
      if (r < 0 || r >= n || c < 0 || c >= n) continue;
      roundRect(PAD + c * CELL + 1.5, PAD + r * CELL + 1.5, CELL - 3, CELL - 3, 4);
      CTX.fill(); CTX.stroke();
    }
  }
}

// 胜负规则：占格多者胜，相同为和局。
// 返回 {text, seat}：seat 是赢家的座位（-1 表示和局），颜色由调用方按引擎取。
function verdict(st) {
  const [a, b] = st.scores;
  if (a === b) return { text: '和局', seat: -1 };
  const win = a > b ? 0 : 1;
  if (st.human_player < 0) return { text: '胜', seat: win };
  return { text: win === st.human_player ? '你赢了' : '你输了', seat: win };
}

// 棋盘下面那一条。左边是当前状态、右边是上一手，终局时整条变成结果横幅。
//
// 之前是三块东西各占一行（终局横幅 + 状态 + 上一手），白白吃掉竖向空间，
// 而且终局横幅一出一进还会顶动棋盘。合成一条之后高度恒定，也不会跳。
function renderStatus(st) {
  const bar = $('status');
  const main = $('status-main');
  const wrap = $('board-wrap');
  bar.className = 'status';
  if (!st) return;

  if (st.terminal) {
    const v = verdict(st);
    const color = v.seat < 0 ? 'var(--muted)' : seatColor(v.seat);
    const s = S.series;
    const nth = (s && s.total > 1) ? `第 ${s.played} / ${s.total} 局　` : '';
    const who = v.seat < 0 ? '' : dotHtml(v.seat);
    bar.classList.add('over');
    bar.style.borderColor = color;
    main.innerHTML = `${who}<b style="color:${color}">${v.text}</b>`;
    $('status-side').innerHTML =
      `${nth}占格 ${st.scores[0]} : ${st.scores[1]}　共 ${st.ply} 手`;
    wrap.style.borderColor = color;    // 棋盘描边跟着变，边框不挡格子
    return;
  }

  bar.style.borderColor = '';
  wrap.style.borderColor = '';
  if (S.phase === 'idle') {
    main.textContent = st.ply === 0
      ? '选好双方与模拟数，点「开始对局」'
      : `已结束对局（停在第 ${st.ply} 手）；点「开始对局」重来一局`;
  } else if (S.phase === 'paused') {
    main.textContent = `已暂停（第 ${st.ply} 手）—— 点「恢复」继续`;
  } else if (st.players[st.current_player] === null) {
    const first = st.ply < 2 ? '（首手必须盖住起点）' : '';
    main.innerHTML = `轮到${dotHtml(st.current_player)}你走，`
      + `共 ${st.legal_actions.length} 种合法着法 ${first}`;
  } else {
    main.innerHTML = `${dotHtml(st.current_player)}思考中…`;
  }
  $('status-side').innerHTML = S.lastMove || '';
}

// 名字太长会把比分格子撑爆，截一下；完整的放 title
function shorten(s) { return s.length > 26 ? s.slice(0, 25) + '…' : s; }

// 往 HTML 属性里塞文本前先转义。引擎名来自 checkpoint 文件名，
// 理论上不会有引号，但拼 HTML 就该转义，不指望输入永远干净。
function esc(t) {
  return String(t).replace(/&/g, '&amp;').replace(/</g, '&lt;')
                  .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

// **一个座位怎么描述，只有这里说了算。**
//
// 之前右上角徽标用服务端返回的 label（带解析后的真实 step），
// 而比分格和战绩表各自去下拉框取选项文字（「最新（跟随训练，当前 step …）」），
// 同一个对手在三个地方写法不一样。现在统一走这里：
// 有服务端 label 就用它，没有（还没开局）才回落到下拉框。
function describeSeat(st, seat) {
  const fallback = seatLabel(seatValues()[engineOfSeat(seat)]);
  const label = (S.labels && S.labels[seat]) || fallback;
  return describe(st.players[seat], label, st.sims[seat]);
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

// ---------------------------------------------------------------- 候选
//
// 点棋盘一个格，右侧列出所有会**盖住这个格**的合法着法。
//
// 必须分两层：实测点开局的起始格有 414 个候选（首手必须盖住它，于是全部合法
// 着法都符合）。按棋子分组之后，第一层最多 21 个（就是全部棋子）、第二层中位数
// 只有 1~4 —— 第 20 手后中位数就是 1，绝大多数时候点一下就定了。

/** 会盖住 (r,c) 的合法着法，三层：Map<pieceId, Map<oriId, action[]>>。
 *
 * 三层各自的规模是实测的（角点口径）：
 *   开局    棋子 21、朝向中位 4、**位置中位 5**
 *   第 6 手 棋子 18、朝向中位 2、**位置中位 1**（最大 2）
 *   第 20 手 棋子 7、朝向中位 1、**位置中位 1**（最大 2）
 *
 * 「位置」这一层绝大多数时候只有 1 个 —— 所以它不该占一次点击，
 * 而是由鼠标在棋盘上滑动决定（见 pickPlacement）。
 */
function candidatesAt(r, c) {
  const out = new Map();
  if (!S.state) return out;
  const n = S.meta.board_n, cells = S.meta.num_cells;
  for (const p of S.meta.pieces) {
    for (const o of p.orientations) {
      const base = o.id * cells;
      for (const [dr, dc] of o.cells) {
        // 该朝向要盖住 (r,c)，锚点就得在 (r-dr, c-dc)
        const ar = r - dr, ac = c - dc;
        if (ar < 0 || ar >= n || ac < 0 || ac >= n) continue;
        const a = base + ar * n + ac;
        if (!S.legal.has(a)) continue;
        if (!out.has(p.id)) out.set(p.id, new Map());
        const byOri = out.get(p.id);
        if (!byOri.has(o.id)) byOri.set(o.id, []);
        const arr = byOri.get(o.id);
        if (!arr.includes(a)) arr.push(a);
      }
    }
  }
  return out;
}

/** 一个格上「棋子 x 朝向」共多少项 —— 决定要不要把这两步合成一步。 */
function pairCount(groups) {
  let n = 0;
  for (const byOri of groups.values()) n += byOri.size;
  return n;
}

// 项数不超过这个数就把「选棋子」和「选朝向」合成一步。
// 实测：开局 91 项、第 6 手中位 44、第 20 手中位 8 —— 于是残局一步到位，
// 而残局正是选择最紧张、最不想多点一下的时候。
const MERGE_LIMIT = 24;

/** 选中朝向后，鼠标停在 (mr,mc) 时该用哪个位置。
 *
 * 优先取「棋子盖住了鼠标所在格」的那个；都没盖住就取重心离鼠标最近的。
 * 位置中位数是 1，所以这个函数多数时候只有一个候选可选。
 */
function pickPlacement(actions, mouse) {
  if (actions.length <= 1 || !mouse) return actions[0];
  const [mr, mc] = mouse;
  let best = actions[0], bestD = Infinity;
  for (const a of actions) {
    const cs2 = actionCells(a);
    if (cs2.some(([r, c]) => r === mr && c === mc)) return a;
    let sr = 0, sc = 0;
    for (const [r, c] of cs2) { sr += r; sc += c; }
    const d = Math.hypot(sr / cs2.length - mr, sc / cs2.length - mc);
    if (d < bestD) { bestD = d; best = a; }
  }
  return best;
}

/** 一个着法覆盖哪些格 —— 预览和高亮都用它。 */
function actionCells(a) {
  const n = S.meta.board_n, cells = S.meta.num_cells;
  const r0 = Math.floor((a % cells) / n), c0 = (a % cells) % n;
  return oriCells(Math.floor(a / cells)).map(([dr, dc]) => [r0 + dr, c0 + dc]);
}

/** 可以点的格：**当前方的角点**，开局时就是那一个起始格。
 *
 * 这是 Blokus 的落子规则本身 —— 新棋子必须斜接自己已有的棋子、不能边接。
 * 服务端直接给（`state.anchors`，来自引擎的 `Board.anchor_cells()`），
 * **前端不重算**：合法性判断只有 C++ 引擎那一份，训练、评测、试玩共用，
 * 在这里抄一份迟早两边对不上（见文件开头那条硬规则）。
 *
 * 早先这里返回的是「任何合法着法盖得住的格」—— 那是个大得多的集合
 * （开局 41 个格），点哪儿都有候选，反而看不出该往哪儿走。
 *
 * 还要**滤掉放不下东西的角点**：实测第 20 手时 12 个角点里有 4 个没有任何
 * 合法着法盖得住（剩下的棋子都太大）。留着它们可点，人点下去只会看到
 * 「这个格放不下任何棋子」—— 算得出来的事不该让人去试。
 */
function playableCells() {
  const st = S.state;
  if (!st || !st.anchors) return [];
  const key = `${S.sid}:${st.ply}`;
  if (_playable.key === key) return _playable.cells;
  const cells = (st.anchors[st.current_player] || [])
    .filter(([r, c]) => candidatesAt(r, c).size > 0);
  _playable = { key, cells };
  return cells;
}
// 缓存键带手数：一局之内每走一手合法集合就变而 sid 不变，只用 sid 会发馊。
let _playable = { key: null, cells: [] };

// 点了格之后，右侧这块从「棋子托盘」切成「这个格能放什么」。
//
// 层次是 棋子 -> 朝向 -> 位置，但**两处自动折叠**：
//   * 项数 <= MERGE_LIMIT 时「棋子」和「朝向」合成一步（残局中位 8 项）
//   * 「位置」不占点击，选中朝向后由鼠标在棋盘上滑动决定（中位数就是 1）
function renderCandidates(wrap) {
  const [r, c] = S.pickCell;
  const groups = candidatesAt(r, c);
  const card = document.createElement('div');
  card.className = 'card';
  const merged = pairCount(groups) <= MERGE_LIMIT;

  const h = document.createElement('h2');
  h.innerHTML = `第 ${r + 1} 行 ${c + 1} 列 可放`
    + `<small>${groups.size} 种棋子 · 点棋盘空白处或按 Esc 返回</small>`;
  card.appendChild(h);

  if (!groups.size) {
    const p = document.createElement('div');
    p.className = 'hint';
    p.textContent = '这个格放不下任何棋子 —— 换一格试试。';
    card.appendChild(p);
    wrap.appendChild(card);
    return;
  }

  const seat = S.state.current_player;
  const list = document.createElement('div');
  list.className = 'cand';

  const item = (cells, count) => {
    const b = document.createElement('div');
    b.className = 'cand-item';
    b.appendChild(miniCanvas(cells, 11, seatColor(seat)));
    if (count > 1) {
      const n = document.createElement('span');
      n.className = 'cand-n';
      n.textContent = count;
      b.appendChild(n);
    }
    return b;
  };

  // 选中某个朝向：不立刻落子，交给鼠标在棋盘上挑位置
  const chooseOri = (oriId, actions) => {
    if (actions.length === 1) { play(actions[0]); return; }   // 只有一处，直接落
    S.pickOri = oriId;
    S.oriActions = actions;
    S.preview = actions[0];
    renderTrays(); draw();
  };

  if (S.pickOri !== null) {
    // 第三层：位置由鼠标决定，这里只给提示和退路
    const tip = document.createElement('div');
    tip.className = 'hint';
    tip.textContent = `这个形态有 ${S.oriActions.length} 处可放 —— 在棋盘上移动鼠标选位置，点击落子。`;
    card.appendChild(tip);
    const back = document.createElement('button');
    back.textContent = '← 换一个形态';
    back.onclick = () => { S.pickOri = null; S.oriActions = null; S.preview = null; renderTrays(); draw(); };
    card.appendChild(back);
  } else if (merged || S.pickPiece !== null) {
    // 一步到位（项数少），或已经选了棋子、现在列它的朝向
    for (const [pid, byOri] of groups) {
      if (!merged && pid !== S.pickPiece) continue;
      for (const [oriId, actions] of byOri) {
        const b = item(oriCells(oriId), actions.length);
        b.onclick = () => chooseOri(oriId, actions);
        b.onmouseenter = () => { S.preview = actions[0]; draw(); };
        b.onmouseleave = () => { S.preview = null; draw(); };
        list.appendChild(b);
      }
    }
    if (!merged) {
      const back = document.createElement('button');
      back.textContent = '← 换一枚棋子';
      back.onclick = () => { S.pickPiece = null; S.preview = null; renderTrays(); draw(); };
      card.appendChild(back);
    }
  } else {
    // 项数多：先选棋子
    for (const [pid, byOri] of groups) {
      const total = [...byOri.values()].reduce((n, a) => n + a.length, 0);
      const b = item(S.meta.pieces[pid].orientations[0].cells, byOri.size);
      b.onclick = () => {
        if (total === 1) { play([...byOri.values()][0][0]); return; }
        S.pickPiece = pid; S.preview = null; renderTrays(); draw();
      };
      b.onmouseenter = () => { S.preview = [...byOri.values()][0][0]; draw(); };
      b.onmouseleave = () => { S.preview = null; draw(); };
      list.appendChild(b);
    }
  }
  card.appendChild(list);
  wrap.appendChild(card);
}

function renderTrays() {
  const wrap = $('pieces-wrap');
  wrap.innerHTML = '';
  const st = S.state;
  if (!st) return;
  if (S.pickCell) { renderCandidates(wrap); return; }

  for (const { seat, interactive } of trayPlan()) {
    const card = document.createElement('div');
    card.className = 'card';

    const h = document.createElement('h2');
    h.innerHTML = dotHtml(seat) + '棋子'
      + `<small>剩 ${st.remaining[seat].filter(Boolean).length} 枚 · 已占 ${st.scores[seat]} 格`
      + (interactive ? ' · 点棋盘格子选位置' : ' · 只读') + '</small>';
    card.appendChild(h);

    const tray = document.createElement('div');
    tray.className = 'tray' + (interactive ? '' : ' readonly');
    for (const p of S.meta.pieces) {
      const div = document.createElement('div');
      const used = !st.remaining[seat][p.id];
      div.className = 'piece' + (used ? ' used' : '')
        ;
      div.appendChild(miniCanvas(p.orientations[0].cells, 9, seatColor(seat)));
      tray.appendChild(div);
    }
    card.appendChild(tray);
    wrap.appendChild(card);
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
  // 比分格子的颜色和名字跟着「这一局谁坐这个座位」走
  for (const seat of [0, 1]) {
    $('dot' + seat).style.background = seatColor(seat);
    const desc = describeSeat(st, seat);
    $('name' + seat).textContent = shorten(desc);
    $('score-box' + seat).title = desc;
  }

  // 战绩表里的引擎名也用同一套描述。swap 要等 recordResult 之后才翻，
  // 所以这里读到的仍是**当前这一局**的座位分配，对得上。
  if (S.series) {
    const a = seatOfA();
    S.series.names[0] = describeSeat(st, a);
    S.series.names[1] = describeSeat(st, 1 - a);
  }

  $('ai-name').innerHTML =
    dotHtml(0) + describeSeat(st, 0) + '　vs　' + dotHtml(1) + describeSeat(st, 1);

  renderStatus(st);

  // 没人在座就不存在「选中的棋子」；有人在座时，选中的棋子被用掉了要清掉。
  // 这里按人的座位索引，noHuman 时直接清空，绝不拿 -1 去索引。
  const noHuman = st.human_player < 0;
  if (noHuman) {
    clearPick();
  } else {
    const seat = st.human_player;             // 走到这里一定 >= 0
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
    list.appendChild(li);
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
// 模拟数下拉里表示「不适用」的那一项（对手是规则基线时）
const NA = '';

function seatSel(i) { return $('seat' + i); }
function simsSel(i) { return $('sims' + i); }
function seatValues() { return [seatSel(0).value, seatSel(1).value]; }
function seatSims() {
  // 「—」是给人看的，后端要一个合法整数。规则基线那侧填什么都无所谓 ——
  // RuleBrain.choose 收下 sims 但完全不用它。
  return [0, 1].map(i => {
    const v = simsSel(i).value;
    return v === NA ? S.meta.default_sims : parseInt(v, 10);
  });
}

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
//
// 三个都是「实心 + 圆角」的同一族，视觉重量接近，在 24x24 里都居中：
// 三角靠 stroke-linejoin=round 把尖角磨圆（纯路径做圆角三角要写贝塞尔，
// 不值当），方块和竖条直接用 rx。
const ICONS = {
  play: '<path d="M8.5 5.5 19 12 8.5 18.5Z" fill="currentColor" stroke="currentColor"'
      + ' stroke-width="2.6" stroke-linejoin="round"/>',
  stop: '<rect x="6.5" y="6.5" width="11" height="11" rx="2.6" fill="currentColor"/>',
  pause: '<rect x="7.6" y="5.5" width="3.4" height="13" rx="1.7" fill="currentColor"/>'
       + '<rect x="13" y="5.5" width="3.4" height="13" rx="1.7" fill="currentColor"/>',
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
  const locked = playing;

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
    seatSel(i).disabled = locked;
    simsSel(i).disabled = locked || simsSel(i).dataset.ruleDisabled === '1';
  }
  $('lock-hint').textContent = paused ? '已暂停 —— 双方都不能落子'
    : playing ? '对局进行中，配置已锁定' : '';
  $('series-count').disabled = locked;
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
  // 规则基线那一侧的模拟数显示成「—」而不是留着一个数字。
  // 置灰但仍显示 64，会让人以为「这局它搜了 64 次」——其实它根本不搜索。
  for (let i = 0; i < 2; i++) {
    const sel = simsSel(i);
    sel.dataset.ruleDisabled = isNet[i] ? '0' : '1';
    if (isNet[i]) {
      if (sel.value === NA) sel.value = sel.dataset.last || String(S.meta.default_sims);
    } else {
      if (sel.value !== NA) sel.dataset.last = sel.value;
      sel.value = NA;
    }
    sel.disabled = !isNet[i] || S.phase !== 'idle';
  }
  $('ai-name').innerHTML =
    `<span class="seat" style="background:${COLORS[0]}"></span>` + seatDesc(0) +
    '　vs　' + `<span class="seat" style="background:${COLORS[1]}"></span>` + seatDesc(1);
  void bothAi;

  $('backend-hint').textContent = (!isNet[0] && !isNet[1])
    ? '两边都不用网络，模拟数不起作用（规则基线不搜索）'
    : '右侧数字 = 该座位每步的 MCTS 模拟数，越大越强也越慢';

  // 连续对战要求双方都不是人（一局完要立刻开下一局，不能停下来等人）。
  // 条件不满足就把整行藏掉 —— 摆一个灰着的控件再配一句「为什么不能用」，
  // 只是把界面弄复杂，并没有多给出信息。
  $('series-row').classList.toggle('gone', !bothAi);
  if (!bothAi) $('series-count').value = '1';
  $('series-count').disabled = S.phase !== 'idle';
}

// 配置改动只留在本地，等「开始对局」时一次性提交给 /api/new。
// 不再中途调 /api/backend —— 对局中根本没有改配置的通道，比事后拦更干净。
function onConfigChange(ev) {
  S.labels = null;                  // 旧 label 已经不对应新选择了
  // 记下这一侧手选的模拟数，切到规则基线再切回来时能恢复
  const t = ev && ev.target;
  if (t && t.classList.contains('sims') && t.value !== NA) t.dataset.last = t.value;
  syncBackendUi();
}

// ---------------------------------------------------------------- 开始 / 结束

// 这一局的随机开局种子。**null = 不注入**。
//
// 只有连打多局才随机：多局是量棋力，随机开局能绕开自博弈那个「所有局都走同一手」
// 的漏斗；单局是看棋，而「网络自己开什么」恰恰是想看的东西。
// 一对里的两局用**同一个**种子 —— 开局逐手相同、先后手已翻，先手优势对双方
// 各记一半，正是 arena 的成对口径。
function openingSeedForThisGame() {
  const s = S.series;
  if (!s || s.total <= 1) return null;
  if (!s.swap) s.openingSeed = Math.floor(Math.random() * 2 ** 31);
  return s.openingSeed;
}

// 开一局新棋（不动战绩）
async function newSession() {
  // 换边在这里做，不在 recordResult 里：终局后棋盘还摆着，那时翻会让颜色对调。
  // 第一局（played === 0）不翻。**必须在取开局种子之前**，否则 `!swap`
  // 判「新一对开始」会错位，成对口径就破了。
  if (S.series && S.series.played > 0) S.series.swap = !S.series.swap;
  const o = orderedForThisGame();          // 连续对战时逐局交换先后手
  const r = await post('/api/new', {
    players: o.players, sims: o.sims,
    opening_seed: openingSeedForThisGame(),
  });
  S.sid = r.sid;
  S.labels = r.labels || null;
  S.analysis = null; S.pickCell = null; S.pickPiece = null;
  S.pickOri = null; S.oriActions = null; S.preview = null;
  applyState(r.state, null);
}

async function startGame() {
  newSeries(seriesCount());
  await guard(async () => {
    setPhase('playing');
    await newSession();
  });
  renderSeries();
  await pump();
}

// 结束只是停手：棋盘留在原样供查看，配置解锁。
// 服务端那边的会话不用管，下次开始会新建一个。
function endGame() {
  setPhase('idle');
  syncBackendUi();          // 解锁后要重算规则基线那一侧的置灰
  if (S.state) applyState(S.state, undefined);
  renderSeries();           // 战绩留在页面上，别一结束就没了
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
// ------------------------------------------------------------------ 连续对战
//
// 只是「开始对局」的一个选项：一局下完自动开下一局，棋盘照常逐手显示，
// 统计在前端累加。服务端不需要知道有这回事 —— 每局就是一次 /api/new。

function seriesCount() { return parseInt($('series-count').value, 10) || 1; }

// 模拟数怎么写给人看。规则基线不搜索，就不写这一项。
function simsText(n) { return n <= 0 ? '纯策略' : n + ' 次模拟'; }

// 一方的完整描述：模型要连模拟数一起说清楚 ——
// 光写 step 数看不出它这局到底搜了多少，而那对棋力的影响不比 step 小。
function describe(backendId, label, sims) {
  if (backendId === null || backendId === HUMAN) return '人类';
  return backendId.startsWith('net:') ? label + ' · ' + simsText(sims) : label;
}

function seatDesc(i) {
  const v = seatValues()[i];
  return describe(v, seatLabel(v), seatSims()[i]);
}

// 连续对战**逐局交换先后手**。
//
// 不交换的话得分率量的是「甲执先 vs 乙执后」，而本项目先手优势极大
// （网络自博弈的先手胜率能到 0.97），那个数字和相对棋力基本无关。
// 交换之后甲一半局执先、一半局执后，先手优势对双方各记一半，就抵消了。
//
// 因此战绩必须按**引擎**记，不能按座位记 —— 座位每局都在换。
function newSeries(total) {
  S.series = {
    total: total, played: 0, swap: false, openingSeed: null,
    wins: [0, 0], draws: 0, squares: [0, 0], plies: 0,
    firstWins: [0, 0], secondWins: [0, 0],
    names: [seatDesc(0), seatDesc(1)],   // 开局前的占位，拿到服务端 label 后覆盖
  };
}

// 这一局「甲」实际坐哪个座位
function seatOfA() { return S.series && S.series.swap ? 1 : 0; }

// 按当前该谁执先，给出这一局的 players / sims
function orderedForThisGame() {
  const p = seatPlayers(), m = seatSims();
  return (S.series && S.series.swap) ? { players: [p[1], p[0]], sims: [m[1], m[0]] }
                                     : { players: p, sims: m };
}

// 一局终局时累加。**每局只能记一次** —— applyState 会被调用很多遍，
// 靠 sid 去重最省事。
function recordResult(st) {
  const s = S.series;
  if (!s || s.lastSid === S.sid) return;
  s.lastSid = S.sid;

  const a = seatOfA(), b = 1 - a;
  const sa = st.scores[a], sb = st.scores[b];
  if (sa > sb) { s.wins[0]++; (a === 0 ? s.firstWins : s.secondWins)[0]++; }
  else if (sb > sa) { s.wins[1]++; (b === 0 ? s.firstWins : s.secondWins)[1]++; }
  else s.draws++;
  s.squares[0] += sa; s.squares[1] += sb;
  s.plies += st.ply;
  s.played++;
  // **这里不翻 swap。** 颜色是从 swap 现算的（seatColor -> engineOfSeat），
  // 终局那一刻翻的话，刚下完、还摆在屏幕上的这一局会当场被按下一局的座位重画 ——
  // 表现就是「一局打完双方颜色对调」。翻边属于「开下一局」，见 newSession()。
  renderSeries();
}

function renderSeries() {
  const s = S.series;
  const card = $('series-card');
  if (!s || s.total <= 1) { card.classList.add('gone'); return; }   // 单局不用摆战绩
  card.classList.remove('gone');
  $('series-progress').textContent =
    s.played + ' / ' + s.total + ' 局' + (S.phase === 'idle' ? '（已结束）' : '');
  if (!s.played) { $('series-result').innerHTML = '<div class="hint">第 1 局进行中…</div>'; return; }

  const n = s.played;
  const scoreA = (s.wins[0] + 0.5 * s.draws) / n;
  // Elo 换算和 cornerstone/elo.py 那条一致；0/1 会发散，钳一下
  const r = Math.min(0.999, Math.max(0.001, scoreA));
  const elo = -400 * Math.log10(1 / r - 1);
  $('series-result').innerHTML =
    '<table class="mstat"><tr><th></th>' +
    `<th><span class="seat" style="background:${COLORS[0]}"></span></th>` +
    `<th><span class="seat" style="background:${COLORS[1]}"></span></th></tr>` +
    // 名字很长，CSS 里按列宽省略号截断；完整的挂在 title 上
    '<tr class="engine"><td>引擎</td>' +
    '<td title="' + esc(s.names[0]) + '">' + esc(s.names[0]) + '</td>' +
    '<td title="' + esc(s.names[1]) + '">' + esc(s.names[1]) + '</td></tr>' +
    '<tr><td>胜</td><td>' + s.wins[0] + '</td><td>' + s.wins[1] + '</td></tr>' +
    '<tr><td>和</td><td colspan="2">' + s.draws + '</td></tr>' +
    // 「执先/执后」说的是这一局谁先走，不是在给某一方起名字 ——
    // 自动换边之后双方都会轮到执先，这两个数正好用来看先手优势有多大，
    // 也用来验证换边确实在起作用（两边执先次数应该各占一半）。
    '<tr><td>执先胜</td><td>' + s.firstWins[0] + '</td><td>' + s.firstWins[1] + '</td></tr>' +
    '<tr><td>执后胜</td><td>' + s.secondWins[0] + '</td><td>' + s.secondWins[1] + '</td></tr>' +
    '<tr><td>平均占格</td><td>' + (s.squares[0] / n).toFixed(1) + '</td>' +
    '<td>' + (s.squares[1] / n).toFixed(1) + '</td></tr></table>' +
    '<div class="hint">' +
    `<span class="seat" style="background:${COLORS[0]}"></span>得分率 ` + scoreA.toFixed(3) +
    '　Elo 差 ' + (elo > 0 ? '+' : '') + elo.toFixed(0) +
    '　平均 ' + (s.plies / n).toFixed(1) + ' 手　自动换边</div>';
}


const sleep = (ms) => new Promise(r => setTimeout(r, ms));

// 连续对战里，一局结束到下一局开始之间停一会儿，让人看清终局。
// 没有这个停顿，棋盘会「啪」地跳到下一局的空盘，等于没看见。
const BETWEEN_GAMES_MS = 2000;

async function pump() {
  await guard(async () => {
    while (S.phase === 'playing' && S.state) {
      if (S.state.terminal) {
        recordResult(S.state);
        if (S.series.played >= S.series.total) break;
        await sleep(BETWEEN_GAMES_MS);
        if (S.phase !== 'playing') break;   // 停顿期间被暂停/结束了
        await newSession();          // 自动开下一局
        continue;
      }
      // 轮到人就停下来等点击
      if (S.state.players[S.state.current_player] === null) break;
      await aiMove();
    }
  });
  if (S.phase !== 'idle' && S.state && S.state.terminal
      && S.series && S.series.played >= S.series.total) {
    endGame();
  }
}

async function aiMove() {
  const r = await post('/api/ai', { sid: S.sid });
  // 用服务端的 label（它带解析后的真实 step）配上本地的模拟数。
  // 连续对战会换边，所以这里按**当前这一局的实际座位**显示，不看下拉框顺序。
  if (r.labels) S.labels = r.labels;
  if (r.ai_seconds !== undefined) {
    // 用 describe() 而不是自己拼 —— 它知道「规则基线不搜索，别写模拟数」。
    // 之前这里无条件拼上模拟数，于是出现过
    // 「上一手：规则基线 greedy-mobility · 64 次模拟」这种自相矛盾的话。
    const who = r.players ? r.players[r.ai_player] : null;
    const sims = r.sims ? r.sims[r.ai_player] : 0;
    S.lastMove =
      `上一手 ${dotHtml(r.ai_player)}${describe(who, r.ai_label, sims)} · ${r.ai_seconds}s`;
  }
  applyState(r, r.analysis || null);
}

async function play(action) {
  await guard(async () => {
    const st = await post('/api/move', { sid: S.sid, action });
    clearPick();
    applyState(st, null);
  });
  // pump 里那个 while 顺带覆盖了「AI 走完对方仍无法落子（停手）」的情况
  await pump();
}

CV.addEventListener('mousemove', (ev) => {
  const cell = cellAt(ev);
  const changed = JSON.stringify(cell) !== JSON.stringify(S.hover);
  S.hover = cell;
  // 选中朝向后，位置由鼠标决定 —— 这一层不占点击，因为实测位置数中位就是 1
  if (S.pickOri !== null && S.oriActions && changed) {
    S.preview = pickPlacement(S.oriActions, cell);
  }
  // 只有角点可点，光标跟着变 —— 省得人对着不能点的格反复试
  const st = S.state;
  const canPick = st && !st.terminal && S.phase === 'playing'
    && st.players[st.current_player] === null && cell && isPlayable(cell);
  CV.style.cursor = canPick ? 'pointer' : 'default';
  if (changed) draw();
});
CV.addEventListener('mouseleave', () => { S.hover = null; draw(); });
// 点棋盘 = **选位置**，不是落子；落子在右侧候选列表里点。
CV.addEventListener('click', (ev) => {
  const st = S.state;
  if (S.phase !== 'playing' || !st || st.terminal) return;
  if (st.players[st.current_player] !== null) return;      // 轮到 AI，人不能替它下
  const cell = cellAt(ev);
  // 已经选好形态、正在用鼠标挑位置：这一下就是落子
  if (S.pickOri !== null && S.preview !== null) { play(S.preview); return; }
  // **只有角点能点**：新棋子必须斜接自己已有的棋子，点别处没有任何合法着法。
  // 点非角点就退回托盘，不进入选位置状态 —— 否则会出现「点了个格、右侧空空」。
  if (!cell || !isPlayable(cell)) { clearPick(); return; }
  S.pickCell = cell; S.pickPiece = null; S.preview = null;
  renderTrays(); draw();
});

function isPlayable([r, c]) {
  return playableCells().some(([ar, ac]) => ar === r && ac === c);
}

function clearPick() {
  S.pickCell = null; S.pickPiece = null; S.pickOri = null;
  S.oriActions = null; S.preview = null;
  renderTrays(); draw();
}

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
on('series-count', 'onchange', onConfigChange);

document.addEventListener('keydown', (ev) => {
  if (S.state && S.state.human_player < 0) return;   // 没人在座就没有选位置这回事
  if (ev.key !== 'Escape') return;
  if (S.pickOri !== null) {            // 退回形态列表
    S.pickOri = null; S.oriActions = null; S.preview = null; renderTrays(); draw();
  } else if (S.pickPiece !== null) {   // 退回棋子列表
    S.pickPiece = null; renderTrays(); draw();
  } else if (S.pickCell) clearPick();  // 退回托盘
});

// ---------------------------------------------------------------- 启动

(async () => {
  S.meta = await api('/api/meta');
  layout();
  for (let i = 0; i < 2; i++) {
    const sel = simsSel(i);
    const na = document.createElement('option');
    na.value = NA; na.textContent = '—';       // 对手是规则基线时显示这个
    sel.appendChild(na);
    for (const n of S.meta.sim_choices) {
      const o = document.createElement('option');
      o.value = String(n);
      // 0 走的是另一条路：取网络先验的 argmax，不看搜索结果（确定性）
      o.textContent = n === 0 ? '纯策略' : String(n);
      if (n === S.meta.default_sims) o.selected = true;
      sel.appendChild(o);
    }
    sel.dataset.last = String(S.meta.default_sims);
  }
  // 默认人执先、AI 执后，和改版前一致
  const sc = $('series-count');
  for (const n of S.meta.series_counts) {
    const o = document.createElement('option');
    o.value = String(n); o.textContent = n === 1 ? '1 局' : n + ' 局';
    sc.appendChild(o);
  }

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
