"use strict";
const Replay = (() => {
    const labels = {
        EndTurn: "结束回合",
        Refresh: "刷新",
        Freeze: "冻结 / 解冻",
        Upgrade: "升本",
        HeroPower: "英雄技能",
        DarkGift: "暗月礼物",
        Cancel: "取消",
        Buy: "购买",
        Sell: "出售",
        Activate: "激活",
        Choose: "选择",
        Play: "打出",
        Magnetize: "磁力",
        Move: "移动",
    };
    const name = (c) => c?.name_zh || c?.name || c?.id || "未知卡牌";
    const text = (c) => (c?.text_zh || c?.text || "").replace(/<[^>]*>/g, "");
    const hero = (p) => name(p?.hero);
    function validate(g) {
        if (
            g?.schema !== "tavernagent_v2.replay" ||
            g.version !== 1 ||
            !Array.isArray(g.events) ||
            !g.events.length
        )
            throw Error("不是受支持的 TavernAgent v2 回放（需要 version=1）");
        if (
            !Array.isArray(g.placements) ||
            g.placements.length !== 8 ||
            !g.placements.every(
                (p) => Number.isInteger(p) && p >= 1 && p <= 8,
            ) ||
            new Set(g.placements).size !== 8
        )
            throw Error("回放缺少完整的八席最终名次");
        if (g.events[0]?.type !== "turn_start") throw Error("回放缺少开局状态");
        if (g.events.at(-1)?.type !== "game_end")
            throw Error("回放尚未完整结束");
        for (const e of g.events) {
            if (!Number.isInteger(e.turn) || e.turn < 1)
                throw Error("回合编号无效");
            if (
                e.type === "action" &&
                (!Number.isInteger(e.seat) ||
                    e.seat < 0 ||
                    e.seat >= 8 ||
                    !e.before ||
                    !e.after)
            )
                throw Error("决策记录不完整");
            if (
                ![
                    "action",
                    "turn_start",
                    "combat_start",
                    "combat_end",
                    "game_end",
                ].includes(e.type)
            )
                throw Error("未知事件类型");
            if (
                e.type !== "action" &&
                (!Array.isArray(e.players) || e.players.length !== 8)
            )
                throw Error("回放缺少玩家状态");
            const states =
                e.type === "action" ? [e.before, e.after] : e.players;
            if (
                states.some(
                    (p) =>
                        !p ||
                        !Array.isArray(p.board) ||
                        !Array.isArray(p.hand) ||
                        !Array.isArray(p.shop) ||
                        !Number.isInteger(p.seat) ||
                        p.seat < 0 ||
                        p.seat >= 8,
                )
            )
                throw Error("玩家状态无效");
            if (
                e.type === "combat_end" &&
                (!Array.isArray(e.combats) ||
                    !g.events.some(
                        (x) => x.type === "combat_start" && x.turn === e.turn,
                    ))
            )
                throw Error("战斗记录不完整");
        }
        return g;
    }
    const turns = (g) =>
        [...new Set(g.events.map((e) => e.turn))].sort((a, b) => a - b);
    const lastBattle = (g) =>
        g.events.findLast((e) => e.type === "combat_start");
    function lastBoard(g, seat) {
        for (let i = g.events.length - 1; i >= 0; i--) {
            const e = g.events[i];
            if (e.type === "combat_start" && e.players[seat]?.alive)
                return e.players[seat];
        }
        // A recruit-phase elimination can occur before the first combat.
        return (
            g.events.findLast((e) => e.type === "action" && e.seat === seat)
                ?.before || g.events[0].players[seat]
        );
    }
    function action(e) {
        if (e.freeze_and_end) return "冻结并结束回合";
        const raw = e.engine_action;
        const kind = typeof raw === "string" ? raw : Object.keys(raw || {})[0];
        const arg = typeof raw === "object" ? raw[kind] : null,
            p = e.before;
        let detail = "";
        if (["Buy", "Sell", "Activate", "Choose", "HeroPower"].includes(kind)) {
            const list = {
                Buy: p.shop,
                Sell: p.board,
                Activate: p.board,
                Choose: p.pending?.options,
                HeroPower: p.hero_powers,
            }[kind];
            const c = list?.[arg];
            detail = ` #${Number(arg) + 1}${c ? " " + name(c) : ""}`;
        }
        if (kind === "Play" || kind === "Magnetize")
            detail = ` 手牌 #${arg.hand + 1} ${name(p.hand?.[arg.hand])} → ${kind === "Magnetize" ? "目标" : "位置"} #${(arg.pos ?? arg.board) + 1}`;
        if (kind === "Move") detail = ` ${JSON.stringify(arg)}`;
        return (labels[kind] || kind || "未知动作") + detail;
    }
    // Accept only files served by the same HTTP origin. All paths are relative to
    // the HTTP server's root, not the directory containing this viewer.
    function serverURL(path, base) {
        const url = new URL(path, base);
        if (
            !/^https?:$/.test(url.protocol) ||
            url.origin !== new URL(base).origin ||
            url.username ||
            url.password
        )
            throw Error("请选择当前服务器上的路径");
        url.hash = "";
        return url;
    }
    function directoryEntries(hrefs, base) {
        const directory = serverURL(base, base);
        if (!directory.pathname.endsWith("/")) directory.pathname += "/";
        const entries = new Map();
        for (const href of hrefs) {
            try {
                const url = serverURL(href, directory.href);
                if (url.search || !url.pathname.startsWith(directory.pathname))
                    continue;
                const relative = url.pathname.slice(directory.pathname.length);
                const folder = relative.endsWith("/");
                const leaf = folder ? relative.slice(0, -1) : relative;
                if (!leaf || leaf.includes("/")) continue;
                const name = decodeURIComponent(leaf);
                if (
                    name.startsWith(".") ||
                    (!folder && !/\.jsonl$/i.test(name))
                )
                    continue;
                entries.set(url.href, { name, folder, url: url.href });
            } catch {
                /* Ignore unrelated or malformed links in directory listings. */
            }
        }
        return [...entries.values()].sort(
            (a, b) =>
                Number(b.folder) - Number(a.folder) ||
                a.name.localeCompare(b.name, "zh-CN", { numeric: true }),
        );
    }
    return {
        validate,
        turns,
        lastBattle,
        lastBoard,
        action,
        name,
        text,
        hero,
        serverURL,
        directoryEntries,
    };
})();
if (typeof module !== "undefined") module.exports = Replay;
if (typeof document !== "undefined") {
    const $ = (id) => document.getElementById(id);
    let games = [];
    const node = (tag, text, cls) => {
        const e = document.createElement(tag);
        if (text !== undefined) e.textContent = text;
        if (cls) e.className = cls;
        return e;
    };
    function option(select, value, label) {
        const e = node("option", label);
        e.value = value;
        select.append(e);
    }
    function list(title, cards, parent) {
        parent.append(node("h3", title));
        if (!cards?.length) {
            parent.append(node("p", "（空）", "muted"));
            return;
        }
        const ol = node("ol");
        for (const c of cards) {
            const stats = c.atk !== undefined ? ` ${c.atk}/${c.hp}` : "";
            const flags = [
                c.golden ? "金色" : "",
                ...(c.keywords || []),
                c.frozen ? "已冻结" : "",
                c.locked ? "锁定" : "",
            ]
                .filter(Boolean)
                .join(" · ");
            const li = node(
                "li",
                `${Replay.name(c)}${stats}${flags ? " [" + flags + "]" : ""}`,
            );
            const desc = Replay.text(c);
            if (desc) li.append(node("span", desc, "card-text"));
            ol.append(li);
        }
        parent.append(ol);
    }
    function state(p, full = false) {
        const box = node("div");
        box.append(
            node(
                "p",
                `生命 ${p.health} + 护甲 ${p.armor} · ${p.tier} 本 · 金币 ${p.gold}${p.alive ? "" : " · 已淘汰"}`,
                "facts",
            ),
        );
        list("阵容", p.board, box);
        if (full) {
            list("手牌", p.hand, box);
            list("商店", p.shop, box);
            if (p.pending) {
                list("待选项", p.pending.options, box);
                box.append(node("p", `选择类型：${p.pending.kind}`, "muted"));
            }
            if (p.hero_powers?.length) list("英雄技能", p.hero_powers, box);
        }
        if (p.trinkets?.length) list("饰品", p.trinkets, box);
        if (p.rewards?.length) list("任务奖励", p.rewards, box);
        return box;
    }
    function player(p) {
        const box = node("article", undefined, "player");
        box.append(node("h3", `玩家 ${p.seat + 1} · ${Replay.hero(p)}`));
        box.append(state(p));
        return box;
    }
    function loadGame() {
        const g = games[Number($("game").value)];
        $("seat").replaceChildren();
        option($("seat"), "all", "全部玩家");
        for (const p of g.events[0].players)
            option($("seat"), p.seat, `${p.seat + 1} · ${Replay.hero(p)}`);
        $("seat").value = "0";
        $("turn").replaceChildren();
        for (const t of Replay.turns(g)) option($("turn"), t, `第 ${t} 回合`);
        $("metadata").textContent =
            `游戏种子 ${g.seed} · 模型 ${g.model?.file ?? "?"} · 训练种子 ${g.model?.training_seed ?? "?"} · ${g.model?.iteration ?? "?"} 轮 · ${g.opponent ?? "?"} · ${g.sample ? "概率采样" : "Greedy"} · ${g.backend ?? "?"} FP32`;
        const warnings = [];
        if (g.turn_limit_end)
            warnings.push("到达回合上限后判定名次，并非自然结束");
        if (g.truncated_combats)
            warnings.push(`${g.truncated_combats} 场战斗被引擎截断`);
        $("warning").hidden = !warnings.length;
        $("warning").textContent = warnings.join("；");
        $("results").replaceChildren();
        const table = node("table");
        const head = node("tr");
        for (const x of ["名次", "玩家", "英雄", "最后战前阵容"])
            head.append(node("th", x));
        table.append(head);
        const rows = g.events
            .at(-1)
            .players.toSorted((a, b) => a.placement - b.placement);
        for (const p of rows) {
            const tr = node("tr");
            const b = Replay.lastBoard(g, p.seat);
            for (const value of [
                p.placement,
                p.seat + 1,
                Replay.hero(p),
                (b.board || [])
                    .map((c) => `${Replay.name(c)} ${c.atk}/${c.hp}`)
                    .join(" / ") || "空场",
            ])
                tr.append(node("td", value));
            table.append(tr);
        }
        $("results").append(table);
        render();
    }
    function render() {
        const g = games[Number($("game").value)],
            turn = Number($("turn").value),
            seat = $("seat").value;
        const events = g.events.filter((e) => e.turn === turn);
        const start = events.find((e) => e.type === "combat_start"),
            end = events.find((e) => e.type === "combat_end");
        const last = Replay.lastBattle(g);
        $("round-title").textContent =
            `第 ${turn} 回合${last?.turn === turn ? " · 最后一战" : ""}${start?.players.filter((p) => p.alive).length === 2 ? " · 二人决赛" : ""}`;
        $("prev").disabled = $("turn").selectedIndex === 0;
        $("next").disabled =
            $("turn").selectedIndex === $("turn").options.length - 1;
        $("final").disabled = !last;
        $("combats").replaceChildren();
        if (end) {
            for (const c of end.combats) {
                const a = start.players[c.a],
                    b = c.b === null ? null : start.players[c.b];
                const rival = b
                    ? `玩家 ${b.seat + 1} ${Replay.hero(b)}`
                    : "幽灵";
                const winner =
                    c.winner === 2
                        ? "平局"
                        : c.winner === 0
                          ? `玩家 ${c.a + 1} 胜`
                          : `${c.b === null ? "幽灵" : `玩家 ${c.b + 1}`} 胜`;
                $("combats").append(
                    node(
                        "p",
                        `玩家 ${a.seat + 1} ${Replay.hero(a)} vs ${rival} → ${winner} · 伤害 ${c.damage}${c.truncated ? " · 战斗截断" : ""}`,
                    ),
                );
            }
        } else
            $("combats").append(
                node(
                    "p",
                    "本回合没有战斗记录（可能在招募阶段结束比赛）。",
                    "muted",
                ),
            );
        $("boards").replaceChildren();
        if (start) {
            for (const p of start.players.filter(
                (p) => p.alive && (seat === "all" || p.seat === Number(seat)),
            ))
                $("boards").append(player(p));
        }
        if (!$("boards").children.length)
            $("boards").append(
                node("p", "该玩家本回合没有战前阵容记录。", "empty"),
            );
        $("actions").replaceChildren();
        const actions = events.filter(
            (e) =>
                e.type === "action" &&
                (seat === "all" || e.seat === Number(seat)),
        );
        actions.forEach((e, i) => {
            const detail = node("details", undefined, "decision");
            const probability =
                typeof e.probability === "number"
                    ? ` · p=${(100 * e.probability).toFixed(1)}%`
                    : "";
            detail.append(
                node(
                    "summary",
                    `${i + 1}. 玩家 ${e.seat + 1} · ${Replay.action(e)} · 金币 ${e.before.gold} → ${e.after.gold}${probability}`,
                ),
            );
            const value =
                typeof e.value === "number" ? ` · V=${e.value.toFixed(3)}` : "";
            detail.append(
                node(
                    "p",
                    `${e.controller}${value}${e.resolved_combat || e.after.turn !== e.turn ? " · 动作后包含战斗结算，可能已进入下一回合；战前阵容见右侧面板" : ""}`,
                    "muted",
                ),
            );
            const compare = node("div", undefined, "compare");
            for (const [title, p] of [
                ["动作前", e.before],
                ["动作后", e.after],
            ]) {
                const column = node("div");
                column.append(node("h3", title), state(p, true));
                compare.append(column);
            }
            detail.append(compare);
            $("actions").append(detail);
        });
        if (!actions.length)
            $("actions").append(node("p", "本回合没有该玩家的决策。", "empty"));
    }
    const maxBytes = 256 * 1024 * 1024;
    let loading = false;
    function busy(value) {
        loading = value;
        $("file").disabled = value;
        for (const control of $("server").querySelectorAll("input, button"))
            control.disabled = value;
        $("server-up").disabled = value || currentDirectory.pathname === "/";
    }
    async function loadSources(files) {
        if (!files.length || loading) return;
        $("viewer").hidden = true;
        busy(true);
        games = [];
        $("status").textContent = `正在读取 ${files[0].name}…`;
        try {
            for (const file of files) {
                const { stream, size } = await file.open();
                const reader = stream.getReader(),
                    decoder = new TextDecoder();
                let lineNo = 0,
                    pending = "",
                    bytes = 0;
                const consume = (line) => {
                    lineNo++;
                    if (!line.trim()) return;
                    try {
                        games.push(Replay.validate(JSON.parse(line)));
                    } catch (error) {
                        throw Error(
                            `${file.name} 第 ${lineNo} 行：${error.message}`,
                        );
                    }
                };
                try {
                    if (size > maxBytes)
                        throw Error(
                            "单个文件超过 256 MiB，请减少导出局数或拆分文件",
                        );
                    while (true) {
                        const { value, done } = await reader.read();
                        bytes += value?.byteLength || 0;
                        if (bytes > maxBytes)
                            throw Error(
                                "单个文件超过 256 MiB，请减少导出局数或拆分文件",
                            );
                        pending += decoder.decode(value, { stream: !done });
                        let index;
                        while ((index = pending.indexOf("\n")) >= 0) {
                            consume(pending.slice(0, index));
                            pending = pending.slice(index + 1);
                        }
                        if (done) break;
                        $("status").textContent =
                            `正在读取 ${file.name}… 已解析 ${games.length} 局`;
                    }
                    if (pending.trim()) consume(pending);
                } finally {
                    await reader.cancel().catch(() => {});
                    reader.releaseLock();
                }
            }
            if (!games.length) throw Error("文件中没有完整对局");
            $("game").replaceChildren();
            games.forEach((g, i) =>
                option(
                    $("game"),
                    i,
                    `${i + 1} · 种子 ${g.seed} · 座位1第${g.placements[0]}名`,
                ),
            );
            $("viewer").hidden = false;
            $("status").textContent =
                `已载入 ${games.length} 局 · ${files.map((f) => f.name).join("、")}`;
            loadGame();
        } catch (error) {
            games = [];
            $("viewer").hidden = true;
            $("status").textContent = `无法加载：${error.message}`;
        } finally {
            busy(false);
        }
    }
    $("file").addEventListener("change", (event) =>
        loadSources(
            Array.from(event.target.files, (file) => ({
                name: file.name,
                open: async () => ({ stream: file.stream(), size: file.size }),
            })),
        ),
    );

    const served = /^https?:$/.test(location.protocol);
    let currentDirectory = new URL("/", location.href),
        listingRequest = 0;
    function readServerFile(path) {
        try {
            const url = Replay.serverURL(path, currentDirectory.href);
            return loadSources([
                {
                    name: decodeURIComponent(url.pathname),
                    open: async () => {
                        const response = await fetch(url, {
                            cache: "no-store",
                        });
                        if (!response.ok)
                            throw Error(
                                `读取失败（HTTP ${response.status}）：${url.pathname}`,
                            );
                        if (!response.body)
                            throw Error("服务器没有返回文件内容");
                        return {
                            stream: response.body,
                            size:
                                Number(
                                    response.headers.get("content-length"),
                                ) || 0,
                        };
                    },
                },
            ]);
        } catch (error) {
            $("status").textContent = `无法加载：${error.message}`;
        }
    }
    async function browse(path) {
        if (loading) return;
        const request = ++listingRequest;
        try {
            const url = Replay.serverURL(path, currentDirectory.href);
            if (!url.pathname.endsWith("/")) url.pathname += "/";
            url.search = "";
            $("server-status").textContent =
                `正在读取目录 ${decodeURIComponent(url.pathname)}…`;
            const response = await fetch(url, { cache: "no-store" });
            if (!response.ok)
                throw Error(`目录读取失败（HTTP ${response.status}）`);
            const html = await response.text();
            if (request !== listingRequest) return;
            const page = new DOMParser().parseFromString(html, "text/html");
            const entries = Replay.directoryEntries(
                Array.from(page.querySelectorAll("a[href]"), (a) =>
                    a.getAttribute("href"),
                ),
                url.href,
            );
            currentDirectory = url;
            $("server-path").value = url.pathname;
            $("server-entries").replaceChildren();
            for (const entry of entries) {
                const button = node(
                    "button",
                    `${entry.folder ? "目录" : "JSONL"} · ${entry.name}`,
                );
                button.type = "button";
                button.disabled = loading;
                button.dataset.path = new URL(entry.url).pathname;
                button.addEventListener("click", () =>
                    entry.folder
                        ? browse(entry.url)
                        : readServerFile(entry.url),
                );
                $("server-entries").append(button);
            }
            $("server-up").disabled = loading || url.pathname === "/";
            $("server-status").textContent = entries.length
                ? `当前目录 ${decodeURIComponent(url.pathname)} · 点击目录进入，点击 JSONL 直接读取`
                : "此目录没有可列出的文件夹或 JSONL。若服务器关闭了目录列表，可在上方填写文件路径直接打开。";
        } catch (error) {
            if (request === listingRequest)
                $("server-status").textContent =
                    `无法浏览：${error.message}。也可以填写 JSONL 路径直接打开。`;
        }
    }
    $("server-form").addEventListener("submit", (event) => {
        event.preventDefault();
        try {
            const url = Replay.serverURL(
                $("server-path").value.trim(),
                currentDirectory.href,
            );
            /\.jsonl$/i.test(decodeURIComponent(url.pathname))
                ? readServerFile(url.href)
                : browse(url.href);
        } catch (error) {
            $("server-status").textContent = error.message;
        }
    });
    $("server-up").addEventListener("click", () =>
        browse(new URL("../", currentDirectory.href).href),
    );
    $("server-root").addEventListener("click", () => browse("/"));
    $("server-refresh").addEventListener("click", () =>
        browse(currentDirectory.href),
    );
    if (served) {
        $("server").hidden = false;
        browse("/");
        const file = new URL(location.href).searchParams.get("replay");
        if (file) readServerFile(file);
    }
    $("game").addEventListener("change", loadGame);
    $("seat").addEventListener("change", render);
    $("turn").addEventListener("change", render);
    for (const [id, delta] of [
        ["prev", -1],
        ["next", 1],
    ])
        $(id).addEventListener("click", () => {
            $("turn").selectedIndex += delta;
            render();
        });
    $("final").addEventListener("click", () => {
        const last = Replay.lastBattle(games[Number($("game").value)]);
        if (last) {
            $("turn").value = last.turn;
            $("seat").value = "all";
            render();
        }
    });
}
