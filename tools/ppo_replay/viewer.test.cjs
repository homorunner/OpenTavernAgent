const test = require("node:test");
const assert = require("node:assert/strict");
const R = require("./viewer.js");
const players = () =>
    Array.from({ length: 8 }, (_, seat) => ({
        seat,
        alive: true,
        placement: seat + 1,
        board: [],
        hand: [],
        shop: [],
    }));
function game() {
    return {
        schema: "tavernagent_v2.replay",
        version: 1,
        seed: "18446744073709551614",
        placements: [1, 2, 3, 4, 5, 6, 7, 8],
        events: [
            { type: "turn_start", turn: 1, players: players() },
            { type: "combat_start", turn: 1, players: players() },
            { type: "game_end", turn: 1, players: players() },
        ],
    };
}
test("validate complete replays and preserve large seeds", () => {
    const g = game();
    assert.equal(R.validate(g).seed, "18446744073709551614");
    assert.deepEqual(R.turns(g), [1]);
});
test("reject wrong schema, partial games and malformed snapshots", () => {
    for (const alter of [
        (g) => (g.version = 2),
        (g) => (g.placements[0] = 0),
        (g) => g.events.pop(),
        (g) => (g.events[0].players[0].board = null),
    ]) {
        const g = game();
        alter(g);
        assert.throws(() => R.validate(g));
    }
});
test("last warband is pre-combat, not cleared terminal state", () => {
    const g = game();
    g.events[1].players[0].board = [{ name_zh: "保留阵容", atk: 12, hp: 34 }];
    assert.equal(R.lastBoard(g, 0).board[0].atk, 12);
    assert.equal(R.lastBattle(g).type, "combat_start");
});
test("actions name the actual bought card and merged magnetic target", () => {
    const before = {
        shop: [{ name_zh: "测试随从" }],
        hand: [{ name_zh: "磁力随从" }],
    };
    assert.equal(
        R.action({ engine_action: { Buy: 0 }, before }),
        "购买 #1 测试随从",
    );
    assert.match(
        R.action({
            engine_action: { Magnetize: { hand: 0, board: 2 } },
            before,
        }),
        /磁力随从.*#3/,
    );
    assert.equal(R.action({ freeze_and_end: true }), "冻结并结束回合");
});
test("directory listings expose direct JSONL files and folders only", () => {
    const base = "http://example.test/runs/";
    const entries = R.directoryEntries(
        [
            "../",
            "./",
            "nested/",
            "result.jsonl",
            "result.jsonl",
            "report.json",
            ".git/",
            "nested/hidden.jsonl",
            "https://other.test/a.jsonl",
            "javascript:alert(1)",
            "%E6%B5%8B%E8%AF%95%20%23.jsonl",
            "broken%ZZ.jsonl",
        ],
        base,
    );
    assert.deepEqual(
        entries.map((e) => e.name),
        ["nested", "测试 #.jsonl", "result.jsonl"],
    );
    assert.equal(entries[0].folder, true);
    assert.equal(
        entries[1].url,
        "http://example.test/runs/%E6%B5%8B%E8%AF%95%20%23.jsonl",
    );
});
test("server paths resolve against the current directory and reject other origins", () => {
    const base = "http://example.test/runs/replay/";
    assert.equal(
        R.serverURL("one.jsonl", base).pathname,
        "/runs/replay/one.jsonl",
    );
    assert.equal(
        R.serverURL("/runs/one.jsonl", base).pathname,
        "/runs/one.jsonl",
    );
    for (const path of [
        "https://other.test/a.jsonl",
        "file:///tmp/a.jsonl",
        "javascript:alert(1)",
        "http://user:pass@example.test/a.jsonl",
    ])
        assert.throws(() => R.serverURL(path, base));
});
