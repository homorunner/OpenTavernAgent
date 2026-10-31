# 接入 AutoTavern：理解、开发与调试记录

这份文档记录一件事：**把 AutoTavern 这个模拟器接到真实游戏里**，从读代码、搭环境、写插件，到一次次被运行时
打回来、最后把商店/赠礼/三连全部对上的完整过程。

它不是 API 参考。参考在别处：

| 想知道 | 看 |
| :-- | :-- |
| tavernplugin 怎么用、协议长什么样 | [tavernplugin/README.md](../tavernplugin/README.md) |
| 游戏内字段怎么映射到 TavernGym | [tavernplugin/docs/ALIGNMENT.md](../tavernplugin/docs/ALIGNMENT.md) |
| 模拟器本身的设计 | [ARCHITECTURE.md](ARCHITECTURE.md) · [TAVERNGYM.md](TAVERNGYM.md) |

这里记录的是**为什么这么做、踩了什么、怎么发现的**。

---

## 1. 项目理解

### 1.1 仓库是什么

AutoTavern 是南京大学一篇论文的代码与数据发布（MIT）。它包含两个产物：

- **TavernGym** —— 一个确定性的酒馆战棋（单人模式）模拟器：Rust 引擎 + PyO3 绑定 + numpy 封装。
- **TavernBench** —— 960 道决策题（368 题评测集），带答案、存档、打分器和模型 runner。

版本锚定客户端 **36.6.3.253932**、服务端规则 **36.6.3**。

仓库没有模型本体：训练好的 TavernAgent、权重、训练代码、题目生成器都没放出来。从答案文件看，
TavernAgent 的概率分母都是 256、延迟只有几毫秒——**它更像"在引擎里跑 256 次模拟取 argmax"的搜索器**
，而不是神经网络（这是推断，不是仓库的声明）。这也意味着：引擎本身给了你复现它的全部能力。

### 1.2 对插件真正有用的部分

读完 `crates/taverngym/src/` 之后，结论是：**真正值钱的是 `debug` 模块**。

`Game` 提供了 `state_json()`（精确存档）、`view(p)`（玩家视角 JSON），而 `DebugCmd` 允许在不改引擎
的前提下构造任意状态：

| 命令 | 用途 |
| :-- | :-- |
| `set_player` | 血量/护甲/等级/金币 |
| `set_hero` | 英雄（决定光环、技能） |
| `add_minion` / `edit_minion` / `remove` | 棋盘/手牌/商店，可指定卡号、位置、金卡、攻防、关键词、附件 |
| `clear_zone` | 清空 |
| `combat_preview` | **对两个 seat 的阵容跑 n 次战斗**，返回 win/tie/loss 与每局伤害 |

`combat_preview` 就是整个插件的地基：只要能把"游戏里两边长什么样"翻译成 `add_minion` 序列，胜率
和决策评估就都有了。TavernBench 的 `engine.py` 也是这么干的（`from_state` + `apply_ops` + `preview`）。

### 1.3 引擎里几个必须知道的设计

- **卡牌即脚本**：每张卡是一个 `Script` 钩子表，金卡不复制代码，脚本读 `me.golden`。
- **确定性**：单一 xoshiro256** 流，seed + 动作序列可完整复现；`Clone` 是深拷贝。
- **状态模型**：`Minion.atk/hp` 是**当前值**（含所有加成），另有 `pbonus_atk/hp`（已烘进去的玩家级
  加成）、`temp_atk/hp`（临时）、`attached`（磁力/赠礼）。
- **动作空间**：245 个离散动作 + 精确掩码；观测 3227 个浮点 + 124 个卡牌索引。
- **待决选择**：`Player::pending`（英雄、发现、三选一、目标、饰品、任务、时涡），有 pending 时其它动作非法。

第 3 条和第 5 条在后面反复出现——它们是"游戏状态能不能塞进模拟器"的核心约束。

---

## 2. 目标与设计

### 2.1 三个目标

1. 战斗中显示胜率；
2. 准备阶段给出建议（初期只做站位，后来才可能做购买）；
3. **用真实游戏验证模拟器**：每次操作后，把游戏状态和模拟器重建的状态逐字段比对。

第 3 条才是最有价值的：它能把 TavernGym 的 bug 变成一份可复现的清单。

### 2.2 client / server 分工

```
游戏进程内（Doorstop 注入的 C# 插件）        本机 Python
  GameState ──采样──► Snapshot ──HTTP──►  bridge.py ──► TavernGym
  截图 / dump（F9）                        /combat /align /advise
```

选择的理由：

- **C# 侧只做"读 + 显示"**，任何解析错误都不会让游戏崩；
- **Python 侧只做"算"**，迭代不用重启游戏；
- 协议是纯 JSON，`Snapshot` 结构在两边各有一份实现，服务器把不认识的字段忽略。

### 2.3 数据源的两次选择

**第一版读 Power.log**。理由是版本无关、可以离线回放、不会崩游戏。它确实跑通了（363,435 行日志、
12 局游戏、`/align` 0 diff）。

**后来换成进程内 `GameState`**，因为日志有三个硬伤：

1. 战棋客户端丢弃战斗复制体时**不发 ZONE 变更**，实体表只能靠"闲置多少行就淘汰"兜底；
2. 商店里旧商品永远不离开 PLAY，同一时刻能数出 23 个（真实是 5~7）；
3. 有读取延迟。

日志源保留下来做离线回放（`tavernreplay`），这是它不可替代的价值。

---

## 3. 开发过程

### 3.1 环境：两个不明显的坑

**Rust 工具链卡住**。`rustup` 停在 "syncing channel updates"，下载目录里躺着一个 0 字节的临时文件。
原因：系统走了 `127.0.0.1:7897` 的代理，而 **rustup/cargo 不读 Windows 的代理设置**，只认环境变量。

```powershell
$env:HTTP_PROXY="http://127.0.0.1:7897"; $env:HTTPS_PROXY="http://127.0.0.1:7897"
```

**PowerShell 默认编码**。这台机器的 `Get-Content`/`Set-Content` 按 ANSI（GBK）处理，用它改含中文的
UTF-8 文档会把文件写花（我踩过一次，整份 ALIGNMENT.md 变成乱码）。改文本一律带 `-Encoding UTF8`，
或者直接用编辑工具。

### 3.2 结构

```
tavernplugin/
  client/Core/     协议与纯逻辑（可被离线工具复用，C# 7.3，Unity Mono 兼容）
  client/Unity/    与游戏交互的一半：GameState 采样、dump、悬浮窗
  server/          stdlib HTTP + bridge.py（唯一摸 TavernGym 的地方）
  tools/ilspec/    反编译游戏程序集（引用 ILSpy 自带的 ICSharpCode.Decompiler）
  tools/replay/    离线回放：同一份 Core 代码跑日志
```

一个刻意的约束：`client/Core` 里**不允许出现 UnityEngine**，因为离线工具要用它。这条约束后来救了很多
时间——所有解析逻辑都能在没有游戏的情况下测。

### 3.3 离线优先

先写好 `tavernreplay`（读日志→快照→POST 服务器），再写游戏内的部分。好处是：解析器改一行，本地跑一次
就知道对不对，不用开游戏。整个项目里凡是"能在本机验证的"，最终都验证了。

---

## 4. 调试过程（这一段是重点）

### 4.1 第一次运行：插件静默失效

部署后进游戏，悬浮窗只显示初始值 `server: starting`。没有日志、没有报错。

线索藏在 `%LOCALAPPDATA%Low\Blizzard Entertainment\Hearthstone\Player-prev.log`：

```
MissingMethodException: Method not found: void System.Net.Http.HttpClient.set_Timeout(System.TimeSpan)
   at HSLibrary.TavernPluginModule.Start ()
```

`Start()` 第三行就炸了，后面全没执行。而 Unity 把这异常写进了一个**这个游戏留空的 Player.log**，
所以表面上什么都没有。

**教训一**：`Start()` 必须有第一行落盘打点，且打点本身不能依赖可疑 API。

### 4.2 发现：游戏的 mscorlib 被裁剪过

换成 `HttpWebRequest` 后再跑，又炸在 `File.AppendAllText(string,string,Encoding)`。于是不再逐个猜，
写了一个 `SelfCheck`：启动时把每个 API 单独 try/catch 跑一遍，结果写日志。

一次运行就拿到了完整矩阵：

| 能用 | 不能用 |
| :-- | :-- |
| `File.AppendAllText(2 参)`、`ReadAllText(1 参)`、`WriteAllText(2 参)` | 所有带 `Encoding` 参数的重载 |
| `FileStream` + `StreamWriter`、`Directory.*`、`Path.*` | `Convert.ToString(object)` |
| `Regex`、`Thread`、`TcpClient`、`HttpWebRequest` | `Process.Kill()`、`TcpClient.Close()`、`HttpClient` 的 `Timeout` |
| `JsonLite` / `JsonWriter` / `StringBuilder` | `HttpWebRequest.ReadWriteTimeout` |
| `string.Substring`、`int.Parse`、`double.ToString("R")` | `Enumerable.OrderByDescending(..., IComparer)` |
| P/Invoke（`CreateProcessW`、`Marshal.SizeOf`） | 带 `StringComparison` 的字符串重载 |

规律很清晰：**游戏自己从不调用的成员，被链接器裁掉了**。

**教训二**：不要假设 .NET 4.x 的成员在 Unity Mono 里都存在；对可疑调用做运行时探测，比读文档可靠。

### 4.3 更关键的教训：整方法 JIT 失败

一开始我以为"把可疑调用包在 try/catch 里"就能兜底。**错了**。

一个方法体里只要出现一个被裁剪的成员，**整个方法在 JIT 时就失败**，第一行都执行不到。所以
`try { request.ReadWriteTimeout = x; } catch { }` 完全无效。

正确写法是把每个可疑成员**单独放一个方法**：

```csharp
private static void SetTimeout(HttpWebRequest r) { r.Timeout = 30000; }   // 失败只影响这一行
...
Try(delegate { SetTimeout(request); });
```

这个模式后来成了整个客户端的基本纪律：`Text`（序数比较）、`GameLog`（文件写）、`NativeProcess`
（P/Invoke）、`GameStateSource`（每个 tag 读取）都是这么写的。

### 4.4 网络层：POST 不可用也要能跑

`ReadWriteTimeout` 缺失后，干脆把 HTTP 层做成两条路：

- 正常走 POST；
- POST 一旦抛异常（包括 JIT 失败），**自动降级成 `GET /combat?body=<URL 编码的 JSON>`**，服务端的
  `do_GET` 支持 `body` 查询参数。编码器自己写（`Uri.EscapeDataString` 也可能被裁）。

这样即使整个请求流不可用，功能仍然完整。

### 4.5 进程管理：不要留下孤儿 python

插件启动时用 P/Invoke `CreateProcessW` 拉起服务器（`Process` 不可信），带两个保险：

1. 游戏正常退出 → `TerminateProcess`；
2. 游戏崩溃/被强杀 → 服务器自己用 `OpenProcess` + `WaitForSingleObject` 盯着父进程，父死则退出。

实测证据（日志）：3 次启动、2 次正常停止、当前 1 个 python 进程，没有堆积。另外新实例发现端口被占用
就不启动，杜绝重复。

唯一漏洞是 Windows 的 PID 复用：极端情况下 watchdog 可能等错进程。彻底堵法是在 `/health` 返回
server 自己的 pid，让下一个游戏实例"接管"并在退出时清理。

---

## 5. 数据对齐分析

### 5.1 锚点是"当前值"

游戏 `GAME_TAG.ATK` 与 `HEALTH - DAMAGE`，对应 TavernGym `Minion.atk` / `hp`。实测验证：

```
AddMinion(atk=11) → view 里 atk=11, base_atk=3, breakdown.other=[8,6]
```

`base_atk` 是卡面值，`other` 只是解释不了的残差。**传当前值就是精确的**，这也是 `/align` 能做到 0 diff 的原因。

### 5.2 商店：六个否定，一个答案

这一块花了最久。先后否定了六条路：

| 假设 | 结果 |
| :-- | :-- |
| 实体表 `GetEntityMap()` 里有商店 | ❌ 只有位置 0 的 `TB_BaconShop_DragBuy` 槽位，`GetEntityDef()` 为 null |
| 商店在 `SETASIDE` | ❌ 那是我方阵容的镜像副本（金卡还是卡面原始数值） |
| 某个 `Zone` 里 | ❌ 遍历 52 个 zone，牌数正好等于棋盘/手牌 |
| `TB_BaconShop` 私有字段 | ❌ 3028 行里只有 UI/表情/协程 |
| `Card.GetActor().GetEntityDef()` | ❌ 槽位是纯 UI |
| `Card.GetController()` | ✅ 只给归属，不给内容 |

甚至一度以为"商店只能靠悬停"（HDT 的 `BigCardState` 做法）。**最后是用户一句"准备阶段对手棋盘就是
商店"点破的**——于是用 `HAS_DRAG_TO_BUY`（tag 2458）做判据，一次验证通过：

| dump | 带 tag 的实体 | "them" 区实体 | 一致 |
| :-- | --: | --: | :-- |
| turn 11 | 5 | 5 | ✓ |
| turn 11 | 5 | 5 | ✓ |
| turn 11（买了一张） | 4 | 4 | ✓ |
| turn 15 | 6 | 6 | ✓ |

**方法论教训**：当所有"代码里存了什么"的路径都走不通时，应该问"游戏里这个区域在*语义*上是什么"——
客户端的实现可能把商店塞进一个看起来无关的容器（对手的 PLAY 区）。

### 5.3 选择、赠礼、三连

拿到 `Network.EntityChoices` 之后，待决选择就完整了：

```json
"pending": {"active": true, "type": 2 (GENERAL), "count_min": 1, "count_max": 1,
            "source": 3692, "options": ["BG34_405","BG34_604","BG32_341"]}
```

黑暗之赐实测：花 3 金币（与 `dark_gift_cost: 3` 一致）、3 个选项（与 `dark_gift_choices: 3` 一致）、
选中后进手牌且是卡面值。赠礼卡本身也会在 GRAVEYARD 留下痕迹，被选中的那个以 `…e` 附魔挂到目标上。

三连那次是最漂亮的一次验证。观测：

```
合并前：BG36_366 46/30、45/27（场上两张）
合并后：BG36_366_G 91/57（进手牌），场上清零
```

TavernGym 的 `merge_golden` 公式是 `金卡基础 + Σ(part − 卡面 − pbonus − temp)`，代入
`BG36_366` 卡面 4/8 → 金卡基础 8/16：

```
攻击 = 8  + (46−4) + (45−4) = 91
生命 = 16 + (30−8) + (27−8) = 57
```

**完全命中**。附件也一致：金卡带 11 个 `attached`，是三个副本的并集，对应引擎里的
`extend_attachments_from`。

### 5.4 附件的一个陷阱

随从带 `attached` 时，数值**已经包含**附件效果。如果重建时先设最终数值、再 attach，效果会翻倍。
所以顺序是：

```
add_minion(卡面) → edit_minion(attach=赠礼) → edit_minion(atk, hp, keywords) 覆盖
```

而且不能什么都挂：`TB_BaconShopBadsongE`（英雄技能附魔）、`BG31_177`（数值附魔）这类只记笔记。
白名单只放两类——

- `BG36_MidGameEffect_*`（黑暗之赐，会持续触发）
- 文本**以 `Magnetic` 开头**的随从（磁力）

第二版才收紧：最初用"文本包含 Magnetic"，结果 `BG34_170`（"Battlecry: Get a random **Magnetic**
Volumizer"）被误判成磁力体。

---

## 6. 现在能做什么

| 能力 | 状态 |
| :-- | :-- |
| 自动拉起/回收 Python 服务器 | ✅ 无孤儿，崩溃也能收尾 |
| 战斗中显示胜率 | ✅ `/combat`，512 次模拟约 50ms |
| 准备阶段显示阵容状态 | ✅ 棋盘/手牌/金币/等级/商店/关键标签 |
| 逐步验证模拟器 | ✅ `/align` 在真实 7v7 棋盘上 `diffs: 0` |
| F9 诊断 | ✅ 原始探针 JSON + 真实快照 JSON + 截图 |

---

## 7. 可复用的方法

1. **先探针，后写逻辑**。`SelfCheck` 一次运行换来的兼容性矩阵，抵得上十次猜。
2. **差集实验**。判断"哪些实体是商店"不靠读代码，靠"刷新前后做差集"。
3. **先写判据，再验证**。判据写死在前面（`zone==PLAY && HAS_DRAG_TO_BUY`），避免事后拟合。
4. **数值验算优先于观感**。91/57 的命中比"看起来对"有说服力得多。
5. **把每次调试沉淀成工具**：`ilspec`（反编译游戏类型）、`tavernreplay`（离线回放）、
   `inspect_dumps.py` / `probe_stats.py`（数据探针）。其中商店的探索脚本和端到端自检后来被删掉了
   ——结论固化进代码和文档之后，它们只剩维护成本。
6. **离线优先**：能在本机跑通的，绝不留到游戏里试。
7. **失败要留痕**：`Start()` 第一行打点、`GameLog` 用已验证可用的 API、worker 异常也写日志。

---

## 8. 附录

### 8.1 常用命令

```powershell
# 起服务（一般由插件自己拉起，手动起用于开发）
python tavernplugin\server\server.py

# 部署插件（需要管理员权限）
.\tavernplugin\deploy.ps1

# 离线回放
dotnet run -c Release --project tavernplugin\tools\replay -- snapshot <Power.log> --out snaps.jsonl --every 2000
dotnet run -c Release --project tavernplugin\tools\replay -- post snaps.jsonl --limit 3

# 反编译游戏类型
dotnet run -c Release --project tavernplugin\tools\ilspec -- members "<Managed>\Assembly-CSharp.dll" '^Player$'


```

### 8.2 关键 tag

| tag | id | 用途 |
| :-- | --: | :-- |
| `ATK` / `HEALTH` / `DAMAGE` | 47 / 45 / 44 | 当前数值 |
| `ARMOR` | 292 | 护甲 |
| `TECH_LEVEL` / `PLAYER_TECH_LEVEL` | 1440 / 1377 | 随从等级 / 酒馆等级 |
| `RESOURCES` | 26 | 金币 |
| `ZONE` / `ZONE_POSITION` / `CONTROLLER` | 49 / 263 / 50 | 位置 |
| `CARDTYPE` | 202 | 5=法术 6=附魔 42=酒馆法术 |
| `HAS_DRAG_TO_BUY` | 2458 | 商店物品 |
| `IS_BACON_POOL_MINION` | 1456 | 池内随从 |
| `BACON_IN_COMBAT_PHASE` | 1522 | 准备/战斗 |
| `TAUNT`/`DIVINE_SHIELD`/`POISONOUS`/`WINDFURY`/`MEGA_WINDFURY`/`REBORN`/`STEALTH`/`RUSH` | 190/194/363/189/1207/1085/191/791 | 关键词 |

### 8.3 当前限制

- 商店随从没有 `COST` tag（价格在槽位 actor 上），客户端按 3 处理；法术用真实值。
- 受伤随从无法完整还原（`set_stats` 会把 `max_hp` 设成 `hp`），所以 `/combat` 必须在战斗开始前采样。
- 英雄皮肤卡号（`TB_BaconShop_HERO_*_SKIN_*`）不在卡表里，`set_hero` 会失败并记 note。
- `CLEAVE` / `IMMUNE` / `MAGNETIC` 没有 tag，只能从卡表推。
- 日志源的商店计数会偏大（旧商品不离开 PLAY），只用于回放。


