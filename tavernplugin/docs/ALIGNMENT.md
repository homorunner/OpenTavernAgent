# 战场状态对齐：游戏内数据 → TavernGym

这份文档回答一个问题：**游戏里的一局酒馆战棋，怎么变成 TavernGym 里的一个 `Game`？**
结论来自实际代码与真机验证：游戏侧用 ILSpy 的反编译库（`tools/ilspec`）读
`Assembly-CSharp.dll`，模拟器侧读 `crates/taverngym/src/*.rs`，数据侧用 F9 dump（含截图）+ 真实对局日志。

## 0. 结论先行

1. **数据源是进程内的 `GameState`**，Power.log 只做离线回放和兜底。客户端写法见
   `client/Unity/GameStateSource.cs`；日志解析器 `client/Core/PowerLogParser.cs` 仍在，
   用来跑 `tavernreplay`。
2. **对齐锚点是"当前值"**：游戏 `GAME_TAG.ATK` / `HEALTH - DAMAGE`，对应 TavernGym
   `Minion.atk` / `hp`。`GetDefATK()` / `base_atk` 才是卡面值，不要混用。
3. **商店已经解决**：准备阶段酒馆内容就在**对手 player 实体的 PLAY 区**里，判据是
   `GAME_TAG.HAS_DRAG_TO_BUY`（2458）。详见第 7 节。
4. **运行时限制比 API 更麻烦**：游戏把 `mscorlib` 裁剪过，许多"本该存在"的成员在运行时抛
   `MissingMethodException`。见第 4 节第 10 条，这是本项目最容易踩的坑。

## 1. 两个数据源

| | Power.log | 进程内 `GameState`（现用） |
| :-- | :-- | :-- |
| 精度 | 需自维护实体表，丢事件就漂 | 就是游戏自己的状态 |
| 战斗复制体 | 结束时**不发 ZONE 变更**，只能按"闲置"推断 | 区域/位置精确 |
| 商店 | 旧商品不离开 PLAY，实测同一时刻会数出 23 个 | 精确 5~7 个 |
| 线程 | 文件 IO，可放工作线程 | **必须主线程** |
| 版本兼容 | 日志格式稳定 | 依赖 `Assembly-CSharp` 类型名（36.6.3 已核对） |
| 用途 | 离线回放、审计、无对局时兜底 | 实际运行 |

## 2. 游戏侧 API（已用 ILSpy 核对）

```csharp
GameState state = GameState.Get();                 // 未开局时为 null
Player me       = state.GetLocalSidePlayer();      // 或 GetFriendlySidePlayer()
Entity gameEnt  = state.GetGameEntity();           // 全局 tag
Player them     = state.GetFirstOpponentPlayer(me); // 准备阶段＝商店宿主，战斗阶段＝对手

foreach (Card card in me.GetBattlefieldZone().GetCards()) {   // 已按位置排序
    Entity e = card.GetEntity();
    e.GetCardId(); e.GetEntityId(); e.GetZone(); e.GetZonePosition();
    e.GetTag(GAME_TAG.ATK); e.GetHealth(); e.GetDamage(); e.HasTag(GAME_TAG.TAUNT);
}
```

`Entity` 的多数访问器在基类 **`EntityBase`** 上（`Entity : EntityBase`），ILSpy 才看得出来：

| 类型 | 关键成员 |
| :-- | :-- |
| `EntityBase` | `GetTag(GAME_TAG)`, `HasTag`, `GetCardId`, `GetEntityId`, `GetZone`, `GetZonePosition`, `GetControllerId`, `GetHealth`, `GetDamage`, `GetArmor` |
| `Entity` | `GetCurrentHealth`, `GetDefATK`, `GetDefHealth`, `GetAttackBonus`, `GetRealTimeAttack`, `GetAttachments`, `GetCard` |
| `Player` | `GetBattlefieldZone`, `GetHandZone`, `GetHero`, `GetPlayerId`, `GetNumAvailableResources`, `BaconFreeRerollLeft` |
| `Zone` | `GetCards()` → `List<Card>`, `GetCardAtIndex`, `GetCardCount`, `GetControllerId` |
| `GameState` | `Get`, `GetLocalSidePlayer`, `GetFriendlySidePlayer`, `GetFirstOpponentPlayer`, `GetEntity(int)`, `GetGameEntity` |
| `BigCard` | `GetCard()`, `Show(Card)` — 悬停大卡（HDT 用它读商店，我们现在不需要） |

## 3. 字段映射表

| TavernGym | 游戏侧 | 备注 |
| :-- | :-- | :-- |
| `card` | `GetCardId()` | 字符串卡号直接对应 `cards.json` 的 `id` |
| `atk` | `GetTag(GAME_TAG.ATK)` | 当前值（含所有加成） |
| `hp` | `GetHealth() - GetDamage()` | 当前生命 |
| `max_hp` | `GetHealth()` | 受伤随从无法完整还原，见坑 5 |
| `tier` | `GetTag(GAME_TAG.TECH_LEVEL)` | 随从等级 |
| `golden` | `GetCardId().EndsWith("_G")` | **没有 `GOLDEN` tag** |
| `TAUNT` / `DIVINE_SHIELD` / `POISONOUS` / `WINDFURY` / `MEGA_WINDFURY` / `REBORN` / `STEALTH` / `RUSH` | 同名 `GAME_TAG` | id 190 / 194 / 363 / 189 / 1207 / 1085 / 191 / 791 |
| `CLEAVE` | **没有 tag** | 只能从卡表推 |
| 手牌 | `player.GetHandZone().GetCards()` | 战棋手牌也是实体 |
| 金币 | `player.GetNumAvailableResources()` 或 `GAME_TAG.RESOURCES`(26) | |
| 英雄 `health` / `armor` | 英雄 `HEALTH`(45) / `ARMOR`(292) | |
| 酒馆等级 | 英雄 `PLAYER_TECH_LEVEL`(1377) | **不是**随从的 `TECH_LEVEL` |
| 阶段 | `BACON_IN_COMBAT_PHASE`(1522) | 0=准备 1=战斗 |
| **商店** | 对手 player 的 `GetBattlefieldZone()` 里 `HAS_DRAG_TO_BUY`(2458) 的那些 | 见第 7 节 |

## 4. 十个坑（按踩到的概率排序）

1. **战斗复制体会污染棋盘**（仅日志源）。每场战斗新建一套实体，打完直接丢弃、**不发 ZONE 变更**。
   进程内读取没有这个问题。
2. **`atk` 是当前值**。`AddMinion(atk: 11)` 后 view 里是 `atk=11, base_atk=3, other=[8,6]`——
   `base_atk` 是卡面值，`other` 只是解释不了的残差。传当前值。
3. **金卡是独立卡号**，`GAME_TAG` 里没有 `GOLDEN`。
4. **`CLEAVE` / `IMMUNE` / `MAGNETIC` 没有 tag**，只能从卡表或 `GetAttachments()` 推。
5. **受伤随从无法完整还原**：`set_stats(atk, hp)` 会把 `max_hp` 一起设成 `hp`。
   所以 `/combat` 必须在战斗开始前（双方满血）采样。
6. **`pbonus_atk/hp` 不会被 `set_stats` 清掉**：重建用的临时 `Game` 若带玩家级加成，后续
   `apply_bonuses_to` 会再加一遍。临时局要保证统计为 0。
7. **必须主线程读** `GameState`/`Entity`。`TavernPluginModule` 在 `Update()` 采样，工作线程只发请求。
8. **`GameState.Get()` 开局前为 null**，`GetGameEntity()` 也可能空，全部要判空。
9. **`Venomous` 与 `Poisonous` 在游戏侧是同一个 tag**，按 `ATK == 0` 判 venomous 属于约定。
10. **游戏裁剪过 `mscorlib`**——运行时实测缺失的成员（全部抛 `MissingMethodException`）：

    ```
    File.AppendAllText(string,string,Encoding)     File.ReadAllText(string,Encoding)
    File.WriteAllText(string,string,Encoding)      Convert.ToString(object)
    Process.Kill()                                 TcpClient.Close()
    HttpClient.set_Timeout(TimeSpan)               HttpWebRequest.set_ReadWriteTimeout(int)
    Enumerable.OrderByDescending(..., IComparer)   以及带 StringComparison 的字符串重载
    ```

    规律是"游戏自己从不调用的成员大概率被裁掉"。**更关键的是**：一个方法体里只要引用了一个
    被裁成员，**整个方法在 JIT 时就失败**，方法内的 `try/catch` 完全无效。所以每个可疑调用
    必须**单独成方法**，让失败局部化。`client/Core/Text.cs`（序数比较）、`GameLog.cs`
    （FileStream+StreamWriter）、`NativeProcess.cs`（P/Invoke）、`BridgeClient.cs`
    （逐成员隔离）都是为此而写。运行时自检见 `client/Core/SelfCheck.cs`。

## 5. 三条链路

| 链路 | 采样时机 | 数据来源 |
| :-- | :-- | :-- |
| `/combat` 胜率 | 进入战斗、第一次攻击前 | 双方棋盘（此时对面才是真对手） |
| `/align` 校验 | 任意准备阶段 | 我的棋盘 + 商店 |
| `/advise` 建议 | 准备阶段 | 我的棋盘 + 手牌 + 金币 + 商店 |

准备阶段 `them` 区是**商店**，战斗阶段才是对手——所以 `/advise` 若要算购买收益，需要一个
"对手参考阵容"（上一轮侦察到的），这是当前 `/advise` 只做站位建议的原因。

## 6. 迁移计划（已完成）

1. ~~`IGameStateSource` 抽象~~ → 直接实现了 `GameStateSource.Capture()`，保留日志源做回放。
2. ~~采样移到主线程~~ → `Update()` 每 250ms 采样一次，`_sourceIsGame` 标记优先级。
3. 保留 `PowerLogParser`，`tavernreplay` 仍用它做离线验证。

## 7. 商店

准备阶段，客户端把酒馆内容放在**对手那个 `Player` 实体的 PLAY 区**里，并且给每个商品打上
`GAME_TAG.HAS_DRAG_TO_BUY`（2458）。判据只有这一条：

```
商店物品 = Zone == PLAY && HasTag(HAS_DRAG_TO_BUY)
```

每个物品都能直接读到卡号 / 攻防 / 等级 / 价格（法术的 `COST` 是真实值，随从没有 `COST` tag，
固定 3 费）。战斗阶段这个区换成了真正的对手阵容，所以读之前先看 `BACON_IN_COMBAT_PHASE`。

实测对照（同一局，中间买过一张）：带该 tag 的实体数量与"对手区"的实体数量逐帧一致，
买了之后少一个。

日志源不要用来数商店：旧商品不离开 PLAY，同一时刻会数出 23 个（真实 5~7），所以这条只在进程内成立。

## 8. 已验证

- `/align` 在真实对局的 3v7 棋盘上：`{"aligned": true, "diffs": 0}`。
- `AddMinion` 数值语义用 `tools/probe_stats.py` 实测：`atk=11` 进入后 `atk=11, base_atk=3`。
- 商店判据用四份 dump 交叉验证（第 7 节表格）。
- 运行时 API 兼容矩阵用 `SelfCheck` 在游戏内跑出（第 4 节第 10 条）。
- `GAME_TAG` 枚举 1217 项导出在 `tools/ilspec/out/GAME_TAG.cs`。

## 9. 工具

```powershell
# 列出名字匹配的类型
dotnet run -c Release --project tavernplugin\tools\ilspec -- list `
  "C:\Program Files (x86)\Hearthstone\Hearthstone_Data\Managed\Assembly-CSharp.dll" 'Bacon'

# 看成员 / 反编译整个类型
dotnet run -c Release --project tavernplugin\tools\ilspec -- members <asm> '^Player$'
dotnet run -c Release --project tavernplugin\tools\ilspec -- type <asm> GAME_TAG

# 汇总 F9 dump（状态 + 截图 + 待决选择）
python tavernplugin\tools\inspect_dumps.py
```

`tools/ilspec` 直接引用 ILSpy 自带的 `ICSharpCode.Decompiler.dll`，不需要 `ilspycmd`，也不联网。

**注意**：本机 PowerShell 的 `Get-Content`/`Set-Content` 默认按 ANSI（GBK）处理，
改这些含中文的文件时务必带 `-Encoding UTF8`，否则会把 UTF-8 文本写坏。



