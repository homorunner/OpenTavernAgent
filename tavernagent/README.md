# TavernAgent

这是基于论文结构自行实现的 NumPy/CPU 训练框架，包含策略/价值网络、PUCT 搜索、自博弈和评测。
当前模型为 **627,510 参数、策略头加一个价值头**；权重格式为v2，搜索协议为`sampled-prep-v2`。

## 使用示例

从项目根目录执行；环境依赖 NumPy 和本仓库编译的 `taverngym`，虚拟环境保留在 `.venv`。

```bash
source .venv/bin/activate
export PYTHONPATH=python:tavernagent
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# 训练100步
python -m tavernagent.selfplay --steps 100 --workers 32 \
  --persistent-workers --save-trainer-state --save-every 50 --out runs/agent/model.npz

# 从中间状态续训到200步
python -m tavernagent.selfplay --resume-state runs/agent/model.trainer.npz --steps 200

# 同批游戏种子下，与七个scripted bot评测
python -m tavernagent.eval --model runs/agent/model.ema.npz --games 1024 \
  --sims 64 --c-puct 1.5 --horizon turn --workers 32 --out runs/agent/eval.json

# 查看开局行为、记录完整动作轨迹、测量推理延迟
python -m tavernagent.probe --model runs/agent/model.npz --games 32 --sims 64
python -m tavernagent.patterns --model runs/agent/model.npz --games 64 --sims 64 --out runs/agent/patterns.json
python -m tavernagent.bench
```

## 网络与观测

`1715维数值 + 69个槽位 × 8维共享卡牌embedding → 192维投影 → 3个残差块 → 245维策略 / 1维tanh价值`。
残差块为 `h ← h + GELU(LN(Wh))`；策略输出按合法动作掩码归一化。

- 环境API保留3237维数值和124个卡牌索引；网络入口裁剪1519维对手历史阵容属性、49个历史阵容卡牌槽位和三个冗余全局标记。
- 对手英雄和公开摘要仍输入网络；摘要包含`last_seen_turn / 20`，0表示未见过，观察到空场也记录回合。引擎继续维护历史阵容，但不将其实体输入网络。
- 随从、法术、英雄、饰品、英雄技能、额外技能、任务及任务奖励共享一张embedding表，索引为`CardId + 1`，0为空槽，普通/金色卡沿用各自ID。
- 使用己方战场、手牌、商店、选择来源与选项、饰品、任务/奖励及对手英雄等69个槽位；奥秘和Deity专用身份槽位未接入。
- 攻击、当前生命、最大生命及Deity身材按`log1p(max(v,0)) / log(11)`缩放；英雄生命除以`log(41)`。
- 其他值线性缩放：等级/7、金币/10、护甲/20、回合/20、费用/5、场上数量/7、手牌数量/10。尺度是典型量级，不截断合法大值。
- 商店物品逐槽编码冻结状态，不使用全局冻结标记；网络也不使用选英雄阶段和终局标记。
- 最后三维为剩余总动作/站位/冻结预算，分别除以100/10/5；0表示耗尽。

英雄由`heroes.select_heroes(game, seed)`在网络外随机选择，不搜索、不生成训练样本；选完后的英雄身份仍输入网络。
发现、饰品及目标选择由模型决策。实战基于胜率数据的英雄选择尚未接入。
动态卡牌计数、附加卡牌身份和持续效果仍未完整编码，相关TODO保留在`obs.rs`。
反向传播为手写实现，重复卡牌embedding梯度通过scatter-add累加，空槽梯度为零；数值梯度检查在`tests/test_agent.py`。

## 搜索与动作预算

默认`horizon=turn`严格止于己方当前准备阶段。选中EndTurn时直接使用该节点价值，**不执行EndTurn、不触发回合结束效果、不模拟其他玩家或战斗**。
每条模拟使用独立随机流；同一动作的不同可见结果分别建节点，动作边汇总访问次数与价值。真实对局及其随机流不变。
`Game.copy()`保留精确复制语义；搜索通过`Game.copy(seed=...)`替换副本未来随机流。

- PUCT使用全部合法动作，未访问动作Q=0；真实决策取访问次数最多的动作，并列保留原`argmax`规则。
- 训练根节点混合Dirichlet噪声，默认比例0.20、alpha=0.3；评测不加根噪声。
- 每玩家每回合最多100个总动作、10次Move、5次Freeze/Unfreeze，均由引擎校验并在下回合重置。
- CLI的`--action-cap`、`--move-cap`、`--freeze-cap`均须为正整数，始终启用对应限制。
- `horizon=game`仅保留为完整状态实验模式，会使用隐藏状态；不用于当前信息受限能力结论。

## 训练默认值与续训

默认采用已完成候选中表现最好的一组；这不意味着各项参数已经分别验证最优。

| 参数 | 默认值 |
| --- | --- |
| 训练步数 / 每步局数 | 200 / 128 |
| 搜索次数 / c_puct | 64 / 1.5 |
| epochs / batch / 保留样本上限 | 1 / 128 / 20,000 |
| LR / 最小LR / 余弦周期 | 3e-4 / 3e-5 / 固定200步 |
| 根噪声比例 / alpha | 0.20 / 0.3 |
| 前5步温度 / 后续温度 | 1.0 / 0.5 |
| 激活 / 残差幅度 / head norm | GELU / 1.0 / 关闭 |
| value_weight / policy smoothing / pg_weight | 0.5 / 0.05 / 0 |
| weight decay / 梯度裁剪 / EMA衰减 | 0.01 / 5.0 / 0.99 |
| replay_rounds / value_samples | 1 / all |
| workers / seed | 10 / 0，按运行需要指定 |

八个席位全部由当前网络自博弈。样本记录观测、合法动作、搜索访问分布、选中动作及真实最终名次；奖励为`1 - 2 × (名次 - 1) / 7`。
损失为访问分布的策略交叉熵加价值MSE，使用AdamW和全局梯度裁剪；EMA每个SGD minibatch更新一次。
`--pg-weight`可启用带价值基线的策略梯度，默认关闭。
`--activation gelu/silu/relu`、`--residual-scale`及无可训练参数的`--head-norm`用于结构对照，配置随权重保存。

`--value-samples end`只用真实选择EndTurn的状态训练**现有唯一价值头**，策略仍使用全部样本，标签仍为最终名次；默认`all`使用全部决策状态。
end模式的价值MSE和梯度按批内结束标签数归一化；无标签时为零，AdamW动量和权重衰减仍正常处理。两种模式的value_loss不宜直接比较。
监督方式all/end随完整训练状态保存。

- `--steps`是累计目标步数；默认余弦周期固定200步，修改总步数不会自动改变周期，可通过`--lr-decay-steps`指定。
- `--save-trainer-state`保存网络、AdamW动量、EMA、训练与采样RNG、历史和配置；`--save-every`额外保留不可变阶段检查点。
- `--resume-state`精确恢复并默认继承保存配置，显式参数可覆盖；`--resume`只加载权重，优化器与步数重新开始。
- 续训沿用检查点的激活、残差和归一化配置，结构冲突会报错。完整续训目前仅支持`--replay-rounds 1`。
- `--persistent-workers`复用worker；worker数量会影响分块采样。原实验使用`--workers 64 --seed 10000000000 --persistent-workers --save-trainer-state --save-every 50 --probe-every 0`。
- 历史指标包含各头梯度、裁剪比例、策略熵、目标KL、价值统计及CPU/墙钟耗时；EMA衰减按minibatch计，缺失的历史计时不能补零。

## 实验结论

### 探索系数与搜索次数

下表为200步、每步128局后的自身推理参数评测，1024个相同游戏种子；每格取raw/EMA中较低均值，属于调参结果。

| 训练c | 64次搜索 | 128次搜索 |
| --- | ---: | ---: |
| 0.25 | 6.008 | 6.628 |
| 0.50 | 5.852 | 6.056 |
| 0.75 | 5.697 | 5.428 |
| 1.50 | 4.320 | 尚未完成 |

## 当前问题与复盘起点

- `all`监督拟合继续执行策略的最终回报，EndTurn边却将同一标量用于停止估值；两种含义是否冲突需要厘清，end的早期改善还不能证明已解决。
- 分开评价策略、价值估计和搜索增益；八席位弱模型自博弈是否提供足够学习信号。
- 升级、凑三连、回合结束触发及战斗收益依赖价值估计；动态效果输入不完整，也尚未实现对对手历史局面的输入。
- 回放窗口可配置，但当前没有使用；当前默认保持完全on-policy。

## 代码与检查

| 模块 | 用途 |
| --- | --- |
| `tavernagent/net.py` | 网络、embedding、手写反向传播及权重存取 |
| `tavernagent/mcts.py` | PUCT、独立随机流及随机结果分支 |
| `tavernagent/selfplay.py` | 自博弈采样、优化器与完整状态续训 |
| `tavernagent/eval.py` / `arena.py` | 对bot配对评测 / 模型对战 |
| `tavernagent/probe.py` / `patterns.py` | 开局摘要 / 完整行为轨迹 |
| `tavernagent/bench.py` / `decision_bench.py` | 网络 / 完整决策延迟 |

在项目根目录执行`.venv/bin/python -m pytest -q`；测试覆盖网络梯度、输入编码、搜索信息边界及完整状态续训。
