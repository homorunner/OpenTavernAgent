# TavernAgent v2

`crates/tavernagent_v2` 是独立的 Rust **PPO-Clip + GAE** 训练项目。模型结构、观测和紧凑动作接口参考 v1；训练改为八座位同步 on-policy 自博弈。

2026-11-11 完成的超参数对照实验支持以下基线：**lr=1e-4、λ=0.95、entropy=0.03、epochs=1、games=64、batch=1024、workers=16**，已设为默认值。五个从头训练 2000 轮的模型，在新的保留游戏种子上，对七个 scripted / teacher 的平均排名分别为 **1.2267 / 1.3126**。

本文是 v2 的统一说明，包含实现、使用方法和实验结论。连续 Freeze 规则仍在对照验证中，默认关闭。训练产物不随源码分发。

- [快速使用](#快速使用)
- [CUDA 后端](#cuda-后端)
- [模型与游戏接口](#模型与游戏接口)
- [训练与恢复](#训练与恢复)
- [评估协议](#评估协议)
- [Checkpoint 联赛与 Elo](#checkpoint-联赛与-elo)
- [定期通知](#定期通知)
- [吞吐与并发](#吞吐与并发)
- [训练实验结果](#训练实验结果)
- [开局轨迹与已知问题](#开局轨迹与已知问题)
- [代码与工具](#代码与工具)

## 快速使用

以下命令从仓库根目录执行。`--iterations` 表示累计训练轮数，每轮采集 `--games` 个完整八人游戏。

```bash
cargo build --release -p tavernagent_v2

# 默认使用实验推荐参数，从头训练到 2000 轮
target/release/tavernagent_v2 train \
  --iterations 2000 \
  --seed 400000000 --save-every 50 --out runs/tavernagent_v2/model.npz

# 继承保存的超参数、优化器和 RNG，继续到累计 3000 轮
target/release/tavernagent_v2 resume \
  --checkpoint runs/tavernagent_v2/model.npz \
  --iterations 3000 --workers 16 --save-every 50

# 两种固定对手分别评估；默认 greedy，加入 --sample 改为概率采样
target/release/tavernagent_v2 eval \
  --model runs/tavernagent_v2/model.npz --opponent scripted \
  --games 1024 --workers 16 --seed 10000000 --out runs/tavernagent_v2/scripted.json
target/release/tavernagent_v2 eval \
  --model runs/tavernagent_v2/model.npz --opponent teacher \
  --games 1024 --workers 16 --seed 10000000 --out runs/tavernagent_v2/teacher.json
```

`train --iterations 0` 只保存初始化模型、零 Adam 状态和初始 RNG，可用于未训练基线并继续训练。完整选项见各子命令的 `--help`。

| 参数 | 默认值（推荐基线） |
| --- | ---: |
| games | 64 |
| batch | 1024 |
| workers | 16 |
| lr | 1e-4 |
| gae-lambda | 0.95 |
| entropy-coef | 0.03 |
| epochs | 1 |
| gamma | 1 |
| clip-range | 0.2 |
| target-kl | 0.02 |
| value-coef | 0.5 |
| max-grad-norm | 0.5 |
| weight-decay | 0.01 |

`games` 为训练每轮完整游戏数；`train`、`resume`、`eval` 均默认 16 workers。新默认值用于新训练，恢复已有 checkpoint 仍使用其保存的训练超参数。

短流程检查可以使用小模型和短回合预算；它不衡量策略强度：

```bash
target/release/tavernagent_v2 train \
  --iterations 2 --games 2 --workers 2 --hidden 16 --blocks 1 --embedding-dim 2 \
  --max-turns 3 --action-cap 8 --epochs 1 --batch 64 \
  --out runs/tavernagent_v2-smoke/model.npz
```

## CUDA 后端

默认仍使用 CPU。启用 `cuda` feature 后，同一个 Rust 可执行文件支持 CPU 和 CUDA 两条路径，无需 Python、PyTorch 或运行时编译器。矩阵乘使用 cuBLAS 的 pedantic FP32 模式；LayerNorm、激活、残差、embedding 梯度、梯度范数和 AdamW 使用嵌入二进制的 PTX。不开启 BF16、FP16 或 TF32。

当前 CUDA 后端面向 Linux 和 compute capability ≥ 8.0 的 GPU。运行依赖能加载 CUDA 12.9 / PTX 8.8 的 NVIDIA 驱动（`libcuda.so.1`），以及支持目标 GPU 的 `libcublas.so.12` 及其依赖库。因此“单个可执行文件”指应用部署形式，并不表示静态包含 NVIDIA 驱动和 cuBLAS。构建普通 Rust 二进制不需要 CUDA SDK；仅重新生成 PTX 时需要 `nvcc`。

```bash
cargo build --release -p tavernagent_v2 --features cuda

# 对目标 GPU 验证前向、梯度、AdamW、状态恢复和 KL 停止
target/release/tavernagent_v2 cuda-check --device 0 --hidden 256 --blocks 6 --batch 64

# 与 CPU 基线相同的训练超参数，只改变执行后端和网络宽度
target/release/tavernagent_v2 train --backend cuda --device 0 \
  --hidden 512 --blocks 6 --workers 4 --iterations 500 \
  --seed 100000000 --save-every 100 --out runs/cuda-wide/model.npz

# 恢复默认继承 checkpoint 中的 backend；device 是本次进程的选择
target/release/tavernagent_v2 resume --checkpoint runs/cuda-wide/model.npz \
  --device 0 --workers 4 --iterations 2000

target/release/tavernagent_v2 eval --model runs/cuda-wide/model.npz \
  --backend cuda --device 0 --workers 4 --inference-batch 64 \
  --opponent teacher --games 1024 --out runs/cuda-wide/teacher.json
```

`arena` 也接受 `--backend cuda --device N --inference-batch N`。评估默认使用 CPU，必须显式选择 CUDA。训练将后端写入 checkpoint；恢复时可显式传入 `--backend` 切换，但跨后端不保证逐位相同。CPU/CUDA 的模型、观测、动作和 checkpoint 张量格式一致，旧 checkpoint 缺省后端为 CPU。

模拟器、GAE、动作采样及 PPO 输出端的 FP64 概率/损失计算仍在 CPU；网络前向、反向、梯度裁剪所需范数和 AdamW 在 GPU。每轮结束把参数和优化器状态同步回标准 checkpoint。CUDA 采样仅在独立游戏之间合批，同一游戏内仍按原座位顺序观察、执行动作，避免使用其他座位行动前的过期观测。评估分批维护游戏，限制主机内存占用。

GPU 显存不能替代主机上的完整轨迹缓冲。先测单任务的主机内存峰值和实际每轮耗时，再决定并行任务数；本实现首先面向单卡训练独立模型，不做多卡 DDP。设备内存池复用中间缓冲，embedding 梯度按固定出现顺序归并，不使用不确定顺序的浮点 atomic add。

跨 CPU 架构部署时，应为目标架构交叉编译并检查系统 ABI，不能直接复制另一架构的可执行文件。模型输入输出相同不意味着跨平台浮点舍入完全一致；精确恢复的验证范围是相同后端、构建和运行环境。

内置核源文件为 [kernels.cu](../crates/tavernagent_v2/src/cuda/kernels.cu)，其 PTX 随源码提供。修改后用 CUDA 编译器重新生成，再构建 Rust 程序：

```bash
nvcc -ptx -arch=compute_80 --fmad=false --ftz=false \
  --prec-div=true --prec-sqrt=true -O3 \
  crates/tavernagent_v2/src/cuda/kernels.cu \
  -o crates/tavernagent_v2/src/cuda/kernels.ptx
```

## 模型与游戏接口

默认模型有 **2,317,685 个参数**，hidden=256、6 个 pre-norm 残差块、共享卡牌 embedding=32。输入、动作和 FP32 输出通过与 v1 的测试校验。

```text
1710 个数值特征 + 22 个回合/星级 one-hot + 57 × 32 卡牌 embedding + 116 个动作 mask
  → Linear(3672, 256) → LayerNorm → GELU
  → 6 个 pre-norm 残差块，FFN 256 → 384 → 256
  → policy：Linear(256, 116) → masked categorical
  → value：Linear(256, 128) → GELU → Linear(128, 1) → tanh
```

宽度、深度、embedding、激活函数、残差缩放和 head normalization 可配置。v2 没有 BF16 推理实现。

### 观测

原始观测为 **3236 个浮点数、124 个卡牌索引、116 位动作 mask**。浮点数包含引擎的 3235 个特征及额外的伤害上限保护位：存活且当前存在战斗伤害上限、生命值加护甲严格大于上限时为 1；使用招募阶段开始时的存活人数判断伤害上限资格。

网络沿用 v1 的特征筛选，保留 1710 个数值特征、57 个卡牌槽，再加入回合与星级 one-hot 和动作 mask。回合 one-hot 共 16 位，T16 及以后共用最后一位；星级共 6 位，6/7 本共用最后一位。原回合、星级标量仍保留。

`observe_seat` 按获得来源映射饰品：普通饰品进入普通槽；英雄技能、任务奖励来源映射到对应槽；Orb 壳本身不占槽，其产生的饰品按来源表示。该映射只改变输入表示，不修改引擎效果，待处理的发现选项也保持原样。

### 动作与训练约束

- **去掉 Move**。0–45 与引擎动作索引一致；46–115 为 `46 + hand * 7 + position`。
- **Play / Magnetize 合并**。在当前状态下，目标为兼容的已占用位置时执行 Magnetize，否则执行普通 Play；法术使用位置 0。合法性检查与执行使用相同映射，引擎原动作 API 不变。
- **金币 ≥ 3 时屏蔽 EndTurn；通常屏蔽 Cancel**。若移除某动作会使合法集合为空，则保留逃生动作。
- 默认每座位每回合总动作上限 100、Freeze 按键上限 5；预算配置仍保留 Move 上限 10，但模型动作空间不提供 Move。
- 可选 `--block-consecutive-freeze`：同一座位在同一回合上次成功动作是 Freeze 时，下一次屏蔽 Freeze；其他成功动作或新回合恢复。其他座位行动不解除限制。该开关默认关闭，作为对照变量写入 checkpoint，训练、固定 bot 评估、混桌和开局采集统一读取。mask 仍保留 EndTurn/Cancel 的唯一动作逃生规则。
- 选英雄使用独立、带种子的均匀策略，不由网络决定，也不产生 PPO 样本。

训练与评估中的模型使用同一 mask。固定 bot 使用引擎动作和自身规则，不套用模型的训练约束。

## 训练与恢复

### 采样与 GAE

每轮固定当前参数，并行采集完整的八座位自博弈游戏，然后执行 PPO 更新。所有实际招募决策按温度 1 的 masked categorical 采样，记录观测、当时的 mask、动作、行为 log probability 和 value。环境 RNG 与动作 RNG 分离。

每个座位有独立的决策序列，EndTurn 和等待战斗不截断序列。座位淘汰或获胜后，其最后一个动作获得终局奖励：

```text
reward = 1 - 2 * (placement - 1) / 7
delta_t = reward_t + gamma * value_(t+1) - value_t
advantage_t = delta_t + gamma * lambda * advantage_(t+1)
return_t = advantage_t + value_t
```

其余奖励为 0，终局 value 为 0，GAE 按座位分别计算；gamma 是同一座位每次决策的折扣。默认 gamma=1 保留最终排名目标。GAE helper 支持未结束序列的 bootstrap，但当前采集器只输出完整 episode。

该流程不使用 MCTS/PUCT、假想结束状态、分支续跑、teacher 模仿、EMA actor 或跨轮 replay。Actor advantage 在整轮新样本上统一标准化；critic 的 lambda-return 不做同样处理。每个 epoch 重新打乱当前样本，更新完成后丢弃整轮数据。

### PPO 更新

```text
ratio = exp(new_log_prob - old_log_prob)
policy_loss = -mean(min(ratio * advantage,
                       clamp(ratio, 1-clip_range, 1+clip_range) * advantage))
loss = policy_loss + value_coef * mean((value - return)^2) - entropy_coef * entropy
approx_kl = mean(exp(log_ratio) - 1 - log_ratio)
```

使用 AdamW 和全局梯度范数裁剪。Value 输出为 tanh，但带噪声的 GAE target 不裁到 [-1, 1]。

`target-kl=0.02` 在某 minibatch 超阈值时，**先停止，再决定不执行该批更新**，并结束剩余 epochs；设为 0 可禁用检查。因此配置 epochs=3 不代表实际使用了三轮完整数据。`clip_fraction` 是概率比超出 PPO 区间的比例，与梯度裁剪不同。

模拟错误或 driver stop 使采集失败；引擎达到配置的游戏回合上限时会判定名次，该名次作为终局奖励，并记为 `turn_limit_games`。战斗截断另记 `truncated_combats`。修改 `--max-turns` 会改变学习任务。

### Checkpoint 与可复现性

每轮结束原子替换 `--out` 指定的 `.npz`，保存权重、AdamW moments/step、shuffle RNG、累计游戏种子位置、超参数、schema/card 元信息和 history。它同时用于推理和轮边界精确恢复。`--save-every N` 额外保留 `model.iterN.npz`，`model.history.json` 提供可读历史。

`resume` 继承保存的全部训练超参数，允许改变累计目标轮数、workers、输出路径和归档频率；另可显式选择设备和切换执行后端。默认覆盖传入的 checkpoint；从归档恢复且想保留原归档时应指定新的 `--out`。不保存中途游戏状态，不支持 v1 权重或 trainer snapshot。

规则对照使用独立的 `fork` 子命令：保持网络、Adam、RNG 和训练计数，显式改变连续 Freeze 开关并写入新路径。旧 checkpoint 没有该字段时按关闭处理；`fork` 拒绝覆盖已有路径，`resume` 本身不覆盖保存的开关。

```bash
target/release/tavernagent_v2 fork \
  --checkpoint runs/tavernagent_v2/model.iter2000.npz \
  --out runs/freeze-test/initial.npz --block-consecutive-freeze true
target/release/tavernagent_v2 resume \
  --checkpoint runs/freeze-test/initial.npz --iterations 10000 --workers 32 \
  --save-every 100 --out runs/freeze-test/model.npz
```

采集和梯度计算在不同阶段复用 Rayon 池。梯度固定按 32 行分块，逐元素归并顺序固定；归并与 AdamW 再按 4096 个参数划分任务。在相同平台和构建下，改变 workers 不改变游戏流、样本顺序和更新结果。

### 诊断指标

历史记录采样/更新耗时、transition 数、回合上限、损失、熵、KL、clip fraction、梯度范数和更新次数。额外区分：

| 指标 | 含义 |
| --- | --- |
| `trained_samples` / `evaluated_samples` | 实际参与更新的样本 / 已计算的样本，后者含被 KL 拒绝的批次 |
| `first_epoch_samples` / `effective_epochs` | 首轮实际覆盖 / 实际数据复用次数 |
| `rollout_seconds` / `prepare_seconds` | 完整游戏采样 / GAE 与样本整理 |
| `gradient_seconds` / `reduction_seconds` / `optimizer_seconds` | 前后向 / 梯度归并 / AdamW 等更新阶段 |
| `cleanup_seconds` / `checkpoint_seconds` / `iteration_seconds` | 缓冲清理 / 保存 / 完整迭代墙钟时间 |

保存完成后才知道的完整迭代、写文件耗时输出在 stdout JSON，不包含在该 snapshot 自己的 history 中。内存随完整游戏批次的决策总数增长，内存不足时优先降低 `--games`。

八个座位的合计平均名次恒为 4.5，不能用来衡量进步；自博弈 seat-0 名次也仅作诊断，强度使用固定对手评估和 checkpoint 联赛。

## 评估协议

固定 bot 评估使用模型在座位 0 对七个相同 bot 的平均排名，越低越好。早期学习主要观察这一指标，接近饱和后主要观察下文的 checkpoint 联赛 Elo。`scripted` 与 `teacher` 分开记录；JSON 还包含前四率、吃鸡率、平均排名标准误、近似正态 95% 区间及逐局排名。

Teacher 原样复制自 v1 的 deterministic no-composition teacher，保留升本曲线、对子/三连偏好及早期 Dark Gift 策略。Bot 决策保留环境 RNG，所有座位使用同一外部选英雄策略。主比较使用 greedy，`--sample` 作为单独诊断。

`arena` 使用四个候选模型对四个冻结参考模型，均为 greedy，各自读取 checkpoint 中的策略规则。每个环境种子跑两个互补座位布局，并轮换分组位置。`--games` 必须为正偶数，1024 局对应 512 个环境种子；标准误按种子配对后的均值计算，不把同一局的四个候选座位当成独立样本。候选组平均排名低于 4.5 表示相对参考组占优，团队吃鸡率的中性值为 50%。

```bash
target/release/tavernagent_v2 arena \
  --candidate runs/freeze-test/model.npz --reference runs/tavernagent_v2/model.iter2000.npz \
  --games 1024 --workers 16 --seed 20000000 --out runs/freeze-test/arena.json
```

### Checkpoint 联赛与 Elo

打固定 bot 的平均排名接近 1.2 后，继续保留这一指标作行为回归检查，训练阶段的相对强度主要看 **每 500 轮 checkpoint 的共享联赛**。`league` 复用 v1 的评估口径：每桌八个不同 checkpoint，随机分配座位，记录八席完整名次；[league_elo.py](../tools/league_elo.py) 用 Plackett–Luce 拟合强度，按 `400 × log10(strength)` 转成 Elo。它与 `arena` 的两模型 4v4 评估分别使用。

联赛使用 CPU `infer_f32` 和 greedy 动作，每席读取自己的 checkpoint 策略规则，因此可以同时比较不同宽深、训练种子及 Freeze 规则。所有模型的引擎规则必须一致。CPU/GPU 训练出的 checkpoint 统一使用同一评估后端；不使用 MCTS。模型池至少需要八个 checkpoint，同桌不重复模型身份。

固定一个已冻结的 checkpoint 为 **Elo 1000**，将所有训练曲线放在同一个连通对手池中；bootstrap 的每次重采样也固定同一锚点。Elo 是池内相对分数，加入新模型或新对局后旧模型分数可能变化。不能直接比较不同池、不同锚点的绝对 Elo。误差来自以完整游戏为单位的 bootstrap，不能代替多个训练种子的波动；仍应分别检查各训练种子的曲线。

```bash
# 同一训练轨迹的八个 checkpoint 混桌；输出文件必须尚不存在
models=()
for step in 500 1000 1500 2000 2500 3000 3500 4000; do
  models+=(--model "run_s${step}=runs/run/model.iter${step}.npz")
done
target/release/tavernagent_v2 league "${models[@]}" \
  --rounds 256 --workers 16 --seed 60000000 --out runs/league/games.jsonl
python3 tools/league_elo.py runs/league/games.jsonl --anchor run_s500 \
  --bootstrap 200 --json runs/league/ratings.json --out runs/league/elo.png
```

[ppo_league.py](../tools/ppo_league.py) 可持续发现多个目录的 `model.iterN.npz`，以及保存了非零整 500 轮状态的 `initial.npz`。默认每个 checkpoint 至少 256 局，每批 64 局；每桌四席优先补足出场较少的模型，另外四席随机抽取其余对手，使新旧 checkpoint 有交叉对局。座位随机化，对局数为下限，成熟 checkpoint 可能继续出场。

```bash
python3 tools/ppo_league.py --binary target/release/tavernagent_v2 \
  --run control=runs/control --run no_repeat=runs/no-repeat --run wider=runs/wider \
  --out runs/shared-league --anchor control_s2000 --interval 500 \
  --games-per-model 256 --target-iteration 10000 --workers 16 --watch
```

监控器只读取归档 checkpoint，不修改训练参数。它保存模型哈希、冻结桌次、独立环境种子、完整名次及 `ratings.json` / `ratings-meta.json` / `status.json`。不足八个模型时等待；新 checkpoint 自动加入并补足对局，全部指定训练轨迹到目标轮数且满足出场下限后退出。相同命令可以恢复中断任务，已完成批次只计入一次。Rust 命令以 `.partial` 暂存，完整批次原子发布；重试使用同一桌次和种子。不合法排名、重复记录或不连通的评分图会报错。达到回合上限或出现战斗截断的对局保留在原始批次及 `excluded-games.json` 中，不计入 Elo 和有效出场数；调度器继续补足有效对局，报告同时列出排除数量。

调度器需要 Linux、Python 3.11+，Elo 拟合需要 NumPy；绘图额外需要 Matplotlib，`league_elo.py --no-plot` 可以仅输出评分。Rust 联赛执行文件本身不需要这些依赖。评估使用独立资源预算，训练与联赛的吞吐应分别统计。

### 定期通知

[ppo_notify.py](../tools/ppo_notify.py) 向已配置的飞书会话发送纯图片卡片，固定依次放置 **scripted bot 平均排名、teacher bot 平均排名、arena Elo** 三张图，不包含卡片标题或文字汇报。默认每 30 分钟发送一次，每分钟检查状态；失败、停止、完成或故障恢复也触发同样的三图通知。读取失败连续三次、运行状态超过 15 分钟未更新时触发通知。全部指定任务完成且最后一张卡片发送成功后退出。

图中按训练种子绘制 CPU 两组与 GPU 模型的历史曲线。Bot 图同时显示完整历史和 500 轮后的排名局部放大，误差条为游戏采样的近似 95% 区间；局部图只显示排名不超过 2.5 的区域，异常高排名仍完整保留在全局图中。Elo 图使用已完成的评分文件，显示对局数及整局 bootstrap 的 95% 区间，不重新拟合 Elo。三张图必须全部上传成功才发送卡片；失败时保留待发消息和 UUID 重试，不发送文字替代或不完整卡片。

运行位置、SSH 参数、凭据文件路径等放在不提交版本控制的私有 JSON 配置中。配置包含 `out`、`credentials_env`、`interval_seconds`、`poll_seconds` 和 `sources`；`sources` 的 `cpu` / `gpu` / `league` 各指定 `ssh` 参数数组及 `files` 映射。训练来源需要 `status`，联赛还需要 `ratings` 和 `ratings_meta`。训练来源另配置 `eval_runs`，每项指定 `name`、`training_seed`、`directories` 和可选 `evaluation_seed`（默认 10000000）；多个目录中的同一 checkpoint 评估去重，保留游戏数较多的结果，过滤 sampled 或其他评估种子的结果。凭据文件使用 `FEISHU_APP_ID`、`FEISHU_APP_SECRET`、`FEISHU_HOME_CHANNEL` 和可选 `FEISHU_DOMAIN`；凭据不写入通知或日志。

```bash
python3 tools/ppo_notify.py --config runs/notifications/config.json --dry-run
python3 tools/ppo_notify.py --config runs/notifications/config.json --once
python3 tools/ppo_notify.py --config runs/notifications/config.json
```

`--dry-run` 生成三张预览图，不发送；`--once` 检查并发送当前到期通知；不带这些选项时持续运行。部署时可由用户级服务管理进程。通知需要 Python 和 Matplotlib。

固定 bot 和 4v4 混桌结果还记录决策数、Freeze 数、连续 Freeze 数、逐回合最后决策时的星级/金币，以及首次 Upgrade 回合直方图。训练 stdout 的 `actions` 汇总全部八个自博弈座位。

### 未训练基线

2026-11-11 对默认模型的三个初始化分别评估两类对手，每项 1024 局，游戏种子为 10000000..10001023。模型来自 `train --iterations 0`，没有采样或更新。

| 初始化种子 | scripted | teacher |
| ---: | ---: | ---: |
| 100000000 | 7.9658 | 7.9277 |
| 200000000 | 7.9688 | 7.9482 |
| 300000000 | 7.9111 | 7.9121 |
| 均值 | 7.9486 | 7.9294 |

6144 局的前四率和吃鸡率均为 0。此结果用于训练前后对照，不能据两个均值判断 bot 相对强弱。

### 正式实验设计

每轮采集 64 个完整游戏，batch=1024、workers=16，使用默认模型及动作预算。gamma=1、clip=0.2、target KL=0.02、value coefficient=0.5、grad norm=0.5、weight decay=0.01。各实验均从自己的初始化模型开始。

| 阶段 | 配置 | 每配置训练种子数 | 每任务轮数 | 任务数 |
| --- | --- | ---: | ---: | ---: |
| 2 | lr={3e-5,1e-4,3e-4} × λ={0.95,0.99,1}，entropy=0.01、epochs=3 | 3 | 500 | 27 |
| 3 | 最佳 lr/λ，entropy={0.003,0.01,0.03} × epochs={1,3} | 3 | 500 | 18 |
| 4 | 前两阶段最好的两个不同配置 | 5 个新种子 | 2000 | 10 |
| λ 补充 | 两个入选配方改用 λ=0.90 | 3 | 500 | 6 |

阶段 2/3 和 λ 补充使用训练种子 100000000、200000000、300000000；阶段 4 使用 400000000 至 800000000，间隔 100000000。

初始模型及每 50 轮 checkpoint 均对每类 bot 评估 1024 局，固定验证种子为 10000000..10001023。阶段 2/3 按 400/450/500 轮的两类对手名次等权平均，再平均三个训练种子排序。阶段 2 胜者须至少在两个种子上较自己的初始综合名次改善 0.02 才继续。阶段 4 从两个阶段全部候选中去重选前二，重复配方采用阶段 3 结果。

阶段 4 只对预先确定的最终 2000 轮模型使用保留种子 9000000000..9000008191，每模型每类对手 8192 局，共 20 项评估。保留集不用于阶段间选参；λ 补充也不使用它。训练种子间共享游戏种子，不能将重复 checkpoint 或全部对局视作独立训练重复。

## 吞吐与并发

`workers` 控制采样和更新阶段共用的 Rayon 线程池。默认值 16 是起点，实际并发应通过 [吞吐工具](../tools/ppo_throughput.py) 在目标运行环境测量。单任务延迟与多任务总吞吐应分别比较，并为评估预留资源；不要直接把单任务最优 workers 乘以任务数。

测试时固定模型、games、batch、训练超参数和随机种子，排除预热轮次。记录决策数/秒、实际训练样本数/秒、每轮耗时和峰值内存；相同轮数的速度差可能来自 KL 提前停止造成的更新次数差异，或策略退化导致对局变长。模型、批次或训练阶段改变后应重新测量。

### 已保留的性能优化

原实现按整张矩阵分配梯度归并和 AdamW 任务；输入投影含 940,032 个参数，约占模型 40.6%，造成调度不均衡。现按 4096 个参数分块，逐元素求和顺序和 Adam 算式保持不变。梯度计算固定按 32 行分块，使 workers 改变时仍保持一致的归并顺序。

配对验证中，最终 142 个网络/Adam/schema 数组逐字节一致，优化器步数、RNG 和训练计数相同；耗时历史不参与比较。后续可评估复用前后向工作区、减少梯度重复清零，以及分块大小的内存/GEMM 折衷。当前分段计时不能单独证明缓存或内存带宽瓶颈。

## 训练实验结果

阶段 2–4 主实验于 **2026-11-11** 完成。55 个训练任务、1810 项验证评估、20 项保留集评估无缺失，每 50 轮 checkpoint 均存在，训练错误日志为空。加上 λ 补充共 61 个任务、291.2 万局训练，各任务至少 500 轮。下方的 Freeze 长训是后续独立实验。

### 阶段 2 学习率与 GAE

以下均为最后三个评估点、三个种子的均值，综合分对两类对手等权，越低越好。

| lr | λ | scripted | teacher | 综合 |
| ---: | ---: | ---: | ---: | ---: |
| 1e-4 | 0.95 | 1.4931 | 1.7816 | **1.6373** |
| 3e-4 | 0.99 | 1.5507 | 2.0384 | 1.7945 |
| 1e-4 | 0.99 | 1.6428 | 2.2320 | 1.9374 |
| 3e-5 | 0.95 | 1.7234 | 2.2061 | 1.9647 |
| 3e-5 | 0.99 | 2.1483 | 3.1247 | 2.6365 |
| 3e-4 | 0.95 | 3.8333 | 4.0918 | 3.9626 |
| 3e-4 | 1 | 4.0145 | 5.6105 | 4.8125 |
| 3e-5 | 1 | 3.9132 | 5.9161 | 4.9147 |
| 1e-4 | 1 | 5.7827 | 7.0399 | 6.4113 |

最佳配方的三个种子分别得到 scripted/teacher：1.5426/1.8486、1.4577/1.7396、1.4788/1.7565。只看第 500 轮，均值为 1.4609/1.7331，前四率 99.28%/95.51%，吃鸡率 65.89%/60.74%。

lr=3e-4、λ=0.95 的种子 200000000 约在 350 轮后退化：第 300 轮为 1.708/2.130，第 400 轮为 7.982/7.951，熵接近零且大量游戏触及 100 回合上限。该任务仍跑满 500 轮，结果保留在排序中。当前预算下 λ=1 明显落后；不能据此断言较小 λ 在所有设置下都更优。

### 阶段 3 熵系数与数据复用

固定 lr=1e-4、λ=0.95，排序方法同阶段 2。

| entropy | epochs | scripted | teacher | 综合 |
| ---: | ---: | ---: | ---: | ---: |
| 0.03 | 1 | 1.4914 | 1.7039 | **1.5977** |
| 0.01 | 3 | 1.4931 | 1.7816 | 1.6373 |
| 0.01 | 1 | 1.5110 | 1.8094 | 1.6602 |
| 0.03 | 3 | 1.5316 | 1.8896 | 1.7106 |
| 0.003 | 3 | 1.6270 | 1.9665 | 1.7967 |
| 0.003 | 1 | 1.6441 | 2.0087 | 1.8264 |

前两名记作 **A：entropy=0.03、epochs=1** 和 **B：entropy=0.01、epochs=3**。三个种子的综合分 A 为 1.6188/1.5957/1.5785，B 为 1.6956/1.5986/1.6177；差距主要来自 teacher。

B 是阶段 2 最佳配方的重复，三个最终 checkpoint 的 142 个数组、配置、优化器和 RNG 均与阶段 2 一致。阶段 3 未出现阶段 2 所见的持续低熵和回合上限退化。

### 阶段 4 长训练与保留集

A/B 各使用五个新训练种子，从头训练到 2000 轮。以下为固定验证集五种子的均值：

| 迭代 | A scripted | A teacher | B scripted | B teacher |
| ---: | ---: | ---: | ---: | ---: |
| 500 | 1.4609 | 1.6746 | 1.5055 | 1.7682 |
| 1000 | 1.3205 | 1.4504 | 1.3314 | 1.5346 |
| 1500 | 1.2781 | 1.3904 | 1.3197 | 1.4730 |
| 2000 | 1.2289 | 1.3203 | 1.2652 | 1.3906 |

最终保留集每模型每对手 8192 局，先计算各模型指标，再等权平均五个种子：

| 配置 | scripted 排名 | teacher 排名 | scripted 吃鸡率 | teacher 吃鸡率 | scripted 前四率 | teacher 前四率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **A** | **1.2267** | **1.3126** | **79.82%** | **77.46%** | **99.91%** | **98.98%** |
| B | 1.2674 | 1.3815 | 76.96% | 74.40% | 99.85% | 98.51% |

| 训练种子 | A scripted | B scripted | A teacher | B teacher |
| ---: | ---: | ---: | ---: | ---: |
| 400000000 | 1.2096 | 1.3348 | 1.2849 | 1.4480 |
| 500000000 | 1.2379 | 1.2535 | 1.3268 | 1.3748 |
| 600000000 | 1.2272 | 1.2434 | 1.3073 | 1.3683 |
| 700000000 | 1.2189 | 1.2461 | 1.3014 | 1.3369 |
| 800000000 | 1.2401 | 1.2590 | 1.3428 | 1.3795 |

A 在五个配对种子、两类对手上全部领先，平均排名改善 0.0406/0.0689。模型间排名样本标准差 A 为 0.0128/0.0225，B 为 0.0382/0.0407；这不是单局标准差或置信区间。保留集衡量固定 bot 和新游戏种子，不代表对人类或其他训练策略的泛化。

各组 64 万局训练中，A 没有触及游戏回合上限，B 有 10 局，未出现持续策略崩塌或非有限训练指标。长训练仍有收益，但不能据此认为已收敛。

### 补充实验 λ 为 0.90

利用阶段 4 的空闲容量，另起 A/B 两配方 × 三个原训练种子，均从头训练 500 轮。与阶段 3 的 λ=0.95 按相同种子、400/450/500 轮比较，不与阶段 4 的 2000 轮混比。

| 配方 | λ=0.95 scripted / teacher | λ=0.90 scripted / teacher |
| --- | --- | --- |
| A | 1.4914 / 1.7039 | 1.5506 / 1.7901 |
| B | 1.4931 / 1.7816 | 1.6263 / 1.9773 |

六个任务、132 项评估全部完成。两种配置、三个种子、两类对手的 12 个配对差值均为 λ=0.90 更差，当前配方保留 0.95；这不证明 0.95 是所有预算下的全局最优。

补充实验未修改阶段 4 配方，也未使用其保留集。

### 连续 Freeze 规则对照长训

后续对照比较原配方与屏蔽连续 Freeze 的效果，尚无最终结论。两组各使用 A 的五个 2000 轮 checkpoint（种子 400000000 至 800000000），均续训到累计 **10000 轮**，即每任务新增 8000 轮、共新增 512 万局训练。

| 组 | 连续 Freeze 规则 | 其余设置 |
| --- | --- | --- |
| `control` | 关闭 | A 原配方，原 Adam/RNG/训练计数继续 |
| `no_repeat_freeze` | 开启 | 与对应 control 从相同 checkpoint 分叉 |

网络、奖励、lr=1e-4、λ=0.95、entropy=0.03、epochs=1、games=64、batch=1024 保持一致。使用 `fork` 仅改变规则开关，两组从相同权重、Adam 状态、RNG 和训练计数继续。关闭规则的新实现与增加规则前的实现，经三轮续训验证，142 个数组及训练状态完全相同；改变 workers 的配对验证也保持一致。历史诊断日志的 JSON 浮点重编码只允许末位误差，训练状态和数组仍要求精确一致。

训练前，同权重规则开关的五种子均值如下。固定 bot 每模型每对手 1024 局，混桌每模型 256 局（128 个配对种子），参考为该种子自身的 A-2000：

| 组 | scripted 排名 | teacher 排名 | 对 A-2000 混桌排名 | scripted / teacher 每局模型决策数 |
| --- | ---: | ---: | ---: | --- |
| control | 1.2289 | 1.3203 | 4.5000 | 262.49 / 265.38 |
| no_repeat_freeze | 1.2375 | 1.3072 | 4.4910 | 245.90 / 246.14 |

实验组连续 Freeze 次数为 0，固定 bot 对战的模型决策数减少约 6%/7%；这些是未续训时的诊断结果，排名差距小，不能据此判断长训胜负。control 的全部逐局名次与上一轮对应 2000 轮评估完全相同。

后续评估协议：

- 每 100 轮保存独立 checkpoint，分别对 scripted/teacher 评估 1024 局，验证种子仍为 10000000 起。
- 每 500 轮对冻结的同种子 A-2000 做 1024 局混桌，使用 20000000 起的 512 个配对种子。中间结果均视为验证数据。
- 仅最终 10000 轮使用新保留集：每模型每种固定 bot 各 8192 局（10000000000 起）；对 A-2000 各 8192 局混桌（11000000000 起的 4096 个配对种子）。
- 最后增加五项同种子 `no_repeat_freeze` 对 `control` 的直接混桌，每项 8192 局。汇总按训练种子配对，包含最终排名、动作效率和种子间差异，不按保留集回选中途 checkpoint。

复现时使用上文的 `fork`、`resume`、`eval` 和 `arena` 命令，分别保存两组输出。任务调度和并发按实际资源配置；一次性批量调度脚本不作为项目接口维护。

## 开局轨迹与已知问题

对 A 的五个 2000 轮模型，各取 1024 个验证种子，对七个 teacher 记录前两回合，共 5120 条 greedy 轨迹；另采集 5120 条概率采样轨迹。采样对照仍是对 teacher，并非八模型共同自博弈。

### 动作分布

下表是该回合至少执行一次该动作的游戏比例，各行不互斥。Play 包含随从、法术及合并后的磁力，Choose 不含外部选英雄。

| 动作 | 第一回合 | 第二回合 |
| --- | ---: | ---: |
| Buy | 93.55% | 97.44% |
| Play | 95.27% | 98.01% |
| Refresh | 25.10% | 51.97% |
| Freeze | 30.74% | 36.99% |
| Choose | 29.82% | 69.65% |
| HeroPower | 11.21% | 13.05% |
| Activate | 0.21% | 38.11% |
| Sell | 0.20% | 0.29% |
| Upgrade | **0%** | **0%** |
| EndTurn | 100% | 100% |

没有 Move、Cancel 或 DarkGift。每回合平均动作数为 4.89/6.92，含 EndTurn。若以全部动作次数为分母，Play/Buy/Freeze/Refresh 第一回合占 25.82%/19.67%/15.84%/7.28%，第二回合占 20.84%/17.03%/15.19%/10.36%。

跳过开局待处理 Choose 后，第一项动作如下：

| 第一项非 Choose 动作 | 第一回合 | 第二回合 |
| --- | ---: | ---: |
| Buy | 79.45% | 65.29% |
| Refresh | 12.50% | 25.04% |
| Freeze | 4.75% | 6.31% |
| HeroPower | 2.36% | 1.95% |
| Play | 0.37% | 1.21% |
| 其他 | 0.57% | 0.20% |

第一回合最常见完整顺序为“买随从 → 打出 → 结束”，占 42.09%；第二回合为“买随从 → 打出 → Activate → Choose → 结束”，占 19.82%，其次“刷新 → 买随从 → 打出 → 结束”占 8.30%。

第一回合买过随从/法术的游戏占 69.90%/25.23%，第二回合为 76.33%/35.16%。高频购买包括可疑的监狱守卫、夺金健将和拔线机；购买频次不等于相同备选下的偏好胜率。

第一回合结束时 83.50% 有一个随从、4.41% 有两个、12.09% 空场；第二回合 74.92% 有两个、19.26% 有一个、4.61% 有三个、1.21% 空场。两回合花光金币的比例为 94.65%/95.31%，所有结束状态少于 3 金币，均显式 EndTurn，没有动作上限强制结束。

### 不升本与冻结循环

五个模型第二回合 greedy 升本次数均为 0，约 95.16% 的第二回合初始状态允许 Upgrade；初始决策点的平均升本概率仅约 0.034%。采样对照也仅有 5/5120 = 0.098% 的游戏第二回合升本。这说明低升本倾向在策略分布中已存在，不能解释为普遍被 mask 禁止，也不能据此推断后续回合都不升本或该开局最优。

第一/二回合分别有 **9.88%/15.23%** 的游戏连续按五次 Freeze，触及既有冻结次数上限，五个模型都有该现象。例如训练种子 400000000、游戏种子 10000000 的苔丝，第二回合英雄技能后连续五次冻结/解冻，接着刷新、买法术联盟旗帜再使用，增加了冗余决策。

概率采样时，至少按一次 Freeze 的比例更高，为 64.18%/72.05%，但连续五次的比例降为 2.09%/1.93%。greedy 会重复某些局部状态的 Freeze 偏好。这促成了上文的连续 Freeze 规则对照；开局策略仍需单独研究。较好的固定 bot 排名不意味着动作效率已理想；这些轨迹来自规则关闭的原始 A 模型。

## 代码与工具

### 实现入口和检查

| 文件 | 职责 |
| --- | --- |
| [train.rs](../crates/tavernagent_v2/src/train.rs) / [rollout.rs](../crates/tavernagent_v2/src/rollout.rs) | 参数、同步训练循环、完整游戏采样 |
| [gae.rs](../crates/tavernagent_v2/src/gae.rs) / [ppo.rs](../crates/tavernagent_v2/src/ppo.rs) | 按座位 GAE、PPO 损失、KL 停止、AdamW |
| [net.rs](../crates/tavernagent_v2/src/net.rs) / [policy.rs](../crates/tavernagent_v2/src/policy.rs) | 模型和 masked categorical |
| [actions.rs](../crates/tavernagent_v2/src/actions.rs) / [observation.rs](../crates/tavernagent_v2/src/observation.rs) | 紧凑动作、训练 mask、额外观测与饰品映射 |
| [checkpoint.rs](../crates/tavernagent_v2/src/checkpoint.rs) | NPZ 保存与精确恢复 |
| [eval.rs](../crates/tavernagent_v2/src/eval.rs) / [teacher.rs](../crates/tavernagent_v2/src/teacher.rs) | 固定对手评估、迁移的 teacher |
| [arena.rs](../crates/tavernagent_v2/src/arena.rs) / [diagnostics.rs](../crates/tavernagent_v2/src/diagnostics.rs) | 配对混桌、动作计数和逐回合诊断 |
| [league.rs](../crates/tavernagent_v2/src/league.rs) / [ppo_league.py](../tools/ppo_league.py) / [league_elo.py](../tools/league_elo.py) | 八 checkpoint 混桌、持续调度、共享池 Elo |
| [backend.rs](../crates/tavernagent_v2/src/backend.rs) / [cuda/mod.rs](../crates/tavernagent_v2/src/cuda/mod.rs) | 执行后端、GPU 网络与优化器 |
| [batched.rs](../crates/tavernagent_v2/src/batched.rs) / [batched_eval.rs](../crates/tavernagent_v2/src/batched_eval.rs) / [batched_arena.rs](../crates/tavernagent_v2/src/batched_arena.rs) | 保持座位顺序的跨游戏合批 |
| [ppo_notify.py](../tools/ppo_notify.py) | 读取训练与联赛结果、绘制三张图、定期发送飞书卡片 |

```bash
cargo test --release -p tavernagent_v2
cargo test --release -p tavernagent_v2 --features cuda
cargo fmt -p tavernagent_v2 -- --check
cargo clippy -p tavernagent_v2 --all-targets -- -D warnings
cargo clippy -p tavernagent_v2 --all-targets --features cuda -- -D warnings
python3 -m unittest discover -s tools -p 'test_ppo_*.py'
```

已有检查覆盖动作和磁力映射、mask 逃生规则、饰品来源、v1 观测/FP32 一致性、terminal/bootstrap GAE、有限差分梯度、actor/learner 概率一致性、按座位奖励、跨 workers 采样与更新一致性、KL 停止和恢复。CPU 检查还覆盖跨游戏合批的完整轨迹、固定 bot 评估和混桌结果一致性；GPU 通过 `cuda-check` 单独验收。其余检查包括 teacher 逐动作 v1 一致性、iteration-0 恢复、连续 Freeze 的座位/回合隔离、规则 fork 状态保留和混桌中性分数。开局工具通过 clippy，试跑验证 1/2 workers 轨迹相同，完整统计校验两个回合都有结束记录。

联赛测试覆盖不同网络大小与策略规则、并行结果一致性、输出保护及引擎规则校验。Python 测试覆盖联赛恢复、截断对局隔离、评估去重、通知故障恢复及三图卡片的完整发送。

### 工具用法

[吞吐工具](../tools/ppo_throughput.py) 的 `--cases` 为 **workers × 任务数**。它自动发现可用物理核和 NUMA 拓扑，默认训练参数采用上文基线；依赖 Linux、Python 标准库、`numactl`、`lscpu` 和 `/usr/bin/time`。每次使用新的输出目录，保存命令、资源指标和 checkpoint；`--repeats` 交替顺序重复扫描。

```bash
cargo build --release -p tavernagent_v2
python3 tools/ppo_throughput.py --binary target/release/tavernagent_v2 \
  --out runs/ppo-throughput --cases 2x1 4x1 \
  --games 64 --batch 1024 --iterations 20 --warmup 2 --repeats 2
```

选择的 workers 与任务数必须符合运行环境的可用物理核和 NUMA 节点容量。可用 `RUSTFLAGS="-C target-cpu=native"` 构建以测量目标 CPU 的性能；这种二进制只应在兼容 CPU 上运行。

[开局采集示例](../crates/tavernagent_v2/examples/opening_trace.rs) 记录模型前两回合的轨迹；加入 `--sample` 改为概率采样。每步包含动作类型概率、所选卡牌，以及前后金币、手牌、战场和商店。它读取 checkpoint 中保存的策略规则。

```bash
cargo build --release -p tavernagent_v2 --example opening_trace
target/release/examples/opening_trace --model runs/tavernagent_v2/model.npz \
  --games 1024 --workers 16 --seed 10000000 --out runs/opening/trace.json
```

### 产物与复现边界

`runs/` 为 Git 忽略的输出目录。Checkpoint、训练日志、逐局评估和开局轨迹由使用者保存，不随源码分发。本文表格记录已完成实验的汇总结论，示例路径用于新建任务，不指向随仓库发布的模型。

复现实验应同时保存源码版本、构建选项、完整命令、训练与评估种子，以及 checkpoint 中的超参数和策略规则。上文已列出已完成实验的配置、预算和评估协议；历史实验使用提交前的开发快照，当前提交并非那些构建的逐字节副本。

算法原始参考：[PPO-Clip](https://arxiv.org/abs/1707.06347)、[Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)。
