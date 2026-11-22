# CS285 Fall 2023 HW1 完成指引

## 0. 你这次作业的目标

这份 HW1 的主题是 imitation learning，具体包括：

1. 把 starter code 里标记为 `TODO` 的部分补完。
2. 跑通 Behavior Cloning, BC。
3. 在同一份代码上跑通 DAgger。
4. 观察、记录、对比不同环境和不同超参数下的表现。
5. 完成报告中要求的表格、曲线和简答题。

这份指引涵盖：

- 项目结构怎么看。
- 每个关键文件在干什么。
- 每个 TODO 应该补在哪里、写什么、实现了什么功能。
- PyCharm 直接点三角运行应该怎么配置。
- 你应该按什么顺序完成、训练、调试和整理结果。

---

## 1. 项目结构总览

`hw1` 根目录下的核心内容：

- `README.md`
  - 最直接的运行说明，给了 BC 和 DAgger 的命令模板。
- `installation.md`
  - 环境安装说明，尤其是 MuJoCo、`pip install -e .`。
- `cs285_fall2023_hw1.pdf`
  - 作业要求本体，里面写了你必须做哪些实验和最后怎么提交。
- `setup.py`
  - 把当前 `hw1` 目录注册成可导入的 `cs285` 包。
- `requirements.txt`
  - 这份作业依赖的 Python 包版本。
- `cs285/`
  - 真正的作业代码。

`cs285/` 下面你这次最重要的目录：

- `cs285/scripts/run_hw1.py`
  - 训练入口。无论 BC 还是 DAgger，最后都从这里开始。
- `cs285/policies/MLP_policy.py`
  - 你要训练的学生策略网络，核心是把 observation 映射成 action distribution。
- `cs285/policies/loaded_gaussian_policy.py`
  - 已经训练好的专家策略，只负责给你打标签，不参与训练。
- `cs285/infrastructure/utils.py`
  - rollout 采样逻辑。策略怎么和环境交互、怎么收一条轨迹，都在这里。
- `cs285/infrastructure/replay_buffer.py`
  - 把 expert data 和 DAgger 新增数据拼起来，供训练时随机采样。
- `cs285/expert_data/*.pkl`
  - 专家演示数据。`itr == 0` 的 BC 初始训练直接用它。
- `cs285/policies/experts/*.pkl`
  - 专家策略参数文件。DAgger 时用它给 learner 采样到的状态重新打标签。

---

## 2. 代码流程总览

建议你按这个顺序读代码，和作业 PDF 第 2 页的建议一致：

1. `cs285/scripts/run_hw1.py`
2. `cs285/policies/MLP_policy.py`
3. `cs285/infrastructure/replay_buffer.py`
4. `cs285/infrastructure/utils.py`
5. `cs285/infrastructure/pytorch_util.py`

整个训练流程可以理解成下面这条线：

1. `main()` 解析参数，创建日志目录。
2. `run_training_loop(params)` 创建环境、学生策略 `actor`、专家策略 `expert_policy`、`replay_buffer`。
3. `itr == 0` 时，直接从 `expert_data/*.pkl` 读取专家轨迹做 BC warm start。
4. `itr > 0` 且 `--do_dagger` 时，用当前学生策略和环境交互，收集 learner 自己访问到的状态。
5. 再用专家策略给这些状态重新标动作标签。
6. 把所有轨迹加入 `ReplayBuffer`。
7. 从 `ReplayBuffer` 随机采样 `(observation, action)` 小批量，训练 `MLPPolicySL`。
8. 用当前学生策略做评估 rollout，记录 `Eval_AverageReturn`、`Eval_StdReturn` 等指标。
9. 反复迭代，得到 DAgger 的 learning curve。

把它画成一句话就是：

`专家数据 / learner rollout -> replay buffer -> 随机采样 batch -> supervised learning 更新策略 -> 环境评估`

---

## 3. 每个关键文件到底在做什么

### 3.1 `cs285/scripts/run_hw1.py`

作用：这是训练总控。

你重点看这些位置：

- 第 31 行附近 `run_training_loop(params)`
  - 训练总循环。
- 第 64 行附近
  - 创建 Gym 环境，读 observation / action 维度。
- 第 88 行附近
  - 创建学生策略 `MLPPolicySL`。
- 第 103 行附近
  - 加载专家策略 `LoadedGaussianPolicy`。
- 第 124 到 160 行附近
  - 这里就是本次要补的训练逻辑核心。

这份文件的职责不是定义模型，而是把“采样、重标注、训练、评估、记录”串起来。

### 3.2 `cs285/policies/MLP_policy.py`

作用：定义学生策略网络。

你重点看这些对象：

- `build_mlp(...)`
  - 创建一个多层感知机，输入是 observation，输出是 action 的均值。
- `MLPPolicySL.mean_net`
  - 学生策略的均值网络。
- `MLPPolicySL.logstd`
  - 连续动作策略的对数标准差参数。
- `forward(...)`
  - 你要把网络输出包装成一个可采样、可算 log probability 的分布。
- `update(...)`
  - 你要实现监督学习更新。

这里的知识点是：

- HW1 不是 Q-learning，也不是 policy gradient。
- 本质上是在做 supervised learning。
- 但因为动作是连续的，所以最自然的写法是输出一个 Gaussian policy，然后最大化专家动作的 log-likelihood。

### 3.3 `cs285/infrastructure/utils.py`

作用：负责 rollout 采样。

你重点看：

- `sample_trajectory(...)`
  - 从当前策略出发，在环境里跑一条完整轨迹。
- `sample_trajectories(...)`
  - 一直采样，直到步数达到 `min_timesteps_per_batch`。
- `sample_n_trajectories(...)`
  - 固定采几条轨迹，主要用于视频记录。
- `compute_metrics(...)`
  - 统计 return、episode length 等日志指标。

知识点：

- BC 的训练数据是 `(s, a)` 配对。
- DAgger 的新数据也是 `(learner 访问到的 s, expert 给出的 a)`。
- 所以 rollout 函数必须把 observation、action、reward、next_observation、terminal 全都存下来。

### 3.4 `cs285/infrastructure/replay_buffer.py`

作用：存数据。

你重点看：

- `add_rollouts(paths)`
  - 把若干条轨迹拆成大数组，拼接到 buffer 中。
- `self.obs`, `self.acs`
  - 训练时真正会随机采样的就是它们。

HW1 里你训练 imitation policy 其实只需要：

- `replay_buffer.obs`
- `replay_buffer.acs`

奖励和 next observation 被保存下来主要是为了统一 rollout 结构和日志统计。

### 3.5 `cs285/policies/loaded_gaussian_policy.py`

作用：加载专家策略并进行前向推理。

你重点看：

- `forward(...)`
  - 专家网络如何把 observation 映射成动作。
- `get_action(...)`
  - 给定一个 observation 或 observation batch，返回 numpy action。

这个文件很重要，因为它告诉你学生策略最好也提供 `get_action(...)` 接口。否则 `utils.sample_trajectory(...)` 里没法统一调用。

### 3.6 数据文件

#### `cs285/expert_data/expert_data_*.pkl`

作用：专家轨迹数据，BC 第 0 轮直接用。

`expert_data_Ant-v4.pkl` 的结构：

- 它是一个 `list`
- 里面每个元素是一条 trajectory `dict`
- 每条轨迹包含这些 key：
  - `observation`
  - `action`
  - `reward`
  - `next_observation`
  - `terminal`

示例形状：

- `observation`: `(1000, 27)`
- `action`: `(1000, 8)`
- `reward`: `(1000,)`
- `next_observation`: `(1000, 27)`
- `terminal`: `(1000,)`

这解释了为什么 `itr == 0` 时可以直接开始训练，因为专家已经帮你把数据准备好了。

#### `cs285/policies/experts/*.pkl`

作用：保存专家模型参数。

DAgger 时不是直接重复使用 `expert_data`，而是：

1. 先让 learner 自己跑。
2. 让 learner 访问到一批新的状态。
3. 再用专家策略给这些状态贴动作标签。

这就是 DAgger 和纯 BC 的核心区别。

---

## 4. 这次真正要你补的 TODO 清单

下面的行号是基于你当前仓库版本。

### 4.1 `cs285/infrastructure/utils.py`

#### TODO A: reset 后拿到初始 observation

位置：

- 文件：`cs285/infrastructure/utils.py`
- 当前行：20

你要完成的功能：

- 新开一条 rollout 时，先 `env.reset()`。
- 把 reset 返回的 observation 存到 `ob`。

推荐写法：

```python
reset_out = env.reset()
ob = reset_out[0] if isinstance(reset_out, tuple) else reset_out
```

为什么这样写：

- 课程给的依赖是 `gym==0.25.2`，通常 `reset()` 直接返回 observation。
- 但如果你在 PyCharm 里选错了解释器，装成了更新版 Gym/Gymnasium，`reset()` 可能返回 `(obs, info)`。
- 这份写法更稳，能少踩环境版本坑。

#### TODO B: 用当前策略根据 observation 选 action

位置：

- 文件：`cs285/infrastructure/utils.py`
- 当前行：35 到 37

你要完成的功能：

- 用当前策略根据最新的 `ob` 选动作。
- 返回值要求是 numpy array。

推荐写法：

```python
ac = policy.get_action(ob)
```

注意：

- 下一行已经有 `ac = ac[0]`，说明这里期望 `get_action` 返回 shape 类似 `(1, ac_dim)`。

#### TODO C: 执行动作，拿到下一个状态和奖励

位置：

- 文件：`cs285/infrastructure/utils.py`
- 当前行：39 到 40

你要完成的功能：

- 把 action 真正喂给环境。
- 拿到 `next_ob`, `rew`, `done`。

推荐写法：

```python
step_out = env.step(ac)
if len(step_out) == 5:
    next_ob, rew, terminated, truncated, _ = step_out
    done = terminated or truncated
else:
    next_ob, rew, done, _ = step_out
```

为什么这样写：

- 老版 Gym 返回 4 个值。
- 新版 Gym/Gymnasium 返回 5 个值。
- PyCharm 直接运行时，解释器选错很常见，这个兼容写法能省很多排错时间。

#### TODO D: 判断 rollout 是否结束

位置：

- 文件：`cs285/infrastructure/utils.py`
- 当前行：42 到 44

你要完成的功能：

- 轨迹结束条件有两个：
  - 环境自己 done 了
  - 走到 `max_path_length`

推荐写法：

```python
rollout_done = done or (steps >= max_path_length)
```

知识点：

- 这一步控制 episode horizon。
- 课程理论部分经常把 horizon 记为 `T`，这里就是代码里的 `max_path_length` / `ep_len`。

---

### 4.2 `cs285/policies/MLP_policy.py`

这个文件最容易让人误解的地方是：它虽然名字叫 policy，但 HW1 的训练方式本质还是 supervised learning。

#### TODO E: `forward(...)`

位置：

- 文件：`cs285/policies/MLP_policy.py`
- 当前行：119 到 132

你要完成的功能：

- 输入 observation batch。
- 输出一个连续动作的 Gaussian distribution。
- 这样你后面才能用专家动作去计算 `log_prob`，做最大似然训练。

推荐写法：

```python
mean = self.mean_net(observation)
std = torch.exp(self.logstd)
action_distribution = distributions.Independent(
    distributions.Normal(mean, std),
    1,
)
return action_distribution
```

为什么是这个写法：

- `mean_net` 负责输出均值。
- `logstd` 是可学习的对数标准差参数。
- `exp(logstd)` 变成标准差。
- `Independent(..., 1)` 的作用是把每个动作维度上的独立高斯合成一个多维动作分布，这样 `log_prob(actions)` 会直接返回每个样本的一个标量。

#### 额外必须补的一点：`get_action(...)`

虽然文件头只明确写了 `forward` 和 `update`，但你实际上还应该在 `MLPPolicySL` 类里补一个 `get_action(...)` 方法。

原因：

- `utils.sample_trajectory(...)` 里会调用 `policy.get_action(ob)`。
- `LoadedGaussianPolicy` 已经提供了这个接口。
- `MLPPolicySL` 现在没有这个方法，运行到 DAgger 或评估时会报：
  - `AttributeError: 'MLPPolicySL' object has no attribute 'get_action'`

建议你插入的位置：

- 文件：`cs285/policies/MLP_policy.py`
- 建议插在 `forward(...)` 后、`update(...)` 前

推荐写法：

```python
@torch.no_grad()
def get_action(self, obs: np.ndarray) -> np.ndarray:
    if len(obs.shape) > 1:
        observation = obs
    else:
        observation = obs[None, :]
    observation = ptu.from_numpy(observation.astype(np.float32))
    action_distribution = self(observation)
    action = action_distribution.sample()
    return ptu.to_numpy(action)
```

知识点：

- BC 训练时优化的是 “专家动作在当前策略下的概率”。
- 真正和环境交互时，策略需要从自己定义的动作分布里采样动作。

#### TODO F: `update(...)`

位置：

- 文件：`cs285/policies/MLP_policy.py`
- 当前行：134 到 145

你要完成的功能：

- 把 numpy 的 observations / actions 转成 torch tensor。
- 前向得到动作分布。
- 计算专家动作的负对数似然损失。
- 反向传播并更新参数。

推荐写法：

```python
observations = ptu.from_numpy(observations.astype(np.float32))
actions = ptu.from_numpy(actions.astype(np.float32))

action_distribution = self(observations)
log_prob = action_distribution.log_prob(actions)
loss = -log_prob.mean()

self.optimizer.zero_grad()
loss.backward()
self.optimizer.step()
```

为什么不是 MSE：

- 很多同学第一反应会写成 `MSE(pred_action, expert_action)`。
- 但这份框架已经把策略建模成 Gaussian policy 了。
- 所以最自然、也最符合 lecture “maximum likelihood estimation” 的方式，是最大化专家动作的 log-likelihood，等价于最小化负 log-likelihood。

---

### 4.3 `cs285/scripts/run_hw1.py`

#### TODO G: DAgger 时收集 learner rollout

位置：

- 文件：`cs285/scripts/run_hw1.py`
- 当前行：132 到 135

你要完成的功能：

- 在 `itr > 0` 时，不再直接读专家数据。
- 而是让当前 learner 自己去环境里跑，至少收集 `params['batch_size']` 步数据。

推荐写法：

```python
paths, envsteps_this_batch = utils.sample_trajectories(
    env,
    actor,
    params['batch_size'],
    params['ep_len'],
)
```

知识点：

- 这一步是 DAgger 的第一半。
- 先让 learner 暴露它自己真实会访问到的状态分布。

#### TODO H: 用专家策略重标 learner 采到的状态

位置：

- 文件：`cs285/scripts/run_hw1.py`
- 当前行：141 到 144

你要完成的功能：

- 保留 learner 自己采到的 observation。
- 但把这些 observation 对应的 action 换成 expert 的动作标签。

推荐写法：

```python
for path in paths:
    path["action"] = expert_policy.get_action(path["observation"])
```

如果你非常想保留原列表变量名，也可以最后写：

```python
paths = paths
```

但实际上不需要额外赋值。

知识点：

- DAgger 不是让 expert 代替 learner 去跑。
- DAgger 是 learner 自己跑，然后 expert 负责告诉它：
  - “如果你走到了这些状态，我会怎么做。”

#### TODO I: 从 replay buffer 随机采样训练 batch

位置：

- 文件：`cs285/scripts/run_hw1.py`
- 当前行：155 到 160

你要完成的功能：

- 从 `replay_buffer.obs` 和 `replay_buffer.acs` 中，用同一组随机下标采样一批训练样本。

推荐写法：

```python
batch_size = params['train_batch_size']
rand_indices = np.random.permutation(replay_buffer.obs.shape[0])[:batch_size]
ob_batch = replay_buffer.obs[rand_indices]
ac_batch = replay_buffer.acs[rand_indices]
```

更稳一点的写法：

```python
batch_size = min(params['train_batch_size'], replay_buffer.obs.shape[0])
rand_indices = np.random.permutation(replay_buffer.obs.shape[0])[:batch_size]
ob_batch = replay_buffer.obs[rand_indices]
ac_batch = replay_buffer.acs[rand_indices]
```

知识点：

- 这是最基本的 SGD / mini-batch training。
- 采样必须对 `obs` 和 `acs` 使用同一批下标，否则监督标签就错位了。

---

## 5. 你可以把整个 HW1 理解成哪些课程知识点

### 5.1 Behavior Cloning = supervised learning / maximum likelihood

课程原话的核心意思是：

- 先收 demonstration trajectories
- 再做 supervised learning / maximum likelihood estimation

你在代码里对应看到的是：

- demonstration data: `expert_data/*.pkl`
- supervised learning 模型: `MLPPolicySL`
- maximum likelihood 训练: `- action_distribution.log_prob(expert_action).mean()`

### 5.2 为什么 BC 会失败：distribution shift

讲义重点是：

- 训练时你只看 expert 访问到的状态。
- 测试时 learner 一旦犯一个小错，就会被带到训练时没见过的状态。
- 接下来错误可能越来越多。

你在代码里对应看到的是：

- `itr == 0` 的纯 BC，只用 `expert_data`。
- 这意味着训练数据只来自 expert 的状态分布。

### 5.3 为什么 DAgger 能缓解这个问题

DAgger 的核心不是“再多加点专家数据”，而是：

1. 用 learner 当前策略去真实环境里跑。
2. 收集 learner 自己会访问到的状态。
3. 让 expert 给这些状态贴动作标签。
4. 聚合到数据集里继续训。

你在代码里对应看到的是：

- `utils.sample_trajectories(env, actor, ...)`
- `path["action"] = expert_policy.get_action(path["observation"])`
- `replay_buffer.add_rollouts(paths)`

---

## 6. 课程资料对应关系

下面这些是和 HW1 最直接相关的 Berkeley CS285 Fall 2023 官方材料，你可以自己打开并截图放到笔记里：

### 6.1 官方 Lecture Slides

来源：

- `https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-2.pdf`

建议重点截图页：

- PDF 第 9 页左右
  - Behavioral Cloning algorithm
- PDF 第 17 页左右
  - BC 不满足 i.i.d. assumption，出现 distribution shift
- PDF 第 35 页左右
  - DAgger: Dataset Aggregation

你在看这些页时可以对照源码：

- BC algorithm 对应 `itr == 0` 读 `expert_data` 然后直接监督训练
- distribution shift 对应为什么单纯 BC 在复杂环境容易崩
- DAgger 对应 `itr > 0` 时的采样、重标注、聚合训练

### 6.2 官方 HW1 PDF

来源：

- `hw1/cs285_fall2023_hw1.pdf`

建议重点看：

- 第 2 页
  - 代码阅读顺序、需要补的文件
- 第 2 到 3 页
  - BC 和 DAgger 分别要跑什么实验
- 第 4 到 5 页
  - 报告和提交格式要求

---

## 7. PyCharm 直接点三角运行的详细指导

### 7.1 先确认 PyCharm 用的是正确解释器

你需要确保 PyCharm 选择的 interpreter 已经安装了：

- `requirements.txt` 里的包
- 当前 `hw1` 的 editable package

也就是你至少做过这两步：

```bash
pip install -r requirements.txt
pip install -e .
```

这里的工作目录一定是在：

- `F:\SchoolWork\课程作业\CS285_Learning\homework\homework_fall2023-main\hw1`

如果没有做 `pip install -e .`，PyCharm 经常会报：

- `ModuleNotFoundError: No module named 'cs285'`

因为 `run_hw1.py` 里有：

```python
from cs285.infrastructure import pytorch_util as ptu
```

### 7.2 最推荐的打开方式

推荐你直接把下面这个目录作为 PyCharm Project Root 打开：

- `F:\SchoolWork\课程作业\CS285_Learning\homework\homework_fall2023-main\hw1`

这样有两个好处：

1. `setup.py`、`requirements.txt`、`cs285/`、`data/` 都在一个项目里。
2. `run_hw1.py` 里使用的相对路径最不容易出错。

### 7.3 Run Configuration 应该怎么配

不要只靠“右键直接 Run 当前文件”，因为这类脚本需要命令行参数。

你应该在 PyCharm 里创建两个固定的 Run Configuration：

- 一个给 BC
- 一个给 DAgger

#### 配置 1：Behavior Cloning

Script path:

- `F:\SchoolWork\课程作业\CS285_Learning\homework\homework_fall2023-main\hw1\cs285\scripts\run_hw1.py`

Working directory:

- `F:\SchoolWork\课程作业\CS285_Learning\homework\homework_fall2023-main\hw1`

Parameters 示例：

```text
--expert_policy_file cs285/policies/experts/Ant.pkl --env_name Ant-v4 --exp_name bc_ant --n_iter 1 --expert_data cs285/expert_data/expert_data_Ant-v4.pkl --video_log_freq -1
```

#### 配置 2：DAgger

Script path:

- 同上

Working directory:

- 同上

Parameters 示例：

```text
--expert_policy_file cs285/policies/experts/Ant.pkl --env_name Ant-v4 --exp_name dagger_ant --n_iter 10 --do_dagger --expert_data cs285/expert_data/expert_data_Ant-v4.pkl --video_log_freq -1
```

### 7.4 为什么 Working Directory 一定要设成 `hw1`

因为参数里写的是相对路径：

- `cs285/policies/experts/Ant.pkl`
- `cs285/expert_data/expert_data_Ant-v4.pkl`

如果 Working Directory 不是 `hw1`，很容易报：

- `FileNotFoundError`

这类错误不是代码错，而是 PyCharm 的运行配置错。

### 7.5 你点运行之后，日志会保存到哪里

`run_hw1.py` 第 258 行附近会把日志目录设成：

- `hw1/data/`

也就是说你跑完后，结果一般会出现在：

- `F:\SchoolWork\课程作业\CS285_Learning\homework\homework_fall2023-main\hw1\data\q1_*`
- `F:\SchoolWork\课程作业\CS285_Learning\homework\homework_fall2023-main\hw1\data\q2_*`

前缀含义：

- `q1_` 对应 BC
- `q2_` 对应 DAgger

### 7.6 调试阶段推荐设置

为了先跑通，建议你一开始固定：

- `--video_log_freq -1`

因为：

- 这样不会录视频
- 更快
- 更适合先查 TODO 有没有写对

如果你已经跑通、想看策略视频，再把这个参数改回默认值或设成较小正数。

### 7.7 PyCharm 里怎么看结果

最直接的办法是看 Run 窗口打印出来的：

- `Eval_AverageReturn`
- `Eval_StdReturn`
- `Training Loss`

这些值来自：

- `utils.compute_metrics(...)`
- `actor.update(...)`

如果你想看 TensorBoard，有两种方式：

1. 直接用 PyCharm 自带 Terminal，进入 `hw1` 后运行：

```bash
tensorboard --logdir data
```

2. 如果你装了 PyCharm 的 TensorBoard 支持，也可以在 IDE 里直接打开事件文件。

### 7.8 PyCharm 常见报错和排查顺序

#### 报错 1

`ModuleNotFoundError: No module named 'cs285'`

排查：

1. 解释器是否是你装了依赖的那个环境。
2. 是否在 `hw1` 根目录执行过 `pip install -e .`。

#### 报错 2

`FileNotFoundError: cs285/...`

排查：

1. Working directory 是否是 `hw1`。
2. Parameters 里的相对路径有没有改错。

#### 报错 3

`AttributeError: 'MLPPolicySL' object has no attribute 'get_action'`

排查：

1. 你是否只补了 `forward` 和 `update`，忘了给学生策略加 `get_action`。

#### 报错 4

`NotImplementedError`

排查：

1. `forward(...)` 是否还在 `raise NotImplementedError`
2. `update(...)` 是否还保留 `loss = TODO`
3. `run_hw1.py` / `utils.py` 中是否还有 `TODO`

#### 报错 5

`too many values to unpack` 或 `not enough values to unpack`

排查：

1. 你装的可能不是课程要求的 `gym==0.25.2`
2. 你可能装成了新版本 Gym/Gymnasium
3. 所以我上面给你的 `reset()` / `step()` 兼容写法很有必要

---

## 8. 完成这份 HW1 的推荐顺序

### 阶段 1：先只把代码补到能跑通

顺序：

1. 先补 `MLP_policy.py`
2. 再补 `utils.py`
3. 最后补 `run_hw1.py`

原因：

- `run_hw1.py` 依赖前两者。
- 如果你先看训练循环，再回头发现 policy 连 `get_action` 都没有，会来回跳。

### 阶段 2：先做一个最小 smoke test

建议：

1. 先只跑 BC。
2. 先只选一个环境。
3. 先把 `--video_log_freq -1` 打开。

你的目标不是马上做出漂亮结果，而是确认：

- 程序不报错
- loss 会动
- 会打印 `Eval_AverageReturn`

### 阶段 3：做 BC 正式实验

作业要求你至少测试两个环境：

- 一个 BC 至少达到 expert 30% 表现
- 一个你自己选的、BC 表现不理想的环境

建议你这样开始：

- 先试 `HalfCheetah-v4` 或 `Hopper-v4`
  - 往往更适合先调通
- 再试 `Ant-v4` 或 `Walker2d-v4`
  - 更容易体现 BC 在分布偏移下的问题

注意：

- 这不是保证规律，只是比较常见的调试顺序。
- 真正写报告时，以你自己跑出来的结果为准。

### 阶段 4：做 BC 超参数实验

PDF 第 2 页要求你选一个超参数做影响分析。

最容易操作的候选：

- `num_agent_train_steps_per_iter`
- `train_batch_size`
- `n_layers`
- `size`
- 专家数据量

如果你想尽量少改代码，最推荐先试：

- `num_agent_train_steps_per_iter`

因为这个参数直接在 PyCharm 的 Run Configuration 里改就行。

### 阶段 5：做 DAgger 正式实验

当 BC 跑通以后，再切到 DAgger：

1. 把 `--do_dagger` 打开
2. 把 `--n_iter` 设成大于 1，例如 10
3. 在和 BC 相同的环境上跑
4. 观察每一轮 `Eval_AverageReturn`

你最终要画的是：

- 横轴：DAgger iteration
- 纵轴：policy mean return
- 误差条：std return
- 再加两条水平线：
  - expert performance
  - BC performance

### 阶段 6：整理报告和提交材料

根据 PDF 第 4 到 5 页，你最后至少要准备：

1. 报告 PDF
2. `run_logs` 目录
3. `cs285` 代码目录
4. 一个 README，写清楚如何复现实验

并且注意：

- 提交代码日志时不要带视频
- 所以提交用的 run 最好保留 `--video_log_freq -1`

---

## 9. 你可以优先盯住的核心函数

如果时间有限，最值得反复理解的是下面这些：

### `run_training_loop(...)`

为什么重要：

- 这是 BC 和 DAgger 的总控逻辑。
- 你一旦理解它，就知道“什么时候在采样，什么时候在训练，什么时候在评估”。

### `sample_trajectory(...)`

为什么重要：

- 这是 agent 和环境真正接触的地方。
- rollout 采样逻辑一旦写错，后面的 DAgger 都会错。

### `MLPPolicySL.forward(...)`

为什么重要：

- 它定义了“策略到底输出什么”。
- 这直接决定 `update(...)` 里 loss 怎么写。

### `MLPPolicySL.update(...)`

为什么重要：

- 这是 BC 的监督学习本体。
- 如果这里写错，你可能程序能跑，但结果会非常怪。

### `ReplayBuffer.add_rollouts(...)`

为什么重要：

- 它解释了为什么后面训练能直接用 `replay_buffer.obs` 和 `replay_buffer.acs` 采样。

---

## 10. 最后给你一个最实用的执行清单

你可以照着这份清单做，不容易乱：

1. 在 PyCharm 里把项目根目录切到 `hw1`
2. 确认 interpreter 正确
3. 安装 `requirements.txt`
4. 在 `hw1` 根目录执行 `pip install -e .`
5. 先补 `MLP_policy.py` 的 `forward`、`get_action`、`update`
6. 再补 `utils.py` 的 `sample_trajectory`
7. 再补 `run_hw1.py` 的 DAgger 采样 / 重标注 / replay buffer 采样
8. 创建 BC 的 Run Configuration
9. 点三角先跑 BC smoke test
10. 跑两个环境，做 BC 对比
11. 选一个超参数做 BC 曲线
12. 创建 DAgger 的 Run Configuration
13. 在同两个环境上跑 DAgger
14. 整理表格、learning curve、报告说明

如果你接下来继续做这份 HW1，最建议你先从 `MLP_policy.py` 下手，因为这是整个作业里“最容易看懂、又最关键”的地方。
