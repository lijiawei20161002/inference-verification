# BF16 问题与技术创新：2026-10-20 研究判断

这是对已有实验和最新相关工作的技术评审，**不是已实现算法或新 GPU 实验的报告**。
基于提交 `57ceb5e` 及截至 10 月 19 日的归档。新增工作仅为离线分数诊断、文献核对与研究设计。

**后续方向约束：用户希望核心创新不以低成本为主。**
下文的按需回放与能力—成本方案保留为此前候选；成本降为工程指标。
新的三条候选及其与近期工作的区别见
[不以成本为主的研究方向](NON_COST_RESEARCH_DIRECTIONS.md)，尚未确定采用其中哪一条。

## 1. 目前离论文核心贡献还有什么距离

用户的担心成立。现有结果能支持“合法执行变化影响验证分数和 timing”的实证结论，
还不能支持“解决了 BF16 验证”或“提出了优于现有工作的验证系统”。
上一版 slides 把“校准 + vLLM + 多硬件”作为贡献候选，创新性判断偏乐观。

BF16 问题至少有四层：

1. 同一执行路径是否可重复。
2. 合法 GPU、batch shape、kernel、prefill/decode 路径变化是否仍能通过。
3. 在独立测试条件下，误报受控的同时是否能检出真实 NF4/INT8。
4. 输出行为能否证明服务端实际使用 BF16。

第一层通过不代表后三层成立。第四层尤其需要收窄：仅凭输出 token 与可被服务端延迟填充的时间，
无法区分观测上等价的实现。合理目标是检验明确执行契约下的行为一致性，而不是无条件证明物理精度。
契约应写清权重、激活、KV cache、累加/舍入、允许的执行配置及采样协议，不能只写“BF16”。

现有真实量化结果：token-DiFR 在 64 条评估序列中对 BF16/NF4/INT8 分别拒绝 0/34/5 条，
每条 128 tokens；32 条 BF16 序列校准。0/64 的 95% Wilson 上界仍为 5.66%，
且不覆盖所有 GPU、调度与语料条件。activation-DiFR 的强检测结果需要 provider 激活，信息条件不同。

## 2. 新的离线发现：少数封顶位置支配分数

来源：[归档 scores.npz](../docs/results/real_quantization_l40s_20261018/pilot/scores.npz)。
仅分析原评估切分，未重新选阈值，未修改归档。

| arm | token 数 | 分数 = 30 的位置 | 封顶位置占总分 | 至少含一次封顶的序列 |
|---|---:|---:|---:|---:|
| 诚实 BF16 | 8,192 | 49 | 98.40% | 32/64 |
| NF4 | 8,192 | 252 | 94.25% | 59/64 |
| INT8 | 8,192 | 94 | 96.49% | 47/64 |

三组这些位置也同时触发本地 token-rank 分数上限 50。
TokenDiFR 在 claimed token 被 reference 的 top-k/top-p 排除时直接返回 30；
足够大的有限 margin 也会被截到 30。因此这是**候选集边界问题的线索，尚非归因**。
封顶 token 比例不是序列误报率；不能直接删掉这些位置，再用原阈值声称效果改善。

可手工构造的机制例子：top-k=2 时，provider 概率为 `[.4,.3001,.2999]`，
reference 为 `[.4,.2999,.3001]`，共享 Gumbel 为 `[0,1,0]`。
前者选择 token 1；后者将 token 1 排除，DiFR 因而封顶。
取消 top-k 后两者都选 token 1。这说明离散候选集变化能放大很小的数值差异；
它不是从真实模型捕获到的事件。

复现：[分析脚本](../slides/mlsys-paper-plan/analyze_score_tails.py)；
[含来源哈希的 JSON](../slides/mlsys-paper-plan/qa/technical/score-tail-diagnostic.json)。

需要先排除协议混杂：

- 本地 provider 过滤 logits 使用 float32，reference 路径使用 float64；应先用同一份 logits 做精度交叉实验。
- 本地 NumPy RNG 明确不复现 vLLM 的随机数流；相同 seed 数值不等于相同噪声张量。
- `apply_top_p` 注释说保留跨过 p 的 token，代码使用 `cumsum <= p`。
  `[.6,.3,.1], p=.7` 当前仅保留第一个 token，包含 crossing token 的约定会保留前两个。
  这需要契约与引擎一致性检查；它本身不能证明历史 pilot 的两端协议不同。
- 本次没有静默修改采样器。修订协议必须另建版本并重新校准，旧结果保留。

## 3. 与最近工作直接比较

文献核对截至 2026-10-20；下述预印本不作为已独立复现的结论。

| 工作 | 已覆盖的核心技术 | 对本项目的影响 |
|---|---|---|
| [DiFR, 2025-11](https://arxiv.org/abs/2511.20621) | 共享随机数下的 token margin、混合合法配置校准、likelihood 变体、Gaussian Monte Carlo、vLLM 集成 | “换分数、加噪声、混合校准、接 vLLM”均不是独立创新；须与这些变体直接比较 |
| [TOPLOC, 2025](https://arxiv.org/abs/2501.16007) | provider 激活的 top-k 指纹及压缩编码 | 本地 `token_toploc` 只是受其启发的 token rank，不能署名为 TOPLOC 复现 |
| [RUT, 2025 / ICLR 2026](https://arxiv.org/abs/2506.06975) | 用本地参考样本校准秩统计，不需要共享 RNG | 是无 seed 黑盒对照；本地 token rank 也不是 RUT；计入本地参考采样成本 |
| [Batch-invariant inference, 2025-09](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) | 固定归约策略，消除 batch 变化引起的数值差异 | 必须用来干预机制，验证差异是否随归约固定而消失 |
| [LLM-42, 2026](https://arxiv.org/abs/2601.17768) | 快速解码、固定形状窗口验证与回滚 | “选择性固定形状回放”已有先例；它改造 serving 以实现确定性，和外部审计的信任模型不同 |
| [Hawkeye, MLSys 2026](https://proceedings.mlsys.org/paper_files/paper/2026/hash/e217c271a57c365a246b0ad39e668ba8-Abstract-Conference.html) | Tensor Core matmul 的精确数值建模与 CPU 回放 | 给出具体硬件知识基线；不能再把“GPU 浮点非结合性”当作新发现 |
| [Bit-Exact AI Inference Verification, 2026-06 预印本](https://arxiv.org/abs/2606.00279) | 在测试配置中重建 Tensor Core、SFU、归约与 cast，精确回放模型块 | 更强执行信息条件下的对照；标题中的无性能牺牲不等于 verifier 无成本 |
| [Taming Bitwise Behavior, 2026-09 预印本](https://arxiv.org/abs/2609.11356) | GEMM 归约顺序描述、编译器约束、静态验证与等价类 | kernel 等价类本身已有人做；机会在跨层误差到采样决定的组合及审计成本 |
| [Accurate Models of NVIDIA Tensor Cores, 2025-12 / 2026 更新](https://arxiv.org/abs/2512.07004) | 不同架构的 Tensor Core 数值模型 | 不能默认“都是 FP32 累加，所以结果应该一样”；模型应绑定架构与指令 |
| [Power-of-Two INT8 Conformance, 2026 预印本](https://arxiv.org/abs/2609.00363) | 特定量化约束与跨 kernel 一致性，分析容差检查盲区 | 改变精度契约是另一条路线，不能当作原 BF16 契约已被解决 |

## 4. 建议主攻：考虑候选集变化的数值一致性验证

暂定题目：**Sampling-Aware Verification of BF16 Inference under Execution Variability**。
这是待验证的研究方向，不是已确定的新颖性或已实现方法。

核心问题：能否把“一个大阈值容忍所有差异”，转化为“根据合法执行的数值机制，
判断这次采样结果是否仍可能出现”，从而同时保住误报、检测能力与验证成本？

### 算法：从点参考到允许的采样结果集合

先固定 prefix、共享随机数及一个允许的执行类别。
无过滤或已证明候选集固定时，设 `z_j = logit_j + T*g_j`，有合法区间 `[L_j,U_j]`。
对 claimed token `y`，若存在始终在候选集内的竞争 token `j` 满足 `L_j > U_y`，
则 y 不可能是该类别下的 argmax；若 y 始终被过滤，同样不可能。
区间重叠只是“尚不能排除”，不是证明其实际可达。

top-k/top-p 会使候选集自身变化，必须同时推导候选成员资格或保守分支。
不能只给 logit margin 加一个 epsilon，却忽略过滤这一不连续操作。
相等边界需遵循 tie-breaking 契约。
先解决固定支持集，再扩展 top-k，最后扩展依赖排序和累计概率的 top-p；
初期对无法有效界定的 top-p 情况返回 inconclusive。

真正的算法挑战在于：避免跨层独立区间膨胀、保留误差相关性、剪枝不可能的候选集合，
以及把复算预算投到能消除歧义的位置。可从冻结输入的单层和少量执行路径开始，
不要直接承诺完整 Transformer 的紧致形式化界。
由测量分位数得到的误差包络只支持经验校准；只有有效的数值上界才可称 certificate。

### 系统：机制约束 + 按需回放

建立受控实验中的合法执行目录：GPU/软件版本、GEMM 与 split-K、
attention split-KV 合并、cast 位置、RMSNorm、RoPE/SFU。
利用已有精确模型或归约描述工具，约束可能产生的数值变化。
用一个便宜的参考 pass 和不确定性估计筛查，再对仍有歧义的位置/窗口复算。

**成本陷阱**：单 token 回放仍依赖整个 prefix/KV；没有可信缓存或可复用中间状态，
“只回放少数 token”可能一点也不便宜。测量必须包含 prefix 重算、缓存构建和存储。
不要暗中使用 provider 激活而仍宣称 token-only。

若不知道实际执行类别，合法类别应事先定义。对有效的分类别 p 值可用
`p_union = max_c p_c` 检验有限并集；真实类别被覆盖时它保守有效。
这条统计构造本身不是创新。运行中类别变化要按允许的执行轨迹建模，
不能逐 token 任意挑选最有利的类别，也不能用测试数据事后扩大白名单。
provider 自报的 kernel/trace 不是可信证据；需要独立约束，或把未验证的类别全部计入不确定性。

预期贡献应是三者的组合及实证收益：
候选集感知的判据、硬件机制约束下的低成本复算、在合法执行变化中的独立验证。
要分别超过 DiFR 的高斯模拟、完整多配置回放与简单阈值校准；否则收窄或放弃该主线。

## 5. 系统知识应如何进入实验，而不是停留在术语

| 机制 / 最新技术 | 可做的干预 | 能检验的因果解释 |
|---|---|---|
| Tensor Core 归约与 cast | 固定输入、只改变 split-K / epilogue / kernel shape | 哪一步产生误差，传播到哪个采样边界 |
| [FlashInfer fixed_split_size](https://docs.flashinfer.ai/api/attention.html) | 在支持的 attention backend 固定 split-KV 与合并方式 | 误差是否随归约稳定而收敛；不是直接给当前 FLASH_ATTN backend 换一个开关 |
| [vLLM batch invariance](https://docs.vllm.ai/en/latest/features/batch_invariance/) | 作为同硬件/版本的执行对照，计入开销 | batch 引起的差异能消除多少；文档保证不外推为跨 GPU 一致 |
| [Sarathi-Serve, OSDI 2024](https://www.usenix.org/conference/osdi24/presentation/agrawal) | 控制 chunked prefill 与预算，保持其他因素尽量一致 | 已有 13→4 次混合迭代现象是否来自预期调度机制 |
| [FlashAttention-4, 2026-03](https://arxiv.org/abs/2603.05451) | 认识 Blackwell 的 TMEM、异步执行、softmax/exp 瓶颈；只在对应硬件检验 | timing 模型应反映实际资源，不能直接套用“字节数/带宽”或搬到 L40S |

clock 适合作为辅助信号或另一篇机制论文。目前没有 DRAM/L2/occupancy 硬件计数器，
预算改变也可能同时改变编译形状；已有 timing 结果不够证明纯内存或纯调度因果。
API 并发 C 不等于活跃 decode batch B。真实 NF4/INT8 在现有 eager pilot 还更慢，
所以不能假设量化一定加速。

若继续 clock 路线，值得研究的是在负载、缓存、排队扰动下仍可辨识的干预响应，
以及最坏条件下的 probe 选择；roofline、主动学习或 token-clock 拼接本身不构成创新。
对服务端主动填充延迟的失效边界必须保留。

## 6. 下一轮只先做三组决定性实验

1. **协议与封顶归因。** 同一份 logits，在 float32/float64、过滤约定、共享 RNG 间逐项对齐；
   对真实 BF16 封顶位置记录未过滤 logits、两端候选集、top-k 边界、top-p 累计概率及 margin。
   若多数封顶由本地协议差异解释，先修复并重跑基线，不能将修复包装为硬件创新。
2. **单机制干预。** 冻结 prefix 和层输入，比较固定/变化的 GEMM shape、归约、attention split。
   用 batch-invariant 路径及可用的精确模型验证归因。
   若封顶与实际硬件机制关联很弱，则停止扩展昂贵的全栈数值建模。
3. **独立的能力—成本比较。** 在新语料、独立校准/测试和匹配的真实量化条件下，
   比较 DiFR margin/likelihood/Monte Carlo、混合校准、多配置完整回放与候选方法。
   RUT 单列无 seed 条件；TOPLOC/Activation-DiFR 单列提供激活条件。
   LLM-42 与确定性 kernels 是执行对照，不混入同一个恶意 provider 检测排行榜。

主指标：每个合法条件的 FPR 及置信上界、NF4/INT8 TPR、拒判率、每次审计 GPU 时间、
prefix/参考采样/校准成本和额外状态。冻结统计量和阈值后评估；多条件报告考虑同时置信覆盖。
若目标是 95% 上界低于 5%，64 条远不适合做稳健的跨条件论证，
应按目标误差界、预期错误数和多重比较预先计算样本量；token 不是独立实验单位。

应有消融：仅变换分数 → 加候选集建模 → 加机制约束 → 加选择性回放。
如果只降低 BF16 分数，同时也抹掉 INT8 信号，方向失败。
如果精确回放开销接近全量重算，系统贡献失败。
如果只有人工挑选的条件有效，不能称部署通用。

论文应围绕一个待验证命题展开：在明确的执行与采样契约内，利用数值机制处理采样边界，
是否能以更低的复算成本获得受控误报下更强的检测。当前证据尚未证明这个命题。
