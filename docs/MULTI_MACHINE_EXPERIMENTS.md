# 推理可信性双通道实验与多机运行指南

[项目首页](../README.md) · [实验入口](../experiments/README.md) · [启动器](../scripts/run_research_job.py)

研究问题是：外部审计者能否利用 **返回的 token 与响应时间**，判断服务商的推理行为是否符合声明？token 通道检验输出与模型、采样规则的一致性；时间通道检验执行过程的可观测特征。主结果比较两个通道各自的边界，以及在相同整体误报率下联合检测增加的证据。成本只在最后报告。

**可以多机器并行。** 每台机器拉取同一个 Git commit，领取独立任务，把结果写入自己的目录。当前任务属于独立实验并行，不需要跨机器通信或分布式训练。Token 任务可按 GPU 并行运行不同模型。一次干净时序基线测量期间，同一主机只运行一个实验，以控制 CPU、PCIe、功耗和散热干扰；这是保守的实验设计，不是双卡不能测时间。受控干扰另用 `clock-interference` 测量。

## 当前可运行范围

本轮分成两个阶段，避免把已有微基准误当成已经完成的联合验证系统。

| 阶段 | 内容 | 当前状态 |
|---|---|---|
| A | token 正常数值变化；上下文长度、服务栈、batch、真实 NF4 对时间的影响；正常 speculative decoding 的输出分块 | 本文启动器可运行已有实验 |
| B | 同一请求的 token 与客户端时间联合采集、跨 GPU 回放、独立校准、固定误报率的联合检测、主动规避 | 按下文实验设计继续实现，当前启动器不产生这些结论 |

阶段 A 主要用于复现、寻找机制和确定实验条件。它不是部署误报率验证。现有 HF/CUDA graph 实验也不是 vLLM 的端到端测试。

**2026-10-16 实测进展：** 单台双 H100 主机的 8 个 baseline 任务已完成，包括修复内存问题后完成的 84/84 个 batching 网格点及第二张 GPU 的受控干扰实验。见[结果、解释与限制](results/h100_campaign_20261016.md)。这批结果尚未完成 A100/L40S 迁移、跨日稳定性或阶段 B 的联合验证；16 次正常测试无告警也不足以证明 0.5% 的部署误报率。

**2026-10-17 实测进展：** A100-SXM4-80GB 的 `clock-window`、`clock-stability`、`token-shape`、`clock-floor` 四个 baseline 任务全部完成，队列于 07:20:02 UTC 结束，耗时 69 分 19 秒。时序覆盖分别为 6/6、6/6 和 48/48 个网格点，token 实验为 80 × 256 个位置。长短上下文差为 61.863 ms，紧接着复测为 61.848 ms。见 [A100 结果与归档](results/a100_campaign_20261017.md)。本轮完成阶段 A 的 A100 独立基线；同一 token 池的 H100/A100 配对回放、L40S、跨日稳定性及阶段 B 仍待完成。

**2026-10-18 配对回放进展：** 已将 H100 冻结的同一份 80 × 256 token 证据
在 A100 上完成四种调度、四个检测器的回放，两台 GPU 的同形状重复对照均精确一致。
batch-1 eager 的三个 token 检测器区间均包含零，但 sequential 的 token-DiFR
偏移为 +0.04596 d′，90% 配对序列 bootstrap 区间为 [+0.02836, +0.06414]。
见 [配对结果、原始数组与核验命令](results/cross_gpu_replay_20261018.md)。本次同时
改变 GPU 和驱动，只完成一个正常池的配对比较，不构成部署误报率或阶段 B 联合检测验证。

## 机器分工

| 机器标签示例 | 配置 | 首批任务 |
|---|---|---|
| `h100-sxm-01` | 2 × H100 SXM 80GB，至少 32 vCPU、256GB RAM、1TB 持久化 SSD | 全部任务；阶段 B 再把服务端和 token 重算放到不同 GPU |
| `a100-80-01` | 1 × A100 80GB，至少 16 vCPU、128GB RAM | token 正常变化、context window、稳定性、floor |
| `l40s-01` | 1 × L40S 48GB，至少 16 vCPU、64GB RAM | token 正常变化、context window、稳定性 |
| `client-01` | 8 vCPU、16GB RAM，无需 GPU | 阶段 B 的独立网络客户端 |

每个 token 任务只暴露一张 GPU，用 `--model` 选择模型；两张卡可分别执行完整任务，分别保存结果，不混合不同模型的统计样本。干净时序任务仍只用一张卡。`clock-interference --gpu 0 --load-gpu 1` 在 GPU 0 测量，在 GPU 1 施加固定负载。双卡主机不会自动变成服务端与验证端分离的系统。

选择按需、整卡独占实例，记录 PCIe 或 SXM 的确切形态。同一区域完成首轮后，再把跨区域网络作为单独因素。不同机型的正常时间基线分别建立；跨硬件迁移是要检验的假设。

Lambda 列有双卡 H100 SXM 实例；Runpod 列有 H100 SXM、A100 80GB 和 L40S 48GB，实际库存以控制台为准。[Lambda 机型](https://docs.lambda.ai/on-demand-cloud/overview/) · [Runpod 机型](https://www.runpod.io/gpu-models)

## 每台机器的部署

### 拉取同一个版本

先把本文、启动器和测试一同提交并推到 GitHub，再把选定的完整 commit SHA 发给所有机器。下面用 `/workspace` 举例；在 Lambda 上可换成实际持久化磁盘路径。

```bash
cd /workspace
git clone https://github.com/lijiawei20161002/inference-verification.git
cd inference-verification
git checkout --detach YOUR_FULL_COMMIT_SHA
git status --short
```

`git status --short` 应为空。运行中不要更新这一轮的代码或依赖。启动器只执行已提交的 `HEAD`，有本地未提交或未跟踪的改动会拒绝启动；`runs/` 已在 `.gitignore` 中。

### 安装并检查环境

优先在各机使用同一镜像及其 digest，统一 Python 和 CUDA 软件栈。可从 Python 3.11、PyTorch 2.8.0 + CUDA 12.8 的环境开始；下面的 Transformers 和 bitsandbytes 版本来自仓库已有的 [H100 协议实验](results/audit_protocol_h100.md)。旧时序脚本仍需在每种 GPU 上通过 smoke 验证。

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -e '.[gpu,service,audit,test]' \
  'transformers==4.57.6' 'bitsandbytes==0.50.2' scipy
python -m pip check
python -c 'import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
nvidia-smi
python -m pytest tests/ -q
```

如果镜像不是所选 PyTorch/CUDA 版本，先统一环境再做跨硬件比较。不要把换硬件和换软件栈混成一个变量。阶段 B 的 vLLM 使用独立环境，另行固定版本。

模型缓存和运行结果都应落在持久化磁盘。下面的命令在每个新 shell 中执行：

```bash
export HF_HOME="$PWD/runs/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
mkdir -p runs/campaign-lock
```

Runpod 的容器盘、Pod volume 和 network volume 生命周期不同；删除实例前，先确认结果已经复制到独立存储。[Runpod 存储说明](https://docs.runpod.io/pods/storage/types)

### 固定模型版本

在首台机器上下载一次模型，之后冻结并分发 Hub 缓存。不能让每台机器各自随时解析最新 `main`。

```bash
hf download Qwen/Qwen3-0.6B
hf download Qwen/Qwen3-1.7B
hf download Qwen/Qwen3-4B
python - <<'PY'
import json, os, re
from pathlib import Path
cache = Path(os.environ['HF_HUB_CACHE'])
models = ['Qwen/Qwen3-0.6B', 'Qwen/Qwen3-1.7B', 'Qwen/Qwen3-4B']
lock = {}
for model in models:
    revision = (cache / ('models--' + model.replace('/', '--')) / 'refs/main').read_text().strip()
    assert re.fullmatch(r'[0-9a-f]{40}', revision), (model, revision)
    lock[model] = revision
with Path('runs/campaign-lock/model-refs.json').open('x') as f:
    json.dump(lock, f, indent=2)
print(json.dumps(lock, indent=2))
PY
python -m pip list --format=json > runs/campaign-lock/packages.json
```

把首台机器的 `runs/huggingface/hub/` 和 `runs/campaign-lock/` 复制到其他机器的相同相对目录。例如下面的 `a100-host` 是已配置的 SSH 别名：

```bash
ssh a100-host 'mkdir -p /workspace/inference-verification/runs/huggingface/hub /workspace/inference-verification/runs/campaign-lock'
rsync -a runs/huggingface/hub/ a100-host:/workspace/inference-verification/runs/huggingface/hub/
rsync -a runs/campaign-lock/ a100-host:/workspace/inference-verification/runs/campaign-lock/
```

同样复制到 L40S。这里仅分发 `hub/`，无需分发登录凭据。缓存使用相对符号链接，`rsync -a` 会保留结构。首次复制完成后，在所有机器上开启离线模式：

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

启动器在 baseline 运行前后核对所需模型的缓存引用与锁文件。若缓存缺失或不同，先修复缓存；不要在一轮实验中更新锁文件。[Hugging Face 下载与 revision 说明](https://huggingface.co/docs/huggingface_hub/guides/download)

## 任务目录

| 任务名 | 研究用途 | 应如何解释 |
|---|---|---|
| `token-shape` | 相同模型与返回 token，在 batch、attention kernel、逐 token 回放变化下的分数偏移 | 使用 `clean` arm，关闭该实验的人工参考噪声；不是跨 GPU 的同一 capture 回放 |
| `clock-window` | 有效上下文长度从 512 到 32768 时，长短上下文时间差如何变化 | GPU 上的有效长度微基准；还不是对同一真实长 prompt 截断后的网络审计 |
| `clock-stability` | 重新测量正常上下文长度曲线，观察跨进程和时段变化 | 需在不同时间重新启动；重复请求不是新的独立文本样本 |
| `clock-floor` | 0.6B、1.7B、4B 及真实 NF4，比较 eager 与 CUDA graph 的时间曲线 | 区分模型、量化和栈的影响，不把绝对速度直接解释为违规 |
| `clock-batching` | batch 1 至 64 对时间信号的影响 | 先在 H100 上运行；旧脚本的内存阈值未按机型自适应，OOM/skip 要保留 |
| `clock-specdec` | speculative decoding 的分块输出如何影响逐 token 时间统计 | 测量设备端 block 时间，再构造 block 内零间隔；不是实测网络到达时间 |
| `clock-interference` | GPU 0 的时序判别在 GPU 1 有负载时怎样变化 | 固定 BF16 GEMM 干扰；独立空闲校准、配对空闲/负载测试；本地有效上下文微基准 pilot |

每个时序任务的 `itl_ms` 来自主机计时加 CUDA 同步，包含该测量方式的开销。不要把它称作客户端网络 ITL 或纯 kernel 时间。`activation_difr` 需要内部激活证据，单独作为有额外可见性的对照，不纳入仅凭公开 token 和时间的主结果。

`exp_clock_channel_gpu.itl_samples` 的 prompt prefill 在计时区间外按 256 token 分块，模型支持时仅生成最后一个位置的 logits，并在每个 repetition 后释放 cache 和 CUDA graph。这样避免完整 `[batch, context, vocab]` logits 和前后两轮 cache 同时驻留导致的 OOM；被测 decode 仍保留原来的完整 batch、上下文、步数及重复数。结果 JSON 的 `prefill` 字段记录此设置。旧版 OOM 结果应保留，新版重新运行完整 batching 网格并单独记录 commit，不将不同 prefill 实现的 cell 混成一轮结果。

`clock-batching` 启动器固定 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，减少动态 KV cache 增长时的显存碎片；manifest 的 `settings` 及结果的 `cuda_allocator_config` 均记录此设置。它不减少实际 cache 大小，也不将两个 GPU 的显存合并。

`--size smoke` 减少可配置的 token/step/repetition 数，用于检查兼容性；`--size baseline` 使用更大的已有基线配置。两者都不是阶段 B 的正式样本量。特别是 `clock-specdec` 的旧入口有固定的 block/repetition 数，smoke 不会缩短它。

## 开始运行

### 每台机器先做 smoke

修改 `--node` 为该机器唯一标签；`--gpu` 是该进程能看到的 GPU 索引，也可以是完整 GPU UUID。不同机器上都写 `--gpu 0` 没有冲突。

```bash
python scripts/run_research_job.py --node h100-sxm-01 --gpu 0 \
  --job token-shape --size smoke --dry-run
python scripts/run_research_job.py --node h100-sxm-01 --gpu 0 \
  --job token-shape --size smoke --model-lock runs/campaign-lock/model-refs.json
python scripts/run_research_job.py --node h100-sxm-01 --gpu 0 \
  --job clock-window --size smoke --model-lock runs/campaign-lock/model-refs.json
```

启动器打印运行目录和日志路径。检查日志后再扩大：

```bash
tail -f runs/campaign/h100-sxm-01/ACTUAL_RUN_DIRECTORY/run.log
```

### 三台机器同时执行

在三个独立的 SSH 会话或 `tmux` 会话中运行以下队列。每个队列内部串行，队列之间并行。新终端先激活 `.venv` 并设置上一节的缓存与离线环境变量。

H100：

```bash
python scripts/run_research_job.py --node h100-sxm-01 --gpu 0 \
  --job token-shape --model Qwen/Qwen3-1.7B --size baseline \
  --model-lock runs/campaign-lock/model-refs.json &
token_pid0=$!
python scripts/run_research_job.py --node h100-sxm-01 --gpu 1 \
  --job token-shape --model Qwen/Qwen3-4B --size baseline \
  --model-lock runs/campaign-lock/model-refs.json &
token_pid1=$!
wait "$token_pid0" && wait "$token_pid1" &&
for job in clock-window clock-stability clock-floor clock-specdec clock-batching; do
  python scripts/run_research_job.py --node h100-sxm-01 --gpu 0 \
    --job "$job" --size baseline --model-lock runs/campaign-lock/model-refs.json || break
done
```

A100：

```bash
for job in token-shape clock-window clock-stability clock-floor; do
  python scripts/run_research_job.py --node a100-80-01 --gpu 0 \
    --job "$job" --size baseline --model-lock runs/campaign-lock/model-refs.json || break
done
```

L40S：

```bash
for job in token-shape clock-window clock-stability; do
  python scripts/run_research_job.py --node l40s-01 --gpu 0 \
    --job "$job" --size baseline --model-lock runs/campaign-lock/model-refs.json || break
done
```

稳定性至少安排早晚及另一天的新进程复测，沿用同一代码、软件和模型版本。每次启动都会创建新目录，不会覆盖上一次。

启动器使用同用户、同临时目录下的主机锁和 GPU UUID 锁。Token 任务共享主机锁，各自独占一张 GPU；所有时序任务独占主机锁。干扰任务还锁定负载 GPU，负载仅由该任务管理。锁不能检测其他用户、其他容器或绕过启动器的程序；干净基线测量时仍需确认没有其他主机负载。

### 双卡受控干扰 pilot

先完成干净时序基线，再执行：

```bash
python scripts/run_research_job.py --node h100-sxm-01 --gpu 0 --load-gpu 1 \
  --job clock-interference --size smoke --model-lock runs/campaign-lock/model-refs.json
python scripts/run_research_job.py --node h100-sxm-01 --gpu 0 --load-gpu 1 \
  --job clock-interference --size baseline --model-lock runs/campaign-lock/model-refs.json
```

GPU 1 运行固定 8192×8192 BF16 矩阵乘法。空闲条件暂停负载 kernel，但保留进程与显存。GPU 0 使用 1.7B 模型与 CUDA graph，在每个独立 prefill/decode 周期测量 256 至 32768 的六个有效上下文。测试周期交替采用空闲→负载、负载→空闲的顺序；每对条件使用相同的随机上下文顺序。记录原始 ITL、执行顺序、负载迭代数、GPU 利用率和功耗快照。

只用空闲校准周期确定长短上下文差值的低尾阈值，然后固定阈值比较两种测试条件的告警计数。baseline 有 16 个校准周期、每个条件 16 个测试周期；smoke 各为 2 个。计数单位是周期，不把单个 token 当作独立样本。这些数量及重复文本不足以验证 0.5% 部署误报率，结果仅用于发现同机负载是否改变正常差值或有效上下文偏离的可分性；正式网络与联合检测仍属阶段 B。

## 保存与汇总

每个任务的目录结构如下：

```text
runs/campaign/h100-sxm-01/clock-window-baseline-TIMESTAMP/
  manifest.json       # commit、任务、参数、模型引用、状态、结果 SHA256
  packages.json       # 安装的软件版本，带命令执行状态
  nvidia-before.json  # 测量前的驱动、显存、功耗和时钟等信息
  nvidia-after.json   # 测量后的对应信息
  run.log
  results/            # 本次新生成的结果
  source/             # HEAD 的独立代码副本，旧脚本只会改这个副本
```

`complete` 只说明进程成功且基本结果覆盖检查通过，不代表统计结论成立。`partial` 表示网格有缺失，退出码为 2；`failed` 需要检查日志。被强制终止的任务可能停留在 `running`，不能算完成。启动器会先删除副本中该任务的旧结果，防止将历史结果误收为新测量。

在汇总机上使用 SSH 别名拉取结果，按节点保留目录。排除完整源码副本可以缩小传输量；Git commit 已记录：

```bash
mkdir -p runs/collected
rsync -a --exclude=source/ h100-host:/workspace/inference-verification/runs/campaign/ runs/collected/
rsync -a --exclude=source/ a100-host:/workspace/inference-verification/runs/campaign/ runs/collected/
rsync -a --exclude=source/ l40s-host:/workspace/inference-verification/runs/campaign/ runs/collected/
```

用下面的命令列出状态并检查完整结果的哈希：

```bash
python - <<'PY'
import hashlib, json
from pathlib import Path
for path in sorted(Path('runs/collected').glob('*/*/manifest.json')):
    m = json.loads(path.read_text())
    if m['status'] in ('complete', 'partial'):
        result = path.parent / 'results' / m['output_artifact']
        assert hashlib.sha256(result.read_bytes()).hexdigest() == m['artifact_sha256'], result
    print(m['node'], m['job'], m['size'], m['status'], m['commit'][:12])
PY
```

汇总时先对齐 commit、模型 revision、软件栈、任务参数和观测类型。按硬件、服务栈、日期分别分析；不能把三台机器的 token 或间隔直接拼成一个巨大独立样本池。原有绘图脚本读取固定的历史文件，当前不应直接用它们合并新目录，否则可能混入旧结果。

`runs/` 不进入 Git。收集完的原始数据放在独立持久化存储，审核后的报告、小型数据和绘图脚本再按唯一实验名称提交。无需每台机器各自向同一个 Git 分支提交结果。

## 双通道正式实验设计

### 声明与证据

先冻结每项声明允许的变化范围。例如，声明模型与采样分布是否允许不同 batch、硬件和分布保持的 speculative decoding？如果硬件型号不属于声明，就不能仅因换硬件更快而判违规。

同一次请求应保存 prompt 与 token IDs、模型及 tokenizer revision、采样设置、客户端单调时钟时间戳、SSE 事件边界、每事件 token 数（若可得）、首 token 时间和完成时间。服务端诊断时间只用于机制解释，不能充当外部审计者独立取得的证据。SSE chunk 不等于单个 token；没有可靠 token 边界时按 chunk 分析。

### 检验矩阵

| 问题 | 正常对照 | 偏离条件或操作 | 主要结果 |
|---|---|---|---|
| 两个通道各自能发现什么 | 同一声明下的合规推理 | 替换模型、真实 NF4/INT8、改变 sampler、上下文截断 | 两通道检出率与盲区矩阵 |
| 时间有没有额外证据 | 相同请求集和冻结阈值 | token 分数相近但执行过程不同的条件 | time 在 token 未检出样本上的额外检出率 |
| 联合检测是否改善可信性 | 相同整体 FPR 预算 | 同一份请求证据计算三个版本 | token-only、time-only、联合检测的对照曲线 |
| 正常变化是否被误判 | 硬件、batch、eager/vLLM、缓存、正常 speculative decoding、负载变化 | 无违规，逐个改变因素 | 每种正常条件的 FPR 和置信区间 |
| 能否主动规避 | 冻结检测器，独立攻击开发集 | context-dependent padding、缓冲成块、混合路由、识别探针 | 固定查询预算下的逃逸率及两通道失效条件 |
| 是否跨模型和规模成立 | Qwen 与另一模型家族 | 先 1.7B/8B，再扩展更大模型 | 跨家族与规模的泛化边界 |

长短 prompt 探针使用随机顺序和配对设计，控制缓存复用，并保存实际执行顺序。加性延迟只有在满足相应共享条件时才会抵消。长短请求受到不同排队、负载或刻意 padding 时，时间差可以向任一方向变化；现有材料中“co-tenancy 不会造成误报”的表述不作为正式实验前提。

时间信号的亮点是补充执行过程证据。服务商可以人为拖慢输出，因此通过时间检测不等于证明指定模型确实运行。联合方法应明确给出可发现的偏离与不可识别的情况。

### 数据划分与统计

开发、校准、测试在完整请求或探针对层面分开；同一长流的不同 token 不能分别进入校准与测试。多机任务的 `node` 只是运行标签，不能代替不同的 prompt ID、采样 seed 或 partition。

正式采集器应先生成固定任务清单，至少包含 `experiment_id`、`condition`、`partition`、`source_id`、`seed`、`replicate` 和 `shard_id`。机器按 shard 领取互不重叠的请求；汇总时检查 ID 重复和遗漏。当前旧脚本有固定 prompt 与 seed，换机器重跑主要测系统差异，不能当成新增独立文本证据。

建议主操作点为整体 FPR ≤ 0.5%，另报告 0.1% 的敏感性分析。联合检测需要联合校准，或明确分配两个通道的误报预算，不能让两个通道各用 0.5% 后直接取 OR。

样本量规划先以每个主要正常条件约 10,000 个独立校准块、10,000 个独立测试块，以及每个偏离条件约 1,000 个测试块为起点。最终数量由 pilot 的变异和目标置信区间确定。这是阶段 B 的采集目标，不能通过增加旧脚本的 token bootstrap 次数实现。报告 FPR/TPR 置信区间、无法判定比例，并按请求或会话层面的相关性处理不确定性。

对比的核心是同样误报约束下的检测能力，以及时间是否提供额外信息。效应量、partial AUC 和机制分析作为解释，不能替代独立测试上的误报与检出率。

## 计算资源记录

最后报告生成与重算的 GPU 时间、端到端耗时、显存、请求数和数据量。启动器的 `elapsed_seconds` 是整个子进程墙钟时间，包含模型加载和预热，不是 GPU 活跃时间；这些分项在正式采集器中分别记录。资源开销作为部署属性，不作为论文的主要贡献。
