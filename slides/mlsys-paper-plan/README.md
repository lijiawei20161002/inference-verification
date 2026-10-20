# Inference Verification：实验进展与 MLSys 论文主线

2026-10-20，基于本地提交 `57ceb5e`，最新归档测量日期为 2026-10-19。

- [可编辑 PowerPoint：8 页](inference-verification-mlsys.pptx)
- [PDF：8 页](../../output/pdf/inference-verification-mlsys.pdf)
- [全部页面预览](qa/contact-sheet.png)
- [逐页讲稿、来源和状态](qa/slide-index.json)
- [数据来源与 SHA-256](qa/evidence.json)

## 技术复审补充（同日更新）

后续讨论明确：**不以低成本作为核心创新**。
10 页技术补充保留为上一轮方案；新的候选方向见
[整段执行一致性、因果故障定位与可验证边界](../../planning/NON_COST_RESEARCH_DIRECTIONS.md)。

- [BF16 与技术创新：10 页可编辑 PowerPoint](inference-verification-technical-novelty.pptx)
- [技术补充 PDF：10 页](../../output/pdf/inference-verification-technical-novelty.pdf)
- [技术评审与文献对照](../../planning/BF16_TECHNICAL_NOVELTY.md)
- [新增离线分数诊断](qa/technical/score-tail-diagnostic.json)
- [技术补充讲稿与来源](qa/technical/slide-index.json)

**更新判断：BF16 尚未解决，核心技术创新尚未建立。**
DiFR 已包含合法执行配置混合校准与 vLLM 集成；2026 年工作进一步覆盖确定性和精确回放。
因此原 8 页中的框架不能作为独立的新颖性主张，其第 6–7 页已增加修正说明。
技术补充提出“候选集感知的数值一致性 + 按需回放”作为待验证主线，未声称已实现或胜过基线。
本次新增分析发现，BF16 评估集 8,192 个 token 中 49 个封顶分数贡献总分的 98.4%；
这只是过滤边界的归因线索，不是误报率或方法改进结果。

## 当前判断

已有真实权重量化、固定 token 跨 GPU 回放、生产 serving 引擎和调度追踪的机制证据。
目前可以撰写问题与机制部分；算法、独立校准下的部署误报率和端到端成本尚未闭环。
这是研究与写作判断，不是录用预测。

真实量化实验的 token-DiFR 在 128-token 预算下检出 NF4 34/64、INT8 5/64，
BF16 误报 0/64（95% Wilson 上界仍为 5.66%）。DiFR 需要共享 RNG 协议。
这次 Qwen3-1.7B 实验与旧的 Qwen3-0.6B signed-audit NF4 0/10 实验条件不同。

vLLM 已完成 600 请求 pilot 和 600 请求同机同输入时间复测。
调度诊断另有 26 次运行、98 个测量请求。它们是机制测量，不能作为部署 FPR/TPR。
不同 campaign 不能合并成同请求的 token + clock 联合评估。

## 页面

| 页 | 内容 | 状态 |
|---|---|---|
| 1 | 当前进展与建议论文主线 | 测量 + 判断 |
| 2 | 六条已归档证据链 | 已测量 |
| 3 | 真实 NF4 / INT8 的检出率和不确定性 | 已测量 |
| 4 | vLLM 执行模式、并发与缓存控制 | 已测量 |
| 5 | 跨 GPU 诚实漂移与调度预算干预 | 已测量 |
| 6 | 核心贡献和校准架构 | 建议；部分设计待验证 |
| 7 | 10 页正文结构、主图和旧稿重写点 | 写作建议 |
| 8 | P0-P3 补实验及验收标准 | 尚未运行 |

图表数字在构建时直接读取归档 JSON，PPTX 中的文字、图表与形状均可编辑。
每页含 speaker notes，解释样本单位、限制和建议的讲述方式。
现有 `paper/paper.tex` 未修改；它仍是 8 月的单 H100 实验汇总。

## MLSys 来源

核对的是 [MLSys 2027 官方 CFP](https://mlsys.org/Conferences/2027/CallForResearchPapers)，查询日期 2026-10-20。
正文上限 10 页双栏，不含参考文献；附录单独提交；参考文献后要求 AI Usage Disclosure。
论文相关工作与创新性仍需作者单独审查，本 deck 不替代完整文献综述。

CFP 明文截止时间为 2026-10-30 20:00 UTC，即北京时间 10 月 31 日 04:00。
官网首页同时显示 12:00 PDT，比该 UTC 时间早一小时；实际提交前须核对 OpenReview，
不要把最后一小时当作已确定的余量。

本次 AI 协助包括归档检查、写作框架、图表脚本与 slides，后续论文披露应如实覆盖这些使用方式。
技术补充还包含文献检索、离线分数诊断与算法研究建议。

## 复现与检查

可复用诊断代码已整理到 [experiments/numerical](../../experiments/numerical/README.md)，
包含归档校验、事件清单导出与采样边界检查。本目录的分析脚本保留为生成 slides
派生 JSON 的兼容入口，导入时不再读取数据或写文件。
当前英文实施计划见 [NEXT_EXPERIMENTS.md](../../planning/NEXT_EXPERIMENTS.md)；
slides 保留制作时的研究方案，不替代当前计划。

需要 `python-pptx`、`reportlab`，并使用脚本中列出的 macOS Heiti SC / Arial 字体。
本机可用环境：

```sh
cd /Users/lijiawei/Desktop/inference-verification
/Users/lijiawei/Desktop/.venv/bin/python slides/mlsys-paper-plan/build_deck.py
/Users/lijiawei/Desktop/.venv/bin/python slides/mlsys-paper-plan/analyze_score_tails.py
/Users/lijiawei/Desktop/.venv/bin/python slides/mlsys-paper-plan/build_technical_deck.py
pdftoppm -scale-to 1440 -png \
  output/pdf/inference-verification-mlsys.pdf \
  slides/mlsys-paper-plan/qa/slide
```

PDF 与 PPTX 共用布局坐标；PDF 使用嵌入字体和矢量形状。
PDF 已逐页渲染检查；未通过 PowerPoint / Keynote 原生渲染器验证字体布局。
打开 PPTX 的系统如缺少 Heiti SC，可能发生字体替换。

此次运行了四项离线结果复核，无 GPU 作业、模型下载或推理 API 调用：

```sh
python3 -m experiments.analyze_cross_gpu_replay --check
python3 -m experiments.analyze_real_quantization \
  docs/results/real_quantization_l40s_20261018/pilot --output /tmp/quantization-new.json
python3 -m experiments.analyze_vllm_serving_campaign \
  docs/results/vllm_serving_l40s_20261019 --output /tmp/serving-new.json
python3 -m experiments.analyze_scheduling_campaign \
  docs/results/scheduling_l40s_20261019 --output /tmp/scheduling-new.json
```

上述 output 应使用不存在的新路径。此次复核输出及布局检查保存在 `qa/`。
