# 评估说明

本文档记录当前评估数据、指标定义、真实报告和失败分析。面试或 README 中的数字必须来自 `reports/` 下的可复现报告。

## 数据集构造

默认评测集是 `data/eval/sample_eval.jsonl`，共 60 条样本。

| 维度 | 数量 |
| --- | ---: |
| 总样本 | 60 |
| 正例 | 52 |
| 负例 | 8 |
| 中文问题 | 33 |
| 英文问题 | 27 |
| BYD 中文公开文档相关 | 31 |
| SEC filings 相关 | 27 |

按类别分布：

| 类别 | 数量 |
| --- | ---: |
| exact_fact | 14 |
| summary | 14 |
| policy | 10 |
| sec_filing | 6 |
| comparison | 4 |
| risk | 4 |
| negative | 8 |

按难度分布：

| 难度 | 数量 |
| --- | ---: |
| easy | 20 |
| medium | 28 |
| hard | 12 |

每条样本包含：

- `id`：稳定 case id。
- `question`：用户问题。
- `ground_truth`：人工写出的参考答案。
- `expected_sources`：期望命中的源文档路径。
- `expected_answer_keywords`：答案诊断用关键词。
- `expected_pages`：可选，期望命中的 PDF 页码，按 source 路径分组。
- `expected_chunk_keywords`：可选，证据 chunk 中应出现的关键短语。
- `evidence_notes`：可选，说明证据标注依据。
- `category`、`difficulty`、`language`、`notes`：分析维度。
- `is_negative`：负例问题标记；负例允许 `expected_sources=[]`。

数据集覆盖：

- BYD 中文年度报告、制度文件、可持续发展和人权政策。
- Apple、Microsoft、NVIDIA、Amazon、Alphabet 的 SEC 10-K。
- 精确事实、摘要、制度条款、风险披露、跨文档对比。
- 不在语料中或时间敏感的负例，例如今日股价、实时产量、私密合同、未披露财务数据。

验证命令：

```bash
python -m scripts.evaluate --validate-dataset --check-source-files
```

## 检索指标定义

检索评估回答的是：正确资料有没有被 Top-K 找回来，以及排得是否靠前。

| 指标 | 含义 |
| --- | --- |
| Hit Rate@K | 正例 Top-K 中是否至少命中一个标注来源 |
| MRR@K | 第一个命中来源的倒数排名均值 |
| Source Recall@K | 标注来源被 Top-K 找回的比例 |
| Precision@K | Top-K 来源中属于标注来源的比例 |
| NDCG@K | 考虑命中排名位置的归一化折损累计增益 |
| Negative Rejection | 负例问题没有返回检索结果的比例 |
| Page Hit Rate@K | 标注了 `expected_pages` 的样本中，Top-K 是否命中期望 source+page |
| Evidence Keyword Recall@K | 标注了 `expected_chunk_keywords` 的样本中，Top-K 证据文本覆盖关键词的比例 |
| Evidence Strict Hit@K | 标注了页码或证据关键词的样本中，同时满足 source 命中、页码命中和关键词全命中的比例 |

运行命令：

```bash
python -m scripts.evaluate --variant dense --top-k 5
python -m scripts.evaluate --variant hybrid --top-k 5
python -m scripts.evaluate --variant hybrid-rerank --top-k 5
python -m scripts.evaluate --variant hybrid-rerank-parent --top-k 5
python -m scripts.evaluate --compare --top-k 5
```

`full` 变体还会启用 Query Rewrite/HyDE，通常需要可用的 chat model 凭据：

```bash
python -m scripts.evaluate --variant full --top-k 5
```

## 当前检索报告

最新 runtime-aligned hybrid-rerank-parent 报告：`reports/retrieval-20260623-185716.json`

数据集：`data/eval/sample_eval.jsonl`

Top-K：5

生成时间：2026-06-23 18:57:16 Asia/Shanghai

有效索引路径：`data/chroma`、`data/chroma/bm25_corpus.jsonl`、`data/chroma/parent_corpus.jsonl`

| metric | value |
| --- | ---: |
| hit_rate@5 | 1.000 |
| mrr@5 | 0.828 |
| source_recall@5 | 0.966 |
| precision@5 | 0.610 |
| ndcg@5 | 0.874 |
| negative_rejection | 1.000 |
| page_hit_rate@5 | 1.000 |
| evidence_keyword_recall@5 | 0.752 |
| evidence_strict_hit@5 | 0.500 |

该报告使用 `scripts.evaluate --variant hybrid-rerank-parent --top-k 5` 生成，配置快照中的 effective paths 与默认 `/ask` 运行路径一致。source-level 指标回答“是否找到了正确文件”；page/chunk/evidence 指标更严格，回答“是否找到了能支撑答案的具体证据”。`evidence_strict_hit@5=0.500` 仍低于 broad source hit rate，这是后续最有价值的改进信号。

历史报告 `reports/retrieval-20260616-132031.json`、`reports/retrieval-20260616-125153.json` 和 `reports/retrieval-comparison-phase6-local.json` 可用于说明问题如何被发现和指标如何演进，但不再作为当前 runtime/evaluation 对齐后的最新质量结论。

最新分组对比报告：`reports/retrieval-comparison-20260623-190622.json`

| variant | hit_rate@5 | mrr@5 | source_recall@5 | precision@5 | ndcg@5 | negative_rejection |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dense | 0.962 | 0.836 | 0.948 | 0.662 | 0.872 | 1.000 |
| hybrid | 1.000 | 0.856 | 0.966 | 0.596 | 0.899 | 1.000 |
| hybrid-rerank | 1.000 | 0.806 | 0.966 | 0.587 | 0.860 | 1.000 |
| hybrid-rerank-parent | 1.000 | 0.828 | 0.966 | 0.610 | 0.874 | 1.000 |

分组观察：

- 语言维度：dense 的中文 hit_rate@5 为 0.929、英文为 1.000；hybrid 系列中英文均为 1.000。
- 负例维度：四个本地变体 negative_rejection 均为 1.000，拒答来自 confidence gating 和显式不可回答规则。
- 难度维度：hybrid-rerank-parent 在 hard 组 hit_rate@5 为 1.000，source_recall@5 为 0.889。
- 风险类问题：四个本地变体 risk 组 hit_rate@5 均为 1.000，但这仍是 source-level；需要结合 evidence metrics 看证据质量。

结论要谨慎：

- 负例拒答已经从旧报告的 0.000 提升到 1.000，但这依赖显式规则和 confidence gating，不应包装成纯检索能力。
- source-level hit rate 仍然较高，但 strict evidence metrics 暴露了“命中文档但未命中证据页/关键词”的问题。
- dense、hybrid、reranker、parent-child 的优劣需要继续看分组和 strict evidence 指标，而不能只看整体 hit rate。

## 答案级指标定义

答案级评估回答的是：模型有没有基于检索上下文生成忠实、相关、完整的答案。

本地确定性诊断：

| 指标 | 含义 |
| --- | --- |
| keyword_coverage | 生成答案覆盖 `expected_answer_keywords` 的比例 |
| citation_present | 是否返回引用 |
| answer_non_empty | 答案是否非空 |
| unknown_answer | 是否识别为无法回答 |
| negative_case_pass | 负例是否正确拒答且无引用 |
| passed | 当前启发式综合通过 |

RAGAS 指标：

| 指标 | 含义 | 当前目标参考值 |
| --- | --- | ---: |
| faithfulness | 答案是否忠实于上下文 | > 0.85 |
| answer_relevancy | 答案是否切题 | > 0.80 |
| context_precision | 检索上下文是否相关 | > 0.75 |
| context_recall | 上下文是否覆盖标准答案所需信息 | > 0.80 |

运行命令：

```bash
python -m scripts.evaluate_answers --limit 10 --sample mixed --include-categories exact_fact,summary,policy,sec_filing,comparison,risk,negative --language all
```

失败行为：

- 缺少 RAGAS 依赖、judge 凭据、模型调用失败或 embedding 调用失败时，命令会非零退出。
- 失败时保存 `reports/answer-eval-failed-*.json` 部分报告，便于排查生成答案和上下文。

## 当前答案级报告

可引用的代表性 RAGAS 报告：`reports/answer-eval-20260623-191226.json`

该报告覆盖 10 条 mixed 样本，包含 exact_fact、summary、policy、sec_filing、comparison、risk、negative，包含中文和英文，并包含 1 条负例。它使用 runtime-aligned 默认索引路径 `data/chroma` 生成，比旧的 1-case smoke 更适合面试引用，但仍是小样本报告，不能代表完整 60 条数据集的总体答案质量。

| 指标 | 数值 |
| --- | ---: |
| faithfulness | 0.896 |
| answer_relevancy | 0.768 |
| context_precision | 0.694 |
| context_recall | 0.750 |
| local pass_rate | 0.800 |
| local keyword_coverage | 0.542 |
| citation_rate | 0.900 |
| unknown_answer_rate | 0.100 |
| negative_case_pass_rate | 1.000 |

Judge 配置：`deepseek-v4-pro` via `https://api.deepseek.com`，embedding 为本地 `bge-m3`。RAGAS target thresholds 仍作为参考：faithfulness > 0.85、answer_relevancy > 0.80、context_precision > 0.75、context_recall > 0.80。本次报告 faithfulness 达标；answer relevancy、context precision、context recall 仍低于目标。

`reports/answer-eval-20260623-134526.json` 是索引路径错配期间生成的过期报告，`pass_rate=0.0`、`unknown_answer_rate=1.0`，不能作为当前质量证据引用。

面试中应这样表述：

- 已经实现答案级评估流水线、mixed sampling、RAGAS 集成和本地诊断。
- 当前 10-case mixed 报告支持“比 smoke 更强的答案质量证据”，但样本仍小，不能夸大为总体质量承诺。
- 本地关键词诊断和 RAGAS context 指标提示后续重点应放在证据覆盖和答案完整性上。

## 失败案例与改进方向

### 负例拒答

旧报告中所有变体的 negative_rejection 都是 0.000。Phase A 已加入实时/私密/不可用数据规则和 confidence gating，最新报告中负例拒答率为 1.000。

后续仍需注意：

- 当前规则覆盖了数据集负例，但真实用户问题会更开放。
- 应继续记录 refusal reason，并按真实 query log 扩展规则或模型分类器。
- 负例不能只看“是否没有引用”，还要确认回答没有暗示模型知道实时或私密信息。

### Hybrid 不一定在所有指标上优于 Dense

旧报告中 dense hit_rate@5 高于 hybrid 系列；最新分组对比中 hybrid 系列整体 hit_rate@5 为 1.000，但 precision@5 低于 dense。可能原因：

- 当前标注来源粒度是 source file，dense 已经足够命中目标文件。
- BM25 引入了同公司、相邻年份、类似字段的候选。
- RRF 参数和中文分词仍有调优空间。

改进：

- 把评估粒度从 source file 扩展到 page/chunk。
- 为中文 BM25 增加更合适的分词和领域词典。
- 调整 dense/BM25 candidate 数和 RRF `k`。
- 继续扩展 grouped report，尤其比较 category、difficulty 和 strict evidence 指标，而不是只看整体均值。

### 答案单位和关键词覆盖

本地答案诊断显示 citation_rate 高，但 keyword coverage 不高。典型问题是报告里单位为千元，ground truth 写成人民币百万元，模型可能直接引用千元数字。

改进：

- 在 prompt 中要求保留原文单位并在必要时换算。
- 对财务问题增加单位规范化后处理。
- 在评估里增加数值等价判断，而不只做字符串关键词匹配。

### RAGAS 样本量仍偏小

当前 RAGAS 报告已经从 1 条 smoke 扩展到 10 条 mixed 样本，但仍不足以覆盖全部 60 条。

- 继续扩大到 30 条或全量 60 条。
- 固定 judge model、temperature 和 embedding。
- 保存每次报告到 `reports/answer-eval-YYYYMMDD-HHMMSS.json`，不要覆盖。

## 下一步优化计划

1. 加入拒答阈值：综合 dense score、BM25/RRF rank、reranker score 和来源数量。
2. 扩展 evidence 标注覆盖面：从 18 条扩展到 40+ 条，减少 strict metrics 方差。
3. 扩展分组报告分析：把 group-level 失败 case 接入文档和前端 drilldown。
4. 调中文 BM25：加入中文分词、术语词典和停用词。
5. 增加 full 变体正式报告：记录 Query Rewrite/HyDE 对各类问题的收益和失败。
6. 扩大 RAGAS 样本：目标跑完整 60 条或代表性 30 条分层样本。
7. 加入答案数值评估：对金额、百分比、单位做 normalize 后比较。
8. 把报告接入 CI 或定期任务，形成趋势而不是单次截图。
