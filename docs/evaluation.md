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

报告路径：`reports/retrieval-comparison-phase6-local.json`

数据集：`data/eval/sample_eval.jsonl`

Top-K：5

| variant | cases | hit_rate@5 | mrr@5 | source_recall@5 | precision@5 | ndcg@5 | negative_rejection |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense | 60 | 1.000 | 0.933 | 0.983 | 0.727 | 0.950 | 0.000 |
| hybrid | 60 | 0.981 | 0.923 | 0.948 | 0.654 | 0.932 | 0.000 |
| hybrid-rerank | 60 | 0.981 | 0.933 | 0.948 | 0.654 | 0.939 | 0.000 |
| hybrid-rerank-parent | 60 | 0.981 | 0.942 | 0.948 | 0.654 | 0.946 | 0.000 |

结论要谨慎：

- dense baseline 在当前小数据集上非常强，hit_rate@5 达到 1.000。
- hybrid 没有在 hit_rate 上超过 dense，说明“BM25 一定提升召回”不是当前报告支持的结论。
- reranker 和 parent-child 对排序指标有帮助：MRR 和 NDCG 从 hybrid 到 hybrid-rerank-parent 有提升。
- negative_rejection 为 0.000，是明确缺口。当前检索层只看是否返回结果，还缺少相似度阈值、时间敏感问题识别和答案层拒答策略。

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
python -m scripts.evaluate_answers --limit 10
```

失败行为：

- 缺少 RAGAS 依赖、judge 凭据、模型调用失败或 embedding 调用失败时，命令会非零退出。
- 失败时保存 `reports/answer-eval-failed-*.json` 部分报告，便于排查生成答案和上下文。

## 当前答案级报告

可引用的 RAGAS smoke 报告：`reports/answer-eval-ragas-optimized.json`

注意：该报告只有 1 条样本，只能证明 RAGAS pipeline 跑通，不能代表整体答案质量。

| 指标 | 数值 |
| --- | ---: |
| faithfulness | 1.000 |
| answer_relevancy | 0.968 |
| context_precision | 0.806 |
| context_recall | 1.000 |
| local pass_rate | 0.000 |
| local keyword_coverage | 0.333 |
| citation_rate | 1.000 |

另一个本地诊断报告 `reports/answer-eval-phase7.json` 覆盖 2 条样本，但没有 RAGAS 字段：pass_rate 0.500、average_keyword_coverage 0.417、citation_rate 1.000。

面试中应这样表述：

- 已经实现答案级评估流水线和 RAGAS 集成。
- 当前 RAGAS 报告是小样本 smoke，不足以支持总体质量承诺。
- 本地关键词诊断暴露出单位和数字表述不一致问题，例如模型答“千元”而标准关键词期待“百万元”。

## 失败案例与改进方向

### 负例拒答不足

当前检索报告所有变体的 negative_rejection 都是 0.000。原因是负例问题仍可能在大语料中检索到某些表面相关片段，例如“今天股价”会命中公司名或报告文本。

改进：

- 增加检索置信阈值和 reranker score 阈值。
- 增加时间敏感/实时数据 intent 分类。
- 对负例或低置信问题要求 LLM 明确基于上下文拒答。
- 把 negative_case_pass 纳入回归评估。

### Hybrid 不一定优于 Dense

当前 dense hit_rate@5 高于 hybrid 系列。可能原因：

- 当前标注来源粒度是 source file，dense 已经足够命中目标文件。
- BM25 引入了同公司、相邻年份、类似字段的候选。
- RRF 参数和中文分词仍有调优空间。

改进：

- 把评估粒度从 source file 扩展到 page/chunk。
- 为中文 BM25 增加更合适的分词和领域词典。
- 调整 dense/BM25 candidate 数和 RRF `k`。
- 对不同 query category 分组评估，而不是只看整体均值。

### 答案单位和关键词覆盖

本地答案诊断显示 citation_rate 高，但 keyword coverage 不高。典型问题是报告里单位为千元，ground truth 写成人民币百万元，模型可能直接引用千元数字。

改进：

- 在 prompt 中要求保留原文单位并在必要时换算。
- 对财务问题增加单位规范化后处理。
- 在评估里增加数值等价判断，而不只做字符串关键词匹配。

### RAGAS 样本量不足

当前 RAGAS 报告是 smoke。要作为面试中的质量证明，至少应跑 10-30 条代表性样本，并记录失败样本。

改进：

- 优先跑 mixed category 子集：事实、摘要、制度、SEC、负例。
- 固定 judge model、temperature 和 embedding。
- 保存每次报告到 `reports/answer-eval-YYYYMMDD-HHMMSS.json`，不要覆盖。

## 下一步优化计划

1. 加入拒答阈值：综合 dense score、BM25/RRF rank、reranker score 和来源数量。
2. 扩展评估标注：从 source-level 细化到 page/chunk-level，减少“命中文档但没命中证据”的误判。
3. 分组报告：按 language、category、difficulty 分别输出指标。
4. 调中文 BM25：加入中文分词、术语词典和停用词。
5. 增加 full 变体正式报告：记录 Query Rewrite/HyDE 对各类问题的收益和失败。
6. 扩大 RAGAS 样本：至少跑 10 条，目标跑完整 60 条或代表性分层样本。
7. 加入答案数值评估：对金额、百分比、单位做 normalize 后比较。
8. 把报告接入 CI 或定期任务，形成趋势而不是单次截图。
