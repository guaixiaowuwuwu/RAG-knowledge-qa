# 技术取舍

本文档回答“为什么这样做”和“什么时候不该这样做”。事实依据来自当前代码和报告；没有压测或大规模评估支持的内容只作为工程推断。

## 为什么本地用 Chroma

Chroma 的价值是本地可复现、部署轻、调试快。这个项目的目标是面试可演示和实验可复现，所以默认把向量库持久化到 `data/chroma`，避免要求面试环境额外启动 Milvus、Qdrant 或 Elasticsearch。

代价是 Chroma 不应该被包装成生产规模证明。生产环境通常需要服务化向量库、备份、权限隔离、索引版本、水平扩展和监控。迁移路径应是抽象 vector store adapter，然后用同一套 `scripts.evaluate --compare --top-k 5` 验证迁移前后召回质量。

不适合继续用本地 Chroma 的场景：

- 文档量和并发请求远超单机目录能力。
- 需要多租户权限隔离、索引回滚和在线扩缩容。
- 需要集中化备份、监控、容量规划和高可用。

## 为什么 BM25 + Dense

Dense retrieval 擅长语义相似，适合用户问题和文档表达不完全一致的场景。BM25 擅长精确词匹配，适合制度名、财务科目、公司名、产品名、Form 10-K 章节、数字和英文缩写。

企业知识库经常同时包含两类问题：一类问“这项制度大概规定了什么”，另一类问“募集资金专户存储由哪些主体签协议”。前者偏语义，后者偏关键词。混合检索可以提高候选覆盖面。

当前报告也提醒了一个风险：在 `reports/retrieval-comparison-phase6-local.json` 的 60 条样本上，dense hit rate@5 是 1.000，hybrid 是 0.981。也就是说 BM25 不是无条件提升，它可能把额外候选带进排序链路，影响 Top-K。更稳妥的说法是：BM25 提供召回互补，需要用数据集验证权重、融合和排序。

不适合或应谨慎使用 BM25 的场景：

- 语料经过强规范化，问题主要是同义改写，关键词稀疏价值不大。
- 查询包含大量噪声词，BM25 把噪声词匹配排得过高。
- 没有合适的分词、停用词和字段权重配置，尤其是中文领域语料。

## 为什么用 RRF

RRF 只依赖各检索器内部排名，不要求 dense score 和 BM25 score 在同一尺度上可比。对本项目这种 Chroma 向量相似度和 BM25 分数混合的链路，RRF 是简单、稳定、可解释的融合方法。

代价是 RRF 不理解分数置信度，也不会自动学习某个领域里 dense 或 BM25 更可靠。它是一个稳健 baseline，不是最优排序器。如果有大量点击日志或人工标注，可以进一步训练 learning-to-rank 或按 query intent 动态加权。

不适合只用 RRF 的场景：

- 需要充分利用 calibrated score 或业务权重。
- 有足够训练数据支撑监督排序。
- 不同字段、文档类型、权限、时间衰减需要复杂策略融合。

## 为什么用 Reranker

Dense/BM25/RRF 更像“召回候选”，reranker 更像“精排”。BGE reranker 会重新看 query 与候选片段的匹配关系，通常能把更相关片段提前，减少 LLM 读到错误上下文的概率。

当前报告中，hybrid 的 MRR@5 是 0.923，hybrid-rerank 是 0.933，hybrid-rerank-parent 是 0.942；NDCG@5 也从 0.932 提升到 0.946。这个结果支持“reranker/parent-child 对排序有帮助”，但不支持夸大成所有指标显著提升。

代价：

- 首次加载本地模型慢。
- 每次问答要对候选做额外推理，增加延迟。
- 候选数越多成本越高。
- reranker 可能把关键词精确但语义短的证据排低。

什么时候不用或降级：

- 延迟预算极低，且 Top-K dense/BM25 已经足够可靠。
- 本地机器无法承载 reranker。
- reranker 服务不可用；此时应退回 RRF 排序并在 trace 中标记 degraded。

## 为什么 Parent-Child Retrieval

小 chunk 适合召回，大 chunk 适合回答。Parent-child retrieval 用 child chunk 做精确匹配，命中后返回 parent chunk 给 LLM，避免答案缺少上下文。

这个策略特别适合年度报告、制度文件和表格附近的解释文字。比如命中某个财务数字所在行后，parent chunk 能提供章节名、年份、单位和上下文，减少模型误读。

代价：

- parent chunk 会增加 prompt 长度和 token 成本。
- parent 粒度过大时会引入不相关信息。
- 需要维护 child-parent 映射和索引版本一致性。

不适合启用的场景：

- 单条知识本身极短，例如 FAQ 或字段型 KB。
- LLM context 很紧张，parent chunk 容易挤掉其他证据。
- 数据权限要求非常细，如果 parent 包含用户无权访问的相邻内容，必须先做权限过滤。

## 为什么 Query Rewrite 和 HyDE

Query Rewrite 用 LLM 把口语化问题改写成更适合检索的表达。HyDE 让 LLM 生成一段假想答案，再用这段文本去检索，适合用户问题很短或文档表达和问题差异较大的情况。

它们的价值是提高召回候选的覆盖面，尤其是跨语言、摘要型、概念型问题。这个项目把 query variants 放在 debug trace 中，便于检查扩展是否有效。

代价：

- 需要额外 LLM 调用，增加延迟和成本。
- 改写可能偏题，HyDE 可能生成不存在的事实，带来检索漂移。
- API 不可用时不能阻断主链路，所以实现里需要 timeout 和 fallback。

什么时候关闭：

- 精确编号、财务数字、股票代码、条款名等问题，原始 query 已经足够强。
- 延迟预算严格。
- 上游 LLM 不稳定。
- 评估发现 query expansion 降低了 MRR 或引入错误来源。

## 为什么做表格感知处理

企业文档里的关键事实经常在表格里，例如收入、成本、分部、地区、年份对比。如果把表格当普通段落切分，很容易拆断行列关系，导致“数字有了、单位或年份没了”。

当前实现把 DOCX/HTML 表格转 Markdown，PDF 表格通过可选 `pdfplumber` 抽取；chunker 尽量保持小表完整，超长表按完整行切分并重复表头。

边界：

- 这不是完整视觉文档理解。
- 扫描 PDF/OCR、复杂跨页表、合并单元格恢复仍是后续方向。
- 表格 Markdown 对检索友好，但对精确计算和结构化查询仍不如专门的表格解析/数据库。

## 为什么答案缓存默认关闭

Redis cache 可以减少重复问题的 LLM 和 reranker 成本，但企业知识库有权限、索引版本和文档更新问题。默认关闭更安全，避免 demo 阶段因为旧缓存误导结果。

适合开启：

- 高频 FAQ。
- 索引更新不频繁。
- key 中包含模型、collection、检索选项、用户/权限版本和 corpus version。

不适合开启：

- 强实时问题。
- 权限高度个性化。
- 没有明确索引版本，无法可靠失效。

## 为什么评估分检索层和答案层

检索层更稳定、便宜、可重复，适合快速比较 dense、hybrid、rerank、parent-child 和 query expansion。答案层更接近用户体验，但受 LLM、prompt、评审模型和随机性影响。

所以项目里两条线都保留：

- `scripts.evaluate` 生成检索指标和逐 case retrieved sources。
- `scripts.evaluate_answers` 生成答案记录、本地诊断指标和 RAGAS 指标。

不要把其中一个替代另一个。检索命中了不代表答案忠实，答案看起来流畅也不代表来源正确。

## 最大已知风险

- 负例拒答不足：当前报告 negative rejection rate 为 0.000。
- `full` 变体依赖 LLM 做 Query Rewrite/HyDE，评估成本和稳定性受凭据影响。
- RAGAS 当前只有 smoke/小样本报告可引用，不能支持总体答案质量结论。
- 本地 Chroma/BM25 JSONL/parent JSONL 适合 demo，不等于生产可用。
- 表格处理已有改进，但复杂版面、图片和扫描件仍需 OCR/版面模型。
