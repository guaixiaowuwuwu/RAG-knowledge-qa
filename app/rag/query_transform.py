from concurrent.futures import ThreadPoolExecutor, TimeoutError


class QueryTransformer:
    def __init__(
        self,
        llm,
        rewrite_enabled: bool,
        hyde_enabled: bool,
        max_variants: int,
        timeout_seconds: float | None = None,
    ):
        self.llm = llm
        self.rewrite_enabled = rewrite_enabled
        self.hyde_enabled = hyde_enabled
        self.max_variants = max(1, max_variants)
        self.timeout_seconds = timeout_seconds

    def expand(
        self,
        query: str,
        *,
        rewrite_enabled: bool | None = None,
        hyde_enabled: bool | None = None,
        max_variants: int | None = None,
    ) -> list[str]:
        rewrite = self.rewrite_enabled if rewrite_enabled is None else rewrite_enabled
        hyde = self.hyde_enabled if hyde_enabled is None else hyde_enabled
        variant_limit = max(1, max_variants or self.max_variants)
        variants = [query]

        if rewrite and len(variants) < variant_limit:
            variants.extend(self._safe_transform(lambda: self._rewrite(query)))

        if hyde and len(variants) < variant_limit:
            variants.extend(self._safe_transform(lambda: [self._hyde(query)]))

        cleaned: list[str] = []
        seen: set[str] = set()
        for variant in variants:
            normalized = variant.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            cleaned.append(normalized)
            if len(cleaned) >= variant_limit:
                break
        return cleaned or [query]

    def _safe_transform(self, transform) -> list[str]:
        try:
            if self.timeout_seconds is None or self.timeout_seconds <= 0:
                return transform()

            executor = ThreadPoolExecutor(max_workers=1)
            try:
                future = executor.submit(transform)
                return future.result(timeout=self.timeout_seconds)
            finally:
                executor.shutdown(wait=False, cancel_futures=True)
        except (Exception, TimeoutError):
            return []

    def _rewrite(self, query: str) -> list[str]:
        prompt = (
            "请将以下用户问题从不同角度改写，生成2个语义相近但表述不同的查询。\n"
            "每行一个查询，不要编号。\n\n"
            f"原始问题：{query}"
        )
        response = self.llm.complete(prompt)
        return [line.strip(" -0123456789.、") for line in response.splitlines() if line.strip()]

    def _hyde(self, query: str) -> str:
        prompt = (
            "请针对以下问题，写一段简短的假设性回答。"
            "不需要保证准确性，只需要包含相关术语和概念，用于检索。\n\n"
            f"问题：{query}"
        )
        return self.llm.complete(prompt).strip()
