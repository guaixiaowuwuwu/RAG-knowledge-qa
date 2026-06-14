from openai import OpenAI


LOCAL_EMBEDDING_MODELS = {
    "bge-m3": "BAAI/bge-m3",
    "BAAI/bge-m3": "BAAI/bge-m3",
}


class OpenAIEmbeddings:
    def __init__(self, api_key: str, base_url: str, model: str):
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for embeddings")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class LocalSentenceTransformerEmbeddings:
    def __init__(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Local embedding model requires sentence-transformers. "
                'Install dependencies with `pip install -e ".[dev]"` or `uv sync --extra dev`.'
            ) from exc

        self.model_name = LOCAL_EMBEDDING_MODELS.get(model_name, model_name)
        self.model = SentenceTransformer(self.model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def build_embeddings(settings):
    if settings.embedding_model in LOCAL_EMBEDDING_MODELS:
        return LocalSentenceTransformerEmbeddings(settings.embedding_model)

    return OpenAIEmbeddings(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=settings.embedding_model,
    )
