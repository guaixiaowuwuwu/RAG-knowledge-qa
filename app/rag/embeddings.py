from openai import OpenAI


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
