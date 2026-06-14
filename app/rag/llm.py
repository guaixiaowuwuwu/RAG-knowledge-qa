from openai import OpenAI


class OpenAIChatLLM:
    def __init__(self, api_key: str, base_url: str, model: str):
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for chat completion")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def complete(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        content = response.choices[0].message.content
        return content or ""
