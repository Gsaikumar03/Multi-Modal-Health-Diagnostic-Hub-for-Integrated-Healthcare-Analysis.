from transformers import pipeline


class Generator:
    def __init__(self):
        self.llm = pipeline(
            "text-generation",
            model="google/flan-t5-base",
            max_new_tokens=256
        )

    def generate(self, question, context):
        prompt = f"""
You are a professional medical assistant.

Context:
{context}

Question:
{question}

Provide a clear and medically accurate answer.
"""

        response = self.llm(prompt)[0]["generated_text"]
        return response