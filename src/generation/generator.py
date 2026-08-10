import ollama


class LLMGenerator:

    def __init__(self, model="llama3.1:8b-instruct-q4_K_M"):

        self.model = model

    def build_prompt(self, question, retrieved_chunks):

        context = "\n\n".join(
            [chunk.text for chunk in retrieved_chunks]
        )

        prompt = f"""
You are a helpful assistant.

Answer the question ONLY using the provided context.

If the answer cannot be found in the context, reply:

"I cannot answer from the provided context."

Context:
{context}

Question:
{question}

Answer:
"""

        return prompt

    def generate(self, question, retrieved_chunks):

        prompt = self.build_prompt(
            question,
            retrieved_chunks
        )

        response = ollama.chat(

            model=self.model,

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]

        )

        return response["message"]["content"]