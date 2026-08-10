"""
LLM answer generator via local Ollama.

The generator is completely decoupled from the retrieval layer:
  it receives a question string and a list of retrieved text chunks,
  builds a RAG prompt, calls Ollama, and returns the response string.

The Ollama model name is configurable — do not hard-code it here.
"""

import logging
from typing import Sequence

import ollama

from retrieval.base import RetrievalResult

logger = logging.getLogger(__name__)

_PROMPT_TEMPLATE = """\
You are a precise, factual assistant.

Answer the question ONLY using information from the provided context.
If the answer cannot be found in the context, reply exactly:
"I cannot answer from the provided context."

Context:
{context}

Question:
{question}

Answer:"""


class LLMGenerator:
    """
    Wraps a locally-running Ollama model for RAG generation.

    Parameters
    ----------
    model : str
        Ollama model tag, e.g. "llama3.1:8b-instruct-q4_K_M".
    """

    def __init__(self, model: str = "llama3.1:8b-instruct-q4_K_M") -> None:
        self.model = model
        logger.info("LLM generator initialised (model=%s)", model)

    def build_prompt(self, question: str, retrieved: Sequence[RetrievalResult]) -> str:
        context = "\n\n".join(r.text for r in retrieved)
        return _PROMPT_TEMPLATE.format(context=context, question=question)

    def generate(self, question: str, retrieved: Sequence[RetrievalResult]) -> str:
        """
        Generate an answer grounded in the retrieved chunks.

        Returns the model's response string, or an error message if the
        Ollama call fails (so the benchmark run is not interrupted).
        """
        prompt = self.build_prompt(question, retrieved)
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )
            return response["message"]["content"]
        except Exception as exc:
            logger.error("Ollama generation failed: %s", exc)
            return f"[GENERATION ERROR: {exc}]"
