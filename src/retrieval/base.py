from dataclasses import dataclass


@dataclass
class RetrievalResult:

    chunk_id: str

    dataset: str

    domain: str

    text: str

    score: float

    rank: int