from rank_bm25 import BM25Okapi
import pickle
import os
import re


class BM25Indexer:

    def __init__(self):
        self.index = None
        self.documents = None

    def tokenize(self, text):
        """
        Simple tokenizer.
        Lowercase and split into words.
        """

        text = text.lower()

        tokens = re.findall(r"\b\w+\b", text)

        return tokens

    def build(self, chunks):

        print("\nBuilding BM25 index...")

        tokenized_documents = []

        for chunk in chunks:
            tokenized_documents.append(
                self.tokenize(chunk["text"])
            )

        self.index = BM25Okapi(tokenized_documents)

        self.documents = chunks

        print(f"Indexed {len(chunks)} chunks.")

    def save(self, dataset_name):

        output_dir = "../results/indexes"

        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(
            output_dir,
            f"{dataset_name}_bm25.pkl"
        )

        with open(output_path, "wb") as f:

            pickle.dump(
                {
                    "bm25": self.index,
                    "documents": self.documents
                },
                f
            )

        print(f"BM25 index saved -> {output_path}")