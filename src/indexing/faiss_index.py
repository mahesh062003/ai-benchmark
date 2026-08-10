import os
import pickle
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer


class FAISSIndexer:

    def __init__(
        self,
        model_name="sentence-transformers/all-mpnet-base-v2"
    ):

        print("\nLoading embedding model...")

        self.model = SentenceTransformer(model_name)

        print("Embedding model loaded.")

        self.index = None
        self.documents = None

    def build(self, chunks):

        print("\nGenerating embeddings...")

        texts = [chunk["text"] for chunk in chunks]

        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)

        self.index.add(embeddings)

        self.documents = chunks

        print(f"Indexed {len(chunks)} chunks.")

    def save(self, dataset_name):

        output_dir = "../results/indexes"

        os.makedirs(output_dir, exist_ok=True)

        faiss.write_index(
            self.index,
            os.path.join(
                output_dir,
                f"{dataset_name}_faiss.index"
            )
        )

        with open(
            os.path.join(
                output_dir,
                f"{dataset_name}_metadata.pkl"
            ),
            "wb"
        ) as f:

            pickle.dump(self.documents, f)

        print("\nFAISS index saved.")