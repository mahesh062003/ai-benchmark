from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentChunker:
    """
    Splits documents into smaller overlapping chunks for retrieval.
    """

    def __init__(self, chunk_size=512, chunk_overlap=50):

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                ""
            ]
        )

    def chunk_documents(self, samples):

        chunks = []

        for sample in samples:

            # ==========================================================
            # Build context depending on dataset structure
            # ==========================================================

            context = sample.get("context", "")

            # Some datasets store context as a list
            if isinstance(context, list):
                context = "\n".join(context)

            # If context is missing, construct one
            if not context:

                question = sample.get("question", "")

                options = sample.get("options", {})

                if isinstance(options, dict) and len(options) > 0:

                    options_text = "\n".join(
                        [f"{k}. {v}" for k, v in options.items()]
                    )

                    context = f"{question}\n\n{options_text}"

                else:

                    context = question

            if context is None:
                continue

            context = str(context).strip()

            if len(context) == 0:
                continue

            # ==========================================================
            # Split into chunks
            # ==========================================================

            pieces = self.splitter.split_text(context)

            for i, piece in enumerate(pieces):

                chunk = {
                    "chunk_id": f"{sample['dataset']}_{sample['id']}_{i}",
                    "sample_id": sample["id"],
                    "dataset": sample["dataset"],
                    "domain": sample["domain"],
                    "question": sample.get("question", ""),
                    "answer": sample.get(
                        "answer",
                        sample.get("answers", [])
                    ),
                    "text": piece,
                    "metadata": {
                        "title": sample.get("title", ""),
                        "paper_id": sample.get("paper_id", ""),
                        "dataset": sample["dataset"],
                        "domain": sample["domain"]
                    }
                }

                chunks.append(chunk)

        return chunks