import pandas as pd


def build_context(full_text):
    """
    Build one continuous document from QASPER sections.
    """

    sections = full_text.get("section_name", [])
    paragraphs = full_text.get("paragraphs", [])

    document = []

    for section, para_list in zip(sections, paragraphs):

        if section:
            document.append(str(section))

        for para in para_list:
            if para:
                document.append(str(para))

    return "\n\n".join(document)


def extract_answer(answer):

    free_form = answer.get("free_form_answer", "")

    if isinstance(free_form, str) and free_form.strip():
        return free_form.strip()

    spans = answer.get("extractive_spans", [])

    if len(spans) > 0:
        return " | ".join(map(str, spans))

    yes_no = answer.get("yes_no", None)

    if yes_no is True:
        return "Yes"

    if yes_no is False:
        return "No"

    return ""


def load_qasper(file_path):

    df = pd.read_parquet(file_path)

    dataset = []

    for _, row in df.iterrows():

        context = build_context(row["full_text"])

        qas = row["qas"]

        questions = list(qas["question"])
        question_ids = list(qas["question_id"])
        answers = list(qas["answers"])

        for qid, question, answer_group in zip(
            question_ids,
            questions,
            answers
        ):

            # Convert numpy array to list if necessary
            annotations = answer_group["answer"]

            if hasattr(annotations, "tolist"):
                annotations = annotations.tolist()

            annotations = list(annotations)

            if len(annotations) == 0:
                continue

            first = annotations[0]

            evidence = first.get("evidence", [])

            if hasattr(evidence, "tolist"):
                evidence = evidence.tolist()

            dataset.append({

                "id": qid,

                "dataset": "QASPER",

                "domain": "Scientific",

                "paper_id": row["id"],

                "title": row["title"],

                "abstract": row["abstract"],

                "context": context,

                "question": question,

                "answer": extract_answer(first),

                "evidence": evidence

            })

    return dataset