import math


def recall_at_k(retrieved_ids, relevant_ids, k):

    retrieved = set(retrieved_ids[:k])
    relevant = set(relevant_ids)

    if len(relevant) == 0:
        return 0.0

    return len(retrieved & relevant) / len(relevant)


def mean_reciprocal_rank(retrieved_ids, relevant_ids):

    relevant = set(relevant_ids)

    for rank, doc in enumerate(retrieved_ids, start=1):

        if doc in relevant:
            return 1 / rank

    return 0.0


def ndcg_at_k(retrieved_ids, relevant_ids, k):

    dcg = 0

    for i, doc in enumerate(retrieved_ids[:k]):

        if doc in relevant_ids:
            dcg += 1 / math.log2(i + 2)

    ideal = min(len(relevant_ids), k)

    idcg = 0

    for i in range(ideal):
        idcg += 1 / math.log2(i + 2)

    if idcg == 0:
        return 0.0

    return dcg / idcg