from evaluation.retrieval_metrics import *

retrieved = ["A", "B", "C", "D", "E"]
relevant = ["C", "E"]

print("Recall@5 :", recall_at_k(retrieved, relevant, 5))
print("MRR       :", mean_reciprocal_rank(retrieved, relevant))
print("NDCG@5    :", ndcg_at_k(retrieved, relevant, 5))