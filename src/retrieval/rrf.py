from collections import defaultdict


class ReciprocalRankFusion:

    def __init__(self, k=60):
        self.k = k

    def fuse(self, *result_lists):

        scores = defaultdict(float)
        lookup = {}

        for results in result_lists:

            for result in results:

                key = result.text.strip()

                scores[key] += 1.0 / (self.k + result.rank)

                lookup[key] = result

        fused = []

        for key, score in scores.items():

            item = lookup[key]

            item.score = score

            fused.append(item)

        fused.sort(
            key=lambda x: x.score,
            reverse=True
        )

        for rank, item in enumerate(fused, start=1):
            item.rank = rank

        return fused