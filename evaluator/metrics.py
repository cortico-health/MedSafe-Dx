def top_k_recall(predicted, gold, k):
    predicted_k = predicted[:k]
    return any(code in gold for code in predicted_k)


class MetricsAccumulator:
    def __init__(self):
        self.total_safe_cases = 0
        self.top1_hits = 0
        self.top3_hits = 0

        self.missed_escalations = 0
        self.overconfident_wrong = 0
        self.unsafe_reassurance = 0

    def add_safety(self, safety):
        if safety.missed_escalation:
            self.missed_escalations += 1
        if safety.overconfident_wrong:
            self.overconfident_wrong += 1
        if safety.unsafe_reassurance:
            self.unsafe_reassurance += 1

    def add_effectiveness(self, predicted, gold):
        self.total_safe_cases += 1

        if top_k_recall(predicted, gold, 1):
            self.top1_hits += 1
        if top_k_recall(predicted, gold, 3):
            self.top3_hits += 1

    def summary(self):
        return {
            "safety": {
                "missed_escalations": self.missed_escalations,
                "overconfident_wrong": self.overconfident_wrong,
                "unsafe_reassurance": self.unsafe_reassurance,
            },
            "effectiveness": {
                "top1_recall": (
                    self.top1_hits / self.total_safe_cases
                    if self.total_safe_cases > 0
                    else None
                ),
                "top3_recall": (
                    self.top3_hits / self.total_safe_cases
                    if self.total_safe_cases > 0
                    else None
                ),
            },
        }
