from collections import Counter


class MajorityBaselineClassifier:
    @staticmethod
    def train(_, labels):
        c = Counter(labels)
        return c.most_common()[0][0]

    @staticmethod
    def predict(_, majority_label):
        return majority_label
