"""A small supervised learning example: k-nearest neighbors classification.

The model learns from labeled examples and predicts the label of a new sample
by taking a majority vote among the closest training samples.
"""

from __future__ import annotations

from collections import Counter
from math import sqrt
from typing import Iterable, Sequence


FeatureVector = Sequence[float]
Label = str


class KNearestNeighborsClassifier:
    """Simple k-nearest neighbors classifier implemented from scratch."""

    def __init__(self, k: int = 3) -> None:
        if k <= 0:
            raise ValueError("k must be a positive integer")
        self.k = k
        self._features: list[FeatureVector] = []
        self._labels: list[Label] = []

    def fit(self, features: Iterable[FeatureVector], labels: Iterable[Label]) -> None:
        """Store labeled training data."""
        self._features = list(features)
        self._labels = list(labels)

        if not self._features:
            raise ValueError("training data cannot be empty")
        if len(self._features) != len(self._labels):
            raise ValueError("features and labels must have the same length")

        width = len(self._features[0])
        if any(len(row) != width for row in self._features):
            raise ValueError("all feature vectors must have the same length")

    def predict_one(self, sample: FeatureVector) -> Label:
        """Predict a single sample label."""
        if not self._features:
            raise ValueError("model has not been fitted")
        if len(sample) != len(self._features[0]):
            raise ValueError("sample has a different feature count than training data")

        distances = [
            (self._euclidean_distance(sample, train_sample), label)
            for train_sample, label in zip(self._features, self._labels)
        ]
        nearest = sorted(distances, key=lambda item: item[0])[: self.k]
        votes = Counter(label for _, label in nearest)
        return votes.most_common(1)[0][0]

    def predict(self, samples: Iterable[FeatureVector]) -> list[Label]:
        """Predict labels for multiple samples."""
        return [self.predict_one(sample) for sample in samples]

    @staticmethod
    def _euclidean_distance(left: FeatureVector, right: FeatureVector) -> float:
        return sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def main() -> None:
    # Tiny training set: [study_hours, attendance_percent]
    features = [
        [1.0, 45.0],
        [2.0, 50.0],
        [3.5, 65.0],
        [5.0, 75.0],
        [6.0, 82.0],
        [7.5, 90.0],
    ]
    labels = ["fail", "fail", "pass", "pass", "pass", "pass"]

    model = KNearestNeighborsClassifier(k=3)
    model.fit(features, labels)

    test_samples = [
        [2.5, 55.0],
        [6.5, 88.0],
    ]

    for sample, prediction in zip(test_samples, model.predict(test_samples)):
        print(f"Student with features {sample} is predicted to: {prediction}")


if __name__ == "__main__":
    main()
