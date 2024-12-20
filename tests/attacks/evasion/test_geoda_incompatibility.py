# test_geoda_incompatibility.py

import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from art.estimators.classification import SklearnClassifier
from art.attacks.evasion import GeoDA


def test_geoda_with_random_forest():
    # Load the Iris dataset
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )

    # Train a RandomForestClassifier
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Wrap the model with ART's SklearnClassifier
    classifier = SklearnClassifier(model=model)

    # Expect GeoDA to raise ValueError when used with RandomForestClassifier
    with pytest.raises(ValueError, match="GeoDA is incompatible with"):
        attack = GeoDA(classifier)
        attack.generate(X_test)