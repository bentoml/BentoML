#!/usr/bin/env python
from sklearn import svm
from sklearn import datasets

from my_test_bento_service import IrisClassifier


if __name__ == "__main__":
    clf = svm.SVC(gamma='scale')
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    clf.fit(X, y)

    # Create a iris classifier service
    iris_classifier_service = IrisClassifier()

    # Pack it with the newly trained model artifact
    iris_classifier_service.pack('model', clf)

    # Save the prediction service to a BentoService bundle
    saved_path = iris_classifier_service.save()

    print(f"{iris_classifier_service.name}:{iris_classifier_service.version}")
