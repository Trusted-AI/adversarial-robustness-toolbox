import numpy as np
import logging
from art import utils
from tests import utils_test

logger = logging.getLogger(__name__)

def _backend_targeted_tabular(attack, classifier, fix_get_iris, fix_mlFramework):
    (x_train_iris, y_train_iris), (x_test_iris, y_test_iris) = fix_get_iris

    if fix_mlFramework in ["scikitlearn"]:
        classifier.fit(x=x_test_iris, y=y_test_iris)

    targets = utils.random_targets(y_test_iris, nb_classes=3)
    x_test_adv = attack.generate(x_test_iris, **{'y': targets})

    utils_test.check_adverse_example_x(x_test_adv, x_test_iris)

    y_pred_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
    target = np.argmax(targets, axis=1)
    assert (target == y_pred_adv).any()

    accuracy = np.sum(y_pred_adv == np.argmax(targets, axis=1)) / y_test_iris.shape[0]
    logger.info('Success rate of targeted boundary on Iris: %.2f%%', (accuracy * 100))