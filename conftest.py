import logging
import pytest
from art import utils

logger = logging.getLogger(__name__)

def pytest_addoption(parser):
    parser.addoption(
        "--cmdopt", action="store", default="type1", help="my option: type1 or type2"
    )



@pytest.fixture(scope="session")
def fix_get_mnist_dataset():
    logging.info("Loading mnist")
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist), _, _ = utils.load_dataset('mnist')


    # if os.environ["mlFramework"] == "pytorch":
    #     # (x_train, y_train), (x_test, y_test) = self.mnist
    #     cls.x_test_mnist = np.reshape(cls.x_test_mnist, (cls.x_test_mnist.shape[0], 1, 28, 28)).astype(np.float32)
    #     # test_mnist = (x_train, y_train), (x_test, y_test)
    yield (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist)
    # Check that the test data has not been modified, only catches changes in attack.generate if self has been used
    # np.testing.assert_array_almost_equal(self._x_train_mnist_original[0:self.n_train], self.x_train_mnist,
    #                                      decimal=3)
    # np.testing.assert_array_almost_equal(self._y_train_mnist_original[0:self.n_train], self.y_train_mnist,
    #                                      decimal=3)
    # np.testing.assert_array_almost_equal(self._x_test_mnist_original[0:self.n_test], self.x_test_mnist,
    #                                      decimal=3)
    # np.testing.assert_array_almost_equal(self._y_test_mnist_original[0:self.n_test], self.y_test_mnist,
    #                                      decimal=3)
    print("teardown")

@pytest.fixture
def cmdopt(request):
    return request.config.getoption("--cmdopt")