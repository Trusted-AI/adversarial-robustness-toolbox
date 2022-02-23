import pytest

from art.defences.preprocessor.preprocessor import PreprocessorPyTorch
from tests.utils import ARTTestException


class DummyPreprocessorPyTorch(PreprocessorPyTorch):
    def forward(self, x, y):
        return x, y


@pytest.mark.parametrize("is_fitted", [True, False])
@pytest.mark.parametrize("apply_fit", [True, False])
@pytest.mark.parametrize("apply_predict", [True, False])
@pytest.mark.only_with_platform("pytorch")
def test_preprocessor_pytorch_init(art_warning, is_fitted, apply_fit, apply_predict):
    try:
        import torch

        preprocessor = DummyPreprocessorPyTorch(
            device_type="cpu",
            is_fitted=is_fitted,
            apply_fit=apply_fit,
            apply_predict=apply_predict,
        )

        assert preprocessor.device == torch.device("cpu")
        assert preprocessor.is_fitted == is_fitted
        assert preprocessor.apply_fit == apply_fit
        assert preprocessor.apply_predict == apply_predict

    except ARTTestException as e:
        art_warning(e)
