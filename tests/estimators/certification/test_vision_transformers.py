# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2023
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import pytest
import os

import numpy as np

from art.utils import load_dataset
from tests.utils import ARTTestException


@pytest.fixture()
def fix_get_mnist_data():
    """
    Get the first 128 samples of the mnist test set with channels first format

    :return: First 128 sample/label pairs of the MNIST test dataset.
    """
    nb_test = 128

    (_, _), (x_test, y_test), _, _ = load_dataset("mnist")
    x_test = np.squeeze(x_test).astype(np.float32)
    x_test = np.expand_dims(x_test, axis=1)
    y_test = np.argmax(y_test, axis=1)

    x_test, y_test = x_test[:nb_test], y_test[:nb_test]
    return x_test, y_test


@pytest.fixture()
def fix_get_cifar10_data():
    """
    Get the first 128 samples of the cifar10 test set

    :return: First 128 sample/label pairs of the cifar10 test dataset.
    """
    nb_test = 128

    (_, _), (x_test, y_test), _, _ = load_dataset("cifar10")
    y_test = np.argmax(y_test, axis=1)
    x_test, y_test = x_test[:nb_test], y_test[:nb_test]
    x_test = np.transpose(x_test, (0, 3, 1, 2))  # return in channels first format
    return x_test.astype(np.float32), y_test


@pytest.mark.only_with_platform("pytorch")
def test_ablation(art_warning, fix_get_mnist_data, fix_get_cifar10_data):
    """
    Check that the ablation is being performed correctly
    """
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from art.estimators.certification.derandomized_smoothing.ablators.pytorch import ColumnAblatorPyTorch

    try:
        cifar_data = fix_get_cifar10_data[0]

        col_ablator = ColumnAblatorPyTorch(
            ablation_size=4,
            channels_first=True,
            to_reshape=False,  # do not upsample initially
            mode="ViT",
            original_shape=(3, 32, 32),
            output_shape=(3, 224, 224),
        )

        cifar_data = torch.from_numpy(cifar_data).to(device)
        # check that the ablation functioned when in the middle of the image
        ablated = col_ablator.forward(cifar_data, column_pos=10)

        assert ablated.shape[1] == 4
        assert torch.sum(ablated[:, :, :, 0:10]) == 0
        assert torch.sum(ablated[:, :, :, 10:14]) > 0
        assert torch.sum(ablated[:, :, :, 14:]) == 0

        # check that the ablation wraps when on the edge of the image
        ablated = col_ablator.forward(cifar_data, column_pos=30)

        assert ablated.shape[1] == 4
        assert torch.sum(ablated[:, :, :, 30:]) > 0
        assert torch.sum(ablated[:, :, :, 2:30]) == 0
        assert torch.sum(ablated[:, :, :, :2]) > 0

        # check that upsampling works as expected
        col_ablator = ColumnAblatorPyTorch(
            ablation_size=4,
            channels_first=True,
            to_reshape=True,
            mode="ViT",
            original_shape=(3, 32, 32),
            output_shape=(3, 224, 224),
        )

        ablated = col_ablator.forward(cifar_data, column_pos=10)

        assert ablated.shape[1] == 4
        assert torch.sum(ablated[:, :, :, : 10 * 7]) == 0
        assert torch.sum(ablated[:, :, :, 10 * 7 : 14 * 7]) > 0
        assert torch.sum(ablated[:, :, :, 14 * 7 :]) == 0

        # check that the ablation wraps when on the edge of the image
        ablated = col_ablator.forward(cifar_data, column_pos=30)

        assert ablated.shape[1] == 4
        assert torch.sum(ablated[:, :, :, 30 * 7 :]) > 0
        assert torch.sum(ablated[:, :, :, 2 * 7 : 30 * 7]) == 0
        assert torch.sum(ablated[:, :, :, : 2 * 7]) > 0

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
def test_ablation_row(art_warning, fix_get_mnist_data, fix_get_cifar10_data):
    """
    Check that the ablation is being performed correctly
    """
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from art.estimators.certification.derandomized_smoothing.ablators.pytorch import ColumnAblatorPyTorch

    try:
        cifar_data = fix_get_cifar10_data[0]

        col_ablator = ColumnAblatorPyTorch(
            ablation_size=4,
            channels_first=True,
            to_reshape=False,  # do not upsample initially
            mode="ViT",
            ablation_mode="row",
            original_shape=(3, 32, 32),
            output_shape=(3, 224, 224),
        )

        cifar_data = torch.from_numpy(cifar_data).to(device)
        # check that the ablation functioned when in the middle of the image
        ablated = col_ablator.forward(cifar_data, column_pos=10)

        assert ablated.shape[1] == 4
        assert torch.sum(ablated[:, :, 0:10, :]) == 0
        assert torch.sum(ablated[:, :, 10:14, :]) > 0
        assert torch.sum(ablated[:, :, 14:, :]) == 0

        # check that the ablation wraps when on the edge of the image
        ablated = col_ablator.forward(cifar_data, column_pos=30)

        assert ablated.shape[1] == 4
        assert torch.sum(ablated[:, :, 30:, :]) > 0
        assert torch.sum(ablated[:, :, 2:30, :]) == 0
        assert torch.sum(ablated[:, :, :2, :]) > 0

        # check that upsampling works as expected
        col_ablator = ColumnAblatorPyTorch(
            ablation_size=4,
            channels_first=True,
            to_reshape=True,
            mode="ViT",
            ablation_mode="row",
            original_shape=(3, 32, 32),
            output_shape=(3, 224, 224),
        )

        ablated = col_ablator.forward(cifar_data, column_pos=10)

        assert ablated.shape[1] == 4
        assert torch.sum(ablated[:, :, : 10 * 7, :]) == 0
        assert torch.sum(ablated[:, :, 10 * 7 : 14 * 7, :]) > 0
        assert torch.sum(ablated[:, :, 14 * 7 :, :]) == 0

        # check that the ablation wraps when on the edge of the image
        ablated = col_ablator.forward(cifar_data, column_pos=30)

        assert ablated.shape[1] == 4
        assert torch.sum(ablated[:, :, 30 * 7 :, :]) > 0
        assert torch.sum(ablated[:, :, 2 * 7 : 30 * 7, :]) == 0
        assert torch.sum(ablated[:, :, : 2 * 7, :]) > 0

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
def test_pytorch_training(art_warning, fix_get_mnist_data, fix_get_cifar10_data):
    """
    Check that the training loop for pytorch does not result in errors
    """
    import torch
    from art.estimators.certification.derandomized_smoothing import PyTorchDeRandomizedSmoothing

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        cifar_data = fix_get_cifar10_data[0][:50]
        cifar_labels = fix_get_cifar10_data[1][:50]

        art_model = PyTorchDeRandomizedSmoothing(
            model="vit_small_patch16_224",
            loss=torch.nn.CrossEntropyLoss(),
            optimizer=torch.optim.SGD,
            optimizer_params={"lr": 0.01},
            input_shape=(3, 32, 32),
            nb_classes=10,
            ablation_size=4,
            load_pretrained=True,
            replace_last_layer=True,
            verbose=False,
        )

        scheduler = torch.optim.lr_scheduler.MultiStepLR(art_model.optimizer, milestones=[1], gamma=0.1)

        head = {
            "weight": torch.tensor(
                np.load(
                    os.path.join(
                        os.path.dirname(os.path.dirname(__file__)),
                        "../../utils/resources/models/certification/smooth_vit/head_weight.npy",
                    )
                )
            ).to(device),
            "bias": torch.tensor(
                np.load(
                    os.path.join(
                        os.path.dirname(os.path.dirname(__file__)),
                        "../../utils/resources/models/certification/smooth_vit/head_bias.npy",
                    )
                )
            ).to(device),
        }
        art_model.model.head.load_state_dict(head)

        art_model.fit(cifar_data, cifar_labels, nb_epochs=2, update_batchnorm=True, scheduler=scheduler)
        preds = art_model.predict(cifar_data)

        gt_preds = np.load(
            os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "../../utils/resources/models/certification/smooth_vit/cumulative_predictions.npy",
            )
        )

        np.array_equal(preds, gt_preds)

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
def test_certification_function(art_warning, fix_get_mnist_data, fix_get_cifar10_data):
    """
    Check that based on a given set of synthetic class predictions the certification gives the expected results.
    """
    from art.estimators.certification.derandomized_smoothing.ablators.pytorch import ColumnAblatorPyTorch
    import torch

    try:
        col_ablator = ColumnAblatorPyTorch(
            ablation_size=4,
            channels_first=True,
            mode="ViT",
            to_reshape=True,  # do not upsample initially
            original_shape=(3, 32, 32),
            output_shape=(3, 224, 224),
        )
        pred_counts = torch.from_numpy(np.asarray([[20, 5, 1], [10, 5, 1], [1, 16, 1]]))
        cert, cert_and_correct, top_predicted_class = col_ablator.certify(
            pred_counts=pred_counts,
            size_to_certify=4,
            label=0,
        )
        assert torch.equal(cert, torch.tensor([True, False, True]))
        assert torch.equal(cert_and_correct, torch.tensor([True, False, False]))
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
@pytest.mark.parametrize("ablation", ["block", "column"])
def test_end_to_end_equivalence(art_warning, fix_get_mnist_data, fix_get_cifar10_data, ablation):
    """
    Assert implementations matches original with a forward pass through the same model architecture.
    There are some differences in architecture between the same model names in timm vs the original implementation.
    We use vit_base_patch16_224 which matches.
    """
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from art.estimators.certification.derandomized_smoothing import PyTorchDeRandomizedSmoothing

    from art.estimators.certification.derandomized_smoothing.ablators import (
        ColumnAblatorPyTorch,
        BlockAblatorPyTorch,
    )

    cifar_data = fix_get_cifar10_data[0][:50]
    torch.manual_seed(1234)
    try:
        art_model = PyTorchDeRandomizedSmoothing(
            model="vit_base_patch16_224",
            loss=torch.nn.CrossEntropyLoss(),
            optimizer=torch.optim.SGD,
            optimizer_params={"lr": 0.01},
            input_shape=(3, 32, 32),
            nb_classes=10,
            ablation_size=4,
            load_pretrained=True,
            replace_last_layer=True,
            verbose=False,
        )

        if ablation == "column":
            ablator = ColumnAblatorPyTorch(
                ablation_size=4,
                channels_first=True,
                to_reshape=True,
                mode="ViT",
                original_shape=(3, 32, 32),
                output_shape=(3, 224, 224),
            )
            ablated = ablator.forward(cifar_data, column_pos=10)
            madry_preds = torch.load(
                os.path.join(
                    os.path.dirname(os.path.dirname(__file__)),
                    "../../utils/resources/models/certification/smooth_vit/madry_preds_column.pt",
                )
            )
            art_preds = art_model.model(ablated)
            assert torch.allclose(madry_preds.to(device), art_preds, rtol=1e-04, atol=1e-04)

        elif ablation == "block":
            ablator = BlockAblatorPyTorch(
                ablation_size=4,
                channels_first=True,
                to_reshape=True,
                original_shape=(3, 32, 32),
                output_shape=(3, 224, 224),
                mode="ViT",
            )
            ablated = ablator.forward(cifar_data, column_pos=10, row_pos=28)
            madry_preds = torch.load(
                os.path.join(
                    os.path.dirname(os.path.dirname(__file__)),
                    "../../utils/resources/models/certification/smooth_vit/madry_preds_block.pt",
                )
            )
            art_preds = art_model.model(ablated)
            assert torch.allclose(madry_preds.to(device), art_preds, rtol=1e-04, atol=1e-04)
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
@pytest.mark.parametrize("ablation", ["block", "column"])
def test_certification_equivalence(art_warning, fix_get_mnist_data, fix_get_cifar10_data, ablation):
    """
    With the forward pass equivalence asserted, we now confirm that the certification functions in the same
    way by doing a full end to end prediction and certification test over the data.
    """
    import torch

    from art.estimators.certification.derandomized_smoothing import PyTorchDeRandomizedSmoothing

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        art_model = PyTorchDeRandomizedSmoothing(
            model="vit_small_patch16_224",
            loss=torch.nn.CrossEntropyLoss(),
            optimizer=torch.optim.SGD,
            optimizer_params={"lr": 0.01},
            input_shape=(3, 32, 32),
            nb_classes=10,
            ablation_type=ablation,
            ablation_size=4,
            load_pretrained=True,
            replace_last_layer=True,
            verbose=False,
        )

        head = {
            "weight": torch.tensor(
                np.load(
                    os.path.join(
                        os.path.dirname(os.path.dirname(__file__)),
                        "../../utils/resources/models/certification/smooth_vit/head_weight.npy",
                    )
                )
            ).to(device),
            "bias": torch.tensor(
                np.load(
                    os.path.join(
                        os.path.dirname(os.path.dirname(__file__)),
                        "../../utils/resources/models/certification/smooth_vit/head_bias.npy",
                    )
                )
            ).to(device),
        }
        art_model.model.head.load_state_dict(head)

        if torch.cuda.is_available():
            num_to_fetch = 100
        else:
            num_to_fetch = 10

        cifar_data = torch.from_numpy(fix_get_cifar10_data[0][:num_to_fetch]).to(device)
        cifar_labels = torch.from_numpy(fix_get_cifar10_data[1][:num_to_fetch]).to(device)

        acc, cert_acc = art_model.eval_and_certify(
            x=cifar_data.cpu().numpy(), y=cifar_labels.cpu().numpy(), batch_size=num_to_fetch, size_to_certify=4
        )

        upsample = torch.nn.Upsample(scale_factor=224 / 32)
        cifar_data = upsample(cifar_data)
        acc_non_ablation = art_model.model(cifar_data)
        acc_non_ablation = art_model.get_accuracy(acc_non_ablation, cifar_labels)

        if torch.cuda.is_available():
            if ablation == "column":
                assert np.allclose(cert_acc.cpu().numpy(), 0.29)
                assert np.allclose(acc.cpu().numpy(), 0.57)
            else:
                assert np.allclose(cert_acc.cpu().numpy(), 0.16)
                assert np.allclose(acc.cpu().numpy(), 0.24)
            assert np.allclose(acc_non_ablation, 0.52)
        else:
            if ablation == "column":
                assert np.allclose(cert_acc.cpu().numpy(), 0.30)
                assert np.allclose(acc.cpu().numpy(), 0.70)
            else:
                assert np.allclose(cert_acc.cpu().numpy(), 0.20)
                assert np.allclose(acc.cpu().numpy(), 0.20)
            assert np.allclose(acc_non_ablation, 0.60)
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
def test_equivalence(art_warning, fix_get_cifar10_data):
    import torch
    from art.estimators.certification.derandomized_smoothing import PyTorchDeRandomizedSmoothing
    from art.estimators.certification.derandomized_smoothing.vision_transformers.pytorch import PyTorchVisionTransformer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:

        class MadrylabImplementations:
            """
            Code adapted from the implementation in https://github.com/MadryLab/smoothed-vit
            to check against our own functionality.

            Original License:

            MIT License

            Copyright (c) 2021 Madry Lab

            Permission is hereby granted, free of charge, to any person obtaining a copy
            of this software and associated documentation files (the "Software"), to deal
            in the Software without restriction, including without limitation the rights
            to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
            copies of the Software, and to permit persons to whom the Software is
            furnished to do so, subject to the following conditions:

            The above copyright notice and this permission notice shall be included in all
            copies or substantial portions of the Software.

            THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
            IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
            FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
            AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
            LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
            OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
            SOFTWARE.

            """

            def __init__(self):
                pass

            @classmethod
            def token_dropper(cls, x, mask):
                """
                The implementation of dropping tokens has been done slightly differently in this tool.
                Here we check that it is equivalent to the original implementation
                """

                class MaskProcessor(torch.nn.Module):
                    def __init__(self, patch_size=16):
                        super().__init__()
                        self.avg_pool = torch.nn.AvgPool2d(patch_size)

                    def forward(self, ones_mask):
                        B = ones_mask.shape[0]
                        ones_mask = ones_mask[0].unsqueeze(0)  # take the first mask
                        ones_mask = self.avg_pool(ones_mask)[0]
                        ones_mask = torch.where(ones_mask.view(-1) > 0)[0] + 1
                        ones_mask = torch.cat([torch.IntTensor(1).fill_(0).to(device), ones_mask]).unsqueeze(0)
                        ones_mask = ones_mask.expand(B, -1)
                        return ones_mask

                mask_processor = MaskProcessor()
                patch_mask = mask_processor(mask)

                # x = self.pos_drop(x) # B, N, C
                if patch_mask is not None:
                    # patch_mask is B, K
                    B, N, C = x.shape
                    if len(patch_mask.shape) == 1:  # not a separate one per batch
                        x = x[:, patch_mask]
                    else:
                        patch_mask = patch_mask.unsqueeze(-1).expand(-1, -1, C)
                        x = torch.gather(x, 1, patch_mask)
                return x

            @classmethod
            def embedder(cls, x, pos_embed, cls_token):
                """
                NB, original code used the pos embed from the divit rather than vit
                (which we pull from our model) which we use here.

                From timm vit:
                self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * .02)

                From timm dvit:
                self.pos_embed = nn.Parameter(torch.zeros(1,
                                                         self.patch_embed.num_patches + self.num_prefix_tokens,
                                                         self.embed_dim))

                From repo:
                self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
                """
                x = torch.cat((cls_token.expand(x.shape[0], -1, -1), x), dim=1)
                return x + pos_embed

        def forward_features(self, x: torch.Tensor) -> torch.Tensor:
            """
            This is a copy of the function in ArtViT.forward_features
            except we also perform an equivalence assertion compared to the implementation
            in https://github.com/MadryLab/smoothed-vit (see MadrylabImplementations class above)

            The forward pass of the ViT.

            :param x: Input data.
            :return: The input processed by the ViT backbone
            """
            import copy

            ablated_input = False
            if x.shape[1] == self.in_chans + 1:
                ablated_input = True

            if ablated_input:
                x, ablation_mask = x[:, : self.in_chans], x[:, self.in_chans : self.in_chans + 1]

            x = self.patch_embed(x)

            madry_embed = MadrylabImplementations.embedder(copy.copy(x), self.pos_embed, self.cls_token)
            x = self._pos_embed(x)
            assert torch.equal(madry_embed, x)

            # pass the x into the token dropping code
            madry_dropped = MadrylabImplementations.token_dropper(copy.copy(x), ablation_mask)

            if self.to_drop_tokens and ablated_input:
                ones = self.ablation_mask_embedder(ablation_mask)
                to_drop = torch.sum(ones, dim=2)
                indexes = torch.gt(torch.where(to_drop > 1, 1, 0), 0)
                x = self.drop_tokens(x, indexes)

            assert torch.equal(madry_dropped, x)

            x = self.norm_pre(x)
            x = self.blocks(x)

            return self.norm(x)

        # Replace the forward_features with the forward_features code with checks.
        PyTorchVisionTransformer.forward_features = forward_features

        art_model = PyTorchDeRandomizedSmoothing(
            model="vit_small_patch16_224",
            loss=torch.nn.CrossEntropyLoss(),
            optimizer=torch.optim.SGD,
            optimizer_params={"lr": 0.01},
            input_shape=(3, 32, 32),
            nb_classes=10,
            ablation_size=4,
            load_pretrained=False,
            replace_last_layer=True,
            verbose=False,
        )

        cifar_data = fix_get_cifar10_data[0][:50]
        cifar_labels = fix_get_cifar10_data[1][:50]

        scheduler = torch.optim.lr_scheduler.MultiStepLR(art_model.optimizer, milestones=[1], gamma=0.1)
        art_model.fit(cifar_data, cifar_labels, nb_epochs=1, update_batchnorm=True, scheduler=scheduler, batch_size=128)
    except ARTTestException as e:
        art_warning(e)
