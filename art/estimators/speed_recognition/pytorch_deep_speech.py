# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2020
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
"""
This module implements the task specific estimator for DeepSpeech, an end-to-end speech recognition in English and
Mandarin in PyTorch.

| Paper link: https://arxiv.org/abs/1512.02595
"""
import logging
from typing import List, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np
from deepspeech_pytorch.utils import load_model

from art.estimators.speed_recognition.speed_recognizer import SpeedRecognizerMixin
from art.estimators.pytorch import PyTorchEstimator
from art.utils import get_file
from art.config import ART_DATA_PATH

if TYPE_CHECKING:
    from deepspeech_pytorch.model import DeepSpeech

    from art.config import CLIP_VALUES_TYPE, PREPROCESSING_TYPE
    from art.defences.preprocessor.preprocessor import Preprocessor
    from art.defences.postprocessor.postprocessor import Postprocessor

logger = logging.getLogger(__name__)


class PyTorchDeepSpeech(SpeedRecognizerMixin, PyTorchEstimator):
    """
    This class implements a model-specific automatic speed recognizer using the end-to-end speech recognizer
    DeepSpeech and PyTorch.

    | Paper link: https://arxiv.org/abs/1512.02595
    """

    def __init__(
        self,
        model: Optional["DeepSpeech"] = None,
        pretrained_model: Optional[str] = None,
        filename: Optional[str] = None,
        url: Optional[str] = None,
        use_half: bool = False,
        clip_values: Optional["CLIP_VALUES_TYPE"] = None,
        preprocessing_defences: Union["Preprocessor", List["Preprocessor"], None] = None,
        postprocessing_defences: Union["Postprocessor", List["Postprocessor"], None] = None,
        preprocessing: "PREPROCESSING_TYPE" = None,
        device_type: str = "gpu"
    ):
        """
        Initialization of an instance PyTorchDeepSpeech.

        :param model: DeepSpeech model.
        :param pretrained_model: The choice of pretrained model if a pretrained model is required. Currently this
                                 estimator supports 3 different pretrained models consisting of `an4`, `librispeech`
                                 and `tedlium`.
        :param filename: Name of the file.
        :param url: Download URL.
        :param use_half: Whether to use FP16 for pretrained model.
        :param clip_values: Tuple of the form `(min, max)` of floats or `np.ndarray` representing the minimum and
               maximum values allowed for features. If floats are provided, these will be used as the range of all
               features. If arrays are provided, each value will be considered the bound for a feature, thus
               the shape of clip values needs to match the total number of features.
        :param preprocessing_defences: Preprocessing defence(s) to be applied by the estimator.
        :param postprocessing_defences: Postprocessing defence(s) to be applied by the estimator.
        :param preprocessing: Tuple of the form `(subtrahend, divisor)` of floats or `np.ndarray` of values to be
               used for data preprocessing. The first value will be subtracted from the input. The input will then
               be divided by the second one.
        :param device_type: Type of device to be used for model and tensors, if `cpu` run on CPU, if `gpu` run on GPU
                            if available otherwise run on CPU.
        """
        import torch

        # Super initialization
        super().__init__(
            clip_values=clip_values,
            preprocessing_defences=preprocessing_defences,
            postprocessing_defences=postprocessing_defences,
            preprocessing=preprocessing
        )

        # Check clip values
        if self.clip_values is not None:
            if not np.all(self.clip_values[0] == -1):
                raise ValueError("This estimator requires normalized input audios with clip_vales=(-1, 1).")
            if not np.all(self.clip_values[1] == 1):
                raise ValueError("This estimator requires normalized input audios with clip_vales=(-1, 1).")

        # Check preprocessing and postprocessing defences
        if self.preprocessing_defences is not None:
            raise ValueError("This estimator does not support `preprocessing_defences`.")
        if self.postprocessing_defences is not None:
            raise ValueError("This estimator does not support `postprocessing_defences`.")

        # Set cpu/gpu device
        self._device: str
        if device_type == "cpu" or not torch.cuda.is_available():
            self._device = torch.device("cpu")
        else:
            cuda_idx = torch.cuda.current_device()
            self._device = torch.device("cuda:{}".format(cuda_idx))

        # Load model
        if model is None:
            if pretrained_model == 'an4':
                filename, url = (
                    "an4_pretrained_v2.pth",
                    "https://github.com/SeanNaren/deepspeech.pytorch/releases/download/v2.0/an4_pretrained_v2.pth"
                )
            elif pretrained_model == 'librispeech':
                filename, url = (
                    "librispeech_pretrained_v2.pth",
                    "https://github.com/SeanNaren/deepspeech.pytorch/releases/download/v2.0/"
                    "librispeech_pretrained_v2.pth"
                )
            elif pretrained_model == 'tedlium':
                filename, url = (
                    "ted_pretrained_v2.pth",
                    "https://github.com/SeanNaren/deepspeech.pytorch/releases/download/v2.0/ted_pretrained_v2.pth"
                )
            elif pretrained_model is None:
                # If model is None and no pretrained model is selected, then we need to have parameters filename and
                # url to download, extract and load the automatic speed recognition model
                if filename is None or url is None:
                    filename, url = (
                        "librispeech_pretrained_v2.pth",
                        "https://github.com/SeanNaren/deepspeech.pytorch/releases/download/v2.0/"
                        "librispeech_pretrained_v2.pth"
                    )
            else:
                raise ValueError("The input pretrained model %s is not supported." % pretrained_model)

            # Download model
            model_path = get_file(filename=filename, path=ART_DATA_PATH, url=url, extract=False)

            # Then load model
            self._model = load_model(device=self._device, model_path=model_path, use_half=use_half)

        else:
            self._model = model

        # Push model to the corresponding device
        self._model.to(self._device)

    def loss_gradient(self, x: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compute the gradient of the loss function w.r.t. `x`.

        :param x: Samples of shape (nb_samples, height, width, nb_channels).
        :param y: Target values of format `List[Dict[Tensor]]`, one for each input image. The
                  fields of the Dict are as follows:

                  - boxes (FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format, with values \
                    between 0 and H and 0 and W
                  - labels (Int64Tensor[N]): the predicted labels for each image
                  - scores (Tensor[N]): the scores or each prediction.
        :return: Loss gradients of the same shape as `x`.
        """
        import torch
        import torchvision  # lgtm [py/repeated-import]

        self._model.train()

        # Apply preprocessing
        x, _ = self._apply_preprocessing(x, y=None, fit=False)

        if y is not None:
            for i, y_i in enumerate(y):
                y[i]["boxes"] = torch.tensor(y_i["boxes"], dtype=torch.float).to(self._device)
                y[i]["labels"] = torch.tensor(y_i["labels"], dtype=torch.int64).to(self._device)
                y[i]["scores"] = torch.tensor(y_i["scores"]).to(self._device)

        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        image_tensor_list = list()

        for i in range(x.shape[0]):
            if self.clip_values is not None:
                img = transform(x[i] / self.clip_values[1]).to(self._device)
            else:
                img = transform(x[i]).to(self._device)
            img.requires_grad = True
            image_tensor_list.append(img)

        output = self._model(image_tensor_list, y)

        # Compute the gradient and return
        loss = None
        for loss_name in self.attack_losses:
            if loss is None:
                loss = output[loss_name]
            else:
                loss = loss + output[loss_name]

        # Clean gradients
        self._model.zero_grad()

        # Compute gradients
        loss.backward(retain_graph=True)  # type: ignore

        grad_list = list()
        for img in image_tensor_list:
            gradients = img.grad.cpu().numpy().copy()
            grad_list.append(gradients)

        grads = np.stack(grad_list, axis=0)

        # BB
        grads = self._apply_preprocessing_gradient(x, grads)
        grads = np.swapaxes(grads, 1, 3)
        grads = np.swapaxes(grads, 1, 2)
        assert grads.shape == x.shape

        if self.clip_values is not None:
            grads = grads / self.clip_values[1]

        return grads

    def predict(self, x: np.ndarray, batch_size: int = 128, **kwargs) -> np.ndarray:
        """
        Perform prediction for a batch of inputs.

        :param x: Samples of shape (nb_samples, height, width, nb_channels).
        :param batch_size: Batch size.
        :return: Predictions of format `List[Dict[Tensor]]`, one for each input image. The
                 fields of the Dict are as follows:

                 - boxes (FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format, with values \
                   between 0 and H and 0 and W
                 - labels (Int64Tensor[N]): the predicted labels for each image
                 - scores (Tensor[N]): the scores or each prediction.
        """
        import torchvision  # lgtm [py/repeated-import]

        self._model.eval()

        # Apply preprocessing
        x, _ = self._apply_preprocessing(x, y=None, fit=False)

        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        image_tensor_list: List[np.ndarray] = list()

        if self.clip_values is not None:
            norm_factor = self.clip_values[1]
        else:
            norm_factor = 1.0
        for i in range(x.shape[0]):
            image_tensor_list.append(transform(x[i] / norm_factor).to(self._device))
        predictions = self._model(image_tensor_list)
        return predictions

    def fit(self, x: np.ndarray, y, batch_size: int = 128, nb_epochs: int = 20, **kwargs) -> None:
        raise NotImplementedError

    def get_activations(
        self, x: np.ndarray, layer: Union[int, str], batch_size: int, framework: bool = False
    ) -> np.ndarray:
        raise NotImplementedError

    def set_learning_phase(self, train: bool) -> None:
        raise NotImplementedError






from deepspeech_pytorch.configs.inference_config import EvalConfig
from deepspeech_pytorch.decoder import GreedyDecoder
from deepspeech_pytorch.loader.data_loader import SpectrogramDataset, AudioDataLoader
from deepspeech_pytorch.utils import load_model, load_decoder

from art.utils import get_file
from art.config import ART_DATA_PATH

from deepspeech_pytorch.loader.data_loader import SpectrogramParser

from deepspeech_pytorch.loader.data_loader import load_audio

from scipy.io import wavfile

from deepspeech_pytorch.configs.inference_config import LMConfig

device = torch.device("cpu")

path = get_file(filename='librispeech_pretrained_v2.pth', path=ART_DATA_PATH, url='https://github.com/SeanNaren/deepspeech.pytorch/releases/download/v2.0/librispeech_pretrained_v2.pth', extract=False)

model = load_model(device=device, model_path=path, use_half=False)

lm = LMConfig()

decoder = load_decoder(labels=model.labels, cfg=lm)
target_decoder = GreedyDecoder(model.labels, blank_index=model.labels.index('_'))


sample_rate = model.audio_conf.sample_rate

window_size = model.audio_conf.window_size

window_stride = model.audio_conf.window_stride

window = model.audio_conf.window.value

n_fft = int(sample_rate * window_size)

win_length = n_fft

hop_length = int(sample_rate * window_stride)

import librosa

D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window)



transformer = torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length, win_length=win_length, window_fn=torch.hamming_window, power=None)
D_ = transformer(torch.tensor(x))
spect_1x, phase_ = torchaudio.functional.magphase(D_)
spect_1 = torch.log1p(spect_1x)
mean1 = spect_1.mean()
std1 = spect_1.std()
spect_1.add_(-mean1)
spect_1.div_(std1)


from deepspeech_pytorch.loader.data_loader import _collate_fn


batch = [(spect_1, l1), (spect_2, l2)]

batch_idx = sorted(range(len(batch)), key=lambda i: batch[i][0].size(1), reverse=True)

inputs, targets, input_percentages, target_sizes = _collate_fn(batch)
input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
out, output_sizes = model(inputs, input_sizes)
output_sizes = output_sizes[batch_idx]
out = out[batch_idx]



from warpctc_pytorch import CTCLoss
criterion = CTCLoss()

out = out.transpose(0, 1)
float_out = out.float()
targets = torch.tensor(l1 + l2)
loss = criterion(float_out, targets, output_sizes, target_sizes).to(device)
loss = loss / inputs.size(0)


model.zero_grad()
loss.backward()

decoder = load_decoder(labels=model.labels, cfg=lm)
decoded_output, _ = decoder.decode(out, output_sizes)   ## Nho batch_idx




In [281]: x[0].grad

In [282]: x[1].grad

In [283]:

In [283]: D_1 = transformer(x[0])

In [284]: D_2 = transformer(x[1])

In [285]: spect_1, phase_1 = torchaudio.functional.magphase(D_1)

In [286]: spect_2, phase_2 = torchaudio.functional.magphase(D_2)

In [287]: spect_1 = torch.log1p(spect_1)

In [288]: spect_2 = torch.log1p(spect_2)

In [289]: mean1 = spect_1.mean()
     ...: std1 = spect_1.std()

In [290]: mean2 = spect_2.mean()
     ...: std2 = spect_2.std()

In [291]:

In [291]: spect_1 = spect_1 - mean1

In [292]: spect_2 = spect_2 - mean2

In [293]: spect_1 = spect_1 / std1

In [294]: spect_2 = spect_2 / std2

In [295]: batch = [(spect_1, l1), (spect_2, l2)]

In [296]: inputs, targets, input_percentages, target_sizes = _collate_fn(batch)
     ...: input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
     ...: out, output_sizes = model(inputs, input_sizes)
     ...: output_sizes = output_sizes[batch_idx]
     ...: out = out[batch_idx]
     ...:

In [297]: out = out.transpose(0, 1)
     ...: float_out = out.float()
     ...: targets = torch.tensor(l1 + l2)
     ...: loss = criterion(float_out, targets, output_sizes, target_sizes).to(device)
     ...: loss = loss / inputs.size(0)

In [298]: model.zero_grad()

In [299]: loss.backward()

In [300]: x[0].grad
Out[300]: tensor([-0.0110, -0.1356,  0.2700,  ..., -2.5302, -1.2972, -1.1815])

In [301]: x[1].grad
Out[301]: tensor([ 0.2616,  0.3763,  0.3973,  ..., -0.0391, -0.1326, -0.0944])

In [302]: x[1].grad.size()
Out[302]: torch.Size([18800])

In [303]: x[0].grad.size()
Out[303]: torch.Size([17040])





In [433]: x1 = load_audio('/home/minhtn/ibm/projects/tmp/deepspeech.pytorch/data/an4_dataset/val/an4/wav/cen3-mwhw-b.wav')

In [434]: x2 = load_audio('/home/minhtn/ibm/projects/tmp/deepspeech.pytorch/data/an4_dataset/val/an4/wav/an3-mblw-b.wav')

In [435]: x = np.array([x1, x2])

In [436]: for i in range(len(x)):
     ...:     x[i] = torch.from_numpy(x[i]).to(device)
     ...:     x[i].requires_grad = True
     ...:

In [437]: x[0].grad.shape
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
<ipython-input-437-ec61ee59a159> in <module>
----> 1 x[0].grad.shape

AttributeError: 'NoneType' object has no attribute 'shape'

In [438]: path = get_file(filename='an4_pretrained_v2.pth', path=ART_DATA_PATH, url='https://github.com/SeanNaren/deepspeech.p
     ...: ytorch/releases/download/v2.0/an4_pretrained_v2.pth', extract=False)

In [439]: path
Out[439]: '/home/minhtn/.art/data/an4_pretrained_v2.pth'

In [440]: model = load_model(device=device, model_path=path, use_half=False)

In [441]: path
Out[441]: '/home/minhtn/.art/data/an4_pretrained_v2.pth'

In [442]: D_1 = transformer(x[0])

In [443]: D_2 = transformer(x[1])

In [444]: spect_1, phase_1 = torchaudio.functional.magphase(D_1)

In [445]: spect_2, phase_2 = torchaudio.functional.magphase(D_2)

In [446]: spect_1 = torch.log1p(spect_1)

In [447]: spect_2 = torch.log1p(spect_2)

In [448]: mean1 = spect_1.mean()
     ...: std1 = spect_1.std()

In [449]: mean2 = spect_2.mean()
     ...: std2 = spect_2.std()

In [450]: spect_1 = spect_1 - mean1

In [451]: spect_2 = spect_2 - mean2

In [452]: spect_1 = spect_1 / std1

In [453]: spect_2 = spect_2 / std2

In [454]: batch = [(spect_1, l1), (spect_2, l2)]
     ...:
     ...: batch_idx = sorted(range(len(batch)), key=lambda i: batch[i][0].size(1), reverse=True)
     ...:

In [455]: inputs, targets, input_percentages, target_sizes = _collate_fn(batch)
     ...: input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
     ...: out, output_sizes = model(inputs, input_sizes)
     ...: output_sizes = output_sizes[batch_idx]
     ...: out = out[batch_idx]
     ...:

In [456]: out = out.transpose(0, 1)
     ...: float_out = out.float()
     ...: targets = torch.tensor(l1 + l2)
     ...: loss = criterion(float_out, targets, output_sizes, target_sizes).to(device)
     ...: loss = loss / inputs.size(0)
     ...:

In [457]: model.zero_grad()

In [458]: loss.backward()

In [459]: x[0].grad.shape
Out[459]: torch.Size([17600])

In [460]: x[0].grad
Out[460]:
tensor([-9.3802e-03, -4.9995e-04, -5.0607e-03,  ..., -4.9801e-01,
         8.0346e-01, -2.0202e-01])

In [461]: x[1].grad
Out[461]: tensor([ 0.0198,  0.0770, -0.0957,  ..., -0.0147, -0.0134, -0.0083])

In [462]: x[1].grad.shape
Out[462]: torch.Size([20800])

In [463]: out, output_sizes = model(inputs, input_sizes)

In [464]: decoded_output, off = decoder.decode(out, output_sizes)

In [465]: decoded_output[0][0]
Out[465]: 'ENTER SIXTY'

In [466]: decoded_output[1][0]
Out[466]: 'ONE FIFTY'





