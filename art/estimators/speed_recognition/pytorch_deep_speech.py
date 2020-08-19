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

            self._model = load_model(device=self._device, model_path=path, use_half=False)

        else:
            self._model = model



        self._model.to(self._device)

        self.attack_losses: Tuple[str, ...] = attack_losses

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


    @staticmethod
    def _load_model(
        images: "Tensor",
        filename: Optional[str] = None,
        url: Optional[str] = None,
        obj_detection_model: Optional["FasterRCNNMetaArch"] = None,
        is_training: bool = False,
        groundtruth_boxes_list: Optional[List["Tensor"]] = None,
        groundtruth_classes_list: Optional[List["Tensor"]] = None,
        groundtruth_weights_list: Optional[List["Tensor"]] = None,
    ) -> Tuple[Dict[str, "Tensor"], ...]:
        """
        Download, extract and load a model from a URL if it not already in the cache. The file at indicated by `url`
        is downloaded to the path ~/.art/data and given the name `filename`. Files in tar, tar.gz, tar.bz, and zip
        formats will also be extracted. Then the model is loaded, pipelined and its outputs are returned as a tuple
        of (predictions, losses, detections).

        :param images: Input samples of shape (nb_samples, height, width, nb_channels).
        :param filename: Name of the file.
        :param url: Download URL.
        :param is_training: A boolean indicating whether the training version of the computation graph should be
                            constructed.
        :param groundtruth_boxes_list: A list of 2-D tf.float32 tensors of shape [num_boxes, 4] containing
                                       coordinates of the groundtruth boxes. Groundtruth boxes are provided in
                                       [y_min, x_min, y_max, x_max] format and also assumed to be normalized and
                                       clipped relative to the image window with conditions y_min <= y_max and
                                       x_min <= x_max.
        :param groundtruth_classes_list: A list of 1-D tf.float32 tensors of shape [num_boxes] containing the class
                                         targets with the zero index which is assumed to map to the first
                                         non-background class.
        :param groundtruth_weights_list: A list of 1-D tf.float32 tensors of shape [num_boxes] containing weights for
                                         groundtruth boxes.
        :return: A tuple of (predictions, losses, detections):

                    - predictions: a dictionary holding "raw" prediction tensors.
                    - losses: a dictionary mapping loss keys (`Loss/RPNLoss/localization_loss`,
                              `Loss/RPNLoss/objectness_loss`, `Loss/BoxClassifierLoss/localization_loss`,
                              `Loss/BoxClassifierLoss/classification_loss`) to scalar tensors representing
                              corresponding loss values.
                    - detections: a dictionary containing final detection results.
        """
        from object_detection.utils import variables_helper

        if obj_detection_model is None:
            from object_detection.utils import config_util
            from object_detection.builders import model_builder

            # If obj_detection_model is None, then we need to have parameters filename and url to download, extract
            # and load the object detection model
            if filename is None or url is None:
                raise ValueError(
                    "Need input parameters `filename` and `url` to download, "
                    "extract and load the object detection model."
                )

            # Download and extract
            path = get_file(filename=filename, path=ART_DATA_PATH, url=url, extract=True)

            # Load model config
            pipeline_config = path + "/pipeline.config"
            configs = config_util.get_configs_from_pipeline_file(pipeline_config)
            configs["model"].faster_rcnn.second_stage_batch_size = configs[
                "model"
            ].faster_rcnn.first_stage_max_proposals

            # Load model
            obj_detection_model = model_builder.build(
                model_config=configs["model"], is_training=is_training, add_summaries=False
            )

        # Provide groundtruth
        groundtruth_classes_list = [
            tf.one_hot(groundtruth_class, obj_detection_model.num_classes)
            for groundtruth_class in groundtruth_classes_list
        ]

        obj_detection_model.provide_groundtruth(
            groundtruth_boxes_list=groundtruth_boxes_list,
            groundtruth_classes_list=groundtruth_classes_list,
            groundtruth_weights_list=groundtruth_weights_list,
        )

        # Create model pipeline
        images *= 255.0
        preprocessed_images, true_image_shapes = obj_detection_model.preprocess(images)
        predictions = obj_detection_model.predict(preprocessed_images, true_image_shapes)
        losses = obj_detection_model.loss(predictions, true_image_shapes)
        detections = obj_detection_model.postprocess(predictions, true_image_shapes)

        # Initialize variables from checkpoint
        # Get variables to restore
        variables_to_restore = obj_detection_model.restore_map(
            fine_tune_checkpoint_type="detection", load_all_detection_checkpoint_vars=True
        )

        # Get variables from checkpoint
        fine_tune_checkpoint_path = path + "/model.ckpt"
        vars_in_ckpt = variables_helper.get_variables_available_in_checkpoint(
            variables_to_restore, fine_tune_checkpoint_path, include_global_step=False
        )

        # Initialize from checkpoint
        tf.train.init_from_checkpoint(fine_tune_checkpoint_path, vars_in_ckpt)

        return predictions, losses, detections



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





