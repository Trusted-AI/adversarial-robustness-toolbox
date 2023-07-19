import torch
import numpy as np
import random

from tqdm import tqdm

from typing import List, Optional, Tuple, Union, Any, TYPE_CHECKING
from art.estimators.classification.pytorch import PyTorchClassifier
from art.utils import check_and_transform_label_format


class HuggingFaceClassifier(PyTorchClassifier):

    def __init__(self, model, loss, input_shape, nb_classes, optimizer, clip_values=(0, 1),
                 preprocessing: "PREPROCESSING_TYPE" = (0.0, 1.0), processor=None):
        import transformers

        assert isinstance(model, transformers.PreTrainedModel)

        self.processor = processor

        super().__init__(
            model=model,
            loss=loss,
            input_shape=input_shape,
            nb_classes=nb_classes,
            optimizer=optimizer,
            channels_first=True,
            clip_values=clip_values,
            preprocessing_defences=None,
            postprocessing_defences=None,
            preprocessing=preprocessing,
            device_type='gpu')

        import functools

        def prefix_function(function, postfunction):
            """
            Huggingface returns logit under outputs.logits.
            To make this compatible with ART we wrap the forward pass function
            of a HF model here, which automatically extracts the logits.
            """
            @functools.wraps(function)
            def run(*args, **kwargs):
                outputs = function(*args, **kwargs)
                return postfunction(outputs)
            return run

        def get_logits(outputs):
            return outputs.logits

        self.model.forward = prefix_function(self.model.forward, get_logits)

    def __call__(self, image):

        if self.processor is not None:
            image = self.processor(images=image, return_tensors="pt")
            image.to(self._device)
            outputs = self.model(**image)
        else:
            if not isinstance(image, torch.Tensor):
                image = torch.from_numpy(image).to(self._device)
            outputs = self.model(image)
        return outputs

    def forward(self, image):
        if self.processor is not None:
            image = self.processor(images=image, return_tensors="pt")
            image.to(self._device)
            outputs = self.model(**image)
        else:
            if not isinstance(image, torch.Tensor):
                image = torch.from_numpy(image).to(self._device)
            outputs = self.model(image)
        return outputs

    def get_grad(self, image, labels, loss_fn):
        """
        Get gradient wrt input image.
        Testing function. To be removed in final PR

        :param image:
        :param labels:
        :return:
        """

        if not isinstance(image, torch.Tensor):
            labels = torch.from_numpy(labels)
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image)
        image.requires_grad = True
        self.model.eval()
        self.model.zero_grad()

        if self.processor is not None:
            image = self.processor(images=image, return_tensors="pt")
            image.to(self._device)
            image['pixel_values'].requires_grad = True
            loss = self.model(**image, labels=labels)[0]
        else:
            out = self.model(image)
            loss = loss_fn(out, labels)
        loss.backward()
        self.model.eval()

        return image.grad

    def make_adv_example(self, x, y):
        """
        Testing function: to be removed in final PR
        """
        self.epsilon = 8 / 255
        self.attack_lr = 1 / 255
        upsampler = torch.nn.Upsample(scale_factor=7, mode='nearest')

        x = x.to(self._device)
        y = y.to(self._device)

        x = upsampler(x)  # hard code resize for now
        model_outputs = self.model(x)
        acc = self.get_accuracy(model_outputs, y)
        print('clean acc is ', acc)

        x_adv = x.detach().clone()
        x_adv.requires_grad = True

        for _ in range(30):
            self.model.zero_grad()
            # x_adv.zero_grad()
            grad = self.get_grad(x_adv, y, loss_fn=torch.nn.CrossEntropyLoss())
            with torch.no_grad():
                grad = grad.sign()
                x_adv = x_adv + self.attack_lr * grad

                # Projection
                noise = torch.clamp(x_adv - x, min=-self.epsilon, max=self.epsilon)
                x_adv = torch.clamp(x + noise, min=0, max=1)

        model_outputs = self.model(x_adv)
        acc = self.get_accuracy(model_outputs, y)
        print('adv acc is ', acc)

    def train(self, x, y,
              batch_size: int = 128,
              nb_epochs: int = 10,
              training_mode: bool = True,
              drop_last: bool = False,
              scheduler: Optional[Any] = None,
              verbose=True,
              **kwargs,):
        import torch

        # Set model mode
        self.model.train()

        if self._optimizer is None:  # pragma: no cover
            raise ValueError("An optimizer is needed to train the model, but none for provided.")

        y = check_and_transform_label_format(y, nb_classes=self.nb_classes)

        # Apply preprocessing
        x_preprocessed, y_preprocessed = self._apply_preprocessing(x, y, fit=True)

        # Check label shape
        y_preprocessed = self.reduce_labels(y_preprocessed)

        num_batch = len(x_preprocessed) / float(batch_size)
        if drop_last:
            num_batch = int(np.floor(num_batch))
        else:
            num_batch = int(np.ceil(num_batch))
        ind = np.arange(len(x_preprocessed))

        # Start training
        for _ in tqdm(range(nb_epochs)):
            # Shuffle the examples
            random.shuffle(ind)
            pbar = tqdm(range(num_batch), disable=not verbose)

            epoch_loss = []
            epoch_acc = []

            # Train for one epoch
            for m in pbar:
                i_batch = np.copy(x_preprocessed[ind[m * batch_size: (m + 1) * batch_size]])
                i_batch = torch.from_numpy(i_batch).to(self._device)
                # i_batch = upsampler(i_batch)  # hard code resize for now
                i_batch = self.processor(i_batch)
                o_batch = torch.from_numpy(y_preprocessed[ind[m * batch_size: (m + 1) * batch_size]]).to(self._device)

                # Zero the parameter gradients
                self._optimizer.zero_grad()

                # Perform prediction
                model_outputs = self.model(i_batch)
                acc = self.get_accuracy(model_outputs.logits, o_batch)

                # Form the loss function
                loss = self._loss(model_outputs.logits, o_batch)

                # Do training
                if self._use_amp:  # pragma: no cover
                    from apex import amp  # pylint: disable=E0611

                    with amp.scale_loss(loss, self._optimizer) as scaled_loss:
                        scaled_loss.backward()

                else:
                    loss.backward()

                self._optimizer.step()
                epoch_loss.append(loss)
                epoch_acc.append(acc)

                if verbose:
                    pbar.set_description(
                        f"Loss {torch.mean(torch.stack(epoch_loss)):.2f} "
                        f"Acc {np.mean(epoch_acc):.2f}"
                    )

            if scheduler is not None:
                scheduler.step()

            torch.save(self.model.state_dict(), 'hf_model.pt')

    @staticmethod
    def get_accuracy(preds: Union[np.ndarray, "torch.Tensor"], labels: Union[np.ndarray, "torch.Tensor"]) -> np.ndarray:
        """
        Helper function to print out the accuracy during training

        :param preds: model predictions
        :param labels: ground truth labels (not one hot)
        :return: prediction accuracy
        """
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()

        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()

        return np.sum(np.argmax(preds, axis=1) == labels) / len(labels)

