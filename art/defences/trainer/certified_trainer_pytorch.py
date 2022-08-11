import logging
from typing import Optional, Union, Any, TYPE_CHECKING

import numpy as np
import random
from tqdm import tqdm

from art.defences.trainer.trainer import Trainer
from art.defences.trainer.adversarial_trainer import AdversarialTrainer
from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent import ProjectedGradientDescent
from art.utils import check_and_transform_label_format

import torch

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_LOSS_GRADIENTS_TYPE
    from art.estimators.certification.deep_z import PytorchDeepZ

logger = logging.getLogger(__name__)


class AdversarialTrainerCertified(Trainer):
    """
    Class performing adversarial training following Madry's Protocol.

    | Paper link: https://arxiv.org/abs/1706.06083

    | Please keep in mind the limitations of defences. While adversarial training is widely regarded as a promising,
        principled approach to making classifiers more robust (see https://arxiv.org/abs/1802.00420), very careful
        evaluations are required to assess its effectiveness case by case (see https://arxiv.org/abs/1902.06705).
    """

    def __init__(
        self,
        classifier: PytorchDeepZ,
        nb_epochs: Optional[int] = 205,
        batch_size: Optional[int] = 128,
        eps: Union[int, float] = 8,
        eps_step: Union[int, float] = 2,
        max_iter: int = 7,
        num_random_init: int = 1,
    ) -> None:
        """
        Create an :class:`.AdversarialTrainerMadryPGD` instance.

        Default values are for CIFAR-10 in pixel range 0-255.

        :param classifier: Classifier to train adversarially.
        :param nb_epochs: Number of training epochs.
        :param batch_size: Size of the batch on which adversarial samples are generated.
        :param eps: Maximum perturbation that the attacker can introduce.
        :param eps_step: Attack step size (input variation) at each iteration.
        :param max_iter: The maximum number of iterations.
        :param num_random_init: Number of random initialisations within the epsilon ball. For num_random_init=0
                                starting at the original input.
        """
        super().__init__(classifier=classifier)  # type: ignore
        self._classifier: PytorchDeepZ
        self.batch_size = batch_size
        self.nb_epochs = nb_epochs

        # Setting up adversary and perform adversarial training:
        self.attack = ProjectedGradientDescent(
            classifier,
            eps=eps,
            eps_step=eps_step,
            max_iter=max_iter,
            num_random_init=num_random_init,
        )

    def fit(  # pylint: disable=W0221
            self,
            x: np.ndarray,
            y: np.ndarray,
            batch_size: int = 128,
            nb_epochs: int = 10,
            training_mode: bool = True,
            scheduler: Optional[Any] = None,
            bound: float = 0.25,
            certification_batch_size: int = 10,
            loss_weighting: float = 0.1,
            use_schedule: bool = True,
            **kwargs,
    ) -> None:
        """
        Fit the classifier on the training set `(x, y)`.

        :param x: Training data.
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or index labels of
                  shape (nb_samples,).
        :param pgd_batch_size: Size of batches to use for PGD training
        :param certification_batch_size: Size of batches to use for certified training. NB, this will run the data
                                         sequentially accumulating gradients over the batch size.
        :param loss_weighting:
        :param nb_epochs: Number of epochs to use for training.
        :param training_mode: `True` for model set to training mode and `'False` for model set to evaluation mode.
        :param scheduler: Learning rate scheduler to run at the start of every epoch.
        :param kwargs: Dictionary of framework-specific arguments. This parameter is not currently supported for PyTorch
               and providing it takes no effect.
        """

        # Set model mode
        # self._model.train(mode=training_mode)
        pgd_batch_size = batch_size

        if self._classifier._optimizer is None:  # pragma: no cover
            raise ValueError("An optimizer is needed to train the model, but none for provided.")

        y = check_and_transform_label_format(y, nb_classes=self._classifier.nb_classes)

        # Apply preprocessing
        x_preprocessed, y_preprocessed = self._classifier._apply_preprocessing(x, y, fit=True)

        # Check label shape
        y_preprocessed = self._classifier.reduce_labels(y_preprocessed)

        num_batch = int(np.ceil(len(x_preprocessed) / float(pgd_batch_size)))
        ind = np.arange(len(x_preprocessed))
        from sklearn.utils import shuffle

        x_cert = np.copy(x_preprocessed)
        y_cert = np.copy(y_preprocessed)

        # Start training
        if use_schedule:
            step_per_epoch = bound / nb_epochs
            bound = 0.0

        for epoch in tqdm(range(nb_epochs)):
            if use_schedule:
                bound += step_per_epoch
            # Shuffle the examples
            random.shuffle(ind)

            # Train for one epoch
            for m in range(num_batch):
                certified_loss = 0.0
                samples_certified = 0
                # Zero the parameter gradients
                self._classifier._optimizer.zero_grad()

                # get the certified loss
                x_cert, y_cert = shuffle(x_cert, y_cert)
                for i, (sample, label) in enumerate(zip(x_cert, y_cert)):
                    print(i)
                    eps_bound = np.eye(784) * bound
                    self._classifier.set_forward_mode('concrete')
                    concrete_pred = self._classifier.model.forward(sample)
                    concrete_pred = torch.argmax(concrete_pred)
                    processed_sample, eps_bound = self._classifier.pre_process(cent=np.copy(sample), eps=eps_bound)
                    processed_sample = np.expand_dims(processed_sample, axis=0)

                    # Perform prediction
                    self._classifier.set_forward_mode('abstract')
                    bias, eps = self._classifier.model.forward(eps=eps_bound, cent=processed_sample)
                    # Form the loss function
                    bias = torch.unsqueeze(bias, dim=0)
                    certified_loss += self._classifier.max_logit_loss(
                        output=torch.cat((bias, eps)), target=np.expand_dims(label, axis=0)
                    )

                    certification_results = []
                    bias = torch.squeeze(bias).detach().cpu().numpy()
                    eps = eps.detach().cpu().numpy()

                    for k in range(self._classifier.nb_classes):
                        if k != concrete_pred:
                            cert_via_sub = self._classifier.certify_via_subtraction(
                                predicted_class=concrete_pred, class_to_consider=k, cent=bias, eps=eps
                            )
                            certification_results.append(cert_via_sub)

                    if all(certification_results):
                        samples_certified += 1

                    if (i + 1) % certification_batch_size == 0 and i > 0:
                        break

                certified_loss /= certification_batch_size
                # Concrete PGD loss
                i_batch = np.copy(x_preprocessed[ind[m * pgd_batch_size: (m + 1) * pgd_batch_size]]).astype("float32")
                o_batch = y_preprocessed[ind[m * pgd_batch_size: (m + 1) * pgd_batch_size]]

                # Perform prediction
                self._classifier.set_forward_mode('concrete')
                self.attack = ProjectedGradientDescent(
                    estimator=self._classifier,
                    eps=0.25,
                    eps_step=0.05,
                    max_iter=20,
                    num_random_init=1,
                )
                i_batch = self.attack.generate(i_batch, y=o_batch)
                self._classifier.model.zero_grad()
                model_outputs = self._classifier.model.forward(i_batch)
                acc = self._classifier.get_accuracy(model_outputs, o_batch)

                # Form the loss function
                pgd_loss = self._classifier._loss(model_outputs, torch.from_numpy(o_batch).to(self._classifier.device))
                print('')
                print("Epoch {}, Batch {}/{}:".format(epoch, m, num_batch))
                print("Loss is {} Cert Loss is {}".format(pgd_loss, certified_loss))
                print("Acc is {} Cert Acc is {}".format(acc, samples_certified / certification_batch_size))
                loss = certified_loss * loss_weighting + pgd_loss * (1 - loss_weighting)
                # Do training
                if self._classifier._use_amp:  # pragma: no cover
                    from apex import amp  # pylint: disable=E0611

                    with amp.scale_loss(loss, self._classifier._optimizer) as scaled_loss:
                        scaled_loss.backward()

                else:
                    loss.backward()

                self._classifier._optimizer.step()
