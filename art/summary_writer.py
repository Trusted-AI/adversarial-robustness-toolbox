# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2021
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
This module defines and implements the summary writers for TensorBoard output.
"""

from abc import ABC, abstractmethod
from math import sqrt
from typing import Dict, List, Optional, Union

import numpy as np


class SummaryWriter(ABC):
    """
    This abstract base class defines the API for summary writers.
    """

    def __init__(self, summary_writer: Union[str, bool]):
        """
        Create summary writer.

        :param summary_writer: Activate summary writer for TensorBoard.
                       Default is `False` and deactivated summary writer.
                       If `True` save runs/CURRENT_DATETIME_HOSTNAME in current directory.
                       If of type `str` save in path.
                       Use hierarchical folder structure to compare between runs easily. e.g. pass in
                       ‘runs/exp1’, ‘runs/exp2’, etc. for each new experiment to compare across them.
        """
        self._summary_writer_arg = summary_writer
        self._init_counter = 0

        self._init_summary_writer(summary_writer, init_counter=0)

    @property
    def summary_writer(self):
        """
        Return the TensorBoardX summary writer instance.
        """
        return self._summary_writer

    @abstractmethod
    def update(
        self, batch_id, global_step, grad=None, patch=None, estimator=None, x=None, y=None, targeted=False, **kwargs
    ):
        """
        Update the summary writer.

        :param batch_id: Id of the current mini-batch.
        :param global_step: Global iteration step.
        :param grad: Loss gradients.
        :param patch: Adversarial patch.
        :param estimator: The estimator to evaluate or calculate gradients of `grad` is None to obtain new metrics.
        :param x: Input data.
        :param y: True or target labels.
        :param targeted: Indicates whether the attack is targeted (True) or untargeted (False).
        """
        raise NotImplementedError

    def _init_summary_writer(self, summary_writer, init_counter):
        """
        Initialise the summary writer.

        :param summary_writer: Activate summary writer for TensorBoard.
                       Default is `False` and deactivated summary writer.
                       If `True` save runs/CURRENT_DATETIME_HOSTNAME in current directory.
                       If of type `str` save in path.
                       Use hierarchical folder structure to compare between runs easily. e.g. pass in
                       ‘runs/exp1’, ‘runs/exp2’, etc. for each new experiment to compare across them.
        """
        from tensorboardX import SummaryWriter as SummaryWriterTbx

        if isinstance(summary_writer, str):
            comment = "generate-{}".format(init_counter)
            logdir = summary_writer + "/" + comment
            self._summary_writer = SummaryWriterTbx(logdir=logdir)
        else:
            comment = "-generate-{}".format(init_counter)
            self._summary_writer = SummaryWriterTbx(comment=comment)

    def reset(self):
        """
        Flush and reset the summary writer.
        """
        self.summary_writer.flush()
        self._init_counter += 1
        self._init_summary_writer(self._summary_writer_arg, init_counter=self._init_counter)


class SummaryWriterDefault(SummaryWriter):
    """
    Implementation of the default ART Summary Writer.
    """

    def __init__(
        self,
        summary_writer: Union[str, bool],
        ind_1: bool = False,
        ind_2: bool = False,
        ind_3: bool = False,
        ind_4: bool = False,
    ):
        super().__init__(summary_writer=summary_writer)

        self.ind_1 = ind_1
        self.ind_2 = ind_2
        self.ind_3 = ind_3
        self.ind_4 = ind_4

        self.loss = None
        self.loss_prev: Dict[str, np.ndarray] = dict()
        self.losses: Dict[str, List[np.ndarray]] = dict()

        self.i_3: Dict[str, np.ndarray] = dict()
        self.i_4: Dict[str, np.ndarray] = dict()

    def update(
        self,
        batch_id: int,
        global_step: int,
        grad: Optional[np.ndarray] = None,
        patch: Optional[np.ndarray] = None,
        estimator=None,
        x: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        targeted: bool = False,
        **kwargs,
    ):
        """
        Update the summary writer.

        :param batch_id: Id of the current mini-batch.
        :param global_step: Global iteration step.
        :param grad: Loss gradients.
        :param patch: Adversarial patch.
        :param estimator: The estimator to evaluate or calculate gradients of `grad` is None to obtain new metrics.
        :param x: Input data.
        :param y: True or target labels.
        :param targeted: Indicates whether the attack is targeted (True) or untargeted (False).
        """

        # Gradients
        if grad is not None:

            l_1 = np.linalg.norm(grad.reshape(grad.shape[0], -1), axis=1, ord=1)
            self.summary_writer.add_scalars(
                "gradients/norm-L1/batch-{}".format(batch_id),
                {str(i): v for i, v in enumerate(l_1)},
                global_step=global_step,
            )

            l_2 = np.linalg.norm(grad.reshape(grad.shape[0], -1), axis=1, ord=2)
            self.summary_writer.add_scalars(
                "gradients/norm-L2/batch-{}".format(batch_id),
                {str(i): v for i, v in enumerate(l_2)},
                global_step=global_step,
            )

            l_inf = np.linalg.norm(grad.reshape(grad.shape[0], -1), axis=1, ord=np.inf)
            self.summary_writer.add_scalars(
                "gradients/norm-Linf/batch-{}".format(batch_id),
                {str(i): v for i, v in enumerate(l_inf)},
                global_step=global_step,
            )

        # Patch
        if patch is not None:
            if patch.shape[2] in [1, 3, 4]:
                patch = np.transpose(patch, (2, 0, 1))
            self.summary_writer.add_image(
                "patch",
                patch,
                global_step=global_step,
            )

        # Losses
        if estimator is not None and x is not None and y is not None:
            if hasattr(estimator, "compute_losses"):
                losses = estimator.compute_losses(x=x, y=y)

                for key, value in losses.items():
                    if np.ndim(value) == 0:
                        self.summary_writer.add_scalar(
                            "loss/{}/batch-{}".format(key, batch_id),
                            value,
                            global_step=global_step,
                        )
                    else:
                        self.summary_writer.add_scalars(
                            "loss/{}/batch-{}".format(key, batch_id),
                            {str(i): v for i, v in enumerate(value)},
                            global_step=global_step,
                        )

            elif hasattr(estimator, "compute_loss"):
                loss = estimator.compute_loss(x=x, y=y)

                if np.ndim(loss) == 0:
                    self.summary_writer.add_scalar(
                        "loss/batch-{}".format(batch_id),
                        loss,
                        global_step=global_step,
                    )
                else:
                    self.summary_writer.add_scalars(
                        "loss/batch-{}".format(batch_id),
                        {str(i): v for i, v in enumerate(loss)},
                        global_step=global_step,
                    )

        # Indicators of Attack Failure by Pintor et al. (2021)
        # Paper link: https://arxiv.org/abs/2106.09947
        if self.ind_1:  # Silent Success
            from art.estimators.classification.classifier import ClassifierMixin

            if isinstance(estimator, ClassifierMixin):
                y_pred = estimator.predict(x)  # type: ignore
                self.i_1 = np.argmax(y_pred, axis=1) == np.argmax(y, axis=1)
                self.summary_writer.add_scalars(
                    "Attack Failure Indicator 1 - Silent Success/batch-{}".format(batch_id),
                    {str(i): v for i, v in enumerate(self.i_1)},
                    global_step=global_step,
                )
            else:
                raise ValueError(
                    "Attack Failure Indicator 1 is only supported for classification, for the current "
                    "`estimator` set `ind_1=False`."
                )

        if self.ind_2:  # Break-point Angle
            losses = estimator.compute_loss(x=x, y=y)

            if str(batch_id) not in self.losses:
                self.losses[str(batch_id)] = list()

            self.losses[str(batch_id)].append(losses)

            self.i_2 = np.ones_like(losses)

            if len(self.losses[str(batch_id)]) >= 3:

                delta_loss = self.losses[str(batch_id)][0] - self.losses[str(batch_id)][-1]
                delta_step = global_step

                side_b = sqrt(2.0)

                for i_step in range(1, len(self.losses[str(batch_id)]) - 1):

                    side_a = np.sqrt(
                        np.square((self.losses[str(batch_id)][0] - self.losses[str(batch_id)][i_step]) / delta_loss)
                        + (i_step / delta_step) ** 2
                    )
                    side_c = np.sqrt(
                        np.square((self.losses[str(batch_id)][i_step] - self.losses[str(batch_id)][-1]) / delta_loss)
                        + ((delta_step - i_step) / delta_step) ** 2
                    )
                    cos_beta = -(side_b ** 2 - (side_a ** 2 + side_c ** 2)) / (2 * side_a * side_c)

                    i_2_step = 1 - np.abs(cos_beta)
                    self.i_2 = np.minimum(self.i_2, i_2_step)

                if np.ndim(self.i_2) == 0:
                    self.summary_writer.add_scalar(
                        "Attack Failure Indicator 2 - Break-point Angle/batch-{}".format(batch_id),
                        self.i_2,
                        global_step=global_step,
                    )
                else:
                    self.summary_writer.add_scalars(
                        "Attack Failure Indicator 2 - Break-point Angle/batch-{}".format(batch_id),
                        {str(i): v for i, v in enumerate(self.i_2)},
                        global_step=global_step,
                    )

        if self.ind_3:  # Diverging (Increasing) Loss
            loss = estimator.compute_loss(x=x, y=y)

            if str(batch_id) in self.i_3:
                if targeted:
                    if isinstance(loss, float):
                        loss_add = loss
                    else:
                        loss_add = loss[loss > self.loss_prev[str(batch_id)]]
                    self.i_3[str(batch_id)][loss > self.loss_prev[str(batch_id)]] += loss_add
                else:
                    if isinstance(loss, float):
                        loss_add = loss
                    else:
                        loss_add = loss[loss < self.loss_prev[str(batch_id)]]
                    self.i_3[str(batch_id)][loss < self.loss_prev[str(batch_id)]] += loss_add
            else:
                self.i_3[str(batch_id)] = np.zeros_like(loss)

            if np.ndim(self.i_3[str(batch_id)]) == 0:
                self.summary_writer.add_scalar(
                    "Attack Failure Indicator 3 - Diverging Loss/batch-{}".format(batch_id),
                    self.i_3[str(batch_id)],
                    global_step=global_step,
                )
            else:
                self.summary_writer.add_scalars(
                    "Attack Failure Indicator 3 - Diverging Loss/batch-{}".format(batch_id),
                    {str(i): v for i, v in enumerate(self.i_3[str(batch_id)])},
                    global_step=global_step,
                )

            self.loss_prev[str(batch_id)] = loss

        if self.ind_4:  # Zero Gradients

            threshold = 0.0

            if str(batch_id) not in self.i_4:
                self.i_4[str(batch_id)] = np.zeros(grad.shape[0])

            self.i_4[str(batch_id)][np.linalg.norm(grad.reshape(grad.shape[0], -1), axis=1, ord=2) <= threshold] += 1

            self.summary_writer.add_scalars(
                "Attack Failure Indicator 4 - Zero Gradients/batch-{}".format(batch_id),
                {str(i): v for i, v in enumerate(self.i_4[str(batch_id)] / global_step)},
                global_step=global_step,
            )
