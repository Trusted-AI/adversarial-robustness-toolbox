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
This module implements the TensorBoard support.
"""

from abc import ABC, abstractmethod

import numpy as np


class SummaryWriter(ABC):
    def __init__(self, tensor_board):
        """
        Create summary writer.

        :param tensor_board:
        """
        from tensorboardX import SummaryWriter as SummaryWriterTbx

        if isinstance(tensor_board, str):
            self._summary_writer = SummaryWriterTbx(tensor_board)
        else:
            self._summary_writer = SummaryWriterTbx()

    @property
    def summary_writer(self):
        return self._summary_writer

    @abstractmethod
    def update(self, batch_id, global_step, grad=None, patch=None, estimator=None, x=None, y=None):
        """
        Update the summary writer.

        :param batch_id:
        :param global_step:
        :param grad:
        :param patch:
        :param estimator:
        :param x:
        :param y:
        """
        raise NotImplementedError


class SummaryWriterDefault(SummaryWriter):
    def update(self, batch_id, global_step, grad=None, patch=None, estimator=None, x=None, y=None):
        """
        Update the summary writer.

        :param batch_id:
        :param global_step:
        :param grad:
        :param patch:
        :param estimator:
        :param x:
        :param y:
        """

        if grad is not None:
            self.summary_writer.add_scalar(
                "gradients/norm-L1/batch-{}".format(batch_id),
                np.linalg.norm(grad.flatten(), ord=1),
                global_step=global_step,
            )
            self.summary_writer.add_scalar(
                "gradients/norm-L2/batch-{}".format(batch_id),
                np.linalg.norm(grad.flatten(), ord=2),
                global_step=global_step,
            )
            self.summary_writer.add_scalar(
                "gradients/norm-Linf/batch-{}".format(batch_id),
                np.linalg.norm(grad.flatten(), ord=np.inf),
                global_step=global_step,
            )

        if patch is not None:
            self.summary_writer.add_image(
                "patch",
                patch,
                global_step=global_step,
            )

        if estimator is not None and x is not None and y is not None:
            if hasattr(estimator, "compute_losses"):
                losses = estimator.compute_losses(x=x, y=y)

                for key, value in losses.items():
                    self.summary_writer.add_scalar(
                        "loss/{}/batch-{}".format(key, batch_id),
                        np.mean(value),
                        global_step=global_step,
                    )
