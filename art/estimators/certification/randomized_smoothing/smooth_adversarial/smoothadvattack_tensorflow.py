# MIT License

# Copyright (c) 2019 Hadi Salman, Greg Yang, Jerry Li, Huan Zhang, Pengchuan Zhang, Ilya Razenshteyn, Sebastien Bubeck

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
This is authors' implementation of Smooth Adversarial Attack using PGD and DDN
in Tensorflow

| Paper link: https://arxiv.org/pdf/1906.04584.pdf

"""
from abc import ABCMeta, abstractmethod

import tensorflow as tf


class Attacker(metaclass=ABCMeta):
    """
    Abstract class for the Attacker. Consists of the single attack function
    extended by the implementation

    """

    @abstractmethod
    def attack(self, model, inputs, labels):
        """
        Abstract function definition for the attack call

        """
        raise NotImplementedError


def get_tensor_mode(input_tensor: tf.Tensor, dim=-1):
    """
    Computes the mode of the tensor along the dimension specified.

    :param input_tensor: Input tensor.
    :param dim: Dimension or axis along which mode is to be calculated.
    :return: `Tensor with mode computed as defined above`
    """
    datatype = input_tensor.dtype
    minval = tf.math.reduce_min(input_tensor)
    input_tensor = input_tensor - minval
    c_int = tf.math.bincount(
      input_tensor, axis = -1, dtype=datatype, minlength=tf.math.reduce_max(input_tensor) + 1
    )
    idx = tf.math.argmax(c_int, axis=-1, output_type=datatype)
    mode_tensor = idx + minval
    return mode_tensor


# Modification of the code from https://github.com/jeromerony/fast_adversarial
class PgdL2(Attacker):
    """
    PGD attack

    Parameters
    ----------
    steps : int
        Number of steps for the optimization.
    max_norm : float or None, optional
        If specified, the norms of the perturbations will not be greater than this value which might lower success rate.
    device : torch.device, optional
        Device on which to perform the attack.

    """

    def __init__(
        self, steps: int, random_start: bool = True, max_norm: float = 1.0, device: tf.device = tf.device("cpu")
    ) -> None:
        super().__init__()
        self.steps = steps
        self.random_start = random_start
        self.max_norm = max_norm
        self.device = device

    def attack(
        self,
        model: tf.Module,
        inputs: tf.Tensor,
        labels: tf.Tensor,
        noise: tf.Tensor = None,
        num_noise_vectors=1,
        targeted: bool = False,
        no_grad=False,
    ) -> tf.Tensor:
        if num_noise_vectors == 1:
            return self._attack(model, inputs, labels, noise, targeted)
        if no_grad:
            return self._attack_multinoise_no_grad(model, inputs, labels, noise, num_noise_vectors)
        return self._attack_multinoise(model, inputs, labels, noise, num_noise_vectors, targeted)

    def _attack(
        self, model: tf.Module, inputs: tf.Tensor, labels: tf.Tensor, noise: tf.Tensor = None, targeted: bool = False
    ) -> tf.Tensor:
        """
        Performs the attack of the model for the inputs and labels.

        Parameters
        ----------
        model : tf.Module
            Model to attack.
        inputs : tf.Tensor
            Batch of samples to attack. Values should be in the [0, 1] range.
        labels : tf.Tensor
            Labels of the samples to attack if untargeted, else
            labels of targets.
        targeted : bool, optional
            Whether to perform a targeted attack or not.

        Returns
        -------
        tf.Tensor
            Batch of samples modified to be adversarial to the model.

        """
        if tf.math.reduce_min(inputs) < 0 or tf.math.reduce_max(inputs) > 1:
            raise ValueError("Input values should be in the [0, 1] range.")

        batch_size = inputs.shape[0]
        multiplier = 1 if targeted else -1
        delta = tf.zeros_like(inputs)
        deltavar = tf.Variable(delta, trainable=True)

        # Setup optimizer
        optimizer = tf.keras.optimizers.SGD(learning_rate=self.max_norm / self.steps * 2)

        for _ in range(self.steps):
            with tf.GradientTape() as tape:
                tape.watch(deltavar)
                adv = inputs + deltavar
                if noise is not None:
                    adv = adv + noise
                adv = tf.transpose(adv, (0, 2, 3, 1))
                logits = model(adv, training=False)
                ce_loss = tf.reduce_sum(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(tf.stop_gradient(labels), logits)
                )
                loss = multiplier * ce_loss
            grad = tape.gradient(loss, deltavar)
            grad_norms = tf.norm(tf.reshape(grad, [batch_size, -1]), ord=2, axis=1)
            grad = tf.math.divide(grad, tf.reshape(grad_norms, (-1, 1, 1, 1)))

            if tf.reduce_any(grad_norms == 0):
                grad = tf.where(tf.reshape(grad_norms, (-1, 1, 1, 1)) != 0, grad, tf.random.normal(grad.shape, 0, 1))

            # optimizer.step()
            optimizer.apply_gradients(zip([grad], [deltavar]))
            delta = deltavar + inputs
            delta = tf.clip_by_norm(tf.math.subtract(tf.clip_by_value(delta, 0, 1), inputs), self.max_norm, axes=[0])
            deltavar.assign(delta)
        return inputs + delta

    def _attack_multinoise(
        self,
        model: tf.Module,
        inputs: tf.Tensor,
        labels: tf.Tensor,
        noise: tf.Tensor = None,
        num_noise_vectors: int = 1,
        targeted: bool = False,
    ) -> tf.Tensor:
        """
        Performs the attack of the model for the inputs and labels.

        Parameters
        ----------
        model : tf.Module
            Model to attack.
        inputs : tf.Tensor
            Batch of samples to attack. Values should be in the [0, 1] range.
        labels : tf.Tensor
            Labels of the samples to attack if untargeted, else labels of
            targets.
        targeted : bool, optional
            Whether to perform a targeted attack or not.

        Returns
        -------
        tf.Tensor
            Batch of samples modified to be adversarial to the model.

        """
        if tf.math.reduce_min(inputs) < 0 or tf.math.reduce_max(inputs) > 1:
            raise ValueError("Input values should be in the [0, 1] range.")
        batch_size = labels.shape[0]
        multiplier = 1 if targeted else -1
        delta = tf.zeros((len(labels), *inputs.shape[1:]))
        deltavar = tf.Variable(delta, trainable=True)

        # Setup optimizers
        optimizer = tf.keras.optimizers.SGD(learning_rate=self.max_norm / self.steps * 2)

        for _ in range(self.steps):
            with tf.GradientTape() as tape:
                tape.watch(deltavar)
                adv = inputs + tf.reshape(tf.tile(deltavar, (1, num_noise_vectors, 1, 1)), inputs.shape)
                if noise is not None:
                    adv = adv + noise
                adv = tf.transpose(adv, (0, 2, 3, 1))
                logits = model(adv, training=False)
                # safe softmax
                softmax = tf.nn.softmax(logits, axis=1)

                # average the probabilities across noise
                average_softmax = tf.squeeze(
                    tf.math.reduce_mean(
                        tf.reshape(softmax, (-1, num_noise_vectors, logits.shape[-1])), axis=1, keepdims=True
                    ),
                    axis=1,
                )
                logsoftmax = tf.math.log(
                    tf.clip_by_value(average_softmax, clip_value_min=1e-20, clip_value_max=tf.float32.max)
                )

                ce_loss = tf.reduce_mean(
                    tf.keras.losses.sparse_categorical_crossentropy(
                        tf.stop_gradient(labels), logsoftmax, from_logits=False
                    )
                )
                loss = multiplier * ce_loss
            grad = tape.gradient(loss, deltavar)
            grad_norms = tf.norm(tf.reshape(grad, [batch_size, -1]), ord=2, axis=1)
            grad = tf.math.divide(grad, tf.reshape(grad_norms, (-1, 1, 1, 1)))

            # renorming gradient
            if tf.reduce_any(grad_norms == 0):
                grad = tf.where(tf.reshape(grad_norms, (-1, 1, 1, 1)) != 0, grad, tf.random.normal(grad.shape, 0, 1))

            # optimizer.step()
            optimizer.apply_gradients(zip([grad], [deltavar]))

            delta = deltavar + inputs[::num_noise_vectors]
            delta = tf.clip_by_norm(
                tf.math.subtract(tf.clip_by_value(delta, 0, 1), inputs[::num_noise_vectors]), self.max_norm, axes=[0]
            )
            deltavar.assign(delta)
        return inputs + tf.reshape(tf.tile(delta, (1, num_noise_vectors, 1, 1)), inputs.shape)

    def _attack_multinoise_no_grad(
        self,
        model: tf.Module,
        inputs: tf.Tensor,
        labels: tf.Tensor,
        noise: tf.Tensor = None,
        num_noise_vectors: int = 1,
    ) -> tf.Tensor:
        """
        Performs the attack of the model for the inputs and labels.

        Parameters
        ----------
        model : tf.Module
            Model to attack.
        inputs : tf.Tensor
            Batch of samples to attack. Values should be in the [0, 1] range.
        labels : tf.Tensor
            Labels of the samples to attack if untargeted, else labels of targets.
        targeted : bool, optional
            Whether to perform a targeted attack or not.

        Returns
        -------
        tf.Tensor
            Batch of samples modified to be adversarial to the model.

        """
        if tf.math.reduce_min(inputs) < 0 or tf.math.reduce_max(inputs) > 1:
            raise ValueError("Input values should be in the [0, 1] range.")
        batch_size = labels.shape[0]
        delta = tf.zeros((len(labels), *inputs.shape[1:]))

        for _ in range(self.steps):
            adv = inputs + tf.reshape(tf.tile(delta, (1, num_noise_vectors, 1, 1)), inputs.shape)
            if noise is not None:
                adv = adv + noise
            adv = tf.transpose(adv, (0, 2, 3, 1))
            logits = model(adv, training=False)

            # safe softmax
            labels_reshaped = tf.reshape(
                tf.tile(tf.expand_dims(tf.cast(labels, dtype=tf.dtypes.int32), axis=1), (1, num_noise_vectors)),
                (batch_size * num_noise_vectors, 1),
            )
            grad = tf.nn.softmax_cross_entropy_with_logits(labels_reshaped, logits)

            # average the probabilities across noise
            grad = tf.expand_dims(tf.expand_dims(tf.expand_dims(grad, 0), 0), 0)
            grad = tf.tile(grad, (*noise.shape[1:], 1))
            grad = tf.transpose(grad, perm=[3, 0, 1, 2]) * noise
            grad = tf.reshape(grad, (-1, num_noise_vectors, *inputs.shape[1:]))
            grad = tf.math.reduce_mean(grad, axis=1)

            grad_norms = tf.cast(tf.norm(tf.reshape(grad, [batch_size, -1]), ord=2, axis=1), tf.dtypes.float32)
            grad = tf.math.divide(grad, tf.reshape(grad_norms, (-1, 1, 1, 1)))
            # avoid nan or inf if gradient is 0
            if tf.reduce_any(grad_norms == 0):
                grad = tf.where(tf.reshape(grad_norms, (-1, 1, 1, 1)) != 0, grad, tf.random.normal(grad.shape, 0, 1))

            # optimizer.step()
            delta = delta + grad * self.max_norm / self.steps * 2
            delta = delta + inputs[::num_noise_vectors]
            delta = tf.clip_by_value(delta, 0, 1)
            delta = tf.math.subtract(delta, inputs[::num_noise_vectors])
            delta = tf.clip_by_norm(delta, self.max_norm, axes=[0])

        return inputs + tf.reshape(tf.tile(delta, (1, num_noise_vectors, 1, 1)), inputs.shape)


# Source code from https://github.com/jeromerony/fast_adversarial
class DDN(Attacker):
    """
    DDN attack: decoupling the direction and norm of the perturbation to achieve a small L2 norm in few steps.

    Parameters
    ----------
    steps : int
        Number of steps for the optimization.
    gamma : float, optional
        Factor by which the norm will be modified. new_norm = norm * (1 + or - gamma).
    init_norm : float, optional
        Initial value for the norm.
    quantize : bool, optional
        If True, the returned adversarials will have quantized values to the specified number of levels.
    levels : int, optional
        Number of levels to use for quantization (e.g. 256 for 8 bit images).
    max_norm : float or None, optional
        If specified, the norms of the perturbations will not be greater than this value which might lower success rate.
    device : tf.device, optional
        Device on which to perform the attack.
    callback : object, optional
        Visdom callback to display various metrics.

    """

    def __init__(
        self,
        steps: int,
        gamma: float = 0.05,
        init_norm: float = 1.0,
        quantize: bool = True,
        levels: int = 256,
        max_norm: float = 1.0,
        device: tf.device = tf.device("cpu"),
    ) -> None:
        super().__init__()
        self.steps = steps
        self.gamma = gamma
        self.init_norm = init_norm
        self.quantize = quantize
        self.levels = levels
        self.max_norm = max_norm
        self.device = device

    def attack(
        self,
        model: tf.Module,
        inputs: tf.Tensor,
        labels: tf.Tensor,
        noise: tf.Tensor = None,
        num_noise_vectors=1,
        targeted: bool = False,
        no_grad=False,
    ) -> tf.Tensor:
        if num_noise_vectors == 1:
            return self._attack(model, inputs, labels, noise, targeted)
        if no_grad:
            raise NotImplementedError
        return self._attack_multinoise(model, inputs, labels, noise, num_noise_vectors, targeted)

    def _attack(
        self, model: tf.Module, inputs: tf.Tensor, labels: tf.Tensor, noise: tf.Tensor = None, targeted: bool = False
    ) -> tf.Tensor:
        """
        Performs the attack of the model for the inputs and labels.

        Parameters
        ----------
        model : tf.Module
            Model to attack.
        inputs : tf.Tensor
            Batch of samples to attack. Values should be in the [0, 1] range.
        labels : tf.Tensor
            Labels of the samples to attack if untargeted, else labels of targets.
        targeted : bool, optional
            Whether to perform a targeted attack or not.

        Returns
        -------
        tf.Tensor
            Batch of samples modified to be adversarial to the model.

        """
        if tf.math.reduce_min(inputs) < 0 or tf.math.reduce_max(inputs) > 1:
            raise ValueError("Input values should be in the [0, 1] range.")

        batch_size = inputs.shape[0]
        multiplier = 1 if targeted else -1
        delta = tf.zeros_like(inputs)
        deltavar = tf.Variable(delta, trainable=True)
        norm = tf.fill((batch_size,), self.init_norm)
        worst_norm = tf.norm(tf.reshape(tf.math.maximum(inputs, 1 - inputs), [batch_size, -1]), ord=2, axis=1)

        # Setup optimizers
        optimizer = tf.keras.optimizers.SGD(learning_rate=1)
        scheduler = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=1.0, decay_steps=self.steps, alpha=0.01
        )

        best_l2 = tf.identity(worst_norm)
        best_delta = tf.zeros_like(inputs)
        adv_found = tf.zeros(inputs.shape[0], dtype=tf.dtypes.uint8)

        for i in range(self.steps):
            optimizer.learning_rate = scheduler(i).numpy()
            with tf.GradientTape() as tape:
                tape.watch(deltavar)
                l2_var = tf.norm(tf.reshape(deltavar, [batch_size, -1]), ord=2, axis=1)
                adv = inputs + deltavar
                if noise is not None:
                    adv = adv + noise
                adv = tf.transpose(adv, (0, 2, 3, 1))
                logits = model(adv, training=True)
                pred_labels = tf.math.argmax(logits, axis=1, output_type=tf.dtypes.int32)
                ce_loss = tf.reduce_sum(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(tf.stop_gradient(labels), logits)
                )
                loss = multiplier * ce_loss

                is_adv = (pred_labels == labels) if targeted else (pred_labels != labels)
                is_smaller = tf.math.less(l2_var, best_l2)
                is_both = tf.cast(
                    tf.math.multiply(tf.cast(is_adv, tf.int32), tf.cast(is_smaller, tf.int32)), tf.dtypes.bool
                )
                adv_found = tf.where(is_both, 1, adv_found)
                best_l2 = tf.where(is_both, l2_var, best_l2)
                is_both_reshaped = tf.expand_dims(tf.expand_dims(tf.expand_dims(is_both, axis=1), axis=1), axis=1)
                is_both_reshaped = tf.tile(is_both_reshaped, (1, *deltavar.shape[1:]))
                best_delta = tf.where(is_both_reshaped, deltavar, best_delta)

            # renorming gradient
            grad = tape.gradient(loss, deltavar)
            grad_norms = tf.norm(tf.reshape(grad, [batch_size, -1]), ord=2, axis=1)
            grad = tf.math.divide(grad, tf.reshape(grad_norms, (-1, 1, 1, 1)))

            # avoid nan or inf if gradient is 0
            if tf.reduce_any(grad_norms == 0):
                grad = tf.where(tf.reshape(grad_norms, (-1, 1, 1, 1)) != 0, grad, tf.random.normal(grad.shape, 0, 1))

            optimizer.apply_gradients(zip([grad], [deltavar]))

            norm = tf.math.multiply(norm, (1 - (2 * tf.cast(is_adv, tf.double) - 1) * self.gamma))
            norm = tf.math.minimum(norm, tf.cast(worst_norm, tf.double))

            delta = tf.math.multiply(
                deltavar,
                tf.reshape(
                    tf.cast(norm, tf.float32) / tf.norm(tf.reshape(deltavar, [batch_size, -1]), ord=2, axis=1),
                    (-1, 1, 1, 1),
                ),
            )
            delta = delta + inputs
            if self.quantize:
                delta = tf.math.divide(tf.math.round(tf.math.multiply(delta, (self.levels - 1))), (self.levels - 1))
            delta = tf.math.subtract(tf.clip_by_value(delta, 0, 1), inputs)
            deltavar.assign(delta)

        if self.max_norm is not None:
            best_delta = tf.clip_by_norm(best_delta, self.max_norm, axes=[0])
            if self.quantize:
                best_delta = tf.math.divide(
                    tf.math.round(tf.math.multiply(best_delta, (self.levels - 1))), (self.levels - 1)
                )
        return inputs + best_delta

    def _attack_multinoise(
        self,
        model: tf.Module,
        inputs: tf.Tensor,
        labels: tf.Tensor,
        noise: tf.Tensor = None,
        num_noise_vectors: int = 1,
        targeted: bool = False,
    ) -> tf.Tensor:
        """
        Performs the attack of the model for the inputs and labels.

        Parameters
        ----------
        model : tf.Module
            Model to attack.
        inputs : tf.Tensor
            Batch of samples to attack. Values should be in the [0, 1] range.
        labels : tf.Tensor
            Labels of the samples to attack if untargeted, else labels of targets.
        targeted : bool, optional
            Whether to perform a targeted attack or not.

        Returns
        -------
        tf.Tensor
            Batch of samples modified to be adversarial to the model.

        """
        if tf.math.reduce_min(inputs) < 0 or tf.math.reduce_max(inputs) > 1:
            raise ValueError("Input values should be in the [0, 1] range.")

        batch_size = labels.shape[0]
        multiplier = 1 if targeted else -1
        delta = tf.zeros((len(labels), *inputs.shape[1:]))
        deltavar = tf.Variable(delta, trainable=True)
        norm = tf.fill((batch_size,), self.init_norm)
        worst_norm = tf.norm(
            tf.reshape(tf.math.maximum(inputs[::num_noise_vectors], 1 - inputs[::num_noise_vectors]), [batch_size, -1]),
            ord=2,
            axis=1,
        )

        # Setup optimizers
        optimizer = tf.keras.optimizers.SGD(learning_rate=1)
        scheduler = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=1.0, decay_steps=self.steps, alpha=0.01
        )

        best_l2 = tf.identity(worst_norm)
        best_delta = tf.zeros_like(inputs[::num_noise_vectors])
        adv_found = tf.zeros(inputs[::num_noise_vectors].shape[0], dtype=tf.dtypes.uint8)

        for i in range(self.steps):
            optimizer.learning_rate = scheduler(i).numpy()
            with tf.GradientTape() as tape:
                tape.watch(deltavar)
                l2_var = tf.norm(tf.reshape(deltavar, [batch_size, -1]), ord=2, axis=1)
                adv = inputs + tf.reshape(tf.tile(deltavar, (1, 1, 1, num_noise_vectors)), inputs.shape)

                if noise is not None:
                    adv = adv + noise
                adv = tf.transpose(adv, (0, 2, 3, 1))
                logits = model(adv, training=True)

                pred_labels = tf.math.argmax(logits, axis=1, output_type=tf.dtypes.int32)
                pred_labels = get_tensor_mode(pred_labels, 1)[0]
                # safe softmax
                softmax = tf.nn.softmax(logits, axis=1)
                # average the probabilities across noise
                average_softmax = tf.squeeze(
                    tf.math.reduce_mean(
                        tf.reshape(softmax, (-1, num_noise_vectors, logits.shape[-1])), axis=1, keepdims=True
                    ),
                    axis=1,
                )

                logsoftmax = tf.math.log(
                    tf.clip_by_value(average_softmax, clip_value_min=1e-20, clip_value_max=tf.float32.max)
                )
                ce_loss = tf.reduce_mean(
                    tf.keras.losses.sparse_categorical_crossentropy(
                        tf.stop_gradient(labels), logsoftmax, from_logits=False
                    )
                )

                loss = multiplier * ce_loss

                is_adv = (pred_labels == labels) if targeted else (pred_labels != labels)
                is_smaller = tf.math.less(l2_var, best_l2)
                is_both = tf.cast(
                    tf.math.multiply(tf.cast(is_adv, tf.int32), tf.cast(is_smaller, tf.int32)), tf.dtypes.bool
                )
                adv_found = tf.where(is_both, 1, adv_found)
                best_l2 = tf.where(is_both, l2_var, best_l2)
                is_both_reshaped = tf.expand_dims(tf.expand_dims(tf.expand_dims(is_both, axis=1), axis=1), axis=1)
                is_both_reshaped = tf.tile(is_both_reshaped, (1, *deltavar.shape[1:]))
                best_delta = tf.where(is_both_reshaped, deltavar, best_delta)

            # renorming gradient
            grad = tape.gradient(loss, deltavar)
            grad_norms = tf.norm(tf.reshape(grad, [batch_size, -1]), ord=2, axis=1)
            grad = tf.math.divide(grad, tf.reshape(grad_norms, (-1, 1, 1, 1)))

            # avoid nan or inf if gradient is 0
            if tf.reduce_any(grad_norms == 0):
                grad = tf.where(tf.reshape(grad_norms, (-1, 1, 1, 1)) != 0, grad, tf.random.normal(grad.shape, 0, 1))

            optimizer.apply_gradients(zip([grad], [deltavar]))

            norm = tf.math.multiply(norm, (1 - (2 * tf.cast(is_adv, tf.double) - 1) * self.gamma))
            norm = tf.math.minimum(norm, tf.cast(worst_norm, tf.double))

            delta = tf.math.multiply(
                deltavar,
                tf.reshape(
                    tf.cast(norm, tf.float32) / tf.norm(tf.reshape(deltavar, [batch_size, -1]), ord=2, axis=1),
                    (-1, 1, 1, 1),
                ),
            )
            delta = delta + inputs[::num_noise_vectors]
            if self.quantize:
                delta = tf.math.divide(tf.math.round(tf.math.multiply(delta, (self.levels - 1))), (self.levels - 1))
            delta = tf.math.subtract(tf.clip_by_value(delta, 0, 1), inputs[::num_noise_vectors])
            deltavar.assign(delta)

        if self.max_norm is not None:
            best_delta = tf.clip_by_norm(best_delta, self.max_norm, axes=[0])
            if self.quantize:
                best_delta = tf.math.divide(
                    tf.math.round(tf.math.multiply(best_delta, (self.levels - 1))), (self.levels - 1)
                )
        return inputs + tf.reshape(tf.tile(best_delta, (1, 1, 1, num_noise_vectors)), inputs.shape)
