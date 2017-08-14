from config import config_dict
from keras.utils.np_utils import to_categorical
import tensorflow as tf

from cleverhans.attacks_tf import CarliniWagnerL2 as CWL2
from cleverhans.utils_keras import KerasModelWrapper

from src.attackers.attack import Attack

# ad462c9b


class CarliniL2Method(Attack):
    """
    The L_2 optimized attack of Carlini and Wagner (2016). This attack is the most efficient and should be used as
    the primary attack to evaluate potential defenses (wrt to the L_0 and L_inf attacks). This code is a wrapper on top
    of the code provided by Cleverhans, which reproduces the authors' original code
    (https://github.com/carlini/nn_robust_attacks). Paper link: https://arxiv.org/pdf/1608.04644.pdf
    """
    attack_params = ['batch_size', 'confidence', 'targeted', 'learning_rate', 'binary_search_steps', 'max_iterations',
                     'abort_early', 'initial_const', 'clip_min', 'clip_max', 'num_labels']

    def __init__(self, classifier, sess, batch_size=1, confidence=0, targeted=True, learning_rate=5e-3,
                 binary_search_steps=5, max_iterations=1000, abort_early=True, initial_const=1e-2, clip_min=0,
                 clip_max=1):
        """
        Create a Carlini L_2 attack instance.
        :param batch_size: (optional integer) Number of attacks to run simultaneously.
        :param confidence: Confidence of adversarial examples: higher produces examples that are farther away, but more
               strongly classified as adversarial.
        :param targeted: (optional boolean) should the attack target one specific class
        :param learning_rate: The learning rate for the attack algorithm. Smaller values produce better results but are
               slower to converge.
        :param binary_search_steps: (optional positive integer) number of times to adjust the constant with binary search
        :param max_iterations: (integer) The maximum number of iterations.
        :param abort_early: (optional boolean) if we stop improving, abort gradient descent early
        :param initial_const: (optional float, positive) The initial trade-off constant c to use to tune the relative
               importance of distance and confidence. If binary_search_steps is large,
               the initial constant is not important.
        :param clip_min: (optional float) Minimum input component value.
        :param clip_max: (optional float) Maximum input component value.
        """
        super(CarliniL2Method, self).__init__(classifier, sess)

        kwargs = {'batch_size': batch_size,
                  'confidence': confidence,
                  'targeted': targeted,
                  'learning_rate': learning_rate,
                  'binary_search_steps': binary_search_steps,
                  'max_iterations': max_iterations,
                  'abort_early': abort_early,
                  'initial_const': initial_const,
                  'clip_min': clip_min,
                  'clip_max': clip_max
                  }
        self.set_params(**kwargs)

    def generate(self, x_val, **kwargs):
        """
        Generate adversarial samples and return them in a Numpy array.
        :param x_val:
        :param y_val: If self.targeted is true, then y_val represents the target labels. If self.targeted is false, then
                      targets are the original class labels.
        :return: A Numpy array holding the adversarial examples.
        """
        # Parse and save attack-specific parameters
        params_cpy = dict(kwargs)
        y_val = params_cpy.pop('y_val', None)
        assert self.set_params(**params_cpy)

        model = KerasModelWrapper(self.classifier.model)
        attack = CWL2(self.sess, model, self.batch_size, self.confidence, self.targeted, self.learning_rate,
                      self.binary_search_steps, self.max_iterations, self.abort_early, self.initial_const,
                      self.clip_min, self.clip_max, self.classifier.model.output_shape[1], x_val.shape[1:])

        if y_val is None:
            # No labels provided, use model prediction as correct class
            x = tf.placeholder(dtype=tf.float32, shape=self.classifier.model.get_input_shape_at(0))
            y_val = self.sess.run(tf.argmax(self.classifier.model(x), axis=1), {x: x_val})
            y_val = to_categorical(y_val, self.classifier.model.get_output_shape_at(-1)[-1])

        return attack.attack(x_val, y_val)

    def set_params(self, **kwargs):
        """Take in a dictionary of parameters and applies attack-specific checks before saving them as attributes.

        Attack-specific parameters:
        :param batch_size: (optional integer) Number of attacks to run simultaneously.
        :param confidence: Confidence of adversarial examples: higher produces examples that are farther away, but more
               strongly classified as adversarial.
        :param targeted: (optional boolean) should the attack target one specific class
        :param learning_rate: The learning rate for the attack algorithm. Smaller values produce better results but are
               slower to converge.
        :param binary_search_steps: (optional positive integer) number of times to adjust the constant with binary search
        :param max_iterations: (integer) The maximum number of iterations.
        :param abort_early: (optional boolean) if we stop improving, abort gradient descent early
        :param initial_const: (optional float, positive) The initial trade-off constant c to use to tune the relative
               importance of distance and confidence. If binary_search_steps is large,
               the initial constant is not important.
        :param clip_min: (optional float) Minimum input component value.
        :param clip_max: (optional float) Maximum input component value.
        :param num_labels: The number of classes the model has.
        """
        # Save attack-specific parameters
        super(CarliniL2Method, self).set_params(**kwargs)
        return True

# class CarliniL2Method(Attack):
#     """
#     The L_2 optimized attack of Carlini and Wagner (2016). This attack is the most efficient and should be used as
#     the primary attack to evaluate potential defenses (wrt to the L_0 and L_inf attacks). This code is a wrapper on top
#     of the original code provided by the authors (https://github.com/carlini/nn_robust_attacks).
#     Paper link: https://arxiv.org/pdf/1608.04644.pdf
#     """
#     attack_params = ['batch_size', 'confidence', 'targeted', 'learning_rate', 'binary_search_step', 'max_iterations',
#                      'abort_early', 'initial_const']
#
#     def __init__(self, model, sess, batch_size=1, confidence=0, targeted=True, learning_rate=1e-2,
#                  binary_search_steps=9, max_iterations=10000, abort_early=True, initial_const=1e-3):
#         """
#         Create a Carlini L_2 attack instance.
#         :param batch_size: (optional integer) Number of attacks to run simultaneously.
#         :param confidence: Confidence of adversarial examples: higher produces examples that are farther away, but more
#                strongly classified as adversarial.
#         :param targeted: (optional boolean) should the attack target one specific class
#         :param learning_rate: The learning rate for the attack algorithm. Smaller values produce better results but are
#                slower to converge.
#         :param binary_search_steps: (optional positive integer) number of times to adjust the constant with binary search
#         :param max_iterations: (integer) The maximum number of iterations.
#         :param abort_early: (optional boolean) if we stop improving, abort gradient descent early
#         :param initial_const: (optional float, positive) The initial trade-off constant c to use to tune the relative
#                importance of distance and confidence. If binary_search_steps is large,
#                the initial constant is not important.
#         """
#         super(CarliniL2Method, self).__init__(model, sess)
#
#         kwargs = {'batch_size': batch_size,
#                   'confidence': confidence,
#                   'targeted': targeted,
#                   'learning_rate': learning_rate,
#                   'binary_search_steps': binary_search_steps,
#                   'max_iterations': max_iterations,
#                   'abort_early': abort_early,
#                   'initial_const': initial_const}
#         self.set_params(**kwargs)
#
#     def generate(self, x_val, **kwargs):
#         """
#         Generate adversarial samples and return them in a Numpy array.
#         :param x_val:
#         :param y_val:
#         :return:
#         """
#         # Parse and save attack-specific parameters
#         params_cpy = dict(kwargs)
#         y_val = params_cpy.pop('y_val', None)
#         x_val_cpy = x_val.copy()
#         if np.all(x_val >= 0):
#             # Recenter data around 0
#             x_val_cpy = x_val_cpy - np.max(x_val_cpy) / 2.
#         assert self.set_params(**params_cpy)
#
#         if self.targeted:
#             if y_val is None:
#                 warnings.warn('A targeted attack was required, but not labels were provided. Performing untargeted '
#                               'attack instead.')
#                 # No labels provided, use model prediction as correct class
#                 x = tf.placeholder(dtype=tf.float32, shape=self.model.get_input_shape_at(0))
#                 y_val = self.sess.run(tf.argmax(self.model(x), axis=1), {x: x_val})
#                 y_val = to_categorical(y_val, self.model.get_output_shape_at(-1)[-1])
#                 res = self._attack_instance.attack(x_val_cpy, y_val)
#             else:
#                 res = self._attack_instance.attack(x_val_cpy, y_val)
#         else:
#             if y_val is None:
#                 # No labels provided, use model prediction as correct class
#                 x = tf.placeholder(dtype=tf.float32, shape=self.model.get_input_shape_at(0))
#                 y_val = self.sess.run(tf.argmax(self.model(x), axis=1), {x: x_val})
#                 y_val = to_categorical(y_val, self.model.get_output_shape_at(-1)[-1])
#                 res = self._attack_instance.attack(x_val_cpy, y_val)
#             else:
#                 # Correct labels provided, use them as targets
#                 res = self._attack_instance.attack(x_val_cpy, y_val)
#
#         # Bring the adversarial examples back to the original scale
#         return res + np.max(x_val) / 2.
#
#     def set_params(self, **kwargs):
#         """Take in a dictionary of parameters and applies attack-specific checks before saving them as attributes.
#
#         Attack-specific parameters:
#         :param batch_size: (optional integer) Number of attacks to run simultaneously.
#         :param confidence: Confidence of adversarial examples: higher produces examples that are farther away, but more
#                strongly classified as adversarial.
#         :param targeted: (optional boolean) should the attack target one specific class
#         :param learning_rate: The learning rate for the attack algorithm. Smaller values produce better results but are
#                slower to converge.
#         :param binary_search_steps: (optional positive integer) number of times to adjust the constant with binary search
#         :param max_iterations: (integer) The maximum number of iterations.
#         :param abort_early: (optional boolean) if we stop improving, abort gradient descent early
#         :param initial_const: (optional float, positive) The initial trade-off constant C to use to tune the relative
#                importance of distance and confidence. If binary_search_steps is large,
#                the initial constant is not important.
#         """
#         # Save attack-specific parameters
#         super(CarliniL2Method, self).set_params(**kwargs)
#
#         # Reconstruct the attack instance with the new parameters
#         mock_model = MockModel(self.model)
#         self._attack_instance = CarliniL2(self.sess, mock_model, **kwargs)
#         return True
#
#
# class CarliniL0Method(Attack):
#     """
#     The L_0 optimized attack of Carlini and Wagner (2016). This code is a wrapper on top of the original code provided
#     by the authors (https://github.com/carlini/nn_robust_attacks).
#     Paper link: https://arxiv.org/pdf/1608.04644.pdf
#     """
#     attack_params = ['targeted', 'learning_rate', 'max_iterations', 'abort_early', 'initial_const', 'largest_const',
#                      'reduce_const', 'const_factor', 'independent_channels']
#
#     def __init__(self, model, sess, targeted=True, learning_rate=1e-2, max_iterations=1000, abort_early=True,
#                  initial_const=1e-3, largest_const=2e6, reduce_const=False, const_factor=2.,
#                  independent_channels=False):
#         """
#         Create a Carlini L_0 attack instance.
#         :param targeted: (optional boolean) should the attack target one specific class
#         :param learning_rate: The learning rate for the attack algorithm. Smaller values produce better results but are
#                slower to converge.
#         :param max_iterations: (integer) Number of iterations to perform gradient descent
#         :param abort_early: (optional boolean) Abort gradient descent upon first valid solution
#         :param initial_const: (optional float) The first value of c to start at
#         :param largest_const: (optional float) The largest value of c to go up to before giving up
#         :param reduce_const: (optional boolean) Try to lower c each iteration; faster to set to false
#         :param const_factor: f>1, rate at which we increase constant, smaller better
#         :param independent_channels: Set to false optimizes for number of pixels changed, set to true (not recommended)
#                returns number of channels changed.
#         """
#         super(CarliniL0Method, self).__init__(model, sess)
#
#         kwargs = {'targeted': targeted,
#                   'learning_rate': learning_rate,
#                   'max_iterations': max_iterations,
#                   'abort_early': abort_early,
#                   'initial_const': initial_const,
#                   'largest_const': largest_const,
#                   'reduce_const': reduce_const,
#                   'const_factor': const_factor,
#                   'independent_channels': independent_channels}
#         self.set_params(**kwargs)
#
#     def generate(self, x_val, **kwargs):
#         """
#         Generate adversarial samples and return them in a Numpy array.
#         :param x_val:
#         :param y_val:
#         :return:
#         """
#         # Parse and save attack-specific parameters
#         params_cpy = dict(kwargs)
#         y_val = params_cpy.pop('y_val', None)
#         assert self.set_params(**params_cpy)
#
#         if self.targeted:
#             if y_val is None:
#                 warnings.warn('A targeted attack was required, but not labels were provided. Performing untargeted '
#                               'attack instead.')
#                 # No labels provided, use model prediction as correct class
#                 x = tf.placeholder(dtype=tf.float32, shape=self.model.get_input_shape_at(0))
#                 y_val = self.sess.run(tf.argmax(self.model(x), axis=1), {x: x_val})
#                 y_val = to_categorical(y_val, self.model.get_output_shape_at(-1)[-1])
#                 return self._attack_instance.attack(x_val, y_val)
#             else:
#                 return self._attack_instance.attack(x_val, y_val)
#         else:
#             if y_val is None:
#                 # No labels provided, use model prediction as correct class
#                 x = tf.placeholder(dtype=tf.float32, shape=self.model.get_input_shape_at(0))
#                 y_val = self.sess.run(tf.argmax(self.model(x), axis=1), {x: x_val})
#                 y_val = to_categorical(y_val, self.model.get_output_shape_at(-1)[-1])
#                 return self._attack_instance.attack(x_val, y_val)
#             else:
#                 # Correct labels provided, use them as targets
#                 return self._attack_instance.attack(x_val, y_val)
#
#     def set_params(self, **kwargs):
#         """Take in a dictionary of parameters and applies attack-specific checks before saving them as attributes.
#
#         Attack-specific parameters:
#         :param targeted: (optional boolean) should the attack target one specific class
#         :param learning_rate: The learning rate for the attack algorithm. Smaller values produce better results but are
#                slower to converge.
#         :param max_iterations: (integer) Number of iterations to perform gradient descent
#         :param abort_early: (optional boolean) Abort gradient descent upon first valid solution
#         :param initial_const: (optional float) The first value of c to start at
#         :param largest_const: (optional float) The largest value of c to go up to before giving up
#         :param reduce_const: (optional boolean) Try to lower c each iteration; faster to set to false
#         :param const_factor: f>1, rate at which we increase constant, smaller better
#         :param independent_channels: Set to false optimizes for number of pixels changed, set to true (not recommended)
#                returns number of channels changed.
#         """
#         # Save attack-specific parameters
#         super(CarliniL0Method, self).set_params(**kwargs)
#
#         # Reconstruct the attack instance with the new parameters
#         mock_model = MockModel(self.model)
#         self._attack_instance = CarliniL0(self.sess, mock_model, **kwargs)
#         return True
#
#
# class CarliniLiMethod(Attack):
#     """
#     The L_inf optimized attack of Carlini and Wagner (2016). This code is a wrapper on top of the original code provided
#     by the authors (https://github.com/carlini/nn_robust_attacks).
#     Paper link: https://arxiv.org/pdf/1608.04644.pdf
#     """
#     attack_params = ['targeted', 'learning_rate', 'max_iterations', 'abort_early', 'initial_const', 'largest_const',
#                      'reduce_const', 'decrease_factor', 'const_factor']
#
#     def __init__(self, model, sess, targeted=True, learning_rate=5e-3, max_iterations=1000, abort_early=True,
#                  initial_const=1e-5, largest_const=2e+1, reduce_const=False, decrease_factor=.9,
#                  const_factor = 2.):
#         """
#         Create a Carlini L_inf attack instance.
#         :param targeted: (optional boolean) should the attack target one specific class
#         :param learning_rate: The learning rate for the attack algorithm. Smaller values produce better results but are
#                slower to converge.
#         :param max_iterations: (integer) Number of iterations to perform gradient descent
#         :param abort_early: (optional boolean) Abort gradient descent upon first valid solution
#         :param initial_const: (optional float) The first value of c to start at
#         :param largest_const: (optional float) The largest value of c to go up to before giving up
#         :param reduce_const: (optional boolean) Try to lower c each iteration; faster to set to false
#         :param decrease_factor: 0<f<1, rate at which we shrink tau; larger is more accurate
#         :param const_factor: f>1, rate at which we increase constant, smaller better
#         """
#         super(CarliniLiMethod, self).__init__(model, sess)
#
#         kwargs = {'targeted': targeted,
#                   'learning_rate': learning_rate,
#                   'max_iterations': max_iterations,
#                   'abort_early': abort_early,
#                   'initial_const': initial_const,
#                   'largest_const': largest_const,
#                   'reduce_const': reduce_const,
#                   'decrease_factor': decrease_factor,
#                   'const_factor': const_factor}
#         self.set_params(**kwargs)
#
#     def generate(self, x_val, **kwargs):
#         """
#         Generate adversarial samples and return them in a Numpy array.
#         :param x_val:
#         :param y_val:
#         :return:
#         """
#         # Parse and save attack-specific parameters
#         params_cpy = dict(kwargs)
#         y_val = params_cpy.pop('y_val', None)
#         assert self.set_params(**params_cpy)
#
#         if self.targeted:
#             if y_val is None:
#                 warnings.warn('A targeted attack was required, but not labels were provided. Performing untargeted '
#                               'attack instead.')
#                 # No labels provided, use model prediction as correct class
#                 x = tf.placeholder(dtype=tf.float32, shape=self.model.get_input_shape_at(0))
#                 y_val = self.sess.run(tf.argmax(self.model(x), axis=1), {x: x_val})
#                 y_val = to_categorical(y_val, self.model.get_output_shape_at(-1)[-1])
#                 return self._attack_instance.attack(x_val, y_val)
#             else:
#                 return self._attack_instance.attack(x_val, y_val)
#         else:
#             if y_val is None:
#                 # No labels provided, use model prediction as correct class
#                 x = tf.placeholder(dtype=tf.float32, shape=self.model.get_input_shape_at(0))
#                 y_val = self.sess.run(tf.argmax(self.model(x), axis=1), {x: x_val})
#                 y_val = to_categorical(y_val, self.model.get_output_shape_at(-1)[-1])
#                 return self._attack_instance.attack(x_val, y_val)
#             else:
#                 # Correct labels provided, use them as targets
#                 return self._attack_instance.attack(x_val, y_val)
#
#     def set_params(self, **kwargs):
#         """
#         :param targeted: (optional boolean) should the attack target one specific class
#         :param learning_rate: The learning rate for the attack algorithm. Smaller values produce better results but are
#                slower to converge.
#         :param max_iterations: (integer) Number of iterations to perform gradient descent
#         :param abort_early: (optional boolean) Abort gradient descent upon first valid solution
#         :param initial_const: (optional float) The first value of c to start at
#         :param largest_const: (optional float) The largest value of c to go up to before giving up
#         :param reduce_const: (optional boolean) Try to lower c each iteration; faster to set to false
#         :param decrease_factor: 0<f<1, rate at which we shrink tau; larger is more accurate
#         :param const_factor: f>1, rate at which we increase constant, smaller better
#         """
#         # Save attack-specific parameters
#         super(CarliniLiMethod, self).set_params(**kwargs)
#
#         # Reconstruct the attack instance with the new parameters
#         mock_model = MockModel(self.model)
#         self._attack_instance = CarliniLi(self.sess, mock_model, **kwargs)
#         return True


class MockModel:
    def __init__(self, model):
        self.model = model
        self.num_channels = model.get_input_shape_at(0)[-1]
        self.image_size = model.get_input_shape_at(0)[1]
        self.num_labels = model.get_output_shape_at(-1)[-1]

    def predict(self, data):
        return self.model(data)
