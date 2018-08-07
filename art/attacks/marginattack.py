from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

from art.attacks.attack import Attack


class MarginAttack(Attack):
    """
    Implementation of MarginAttack
    Paper link TBA
    """
    
    attack_params = Attack.attack_params + ['max_iter', 'offset', 'metric', 'targeted',
                                            'batch_size', 'alpha', 'beta0', 'nu', 'verbose']
    
    def __init__(self, classifier,  max_iter=200,  offset = -0.1, metric = 'L2',
                 targeted = False, batch_size = 1, alpha = None, beta0 = None, nu = None,
                 verbose = False):
        """
        Create a DeepFool attack instance.

        :param classifier: A trained model.
        :type classifier: :class:`Classifier`
        :param max_iter: The maximum number of iterations.
        :type max_iter: `int`
        :offset: the decision boundary defined as logit difference = offset
        :type offset: `float`
        :metric: metric of the distance to be minimized
        :type metric: 'L1', 'L2' or 'Linf'
        :targeted: True is targeted attack is performed
        :batch_size: data will be processed in batch mode with batch_size internally
        :type batch_size: `int`
        :alpha: resotration step size
        :type alpha: float in (0, 1)
        :beta0: projection initial step size
        :type beta0: float in (0, 1)
        :nu: projection step size decay rate
        :type nu: float in (0, 1)
        :verbose: output training information if true
        :type verbose: bool
        """
        super(MarginAttack, self).__init__(classifier)
        
        # set the default values for alpha, beta0 and nu
        if alpha is None:
            if metric is 'L2':
                alpha = 1.0
            else:
                alpha = 0.3
                
        if beta0 is None:
            beta0 = 1.0
            
        if nu is None:
            if metric is 'L2':
                nu = 0.5
            else:
                nu = 1.0
                
        params = {'max_iter': max_iter,
                  'offset': offset,
                  'metric': metric,
                  'targeted': targeted,
                  'batch_size': batch_size,
                  'alpha': alpha,
                  'beta0': beta0,
                  'nu': nu,
                  'verbose': verbose}
        self.set_params(**params)
        
        
    def generate(self, x, **kwargs):
        """
        Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs to be attacked.
        :type x: `np.ndarray`
        :param y: If `self.targeted` is true, then `y_val` represents the target labels. Otherwise, the targets are
                the original class labels.
        :type y: `np.ndarray`
        :return: An array holding the adversarial examples.
        :rtype: `np.ndarray`
        """
        
        # Parse and save attack-specific parameters
        params_cpy = dict(kwargs)
        y = params_cpy.pop(str('y'), None)
        
        # if y value is missing
        if y is None:
            if self.targeted:
                raise ValueError('The target class must be specified in y for targeted attacks.')
            else:
                y = np.argmax(self.classifier.predict(x, logits=True), axis=1)
        else:
            # convert potential one-hot labeling to one-dim label
            if y.ndim==2:
                y = np.argmax(y, axis=1)
        
        # batch the input and labels
        num_batches = np.ceil(float(x.shape[0]) / float(self.batch_size))
        x = np.array_split(x, num_batches, axis = 0)
        y = np.array_split(y, num_batches)
        x_adv = []
        x_ndim = x[0].ndim
        
        # clip values
        clip_min, clip_max = self.classifier.clip_values
        
        # number of final restoration iterations
        final_restore_iters = 20
        
        # attack batches
        for batch_id, _x, _y in zip(xrange(len(x)), x, y):
            _x0 = _x.copy()
            num_tokens = _x.shape[0]
            
            for _it in range(self.max_iter):
                
                # prediction and gradients
                f = self.classifier.predict(_x, logits=True)
                grd = self.classifier.class_gradient(_x, logits=True)                
                
                # find the largest non-y class
                _f = f.copy()
                idx = np.arange(_f.shape[0])
                _f[idx, _y] = -np.inf
                _largest_class = np.argmax(_f, axis=1)
                
                # constraint and constraint gradient
                if self.targeted:
                    # targeted attack: largest (non-adv) class - adv class
                    c1 = f[idx, _largest_class] - f[idx, _y] - self.offset
                    c1_mask = 2 * np.float32(c1 > 0) - 1
                    c = c1 *  c1_mask\
                        + self._box_constraint(_x, clip_min, clip_max)
                    c_grd = np.reshape(c1_mask, [-1]+[1] * (x_ndim-1)) \
                            * (grd[idx, _largest_class, ...] - grd[idx, _y, ...]) \
                            + self._box_constraint_grd(_x, clip_min, clip_max)
                else:
                    # nontargeted attack: labeled class - largest (non-labeled) class
                    c1 = f[idx, _y] - f[idx, _largest_class] - self.offset
                    c1_mask = 2 * np.float32(c1 > 0) - 1
                    c = c1 * c1_mask \
                        + self._box_constraint(_x, clip_min, clip_max)
                    c_grd = np.reshape(c1_mask, [-1]+[1] * (x_ndim-1)) \
                            * (grd[idx, _y, ...] - grd[idx, _largest_class, ...]) \
                            + self._box_constraint_grd(_x, clip_min, clip_max)

                # output training information
                if self.verbose:
                    # compute distance
                    if self.metric == 'L2':
                        dist_mean = np.mean(np.sqrt(np.sum(np.square(_x - _x0), axis=tuple(range(1, x_ndim)))))
                    elif self.metric == 'L1':
                        dist_mean = np.mean(np.sum(np.absolute(_x - _x0), axis=tuple(range(1, x_ndim))))
                    elif self.metric == 'Linf':
                        dist_mean = np.mean(np.max(np.absolute(_x - _x0), axis=tuple(range(1, x_ndim))))
                    
                    # compute average constraint
                    c_mean = np.mean(c)
                    
                    print('Batch {}, before iteration {}: average constraint = {}, average perturbation = {}'.format(batch_id, _it, c_mean, dist_mean))
                
                # compute s
                if self.metric == 'L2':
                    s = c_grd
                    # set the unfeasible directions to 0
                    s[np.logical_and(_x <= clip_min, s > 0)] = 0
                    s[np.logical_and(_x >= clip_max, s < 0)] = 0
                elif self.metric == 'Linf':
                    s = np.sign(c_grd)
                    # set the unfeasible directions to 0
                    s[np.logical_and(_x <= clip_min, s > 0)] = 0
                    s[np.logical_and(_x >= clip_max, s < 0)] = 0
                elif self.metric == 'L1':
                    _c_grd = np.reshape(c_grd, [c_grd.shape[0], -1])
                    _x_flat = np.reshape(_x, [c_grd.shape[0], -1])
                    # set the unfeasible directions to 0
                    _c_grd[np.logical_and(_x_flat <= clip_min, _c_grd > 0)] = 0
                    _c_grd[np.logical_and(_x_flat >= clip_max, _c_grd < 0)] = 0
                    max_idx = np.argmax(np.absolute(_c_grd), axis=1)
                    s = np.zeros(_c_grd.shape, dtype = float)
                    s[idx, max_idx] = (_c_grd[idx, max_idx])
                    s = np.reshape(s, c_grd.shape)
                s_normalized = s / np.sum(c_grd * s, axis = tuple(range(1, x_ndim)), keepdims=True)
                
                if _it % 2 == 0 or self.max_iter - _it <= final_restore_iters :
                    # even iterations: restoration step
                    _x = _x - self.alpha * np.reshape(c, [-1] + [1] * (x_ndim-1)) * s_normalized
                    
                    
                        
                else:
                    # odd iterations: projection step
                    beta = self.beta0 / (float(_it)**self.nu)
                    prj_step = _x - _x0 - \
                               np.sum((_x - _x0) * c_grd, axis = tuple(range(1, x_ndim)), keepdims=True) * s_normalized
                    _x = _x - beta * prj_step
                    
                if self.max_iter - _it <= final_restore_iters :
                    # clip to bounding box
                    _x = np.clip(_x, clip_min, clip_max)
                

            # append the adversarial samples
            x_adv.append(_x)
            
        return np.concatenate(x_adv, axis=0)
                
        
                
    def set_params(self, **kwargs):
        """Take in a dictionary of parameters and applies attack-specific checks before saving them as attributes.

        :param max_iter: The maximum number of iterations.
        :type max_iter: `int`
        """
        # Save attack-specific parameters
        super(MarginAttack, self).set_params(**kwargs)

        if type(self.max_iter) is not int or self.max_iter <= 0:
            raise ValueError("The number of iterations must be a positive integer.")
            
        if self.metric != 'L1' and self.metric != 'L2' and self.metric != 'Linf':
            raise ValueError("'metric' must be 'L1', 'L2' or 'Linf.'")
            
        if type(self.targeted) is not bool:
            raise ValueError("'targeted' must be boolean.")
            
        if type(self.batch_size) is not int or self.batch_size <= 0:
            raise ValueError("Batch size must be a positive integer.")
            
        if type(self.alpha) is not float or self.alpha > 1 or self.alpha < 0:
            raise ValueError('alpha must be a float between 0 and 1')
            
        if type(self.beta0) is not float or self.beta0 > 1 or self.beta0 < 0:
            raise ValueError('beta0 must be a float between 0 and 1')
            
        if type(self.nu) is not float or self.nu > 1 or self.nu < 0:
            raise ValueError('nu must be a float between 0 and 1')
            
        if type(self.verbose) is not bool:
            raise ValueError('verbose must be a bool.')

        return True
    
    def _box_constraint(self, x, clip_min, clip_max):
        """
        Implement the barrier function for box constraint
        """
        
        x_clip_min = clip_min - x
        x_clip_max = x - clip_max
        c = np.mean(0.5 * x_clip_min ** 2 * np.float32(np.logical_and(x_clip_min > 0, x_clip_min <= 1)) +
                    (x_clip_min - 0.5) * np.float32(x_clip_min > 1) +
                    0.5 * x_clip_max ** 2 * np.float32(np.logical_and(x_clip_max > 0, x_clip_max <= 1)) +
                    (x_clip_max - 0.5) * np.float32(x_clip_max > 1),
                    axis = tuple(range(1, x.ndim)))
        return c
    
    def _box_constraint_grd(self, x, clip_min, clip_max):
        
        """
        Implement the derivative of barrier function for box constraint
        """
        
        x_clip_min = clip_min - x
        x_clip_max = x - clip_max
        c = (-x_clip_min * np.float32(np.logical_and(x_clip_min > 0, x_clip_min <= 1)) -
            1.0 * np.float32(x_clip_min > 1) +
            x_clip_max * np.float32(np.logical_and(x_clip_max > 0, x_clip_max <= 1)) +
            1.0 * np.float32(x_clip_max > 1)) \
            / float(x[0].size)
        
        return c
            