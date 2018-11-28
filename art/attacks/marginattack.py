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
        target_scan_iters = 20
        
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
                    c = f[idx, _largest_class] - f[idx, _y] - self.offset
                    c_grd = grd[idx, _largest_class, ...] - grd[idx, _y, ...]
                else:
                    # nontargeted attack: labeled class - largest (non-labeled) class
                    c = f[idx, _y] - f[idx, _largest_class] - self.offset
                    c_grd = grd[idx, _y, ...] - grd[idx, _largest_class, ...]

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
                
                # # compute s
                # if self.metric == 'L2':
                #     s = c_grd
                #     # set the unfeasible directions to 0
                #     s[np.logical_and(_x <= clip_min, s > 0)] = 0
                #     s[np.logical_and(_x >= clip_max, s < 0)] = 0
                # elif self.metric == 'Linf':
                #     s = np.sign(c_grd)
                #     # set the unfeasible directions to 0
                #     s[np.logical_and(_x <= clip_min, s > 0)] = 0
                #     s[np.logical_and(_x >= clip_max, s < 0)] = 0
                # elif self.metric == 'L1':
                #     _c_grd = np.reshape(c_grd, [c_grd.shape[0], -1])
                #     _x_flat = np.reshape(_x, [c_grd.shape[0], -1])
                #     # set the unfeasible directions to 0
                #     _c_grd[np.logical_and(_x_flat <= clip_min, _c_grd > 0)] = 0
                #     _c_grd[np.logical_and(_x_flat >= clip_max, _c_grd < 0)] = 0
                #     max_idx = np.argmax(np.absolute(_c_grd), axis=1)
                #     s = np.zeros(_c_grd.shape, dtype = float)
                #     s[idx, max_idx] = (_c_grd[idx, max_idx])
                #     s = np.reshape(s, c_grd.shape)
                # s_normalized = s / np.sum(c_grd * s, axis = tuple(range(1, x_ndim)), keepdims=True)
                
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
    
    
    def _restore_move(self, x0, x_original, y, y_onehot):
        ''' Performs restoration move.
        
        Args:
        feed_dict - feed_dict needed to run the tensors and operations
        '''
        
        # find the largest incorrect class
        f = self.classifier.predict(x0, logits=True)
        incorrect_class = np.argmax(f, axis=-1)
        
        # compute the constraint and its gradient
        grd_correct = self.classifier.class_gradient(x0, y, logits=True)
        grd_incorrect = self.classifier.class_gradient(x0, incorrect_class,
                                                           logits=True)
        f_correct = f[range(self.batch_size), y]
        f_incorrect = f[range(self.batch_size), incorrect_class]
        if self.targeted:
            c = f_incorrect - f_correct - self.offset
            c_grad = grd_incorrect - grd_correct
        else:
            c = f_correct - f_incorrect - self.offset
            c_grad = grd_correct - grd_incorrect
            
        # solve the constrained optimization problem
        b = -c * self.restore_lr
        x = self._min_norm(x0, c_grad, b=b,
                           x_original=x_original)
        return x
    
    def _restore_move_scan(self, x0, x_original, y, y_onehot)
    ''' Performs restoration move.
        
        Args:
        feed_dict - feed_dict needed to run the tensors and operations
        '''
        
        
        # find the top incorrect classes
        top_adv = 9
        f = self.classifier.predict(x0, logits=True)
        incorrect_class = np.argpartition(f + np.log(y_onehot), 
                                          f.shape[-1] - top_adv, 
                                          axis=1)[:, -top_adv:]
        
        # for each incorrect classes, perform the restoration move, 
        # pick the one with smallest perturbation
        xs = []
        dists = []
        grd_correct = self.classifier.class_gradient(x0, y, logits=True)
        f_correct = f[range(self.batch_size), y]
        for i in xrange(top_adv):
            # compute the constraint and its gradient
            grd_incorrect = self.classifier.class_gradient(x0, incorrect_class[:, i],
                                                           logits=True)
            f_incorrect = f[range(self.batch_size), incorrect_class[:, i]]
            
            if self.targeted:
                c = f_incorrect - f_correct - self.offset
                c_grad = grd_incorrect - grd_correct
            else:
                c = f_correct - f_incorrect - self.offset
                c_grad = grd_correct - grd_incorrect

            # solve the constrained optimization problem
            b = -c * self.restore_lr
            _x = self._min_norm(x0, c_grad, b=b,
                                x_original=x_original)
            # evaluate the distance
            if self.metric == 'L2':
                _dist = np.sum((_x-x0)**2, axis=tuple(range(1, x0.ndim)))
            elif self.metric == 'Linfinity':
                _dist = np.amax(np.absolute(_x-x0), axis=tuple(range(1, x0.ndim)))
            elif self.metric == 'L1':
                _dist = np.sum(np.absolute(_x-x0), axis=tuple(range(1, x0.ndim)))
            xs.append(_x)
            dists.append(_dist)
        
        # find the smallest perturbation
        best_class = np.argmin(np.stack(dists, axis=0), axis=0)
        x = np.stack(xs, axis=0)[best_class, 
                                 range(x0.shape[0]), ...]
        feed_dict.pop(self.incorrect_class)

        return x
    
    def _project_move(self, x_original, x1):
        ''' Performs restoration move.
        
        Args:
        feed_dict - feed_dict needed to run the tensors and operations
        '''
        
        # find the largest incorrect class
        f = self.classifier.predict(x1, logits=True)
        incorrect_class = np.argmax(f, axis=-1)
        
        # compute the constraint and its gradient
        grd_correct = self.classifier.class_gradient(x1, y, logits=True)
        grd_incorrect = self.classifier.class_gradient(x1, incorrect_class,
                                                           logits=True)
        f_correct = f[range(self.batch_size), y]
        f_incorrect = f[range(self.batch_size), incorrect_class]
        if self.targeted:
            c = f_incorrect - f_correct - self.offset
            c_grad = grd_incorrect - grd_correct
        else:
            c = f_correct - f_incorrect - self.offset
            c_grad = grd_correct - grd_incorrect
        
        # solve the constrained optimization problem
        x = self._min_norm_nobound(x_original, c_grad, x1 = x1)
        
        return x

    def _project_move3(self, x_original, x1):
        ''' Performs restoration move.
        
        Args:
        feed_dict - feed_dict needed to run the tensors and operations
        '''
        
        # find the largest incorrect class
        f = self.classifier.predict(x1, logits=True)
        incorrect_class = np.argmax(f, axis=-1)
        
        # compute the constraint and its gradient
        grd_correct = self.classifier.class_gradient(x1, y, logits=True)
        grd_incorrect = self.classifier.class_gradient(x1, incorrect_class,
                                                           logits=True)
        f_correct = f[range(self.batch_size), y]
        f_incorrect = f[range(self.batch_size), incorrect_class]
        if self.targeted:
            c = f_incorrect - f_correct - self.offset
            c_grad = grd_incorrect - grd_correct
        else:
            c = f_correct - f_incorrect - self.offset
            c_grad = grd_correct - grd_incorrect
        
        # perform a gradient descent on d(x - x_original) over x
        if self.metric == 'L2':
            x1 = x1 - self.project_lr * self.project_lr_ratio * \
                 c_grad / np.sqrt(np.sum(c_grad**2, axis = (1,2,3), keepdims=True))
            x0 = x1 - (x1 - x_original) * self.project_lr
        elif self.metric == 'Linfinity':
            dinf = np.amax(np.absolute(x1 - x_original), 
                           axis = tuple(range(1, x1.ndim)),
                           keepdims = True)
            dinf_target = (1 - self.project_lr) * dinf
            x0 = x_original + np.clip(x1 - np.sign(c_grad) * self.project_lr * self.project_lr_ratio\
                                      - x_original, -dinf_target, dinf_target)
#             x0 = x_original + np.clip(x1 - np.reshape(np.greater(c, 0.0), [-1] + [1] * (x1.ndim-1)) \
#                                       * np.sign(c_grad) * self.project_lr - x_original,
#                                       -dinf_target, dinf_target)

#             x0 = np.clip(x0, self.input_min, self.input_max)
        
        
        return x0
    
    def _min_norm(self, x0, c_grad, x1 = None, b = 0, x_min = None, x_max = None,
                  x_original = None):
        ''' This method solves the following constrained optimization problem:
        
        min_x d(x - x0)
        s.t. c_grad.transpose() * (x - x1) = b
             x_min[i] <= x[i] <= x_max[i]
        
        if x1 is None, set to x0
        
        The size of x0, x1, c_grad, x_min and x_max is (num_tokens, ...)
        The size of b should be (num_tokens,)
        
        :param x0: the vector from which the distance is computed.
        :type x0: `np.ndarray`
        :param c_grad: the normal direction of the constraint plane.
        :type c_grad: `np.ndarray`
        :param x1: a vector on the constraint plane.
        :type x1: `np.ndarray`
        :param b: the intercept of the constraint plane.
        :type b: `np.ndaaray`
        :param x_min: the lower bound of each pixel
        :type x_min: `np.ndarray`
        :param x_max: the upper bound of each pixel
        :type x_max: `np.ndarray`
        :param x_original: a bool numpy array the same size as x0 specifying which dimension is binding
        "type x_original: `np.ndarray`
        '''
        
        
        # dimension information
        ndim = x0.ndim
        num_tokens = x0.shape[0]
        x_mask = np.zeros(x0.shape) # 1 - hit upper bound, -1 - hit lower ound, 0 - neither
        
        # reshape b
        if b is 0:
            b = np.array([0]).reshape((-1,) + (1,) * (ndim-1))
        else:
            b = b.reshape((-1,) + (1,) * (ndim-1))
            
        # assign x_min and x_max
        if x_min is None:
            x_min = self.input_min * np.ones(x0.shape)
        if x_max is None:
            x_max = self.input_max * np.ones(x0.shape)
        
        # compute the s vector
        if self.metric == 'L2':
            s = c_grad
        elif self.metric == 'Linfinity':
            c_grad_abs = np.absolute(c_grad)

            if x_original is not None:
                # determine binding dimension s
                diff = np.absolute(x0 - x_original)
                diff_max = np.max(diff,
                                  axis = tuple(range(1, ndim)),
                                  keepdims=True)
                nonbinding = (diff < 0.99 * diff_max)
                
                # for binding dimensions, assign the maximum c_grad;
                # for non-binding dimensions, assign c_grad
                c_grad_max = np.max(np.absolute(c_grad),
                                    axis = tuple(range(1, ndim)),
                                    keepdims=True)
                s = np.sign(c_grad) * c_grad_max * 0.1
#                 s[nonbinding] = c_grad[nonbinding]
            else:
                s = np.sign(c_grad)
        elif self.metric == 'L1':
            _c_grd = np.reshape(c_grad, (num_tokens, -1))
            max_idx = np.argmax(np.absolute(_c_grd), axis=1)
            s = np.zeros(_c_grd.shape, dtype = float)
            s[idx, max_idx] = (_c_grd[idx, max_idx])
            s = np.reshape(s, c_grad.shape)
        
        num_iters = 5
        for it in xrange(num_iters):
            # compute the tentative solution
            if x1 is None:
                x_tent = x0 + s * b /\
                                  np.sum(c_grad * s,
                                         axis = tuple(range(1, ndim)),
                                         keepdims=True)
            else:
                x_tent = x0 + s * (np.sum(c_grad * (x1 - x0),
                                          axis = tuple(range(1, ndim)),
                                          keepdims=True) + b)/\
                                  np.sum(c_grad * s,
                                         axis = tuple(range(1, ndim)),
                                         keepdims=True)
                    
            # check if tentative solution hit the bounds
            _x_mask = (np.logical_and(x_tent > x_max, x_mask == 0)).astype(int) -\
                      (np.logical_and(x_tent < x_min, x_mask == 0)).astype(int)
                
            # update the optimization problem
            if it < num_iters-1:
                if x1 is None:
                    b = b + np.sum(((x0 - x_max) * (_x_mask == 1).astype(int) +
                                    (x0 - self.input_min) * (_x_mask == -1).astype(int)) * c_grad,
                                   axis = tuple(range(1, ndim)),
                                   keepdims=True)
                else:
                    b = b + np.sum(((x1 - x_max) * (_x_mask == 1).astype(int) +
                                    (x1 - self.input_min) * (_x_mask == -1).astype(int)) * c_grad,
                                   axis = tuple(range(1, ndim)),
                                   keepdims=True)
                    
                c_grad[_x_mask != 0] = 0
                if self.metric == 'L1':
                    _c_grd = np.reshape(c_grad, (num_tokens, -1))
                    max_idx = np.argmax(np.absolute(_c_grd), axis=1)
                    s = np.zeros(_c_grd.shape, dtype = float)
                    s[idx, max_idx] = (_c_grd[idx, max_idx])
                    s = np.reshape(s, c_grad.shape)
                else:
                    s[_x_mask != 0] = 0
                
            x_mask += _x_mask
                    
        # return the final answer
        x = x_tent
        x[x_mask == 1] = x_max[x_mask == 1]
        x[x_mask == -1] = x_min[x_mask == -1]        
        
        return x
    
    def _min_norm_nobound(self, x0, c_grad, x1 = None, b = 0,
                          x_original = None):
        ''' This function solves the following constrained optimization problem:
        
        min_x d(x - x0)
        s.t. c_grad.transpose() * (x - x1) = b
             x_min <= x[i] <= x_max
        
        if x1 is None, set to x0
        
        The size of x0, x1, c_grad, x_min and x_max is (num_tokens, ...)
        The size of b should be (num_tokens,)
        
        x_original - a bool numpy array the same size as x0 specifying which dimension is binding
        '''
        
        
        # dimension information
        ndim = x0.ndim
        num_tokens = x0.shape[0]
        x_mask = np.zeros(x0.shape) # 1 - hit upper bound, -1 - hit lower ound, 0 - neither
        
        # reshape b
        if b is 0:
            b = np.array([0]).reshape((-1,) + (1,) * (ndim-1))
        else:
            b = b.reshape((-1,) + (1,) * (ndim-1))
        
        # compute the s vector
        if self.metric == 'L2':
            s = c_grad
        elif self.metric == 'Linfinity':
            c_grad_abs = np.absolute(c_grad)

            if x_original is not None:
                # determine binding dimension s
                diff = np.absolute(x0 - x_original)
                diff_max = np.max(diff,
                                  axis = tuple(range(1, ndim)),
                                  keepdims=True)
                nonbinding = (diff < 0.99 * diff_max)
                
                # for binding dimensions, assign the maximum c_grad;
                # for non-binding dimensions, assign c_grad
                c_grad_max = np.max(np.absolute(c_grad),
                                    axis = tuple(range(1, ndim)),
                                    keepdims=True)
                s = np.sign(c_grad) * c_grad_max * 0.1
#                 s[nonbinding] = c_grad[nonbinding]
            else:
                s = np.sign(c_grad)
        elif self.metric == 'L1':
            _c_grd = np.reshape(c_grad, (num_tokens, -1))
            max_idx = np.argmax(np.absolute(_c_grd), axis=1)
            s = np.zeros(_c_grd.shape, dtype = float)
            s[idx, max_idx] = (_c_grd[idx, max_idx])
            s = np.reshape(s, c_grad.shape)
        

        # compute the tentative solution
        if x1 is None:
            x_tent = x0 + s * b /\
                              np.sum(c_grad * s,
                                     axis = tuple(range(1, ndim)),
                                     keepdims=True)
        else:
            x_tent = x0 + s * (np.sum(c_grad * (x1 - x0),
                                      axis = tuple(range(1, ndim)),
                                      keepdims=True) + b)/\
                              np.sum(c_grad * s,
                                     axis = tuple(range(1, ndim)),
                                     keepdims=True)

                    
        # return the final answer
        x = x_tent     
        
        return x
            