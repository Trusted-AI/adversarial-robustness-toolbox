from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

from art.attacks.attack import Attack


class MarginAttack(Attack):
    """
    Implementation of MarginAttack
    Paper link TBA
    """
    
    attack_params = Attack.attack_params + ['max_iter', 'target_scan_iters', 'final_restore_iters', 
                                            'offset', 'metric', 'targeted','input_max', 'input_min', 
                                            'num_classes', 'num_scan_classes', 'restore_lr',
                                            'project_lr_init', 'project_lr_ratio', 'nu', 'verbose']
    
    def __init__(self, classifier,  max_iter=200, target_scan_iters=20, final_restore_iters=20,
                 offset = -0.1, metric = 'L2', targeted = False, input_max = None, input_min = None, 
                 num_classes = None, num_scan_classes = None, restore_lr = None, 
                 project_lr_init = 1.0, project_lr_ratio = 0.1, nu = None, verbose = False):
        """
        Create a DeepFool attack instance.

        :param classifier: A trained model.
        :type classifier: :class:`Classifier`
        :param max_iter: The maximum number of iterations.
        :type max_iter: `int`
        :param target_scan_iters: The number of iterations where restoration move scans candidate adversarial classes
        :type target_scan_iters: `int`
        :param final_restore_iters: The number of final iterations where projection move is removed
        :type final_restore_iters: `int`
        :param offset: the decision boundary defined as logit difference = offset
        :type offset: `float`
        :param metric: metric of the distance to be minimized
        :type metric: 'L1', 'L2' or 'Linf'
        :param targeted: True if targeted attack is performed
        :type targeted: `bool`
        :param input_max: the maximum value to which the input feature is clipped
        :type input_max: `int`
        :param input_min: the minimum value to which the input feature is clipped
        :type input_min: `int`
        :param num_classes: total number of classes
        :type num_classes: `int`
        :param num_scan_classes: total number of candidate adversarial classes to scan in the restoration move
        :type num_scan_classes:
        :param restore_lr: resotration step size, should be within (0, 1)
        :type restore_lr: `float`
        :param project_lr_init: the initial step size of projection move; should be within (0, 1)
        :type project_lr_init: `float`
        :param project_lr_ratio: the ratio between the step size of the projection move component that reduces distance;
                                 only useful in Linf attack; should be between (0, 1)
        :type project_lr_ratio: `float`
        :param nu: projection step size decay rate; should be within (0, 1)
        :type nu: `float`
        :param verbose: print training information at each iteration if true
        :type verbose: `bool`
        """
        super(MarginAttack, self).__init__(classifier)
        
        # set the default values for the parameters
        if input_min is None:
            print("input_min is not specified; set to negative infinity.")
            input_min = -np.inf
        if input_max is None:
            print("input_max is not specified; set to positive infinity.")
            input_max = np.inf
            
        if num_scan_classes is None and num_classes is not None:
            num_scan_classes = min(9, num_classes-1)
            
        if restore_lr is None:
            if metric is 'L2':
                restore_lr = 1.0
            else:
                restore_lr = 0.2
            
        if nu is None:
            if metric is 'L2':
                nu = 0.3
            else:
                nu = 1.0
                
        params = {'max_iter': max_iter,
                  'target_scan_iters': target_scan_iters,
                  'final_restore_iters': final_restore_iters,
                  'offset': offset,
                  'metric': metric,
                  'targeted': targeted,
                  'input_max': input_max,
                  'input_min': input_min,
                  'num_classes': num_classes,
                  'num_scan_classes': num_scan_classes,
                  'restore_lr': restore_lr,
                  'project_lr_init': project_lr_init,
                  'project_lr_ratio': project_lr_ratio,
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
                print('True class label not provided. Using predicted class as label.')
                y = np.argmax(self.classifier.predict(x, logits=True), axis=1)
        else:
            # convert potential one-hot labeling to one-dim label
            if y.ndim==2:
                y = np.argmax(y, axis=1)
                
        # generate a one-hot labeling
        self.num_obs = x.shape[0]
        y_onehot = np.zeros((self.num_obs, self.num_classes), dtype=np.int8)
        y_onehot[range(self.num_obs), y] = 1
        
        # attack batch
        x_adv = x.copy()
        x_original = x.copy()
        for _it in range(self.max_iter):


            # output training information
            if self.verbose:
                # compute distance
                if self.metric == 'L2':
                    dist_mean = np.mean(np.sqrt(np.sum(np.square(x_adv - x_original), axis=tuple(range(1, x.ndim)))))
                elif self.metric == 'Linf':
                    dist_mean = np.mean(np.max(np.absolute(x_adv - x_original), axis=tuple(range(1, x.ndim))))

                # compute average constraint
                f = self.classifier.predict(x_adv, logits=True)
                incorrect_class = np.argmax(f + np.log(1-y_onehot), axis=-1)
                f_correct = f[range(self.num_obs), y]
                f_incorrect = f[range(self.num_obs), incorrect_class]
                if self.targeted:
                    c = f_incorrect - f_correct - self.offset
                else:
                    c = f_correct - f_incorrect - self.offset
                c_mean = np.mean(c)

                print('Before iteration {}: average constraint = {}, average perturbation = {}'\
                      .format(_it, c_mean, dist_mean))


            if _it % 2 == 0 or self.max_iter - _it <= self.final_restore_iters:
                # even iterations: restoration step
                if _it <= self.target_scan_iters:
                    x_adv = self._restore_move_scan(x_adv, x_original, y, y_onehot)
                else:
                    x_adv = self._restore_move(x_adv, x_original, y, y_onehot)

            else:
                # odd iterations: projection step
                self.project_lr = self.project_lr_init / (float(_it+1) ** self.nu) # learning rate schedule
                if self.metric == 'L2':
                    x_adv = self._project_move_l2(x_adv, x_original, y, y_onehot)
                elif self.metric == 'Linf':
                    x_adv = self._project_move_linf(x_adv, x_original, y, y_onehot)
            
        return x_adv
                
        
                
    def set_params(self, **kwargs):
        """Take in a dictionary of parameters and applies attack-specific checks before saving them as attributes.

        :param max_iter: The maximum number of iterations.
        :type max_iter: `int`
        """
        # Save attack-specific parameters
        super(MarginAttack, self).set_params(**kwargs)

        if type(self.max_iter) is not int or self.max_iter <= 0:
            raise ValueError("The number of iterations must be a positive integer.")
        
        if type(self.target_scan_iters) is not int or self.target_scan_iters <= 0:
            raise ValueError("target_scan_iters must be a positive integer.")
        
        if self.target_scan_iters > self.max_iter:
            raise ValueError("target_scan_iters should be no greater than max_iter.")
            
        if type(self.final_restore_iters) is not int or self.final_restore_iters <= 0:
            raise ValueError("final_restore_iters must be a positive integer.")
        
        if self.final_restore_iters > self.max_iter:
            raise ValueError("final_restore_iters should be no greater than max_iter.")
            
        if self.metric != 'L2' and self.metric != 'Linf':
            raise ValueError("'metric' must be 'L2' or 'Linf.'")
            
        if type(self.targeted) is not bool:
            raise ValueError("'targeted' must be boolean.")
        
        if self.num_classes is None:
            raise ValueError("num_classes must be specified.")
        
        if type(self.num_classes) is not int or self.num_classes <= 0:
            raise ValueError("num_classes must be a positive integer.")
        
        if type(self.num_scan_classes) is not int or self.num_scan_classes <= 0:
            raise ValueError("num_scan_classes must be a positive integer.")
        
        if self.num_scan_classes > self.num_classes-1:
            raise ValueError("num_scan_classes should be no greater than self.num_classes-1.")
            
        if self.input_max <= self.input_min:
            raise ValueError("input_max should be greater than input_min.")
            
        if self.restore_lr > 1 or self.restore_lr < 0:
            raise ValueError('restore_lr must be a float between 0 and 1.')
            
        if self.project_lr_init > 1 or self.project_lr_init < 0:
            raise ValueError('self.project_lr_init must be a float between 0 and 1.')
            
        if self.project_lr_ratio > 1 or self.project_lr_ratio < 0:
            raise ValueError('self.project_lr_ratio must be a float between 0 and 1.')
            
        if self.nu > 1 or self.nu < 0:
            raise ValueError('nu must be a float between 0 and 1.')
            
        if type(self.verbose) is not bool:
            raise ValueError('verbose must be a bool.')

        return True
    
    
    def _restore_move(self, x0, x_original, y, y_onehot):
        ''' Performs restoration move. All input arrays must match dimension 0.
        
        :param x0: the adversarial samples at the current iteration
        :type x0: `numpy.ndarray`
        :param x_original: the original input features to be attacked
        :type x_orignal: `numpy.ndarray`
        :param y: the class labels of x_original; one dim array
        :type y: `numpy.ndarray`
        :param y_onehot: the one-hot version of y; two dim array
        :type y_onehot: `numpy.ndarray`
        :return: the updated adversarial samples
        :rtype: `numpy.ndarray`
        '''
        
        # find the largest incorrect class
        f = self.classifier.predict(x0, logits=True)
        incorrect_class = np.argmax(f + np.log(1-y_onehot), axis=-1)
        
        # compute the constraint and its gradient
        grd_correct = self.classifier.class_gradient(x0, y, logits=True)
        grd_incorrect = self.classifier.class_gradient(x0, incorrect_class,
                                                           logits=True)
        f_correct = f[range(self.num_obs), y]
        f_incorrect = f[range(self.num_obs), incorrect_class]
        if self.targeted:
            c = f_incorrect - f_correct - self.offset
            c_grad = grd_incorrect - grd_correct
        else:
            c = f_correct - f_incorrect - self.offset
            c_grad = grd_correct - grd_incorrect
        c_grad = c_grad[:, 0, ...]
            
        # solve the constrained optimization problem
        b = -c * self.restore_lr
        x = self._min_norm(x0, c_grad, b=b,
                           x_original=x_original)
        return x
    
    def _restore_move_scan(self, x0, x_original, y, y_onehot):
        ''' Performs restoration move with target scan. All input arrays must match dimension 0.
        
        :param x0: the adversarial samples at the current iteration
        :type x0: `numpy.ndarray`
        :param x_original: the original input features to be attacked
        :type x_orignal: `numpy.ndarray`
        :param y: the class labels of x_original; one dim array
        :type y: `numpy.ndarray`
        :param y_onehot: the one-hot version of y; two dim array
        :type y_onehot: `numpy.ndarray`
        :return: the updated adversarial samples
        :rtype: `numpy.ndarray`
        '''
        
        
        # find the top incorrect classes
        f = self.classifier.predict(x0, logits=True)
        incorrect_class = np.argpartition(f + np.log(1-y_onehot), 
                                          f.shape[-1] - self.num_scan_classes, 
                                          axis=1)[:, -self.num_scan_classes:]
        
        # for each incorrect classes, perform the restoration move, 
        # pick the one with smallest perturbation
        xs = []
        dists = []
        grd_correct = self.classifier.class_gradient(x0, y, logits=True)
        f_correct = f[range(self.num_obs), y]
        for i in xrange(self.num_scan_classes):
            # compute the constraint and its gradient
            grd_incorrect = self.classifier.class_gradient(x0, incorrect_class[:, i],
                                                           logits=True)
            f_incorrect = f[range(self.num_obs), incorrect_class[:, i]]
            
            if self.targeted:
                c = f_incorrect - f_correct - self.offset
                c_grad = grd_incorrect - grd_correct
            else:
                c = f_correct - f_incorrect - self.offset
                c_grad = grd_correct - grd_incorrect
            c_grad = c_grad[:, 0, ...]

            # solve the constrained optimization problem
            b = -c * self.restore_lr
            _x = self._min_norm(x0, c_grad, b=b,
                                x_original=x_original)
            # evaluate the distance
            if self.metric == 'L2':
                _dist = np.sum((_x-x0)**2, axis=tuple(range(1, x0.ndim)))
            elif self.metric == 'Linf':
                _dist = np.amax(np.absolute(_x-x0), axis=tuple(range(1, x0.ndim)))
            elif self.metric == 'L1':
                _dist = np.sum(np.absolute(_x-x0), axis=tuple(range(1, x0.ndim)))
            xs.append(_x)
            dists.append(_dist)
        
        # find the smallest perturbation
        best_class = np.argmin(np.stack(dists, axis=0), axis=0)
        x = np.stack(xs, axis=0)[best_class, 
                                 range(x0.shape[0]), ...]

        return x
    
    def _project_move_l2(self, x1, x_original, y, y_onehot):
        ''' Performs projection move (the L2 version). All input arrays must match dimension 0.
        
        :param x1: the adversarial samples at the current iteration
        :type x1: `numpy.ndarray`
        :param x_original: the original input features to be attacked
        :type x_orignal: `numpy.ndarray`
        :param y: the class labels of x_original; one dim array
        :type y: `numpy.ndarray`
        :param y_onehot: the one-hot version of y; two dim array
        :type y_onehot: `numpy.ndarray`
        :return: the updated adversarial samples
        :rtype: `numpy.ndarray`
        '''
        
        # find the largest incorrect class
        f = self.classifier.predict(x1, logits=True)
        incorrect_class = np.argmax(f + np.log(1-y_onehot), axis=-1)
        
        # compute the constraint and its gradient
        grd_correct = self.classifier.class_gradient(x1, y, logits=True)
        grd_incorrect = self.classifier.class_gradient(x1, incorrect_class,
                                                           logits=True)
        f_correct = f[range(self.num_obs), y]
        f_incorrect = f[range(self.num_obs), incorrect_class]
        if self.targeted:
            c = f_incorrect - f_correct - self.offset
            c_grad = grd_incorrect - grd_correct
        else:
            c = f_correct - f_incorrect - self.offset
            c_grad = grd_correct - grd_incorrect
        c_grad = c_grad[:, 0, ...]
        
        # solve the constrained optimization problem
        _x = self._min_norm_nobound(x_original, c_grad, x1 = x1)
        
        x = x1 + self.project_lr * (_x-x1)
        
        return x

    def _project_move_linf(self, x1, x_original, y, y_onehot):
        ''' Performs projection move (the Linf version). All input arrays must match dimension 0.
        
        :param x1: the adversarial samples at the current iteration
        :type x1: `numpy.ndarray`
        :param x_original: the original input features to be attacked
        :type x_orignal: `numpy.ndarray`
        :param y: the class labels of x_original; one dim array
        :type y: `numpy.ndarray`
        :param y_onehot: the one-hot version of y; two dim array
        :type y_onehot: `numpy.ndarray`
        :return: the updated adversarial samples
        :rtype: `numpy.ndarray`
        '''
        
        # find the largest incorrect class
        f = self.classifier.predict(x1, logits=True)
        incorrect_class = np.argmax(f + np.log(1-y_onehot), axis=-1)
        
        # compute the constraint and its gradient
        grd_correct = self.classifier.class_gradient(x1, y, logits=True)
        grd_incorrect = self.classifier.class_gradient(x1, incorrect_class,
                                                           logits=True)
        f_correct = f[range(self.num_obs), y]
        f_incorrect = f[range(self.num_obs), incorrect_class]
        if self.targeted:
            c = f_incorrect - f_correct - self.offset
            c_grad = grd_incorrect - grd_correct
        else:
            c = f_correct - f_incorrect - self.offset
            c_grad = grd_correct - grd_incorrect
        c_grad = c_grad[:, 0, ...]
        
        # perform a gradient descent on d(x - x_original) over x
        dinf = np.amax(np.absolute(x1 - x_original), 
                       axis = tuple(range(1, x1.ndim)),
                       keepdims = True)
        dinf_target = (1 - self.project_lr) * dinf
        x0 = x_original + np.clip(x1 - np.sign(c_grad) * self.project_lr * self.project_lr_ratio\
                                  - x_original, -dinf_target, dinf_target)
        
        
        return x0
    
    def _min_norm(self, x0, c_grad, x1 = None, b = 0,
                  x_original = None):
        ''' This method solves the following constrained optimization problem:
        
        min_x d(x - x0)
        s.t. c_grad.transpose() * (x - x1) = b
             self.input_min[i] <= x[i] <= self.input_max[i]
        
        if x1 is None, set to x0
        
        The size of x0, x1, c_grad is (num_tokens, ...)
        The size of b should be (num_tokens,)
        
        :param x0: the vector from which the distance is computed.
        :type x0: `np.ndarray`
        :param c_grad: the normal direction of the constraint plane.
        :type c_grad: `np.ndarray`
        :param x1: a vector on the constraint plane.
        :type x1: `np.ndarray`
        :param b: the intercept of the constraint plane.
        :type b: `np.ndaaray`
        :param x_original: a bool numpy array the same size as x0 specifying which dimension is binding
        :type x_original: `np.ndarray`
        :return: the solution of the constrained optimization problem
        :rtype: `np.ndarray`
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
        elif self.metric == 'Linf':
            s = np.sign(c_grad)
        
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
            _x_mask = (np.logical_and(x_tent > self.input_max, x_mask == 0)).astype(int) -\
                      (np.logical_and(x_tent < self.input_min, x_mask == 0)).astype(int)
                
            # update the optimization problem
            if it < num_iters-1:
                if x1 is None:
                    b = b + np.sum(((x0 - self.input_max) * (_x_mask == 1).astype(int) +
                                    (x0 - self.input_min) * (_x_mask == -1).astype(int)) * c_grad,
                                   axis = tuple(range(1, ndim)),
                                   keepdims=True)
                else:
                    b = b + np.sum(((x1 - self.input_max) * (_x_mask == 1).astype(int) +
                                    (x1 - self.input_min) * (_x_mask == -1).astype(int)) * c_grad,
                                   axis = tuple(range(1, ndim)),
                                   keepdims=True)
                    
                c_grad[_x_mask != 0] = 0
                s[_x_mask != 0] = 0
                
            x_mask += _x_mask
                    
        # return the final answer
        x = x_tent
        x[x_mask == 1] = self.input_max
        x[x_mask == -1] = self.input_min       
        
        return x
    
    def _min_norm_nobound(self, x0, c_grad, x1 = None, b = 0,
                          x_original = None):
        ''' This function solves the following constrained optimization problem:
        
        min_x d(x - x0)
        s.t. c_grad.transpose() * (x - x1) = b
        
        if x1 is None, set to x0
        
        The size of x0, x1, c_grad is (num_tokens, ...)
        The size of b should be (num_tokens,)
        
        :param x0: the vector from which the distance is computed.
        :type x0: `np.ndarray`
        :param c_grad: the normal direction of the constraint plane.
        :type c_grad: `np.ndarray`
        :param x1: a vector on the constraint plane.
        :type x1: `np.ndarray`
        :param b: the intercept of the constraint plane.
        :type b: `np.ndaaray`
        :param x_original: a bool numpy array the same size as x0 specifying which dimension is binding
        :type x_original: `np.ndarray`
        :return: the solution of the constrained optimization problem
        :rtype: `np.ndarray`
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
        elif self.metric == 'Linf':
            s = np.sign(c_grad)
        

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
            