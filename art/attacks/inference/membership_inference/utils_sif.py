from torchvision import transforms
import sys
import json
from pathlib import Path
from datetime import datetime as dt
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import fmin_ncg
from torch.utils.data import DataLoader
from torch.autograd import grad
from torch.autograd.functional import vhp

RGB_MEAN = (0.4914, 0.4822, 0.4465)
RGB_STD = (0.2023, 0.1994, 0.2010)


def normalize(x, rgb_mean, rgb_std):
    """
    :param x: np.ndaaray of image RGB of (3, W, H), normalized between [0,1]
    :param rgb_mean: Tuple of (RED mean, GREEN mean, BLUE mean)
    :param rgb_std: Tuple of (RED std, GREEN std, BLUE std)
    :return np.ndarray transformed by x = (x-mean)/std
    """
    transform = transforms.Normalize(rgb_mean, rgb_std)
    x_tensor = torch.tensor(x)
    x_new = transform(x_tensor)
    x_new = x_new.cpu().numpy()
    return x_new


def grad_z(x, y, model, gpu=-1, loss_func="cross_entropy"):
    """Calculates the gradient z. One grad_z should be computed for each
    training sample.

    Arguments:
        x: torch tensor, training data points
            e.g. an image sample (batch_size, 3, 256, 256)
        y: torch tensor, training data labels
        model: torch NN, model used to evaluate the dataset
        gpu: int, device id to use for GPU, -1 for CPU

    Returns:
        grad_z: list of torch tensor, containing the gradients
            from model parameters to loss"""
    model.eval()

    # initialize
    if gpu >= 0:
        x, y = x.cuda(), y.cuda()

    prediction = model(x)

    loss = calc_loss(prediction, y, loss_func=loss_func)

    # Compute sum of gradients from model parameters to loss
    return grad(loss, model.parameters())


def calc_loss(logits, labels, loss_func="cross_entropy"):
    """Calculates the loss

    Arguments:
        logits: torch tensor, input with size (minibatch, nr_of_classes)
        labels: torch tensor, target expected by loss of size (0 to nr_of_classes-1)
        loss_func: str, specify loss function name

    Returns:
        loss: scalar, the loss"""

    if loss_func == "cross_entropy":
        if logits.shape[-1] == 1:
            loss = F.binary_cross_entropy_with_logits(logits, labels.type(torch.float))
        else:
            loss = F.cross_entropy(logits, labels)
    elif loss_func == "mean":
        loss = torch.mean(logits)
    else:
        raise ValueError("{} is not a valid value for loss_func".format(loss_func))

    return loss


def s_test_sample(
    model,
    x_test,
    y_test,
    train_loader,
    gpu=-1,
    damp=0.01,
    scale=25,
    recursion_depth=5000,
    r=1,
    loss_func="cross_entropy",
):
    """Calculates s_test for a single test image taking into account the whole
    training dataset. s_test = invHessian * nabla(Loss(test_img, model params))

    Arguments:
        model: pytorch model, for which s_test should be calculated
        x_test: test image
        y_test: test image label
        train_loader: pytorch dataloader, which can load the train data
        gpu: int, device id to use for GPU, -1 for CPU (default)
        damp: float, influence function damping factor
        scale: float, influence calculation scaling factor
        recursion_depth: int, number of recursions to perform during s_test
            calculation, increases accuracy. r*recursion_depth should equal the
            training dataset size.
        r: int, number of iterations of which to take the avg.
            of the h_estimate calculation; r*recursion_depth should equal the
            training dataset size.

    Returns:
        s_test_vec: torch tensor, contains s_test for a single test image"""

    inverse_hvp = [torch.zeros_like(params, dtype=torch.float) for params in model.parameters()]

    for i in range(r):
        hessian_loader = DataLoader(
            train_loader.dataset,
            sampler=torch.utils.data.RandomSampler(train_loader.dataset, True, num_samples=recursion_depth),
            batch_size=1,
            num_workers=4,
        )

        cur_estimate = s_test(
            x_test,
            y_test,
            model,
            i,
            hessian_loader,
            gpu=gpu,
            damp=damp,
            scale=scale,
            loss_func=loss_func,
        )

        with torch.no_grad():
            inverse_hvp = [old + (cur / scale) for old, cur in zip(inverse_hvp, cur_estimate)]

    with torch.no_grad():
        inverse_hvp = [component / r for component in inverse_hvp]

    return inverse_hvp


def s_test(x_test, y_test, model, i, samples_loader, gpu=-1, damp=0.01, scale=25.0, loss_func="cross_entropy"):
    """s_test can be precomputed for each test point of interest, and then
    multiplied with grad_z to get the desired value for each training point.
    Here, stochastic estimation is used to calculate s_test. s_test is the
    Inverse Hessian Vector Product.

    Arguments:
        x_test: torch tensor, test data points, such as test images
        y_test: torch tensor, contains all test data labels
        model: torch NN, model used to evaluate the dataset
        i: the sample number
        samples_loader: torch DataLoader, can load the training dataset
        gpu: int, GPU id to use if >=0 and -1 means use CPU
        damp: float, dampening factor
        scale: float, scaling factor

    Returns:
        h_estimate: list of torch tensors, s_test"""

    v = grad_z(x_test, y_test, model, gpu, loss_func=loss_func)
    h_estimate = v

    params, names = make_functional(model)
    # Make params regular Tensors instead of nn.Parameter
    params = tuple(p.detach().requires_grad_() for p in params)

    # TODO: Dynamically set the recursion depth so that iterations stop once h_estimate stabilises
    progress_bar = tqdm(samples_loader, desc=f"IHVP sample {i}")
    for i, (x_train, y_train) in enumerate(progress_bar):
        if gpu >= 0:
            x_train, y_train = x_train.cuda(), y_train.cuda()

        def f(*new_params):
            load_weights(model, names, new_params)
            out = model(x_train)
            loss = calc_loss(out, y_train, loss_func=loss_func)
            return loss

        hv = vhp(f, params, tuple(h_estimate), strict=True)[1]

        # Recursively calculate h_estimate
        with torch.no_grad():
            h_estimate = [_v + (1 - damp) * _h_e - _hv / scale for _v, _h_e, _hv in zip(v, h_estimate, hv)]

            if i % 100 == 0:
                norm = sum([h_.norm() for h_ in h_estimate])
                progress_bar.set_postfix({"est_norm": norm.item()})

    with torch.no_grad():
        load_weights(model, names, params, as_params=True)

    return h_estimate


def save_json(
    json_obj,
    json_path,
    append_if_exists=False,
    overwrite_if_exists=False,
    unique_fn_if_exists=True,
):
    """Saves a json file

    Arguments:
        json_obj: json, json object
        json_path: Path, path including the file name where the json object
            should be saved to
        append_if_exists: bool, append to the existing json file with the same
            name if it exists (keep the json structure intact)
        overwrite_if_exists: bool, xor with append, overwrites any existing
            target file
        unique_fn_if_exsists: bool, appends the current date and time to the
            file name if the target file exists already.
    """
    if isinstance(json_path, str):
        json_path = Path(json_path)

    if overwrite_if_exists:
        append_if_exists = False
        unique_fn_if_exists = False

    if unique_fn_if_exists:
        overwrite_if_exists = False
        append_if_exists = False
        if json_path.exists():
            time = dt.now().strftime("%Y-%m-%d-%H-%M-%S")
            json_path = json_path.parents[0] / f"{str(json_path.stem)}_{time}" f"{str(json_path.suffix)}"

    if overwrite_if_exists:
        append_if_exists = False
        with open(json_path, "w+") as fout:
            json.dump(json_obj, fout, indent=2)
        return

    if append_if_exists:
        if json_path.exists():
            with open(json_path, "r") as fin:
                read_file = json.load(fin)
            read_file.update(json_obj)
            with open(json_path, "w+") as fout:
                json.dump(read_file, fout, indent=2)
            return

    with open(json_path, "w+") as fout:
        json.dump(json_obj, fout, indent=2)


def display_progress(text, current_step, last_step, enabled=True, fix_zero_start=True):
    """Draws a progress indicator on the screen with the text preceeding the
    progress

    Arguments:
        test: str, text displayed to describe the task being executed
        current_step: int, current step of the iteration
        last_step: int, last possible step of the iteration
        enabled: bool, if false this function will not execute. This is
            for running silently without stdout output.
        fix_zero_start: bool, if true adds 1 to each current step so that the
            display starts at 1 instead of 0, which it would for most loops
            otherwise.
    """
    if not enabled:
        return

    # Fix display for most loops which start with 0, otherwise looks weird
    if fix_zero_start:
        current_step = current_step + 1

    term_line_len = 80
    final_chars = [":", ";", " ", ".", ","]
    if text[-1:] not in final_chars:
        text = text + " "
    if len(text) < term_line_len:
        bar_len = term_line_len - (len(text) + len(str(current_step)) + len(str(last_step)) + len("  / "))
    else:
        bar_len = 30
    filled_len = int(round(bar_len * current_step / float(last_step)))
    bar = "=" * filled_len + "." * (bar_len - filled_len)

    bar = f"{text}[{bar:s}] {current_step:d} / {last_step:d}"
    if current_step < last_step - 1:
        # Erase to end of line and print
        sys.stdout.write("\033[K" + bar + "\r")
    else:
        sys.stdout.write(bar + "\n")

    sys.stdout.flush()


def conjugate_gradient(ax_fn, b, debug_callback=None, avextol=None, maxiter=None):
    """Computes the solution to Ax - b = 0 by minimizing the conjugate objective
    f(x) = x^T A x / 2 - b^T x. This does not require evaluating the matrix A
    explicitly, only the matrix vector product Ax.

    From https://github.com/kohpangwei/group-influence-release/blob/master/influence/conjugate.py.

    Args:
      ax_fn: A function that return Ax given x.
      b: The vector b.
      debug_callback: An optional debugging function that reports the current optimization function. Takes two
          parameters: the current solution and a helper function that evaluates the quadratic and linear parts of the
          conjugate objective separately. (Default value = None)
      avextol:  (Default value = None)
      maxiter:  (Default value = None)

    Returns:
      The conjugate optimization solution.

    """

    cg_callback = None
    if debug_callback:

        def cg_callback(x):
            return debug_callback(x, -np.dot(b, x), 0.5 * np.dot(x, ax_fn(x)))

    result = fmin_ncg(
        f=lambda x: 0.5 * np.dot(x, ax_fn(x)) - np.dot(b, x),
        x0=np.zeros_like(b),
        fprime=lambda x: ax_fn(x) - b,
        fhess_p=lambda x, p: ax_fn(p),
        callback=cg_callback,
        avextol=avextol,
        maxiter=maxiter,
    )

    return result


def load_weights(model, names, params, as_params=False):
    for name, p in zip(names, params):
        if not as_params:
            set_attr(model, name.split("."), p)
        else:
            set_attr(model, name.split("."), torch.nn.Parameter(p))


def set_attr(obj, names, val):
    if len(names) == 1:
        setattr(obj, names[0], val)
    else:
        set_attr(getattr(obj, names[0]), names[1:], val)


def tensor_to_tuple(vec, parameters):
    r"""Convert one vector to the parameters

    Adapted from
    https://pytorch.org/docs/master/generated/torch.nn.utils.vector_to_parameters.html#torch.nn.utils.vector_to_parameters

    Arguments:
        vec (Tensor): a single vector represents the parameters of a model.
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.
    """
    if not isinstance(vec, torch.Tensor):
        raise TypeError("expected torch.Tensor, but got: {}".format(torch.typename(vec)))

    # Pointer for slicing the vector for each parameter
    pointer = 0

    split_tensors = []
    for param in parameters:
        # The length of the parameter
        num_param = param.numel()
        # Slice the vector, reshape it, and replace the old data of the
        # parameter
        split_tensors.append(vec[pointer : pointer + num_param].view_as(param))

        # Increment the pointer
        pointer += num_param

    return tuple(split_tensors)


def del_attr(obj, names):
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        del_attr(getattr(obj, names[0]), names[1:])


def make_functional(model):
    orig_params = tuple(model.parameters())
    # Remove all the parameters in the model
    names = []

    for name, p in list(model.named_parameters()):
        del_attr(model, name.split("."))
        names.append(name)

    return orig_params, names
