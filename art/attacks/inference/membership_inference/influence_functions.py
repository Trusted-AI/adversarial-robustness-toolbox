import os
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
import time
import datetime
import numpy as np
import copy
import logging
from tqdm import tqdm

from pathlib import Path

from art.attacks.inference.membership_inference.utils_sif import (
    grad_z,
    s_test_sample,
    save_json,
    display_progress,
    RGB_MEAN,
    RGB_STD
)


def calc_s_test(
    model,
    test_loader,
    train_loader,
    save=False,
    gpu=-1,
    damp=0.01,
    scale=25,
    recursion_depth=5000,
    r=1,
    start=0,
):
    """Calculates s_test for the whole test dataset taking into account all
    training data images.

    Arguments:
        model: pytorch model, for which s_test should be calculated
        test_loader: pytorch dataloader, which can load the test data
        train_loader: pytorch dataloader, which can load the train data
        save: Path, path where to save the s_test files if desired. Omitting
            this argument will skip saving
        gpu: int, device id to use for GPU, -1 for CPU (default)
        damp: float, influence function damping factor
        scale: float, influence calculation scaling factor
        recursion_depth: int, number of recursions to perform during s_test
            calculation, increases accuracy. r*recursion_depth should equal the
            training dataset size.
        r: int, number of iterations of which to take the avg.
            of the h_estimate calculation; r*recursion_depth should equal the
            training dataset size.
        start: int, index of the first test index to use. default is 0

    Returns:
        s_tests: list of torch vectors, contain all s_test for the whole
            dataset. Can be huge.
        save: Path, path to the folder where the s_test files were saved to or
            False if they were not saved."""
    if save and not isinstance(save, Path):
        save = Path(save)
    if not save:
        logging.info("ATTENTION: not saving s_test files.")

    s_tests = []
    for i in range(start, len(test_loader.dataset)):
        z_test, t_test = test_loader.dataset[i]
        z_test = test_loader.collate_fn([z_test])
        t_test = test_loader.collate_fn([t_test])

        s_test_vec = s_test_sample(
            model, z_test, t_test, train_loader, gpu, damp, scale, recursion_depth, r
        )

        if save:
            s_test_vec = [s.cpu() for s in s_test_vec]
            torch.save(
                s_test_vec, save.joinpath(
                    f"{i}_recdep{recursion_depth}_r{r}.s_test")
            )
        else:
            s_tests.append(s_test_vec)
        display_progress(
            "Calc. z_test (s_test): ", i -
            start, len(test_loader.dataset) - start
        )

    return s_tests, save


def calc_self_influence(X, y, net, rec_dep, r):
    influences = []
    for i in range(X.shape[0]):
        tensor_dataset = TensorDataset(torch.from_numpy(np.expand_dims(X[i], 0)),
                                       torch.from_numpy(np.expand_dims(y[i], 0)))
        loader = DataLoader(tensor_dataset, batch_size=1, shuffle=False,
                            pin_memory=False, drop_last=False)
        influence, _, _, _ = calc_influence_single(
            net, loader, loader, 0, 0, rec_dep, r)
        influences.append(influence.item())
    return np.asarray(influences)


def calc_self_influence_average(X, y, net, rec_dep, r):
    influences = []
    img_size = X.shape[2]
    pad_size = int(img_size / 8)
    train_transform = transforms.Compose([
        transforms.RandomCrop(img_size, padding=pad_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    for i in range(X.shape[0]):
        train_transform_gen = MyVisionDataset(
            X[i], y[i], transform=train_transform)  # just for transformations
        test_dataset = TensorDataset(torch.from_numpy(np.expand_dims(X[i], 0)),
                                     torch.from_numpy(np.expand_dims(y[i], 0)))
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            pin_memory=False,
            drop_last=False)
        influences_tmp = []
        for k in range(8):
            X_aug, y_aug = train_transform_gen.__getitem__(0)
            train_dataset = TensorDataset(torch.from_numpy(np.expand_dims(X_aug, 0)),
                                          torch.from_numpy(np.expand_dims(y_aug, 0)))
            train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False,
                                      pin_memory=False, drop_last=False)
            influence, _, _, _ = calc_influence_single(
                net, train_loader, test_loader, 0, 0, rec_dep, r)
            influences_tmp.append(influence.item())

        influences.append(np.mean(influences_tmp))
    return np.asarray(influences)


def calc_self_influence_adaptive(X, y, net, rec_dep, r):
    influences = []
    img_size = X.shape[2]
    pad_size = int(img_size / 8)
    train_transform = transforms.Compose([
        transforms.RandomCrop(img_size, padding=pad_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    for i in range(X.shape[0]):
        train_dataset = MyVisionDataset(X[i], y[i], transform=train_transform)
        test_dataset = TensorDataset(torch.from_numpy(np.expand_dims(X[i], 0)),
                                     torch.from_numpy(np.expand_dims(y[i], 0)))
        train_loader = DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=False,
            pin_memory=False,
            drop_last=False)
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            pin_memory=False,
            drop_last=False)
        influence, _, _, _ = calc_influence_single_adaptive(
            net, train_loader, test_loader, 0, 0, rec_dep, r)
        influences.append(influence.item())
    return np.asarray(influences)


def calc_self_influence_adaptive_for_ref(X, y, net, rec_dep, r):
    influences = []
    img_size = X.shape[2]
    pad_size = int(img_size / 8)
    train_transform = transforms.Compose([
        transforms.RandomCrop(img_size, padding=pad_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(RGB_MEAN, RGB_STD)
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(RGB_MEAN, RGB_STD)
    ])

    for i in range(X.shape[0]):
        train_dataset = MyVisionDataset(X[i], y[i], transform=train_transform)
        test_dataset = MyVisionDataset(X[i], y[i], transform=test_transform)
        train_loader = DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=False,
            pin_memory=False,
            drop_last=False)
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            pin_memory=False,
            drop_last=False)
        influence, _, _, _ = calc_influence_single_adaptive(
            net, train_loader, test_loader, 0, 0, rec_dep, r)
        influences.append(influence.item())
    return np.asarray(influences)

def calc_influence_single_adaptive(
    model,
    train_loader,
    test_loader,
    test_id_num,
    gpu,
    recursion_depth,
    r,
    damp=0.01,
    scale=25,
    s_test_vec=None,
    time_logging=False,
    loss_func="cross_entropy",
):
    # Calculate s_test vectors if not provided
    if s_test_vec is None:
        z_test, t_test = test_loader.dataset[test_id_num]
        z_test = test_loader.collate_fn([z_test])
        t_test = test_loader.collate_fn([t_test])

        # debug
        # z_img = convert_tensor_to_image(z_test[0].cpu().numpy())
        # plt.imshow(z_img)
        # plt.show()

        s_test_vec = s_test_sample(
            model,
            z_test,
            t_test,
            train_loader,
            gpu,
            recursion_depth=recursion_depth,
            r=r,
            damp=damp,
            scale=scale,
            loss_func=loss_func,
        )

    # Calculate the influence function
    train_dataset_size = len(train_loader.dataset)
    assert train_dataset_size == 1
    influences = []
    num_layers = sum(1 for x in model.parameters())
    num_iters = 128
    grad_z_vec = []

    if time_logging:
        time_a = datetime.datetime.now()

    for i in range(num_iters):
        z, t = train_loader.dataset[0]
        z = train_loader.collate_fn([z])
        t = train_loader.collate_fn([t])

        # debug
        # z_img = convert_tensor_to_image(z[0].cpu().numpy())
        # plt.imshow(z_img)
        # plt.show()

        grad_z_vec_tmp = list(grad_z(z, t, model, gpu=gpu))
        for j in range(num_layers):
            if i == 0:
                grad_z_vec.append(torch.zeros_like(grad_z_vec_tmp[j]))
            else:
                grad_z_vec[j] += grad_z_vec_tmp[j]

    for j in range(num_layers):
        grad_z_vec[j] /= num_iters

    if time_logging:
        time_b = datetime.datetime.now()
        time_delta = time_b - time_a
        logging.info(
            f"Time for grad_z iter:" f" {time_delta.total_seconds() * 1000}"
        )
    with torch.no_grad():
        tmp_influence = (
            -sum(
                [
                    ####################
                    # TODO: potential bottle neck, takes 17% execution time
                    # torch.sum(k * j).data.cpu().numpy()
                    ####################
                    torch.sum(k * j).data
                    for k, j in zip(grad_z_vec, s_test_vec)
                ]
            )
            / train_dataset_size
        )

    influences.append(tmp_influence)

    influences = torch.stack(influences)
    influences = influences.cpu().numpy()
    harmful = np.argsort(influences)
    helpful = harmful[::-1]

    return influences, harmful.tolist(), helpful.tolist(), test_id_num



def calc_self_influence_average_for_ref(X, y, net, rec_dep, r):
    influences = []
    img_size = X.shape[2]
    pad_size = int(img_size / 8)
    train_transform = transforms.Compose([
        transforms.RandomCrop(img_size, padding=pad_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(RGB_MEAN, RGB_STD)
    ])
    test_transform = transforms.Normalize(RGB_MEAN, RGB_STD)
    for i in range(X.shape[0]):
        train_transform_gen = MyVisionDataset(
            X[i], y[i], transform=train_transform)  # just for transformations
        X_tensor = torch.tensor(X[i])
        X_transformed = test_transform(X_tensor).cpu().numpy()
        test_dataset = TensorDataset(torch.from_numpy(np.expand_dims(X_transformed, 0)),
                                     torch.from_numpy(np.expand_dims(y[i], 0)))
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            pin_memory=False,
            drop_last=False)
        influences_tmp = []
        for k in range(8):
            X_aug, y_aug = train_transform_gen.__getitem__(0)
            train_dataset = TensorDataset(torch.from_numpy(np.expand_dims(X_aug, 0)),
                                          torch.from_numpy(np.expand_dims(y_aug, 0)))
            train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False,
                                      pin_memory=False, drop_last=False)
            influence, _, _, _ = calc_influence_single(
                net, train_loader, test_loader, 0, 0, rec_dep, r)
            influences_tmp.append(influence.item())

        influences.append(np.mean(influences_tmp))
    return np.asarray(influences)


def calc_self_influence_for_ref(X, y, net, rec_dep, r):
    influences = []
    transform = transforms.Normalize(RGB_MEAN, RGB_STD)
    for i in range(X.shape[0]):
        X_tensor = torch.tensor(X[i])
        X_transformed = transform(X_tensor).cpu().numpy()
        tensor_dataset = TensorDataset(torch.from_numpy(np.expand_dims(X_transformed, 0)),
                                       torch.from_numpy(np.expand_dims(y[i], 0)))
        loader = DataLoader(tensor_dataset, batch_size=1, shuffle=False,
                            pin_memory=False, drop_last=False)
        influence, _, _, _ = calc_influence_single(
            net, loader, loader, 0, 0, rec_dep, r)
        influences.append(influence.item())
    return np.asarray(influences)

def calc_influence_single(
    model,
    train_loader,
    test_loader,
    test_id_num,
    gpu,
    recursion_depth,
    r,
    damp=0.01,
    scale=25,
    s_test_vec=None,
    time_logging=False,
    loss_func="cross_entropy",
):
    """Calculates the influences of all training data points on a single
    test dataset image.

    Arugments:
        model: pytorch model
        train_loader: DataLoader, loads the training dataset
        test_loader: DataLoader, loads the test dataset
        test_id_num: int, id of the test sample for which to calculate the
            influence function
        gpu: int, identifies the gpu id, -1 for cpu
        recursion_depth: int, number of recursions to perform during s_test
            calculation, increases accuracy. r*recursion_depth should equal the
            training dataset size.
        r: int, number of iterations of which to take the avg.
            of the h_estimate calculation; r*recursion_depth should equal the
            training dataset size.
        s_test_vec: list of torch tensor, contains s_test vectors. If left
            empty it will also be calculated

    Returns:
        influence: list of float, influences of all training data samples
            for one test sample
        harmful: list of float, influences sorted by harmfulness
        helpful: list of float, influences sorted by helpfulness
        test_id_num: int, the number of the test dataset point
            the influence was calculated for"""
    # Calculate s_test vectors if not provided
    if s_test_vec is None:
        z_test, t_test = test_loader.dataset[test_id_num]
        z_test = test_loader.collate_fn([z_test])
        t_test = test_loader.collate_fn([t_test])
        s_test_vec = s_test_sample(
            model,
            z_test,
            t_test,
            train_loader,
            gpu,
            recursion_depth=recursion_depth,
            r=r,
            damp=damp,
            scale=scale,
            loss_func=loss_func,
        )

    # Calculate the influence function
    train_dataset_size = len(train_loader.dataset)
    influences = []
    for i in tqdm(range(train_dataset_size)):
        z, t = train_loader.dataset[i]
        z = train_loader.collate_fn([z])
        t = train_loader.collate_fn([t])

        if time_logging:
            time_a = datetime.datetime.now()

        grad_z_vec = grad_z(z, t, model, gpu=gpu)

        if time_logging:
            time_b = datetime.datetime.now()
            time_delta = time_b - time_a
            logging.info(
                f"Time for grad_z iter:" f" {time_delta.total_seconds() * 1000}"
            )
        with torch.no_grad():
            tmp_influence = (
                -sum(
                    [
                        ####################
                        # TODO: potential bottle neck, takes 17% execution time
                        # torch.sum(k * j).data.cpu().numpy()
                        ####################
                        torch.sum(k * j).data
                        for k, j in zip(grad_z_vec, s_test_vec)
                    ]
                )
                / train_dataset_size
            )

        influences.append(tmp_influence)

    influences = torch.stack(influences)
    influences = influences.cpu().numpy()
    harmful = np.argsort(influences)
    helpful = harmful[::-1]

    return influences, harmful.tolist(), helpful.tolist(), test_id_num

def load_grad_z(grad_z_dir=Path("./grad_z/"), train_dataset_size=-1):
    """Loads all grad_z data required to calculate the influence function and
    returns it.

    Arguments:
        grad_z_dir: Path, folder containing files storing the grad_z values
        train_dataset_size: int, number of total samples in dataset;
            -1 indicates to use all available grad_z files

    Returns:
        grad_z_vecs: list of torch tensors, contains the grad_z tensors"""
    if isinstance(grad_z_dir, str):
        grad_z_dir = Path(grad_z_dir)

    grad_z_vecs = []
    logging.info(f"Loading grad_z from: {grad_z_dir} ...")
    available_grad_z_files = len(list(grad_z_dir.glob("*.grad_z")))
    if available_grad_z_files != train_dataset_size:
        logging.warning("Load Influence Data: number of grad_z files mismatches" " the dataset size")
        if -1 == train_dataset_size:
            train_dataset_size = available_grad_z_files
    for i in range(train_dataset_size):
        grad_z_vecs.append(torch.load(os.path.join(grad_z_dir, str(i) + '.grad_z')))
        display_progress("grad_z files loaded: ", i, train_dataset_size)

    return grad_z_vecs

def load_s_test(s_test_dir=Path("./s_test/"), test_dataset_size=10, suffix='recdep500_r1'):
    """Loads all s_test data required to calculate the influence function
    and returns a list of it.

    Arguments:
        s_test_dir: Path, folder containing files storing the s_test values
        s_test_id: int, number of the test data sample s_test was calculated for
        test_dataset_size: int, number of s_tests vectors expected

    Returns:
        e_s_test: list of torch vectors, contains all e_s_tests for the whole dataset.
        s_test: list of torch vectors, contain all s_test for the whole dataset. Can be huge."""
    if isinstance(s_test_dir, str):
        s_test_dir = Path(s_test_dir)

    s_test = []
    logging.info(f"Loading s_test from: {s_test_dir} ...")
    num_s_test_files = len(list(s_test_dir.glob("*.s_test")))
    if num_s_test_files != test_dataset_size:
        logging.warning(
            "Load Influence Data: number of s_test sample files"
            " mismatches the available samples"
        )
    for i in range(num_s_test_files):
        s_test.append(torch.load(os.path.join(s_test_dir, str(i) + '_' + suffix + '.s_test')))
        display_progress("s_test files loaded: ", i, test_dataset_size)

    return s_test

def calc_all_influences(grad_z_dir, train_dataset_size,
                        s_test_dir, test_dataset_size):
    grad_z_vecs = load_grad_z(
        grad_z_dir=grad_z_dir,
        train_dataset_size=train_dataset_size)
    suffix = 'recdep{}_r1'.format(train_dataset_size)
    s_test_vecs = load_s_test(
        s_test_dir=s_test_dir,
        test_dataset_size=test_dataset_size,
        suffix=suffix)

    influences = torch.zeros(test_dataset_size, train_dataset_size)
    for i in tqdm(range(test_dataset_size)):
        s_test_vec = s_test_vecs[i]
        for j in range(train_dataset_size):
            grad_z_vec = grad_z_vecs[j]
            with torch.no_grad():
                tmp_influence = (
                    -sum(
                        [
                            torch.sum(k * j).data
                            for k, j in zip(grad_z_vec, s_test_vec)
                        ]
                    )
                    / train_dataset_size
                )
            influences[i, j] = tmp_influence
    influences = influences.cpu().numpy()
    return influences


def calc_grad_z(model, train_loader, save_pth=False, gpu=-1, start=0):
    """Calculates grad_z and can save the output to files. One grad_z should
    be computed for each training data sample.

    Arguments:
        model: pytorch model, for which s_test should be calculated
        train_loader: pytorch dataloader, which can load the train data
        save_pth: Path, path where to save the grad_z files if desired.
            Omitting this argument will skip saving
        gpu: int, device id to use for GPU, -1 for CPU (default)
        start: int, index of the first test index to use. default is 0

    Returns:
        grad_zs: list of torch tensors, contains the grad_z tensors
        save_pth: Path, path where grad_z files were saved to or
            False if they were not saved."""
    if save_pth and isinstance(save_pth, str):
        save_pth = Path(save_pth)
    if not save_pth:
        logging.info("ATTENTION: Not saving grad_z files!")

    grad_zs = []
    for i in range(start, len(train_loader.dataset)):
        z, t = train_loader.dataset[i]
        z = train_loader.collate_fn([z])
        t = train_loader.collate_fn([t])
        grad_z_vec = grad_z(z, t, model, gpu=gpu)
        if save_pth:
            grad_z_vec = [g.cpu() for g in grad_z_vec]
            torch.save(grad_z_vec, save_pth.joinpath(f"{i}.grad_z"))
        else:
            grad_zs.append(grad_z_vec)
        display_progress("Calc. grad_z: ", i - start,
                         len(train_loader.dataset) - start)

    return grad_zs, save_pth

from typing import Any, Tuple
from torchvision.datasets import VisionDataset
import numpy as np
from PIL import Image

def convert_tensor_to_image(x: np.ndarray):
    """
    :param X: np.array of size (Batch, feature_dims, H, W) or (feature_dims, H, W)
    :return: X with (Batch, H, W, feature_dims) or (H, W, feature_dims) between 0:255, uint8
    """
    X = x.copy()
    X *= 255.0
    X = np.round(X)
    X = X.astype(np.uint8)
    if len(x.shape) == 3:
        X = np.transpose(X, [1, 2, 0])
    else:
        X = np.transpose(X, [0, 2, 3, 1])
    return X

class MyVisionDataset(VisionDataset):

    def __init__(self, data: np.ndarray, y_gt: np.ndarray, *args, **kwargs) -> None:
        root = None
        super().__init__(root, *args, **kwargs)
        assert isinstance(data, np.ndarray), 'type of data must be np.ndarray type, but got {} instead'.format(type(data))
        assert isinstance(y_gt, (np.int32, np.int64)), 'type of y_gt must be np.int type, but got {} instead'.format(type(y_gt))
        self.data = np.expand_dims(convert_tensor_to_image(data), 0)
        self.y_gt = np.expand_dims(y_gt, 0)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        assert index == 0
        img, y_gt = self.data[index], self.y_gt[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            y_gt = self.target_transform(y_gt)

        return img, y_gt
