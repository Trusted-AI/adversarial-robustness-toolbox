# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2024
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
import logging

import numpy as np
import pytest

from art.attacks.evasion import SNAL

from tests.utils import ARTTestException

logger = logging.getLogger(__name__)


@pytest.mark.only_with_platform("pytorch")
def test_generate(art_warning, get_pytorch_detector_yolo):
    try:
        import glob
        from io import BytesIO
        import os
        from PIL import Image
        import requests
        import time
        import torch
        from torchvision import transforms
        from torchvision.datasets.vision import VisionDataset

        object_detector = get_pytorch_detector_yolo
        object_detector.set_params(input_shape=(3, 640, 640))

        # Define a custom function to collect patches from images
        def collect_patches_from_images(detector, imgs):
            bs = imgs.shape[0]
            pred = detector.predict(x=imgs)
            y = []
            for obj in pred:
                y.append(obj["boxes"])

            candidates_patch = []
            candidates_position = []
            for i in range(bs):
                patch = []
                if y[i].shape[0] == 0:
                    candidates_patch.append(patch)
                    candidates_position.append(torch.zeros((0, 4), device=object_detector.device))
                    continue

                pos_matrix = torch.from_numpy(y[i][:, :4]).int()
                pos_matrix[:, 0] = torch.clamp_min(pos_matrix[:, 0], 0)
                pos_matrix[:, 1] = torch.clamp_min(pos_matrix[:, 1], 0)
                pos_matrix[:, 2] = torch.clamp_max(pos_matrix[:, 2], imgs.shape[3])
                pos_matrix[:, 3] = torch.clamp_max(pos_matrix[:, 3], imgs.shape[2])
                for pos_m in pos_matrix:
                    if pos_m[3] - pos_m[1] == 0 or pos_m[2] - pos_m[0] == 0:
                        continue
                    p = imgs[i, :, pos_m[1] : pos_m[3], pos_m[0] : pos_m[2]]
                    patch.append(p.to(detector.device))

                candidates_patch.append(patch)
                candidates_position.append(pos_matrix)

            return candidates_patch, candidates_position

        # Download a sample image
        target = "https://farm2.staticflickr.com/1065/705706084_39a7f28fc9_z.jpg"  # val2017/000000552842.jpg
        headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.flickr.com/"}
        response = requests.get(target, headers=headers)
        org_img = np.asarray(Image.open(BytesIO(response.content)).resize((640, 640)))
        x = np.stack([org_img.transpose((2, 0, 1)) / 255.0], axis=0).astype(np.float32)

        # Prepare dataset
        # Select images randomly from COCO dataset
        list_url = [
            "http://farm4.staticflickr.com/3572/5744200926_082c11c43c_z.jpg",  # 000000460229
            "http://farm4.staticflickr.com/3010/2749181045_ed450e5d36_z.jpg",  # 000000057760
            "http://farm4.staticflickr.com/3826/9451771633_f14cef3a8b_z.jpg",  # 000000468332
            "http://farm7.staticflickr.com/6194/6106161903_e505cbc192_z.jpg",  # 000000190841
            "http://farm1.staticflickr.com/48/140268688_947e2bcc96_z.jpg",  # 000000078420
            "http://farm6.staticflickr.com/5011/5389083366_fdf13f2ee6_z.jpg",  # 000000309655
            "http://farm4.staticflickr.com/3552/5812461870_eb24c8eac5_z.jpg",  # 000000293324
            "http://farm4.staticflickr.com/3610/3361019695_1005dd49fd_z.jpg",  # 000000473821
            "http://farm8.staticflickr.com/7323/9725958435_3359641442_z.jpg",  # 000000025386
            "http://farm4.staticflickr.com/3317/3427794620_9db24fe462_z.jpg",  # 000000347693
            "http://farm6.staticflickr.com/5143/5589997131_22f51b308c_z.jpg",  # 000000058029
            "http://farm5.staticflickr.com/4061/4376326145_7ef66603e3_z.jpg",  # 000000389933
            "http://farm3.staticflickr.com/2028/2188480725_5fbf27a5b3_z.jpg",  # 000000311789
            "http://farm1.staticflickr.com/172/421715600_666b0f6a2b_z.jpg",  # 000000506004
            "http://farm9.staticflickr.com/8331/8100320407_6044d243a5_z.jpg",  # 000000076648
            "http://farm4.staticflickr.com/3236/2487649513_1ef6a6d5c9_z.jpg",  # 000000201646
            "http://farm4.staticflickr.com/3094/2684280938_a5b59c0fac_z.jpg",  # 000000447187
            "http://farm1.staticflickr.com/42/100911501_005e4d3aa8_z.jpg",  # 000000126107
            "http://farm1.staticflickr.com/56/147795701_40d7bc8331_z.jpg",  # 000000505942
            "http://farm5.staticflickr.com/4103/5074895283_71a73d77e5_z.jpg",  # 000000360951
            "http://farm1.staticflickr.com/160/404335548_3bdc1f2ed9_z.jpg",  # 000000489764
            "http://farm9.staticflickr.com/8446/7857456044_401a257790_z.jpg",  # 000000407574
        ]

        root_mscoco = "datasets"
        os.makedirs(root_mscoco, exist_ok=True)
        for idx, img_url in enumerate(list_url):
            response = requests.get(img_url, headers=headers)
            with open(f"{root_mscoco}/{idx:03d}.jpg", "wb") as f:
                f.write(response.content)
            time.sleep(0.5)

        # Collect patches

        class CustomDatasetFolder(VisionDataset):
            def __init__(self, root, transform=None):
                super(CustomDatasetFolder, self).__init__(root)
                self.transform = transform
                samples = glob.glob(f"{root}/*.jpg")

                self.samples = samples

            def __getitem__(self, index):
                sample = self._loader(self.samples[index])
                if self.transform is not None:
                    sample = self.transform(sample)
                return sample

            def __len__(self):
                return len(self.samples)

            @staticmethod
            def _loader(path):
                return Image.open(path).convert("RGB")

        img_dataset = CustomDatasetFolder(
            root_mscoco,
            transforms.Compose(
                [
                    transforms.RandomResizedCrop((640, 640)),
                    transforms.AutoAugment(),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            ),
        )
        img_loader = torch.utils.data.DataLoader(img_dataset, batch_size=1, shuffle=True)

        candidates_list = []
        max_imgs = 25
        img_count = 0
        for x_i in iter(img_loader):
            img_count = img_count + 1
            if img_count == max_imgs:
                break

            candidates, _ = collect_patches_from_images(object_detector, x_i.to(object_detector.device))
            candidates_list = candidates_list + candidates[0]

        attack = SNAL(
            object_detector,
            eps=16.0 / 255.0,
            max_iter=5,
            num_grid=10,
            candidates=candidates_list,
            collector=collect_patches_from_images,
        )

        x_adv = attack.generate(x=x)
        assert x.shape == x_adv.shape
        assert np.min(x_adv) >= 0.0
        assert np.max(x_adv) <= 1.0

        y_pred_adv = object_detector.predict(x=x_adv)
        assert y_pred_adv[0]["boxes"].shape[0] == 25200

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
def test_check_params(art_warning, get_pytorch_detector_yolo):
    try:
        import torch
        import requests

        object_detector = get_pytorch_detector_yolo
        object_detector.set_params(input_shape=(3, 640, 640))

        def dummy_func(model, imags):
            candidates_patch = []
            candidates_position = []
            return candidates_patch, candidates_position

        dummy_list = [[], []]

        with pytest.raises(ValueError):
            _ = SNAL(
                estimator=object_detector,
                eps=-1.0,
                max_iter=5,
                num_grid=10,
                candidates=dummy_list,
                collector=dummy_func,
            )
        with pytest.raises(ValueError):
            _ = SNAL(
                estimator=object_detector, eps=2.0, max_iter=5, num_grid=10, candidates=dummy_list, collector=dummy_func
            )
        with pytest.raises(TypeError):
            _ = SNAL(
                estimator=object_detector,
                eps=8 / 255.0,
                max_iter=1.0,
                num_grid=10,
                candidates=dummy_list,
                collector=dummy_func,
            )
        with pytest.raises(ValueError):
            _ = SNAL(
                estimator=object_detector,
                eps=8 / 255.0,
                max_iter=0,
                num_grid=10,
                candidates=dummy_list,
                collector=dummy_func,
            )
        with pytest.raises(TypeError):
            _ = SNAL(
                estimator=object_detector,
                eps=8 / 255.0,
                max_iter=5,
                num_grid=1.0,
                candidates=dummy_list,
                collector=dummy_func,
            )
        with pytest.raises(ValueError):
            _ = SNAL(
                estimator=object_detector,
                eps=8 / 255.0,
                max_iter=5,
                num_grid=0,
                candidates=dummy_list,
                collector=dummy_func,
            )
        with pytest.raises(TypeError):
            _ = SNAL(
                estimator=object_detector, eps=8 / 255.0, max_iter=5, num_grid=10, candidates=1.0, collector=dummy_func
            )
        with pytest.raises(ValueError):
            _ = SNAL(
                estimator=object_detector, eps=8 / 255.0, max_iter=5, num_grid=10, candidates=[], collector=dummy_func
            )

    except ARTTestException as e:
        art_warning(e)
