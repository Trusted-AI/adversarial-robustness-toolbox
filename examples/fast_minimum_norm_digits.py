# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2026
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
"""Evaluate FMN on held-out real handwritten digits, on CPU with no dataset download.

Run from the repository root after installing ART, scikit-learn and PyTorch::

    python examples/fast_minimum_norm_digits.py --samples 100 --output-dir /tmp/fmn-digits

By default, a small tanh MLP is fitted on a fixed training split. The attack uses the first 100 correctly
classified held-out samples and evaluates L2/Linf in both targeted and untargeted modes. Use --model linear
for logistic regression. The output folder contains results.json (metrics) and samples.npz (exact samples
and split indices). Unsuccessful searches count as failures and are excluded from successful-norm medians.

The optional --reference-source argument points to the authors' original src/attacks/fmn.py,
which requires foolbox and eagerpy. It runs that code on the identical model and selected samples.
No reference source is downloaded or executed automatically.

Paper: Pintor, Roli, Brendel and Biggio, "Fast Minimum-norm Adversarial Attacks through Adaptive Norm
Constraints", NeurIPS 2021. https://arxiv.org/abs/2102.12827
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import sklearn
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import torch

from art.attacks.evasion import FastMinimumNorm
from art.estimators.classification import PyTorchClassifier


def evaluate(
    samples: int, steps: int, output_dir: Path, reference_source: Path | None = None, model_kind: str = "mlp"
) -> dict:
    """Fit on the training split, attack a fixed held-out subset, and save results plus exact input arrays."""
    if samples <= 0 or steps <= 0:
        raise ValueError("samples and steps must be positive.")
    torch.set_num_threads(1)
    torch.manual_seed(42)
    digits = load_digits()
    x = (digits.data / 16).astype(np.float32)
    y = digits.target
    train_indices, test_indices = train_test_split(np.arange(len(y)), test_size=0.25, random_state=42, stratify=y)
    # Copy trained weights to torch so both attack libraries receive identical logits and autograd derivatives.
    if model_kind == "linear":
        fitted = LogisticRegression(C=1.0, max_iter=1000).fit(x[train_indices], y[train_indices])
        model = torch.nn.Sequential(torch.nn.Linear(64, 10))
        weights, biases = [fitted.coef_.T], [fitted.intercept_]
    elif model_kind == "mlp":
        fitted = MLPClassifier(
            hidden_layer_sizes=(32,), activation="tanh", solver="lbfgs", max_iter=1000, random_state=42
        ).fit(x[train_indices], y[train_indices])
        model = torch.nn.Sequential(torch.nn.Linear(64, 32), torch.nn.Tanh(), torch.nn.Linear(32, 10))
        weights, biases = fitted.coefs_, fitted.intercepts_
    else:
        raise ValueError("model_kind must be 'linear' or 'mlp'.")
    with torch.no_grad():
        for layer, weight, bias in zip(
            (layer for layer in model if isinstance(layer, torch.nn.Linear)), weights, biases
        ):
            layer.weight.copy_(torch.from_numpy(weight.T.astype(np.float32)))
            layer.bias.copy_(torch.from_numpy(bias.astype(np.float32)))
    model.eval()
    classifier = PyTorchClassifier(
        model=model,
        loss=torch.nn.CrossEntropyLoss(),
        input_shape=(64,),
        nb_classes=10,
        clip_values=(0, 1),
        device_type="cpu",
    )
    clean_predictions = classifier.predict(x[test_indices]).argmax(axis=1)
    np.testing.assert_array_equal(clean_predictions, fitted.predict(x[test_indices]))
    correct_indices = test_indices[clean_predictions == y[test_indices]]
    if samples > len(correct_indices):
        raise ValueError(f"Only {len(correct_indices)} correctly classified held-out samples are available.")
    # First N in the seeded split: no selection based on attack outcomes.
    selected = correct_indices[:samples]
    inputs, labels = x[selected], y[selected]
    targets = (labels + 1) % 10
    arrays = {
        "x": inputs,
        "y": labels,
        "targets": targets,
        "dataset_indices": selected,
        "train_indices": train_indices,
        "test_indices": test_indices,
    }
    report = {
        "dataset": "sklearn.datasets.load_digits (real 8x8 handwritten digits)",
        "seed": 42,
        "train_samples": len(train_indices),
        "test_samples": len(test_indices),
        "clean_test_correct": int(np.sum(clean_predictions == y[test_indices])),
        "clean_test_accuracy": float(np.mean(clean_predictions == y[test_indices])),
        "attacked_samples": samples,
        "selection": "first correctly classified samples in seeded held-out split",
        "max_iter": steps,
        "device": "cpu",
        "model": model_kind,
        "model_parameters": fitted.get_params(),
        "versions": {"torch": torch.__version__, "numpy": np.__version__, "scikit_learn": sklearn.__version__},
        "results": [],
    }
    if reference_source is not None:
        import foolbox as fb

        spec = importlib.util.spec_from_file_location("fmn_author_reference", reference_source)
        if spec is None or spec.loader is None:
            raise ValueError("Cannot load the supplied reference source.")
        reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)
        reference_model = fb.PyTorchModel(model, bounds=(0, 1), device="cpu")
        report["reference"] = "authors' original src/attacks/fmn.py; see source path used for this run"
        report["reference_sha256"] = hashlib.sha256(reference_source.read_bytes()).hexdigest()
        report["versions"]["foolbox"] = fb.__version__

    for norm_name, norm in (("l2", 2), ("linf", np.inf)):
        for targeted in (False, True):
            attack_labels = targets if targeted else labels
            adv = FastMinimumNorm(classifier, norm=norm, targeted=targeted, max_iter=steps, verbose=False).generate(
                inputs, attack_labels
            )
            predictions = classifier.predict(adv).argmax(axis=1)
            success = predictions == targets if targeted else predictions != labels
            distances = np.linalg.norm(adv - inputs, ord=norm, axis=1)
            key = norm_name + ("_targeted" if targeted else "_untargeted")
            arrays[key] = adv
            result = {
                "attack": key,
                "successes": int(success.sum()),
                "samples": samples,
                "success_rate": float(success.mean()),
                "median_norm_successful": float(np.median(distances[success])) if success.any() else None,
                "within_bounds": bool(np.all((adv >= 0) & (adv <= 1))),
                "sample_predictions": predictions[:6].tolist(),
                "sample_labels": labels[:6].tolist(),
                "sample_norms": distances[:6].tolist(),
            }
            if reference_source is not None:
                reference_class = reference.L2FMNAttack if norm == 2 else reference.LInfFMNAttack
                reference_attack = reference_class(steps=steps, max_stepsize=1, min_stepsize=0.01, gamma=0.05)
                criterion = fb.criteria.TargetedMisclassification if targeted else fb.criteria.Misclassification
                ref_adv = reference_attack.run(
                    reference_model, torch.from_numpy(inputs), criterion(torch.from_numpy(attack_labels))
                ).numpy()
                ref_predictions = classifier.predict(ref_adv).argmax(axis=1)
                ref_success = ref_predictions == targets if targeted else ref_predictions != labels
                ref_norms = np.linalg.norm(ref_adv - inputs, ord=norm, axis=1)
                result["reference_successes"] = int(ref_success.sum())
                result["reference_median_norm_successful"] = (
                    float(np.median(ref_norms[ref_success])) if ref_success.any() else None
                )
                result["reference_max_absolute_difference"] = float(np.max(abs(adv - ref_adv)))
                result["reference_max_norm_difference"] = float(np.max(abs(distances - ref_norms)))
                result["reference_median_norm_difference"] = float(np.median(abs(distances - ref_norms)))
                result["reference_success_agreement"] = int(np.sum(success == ref_success))
                result["reference_prediction_agreement"] = int(np.sum(predictions == ref_predictions))
                arrays[key + "_reference"] = ref_adv
            report["results"].append(result)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "samples.npz", **arrays)
    (output_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--samples", type=int, default=100, help="Number of correctly classified test samples (default: 100)"
    )
    parser.add_argument("--steps", type=int, default=100, help="FMN iterations per sample (default: 100)")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("/tmp/fmn-digits"), help="Results folder (default: /tmp/fmn-digits)"
    )
    parser.add_argument(
        "--reference-source", type=Path, help="Optional path to the authors' original src/attacks/fmn.py"
    )
    parser.add_argument("--model", choices=("linear", "mlp"), default="mlp", help="Classifier to train (default: mlp)")
    args = parser.parse_args()
    evaluate(args.samples, args.steps, args.output_dir, args.reference_source, args.model)
