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

from art.defences.evaluation.evasion import great_score
import numpy as np
from scipy import stats
from robustbench.data import load_cifar10

def load_gan_generated_data(n_examples=500):
    """
    This function is a placeholder for loading GAN-generated data based on CIFAR10.
    In actual implementation, this should be replaced with code to generate or load
    GAN-generated samples.

    Args:
    n_examples (int): Number of examples to generate/load.

    Returns:
    tuple: (x_test, y_test) where x_test is the generated images and y_test is the corresponding labels.
    """
    # NOTE: In actual implementation, replace this with GAN data generation or loading
    # For example:
    # x_test, y_test = generate_gan_samples(n_examples)
    # or
    # x_test, y_test = load_gan_samples_from_file('path/to/gan_samples.npz')

    print("NOTE: Currently using CIFAR10 data as a placeholder.")
    print("In actual implementation, replace this function with GAN-generated data loading.")
    
    x_test, y_test = load_cifar10(n_examples=n_examples)
    
    # Convert to numpy arrays if they aren't already
    x_test = x_test.numpy() if hasattr(x_test, 'numpy') else x_test
    y_test = y_test.numpy() if hasattr(y_test, 'numpy') else y_test

    return x_test, y_test

def test_great_score():
    try:
        # Load GAN-generated data (currently a placeholder using CIFAR10)
        x_test, y_test = load_gan_generated_data(n_examples=500)

        # List of models to test
        model_list = [
            'Rebuffi2021Fixing_70_16_cutmix_extra',
            'Gowal2020Uncovering_extra',
            'Rebuffi2021Fixing_70_16_cutmix_ddpm',
            'Rebuffi2021Fixing_28_10_cutmix_ddpm',
            'Augustin2020Adversarial_34_10_extra',
            'Sehwag2021Proxy',
            'Augustin2020Adversarial_34_10',
            'Rade2021Helper_R18_ddpm',
            'Rebuffi2021Fixing_R18_cutmix_ddpm',
            'Gowal2020Uncovering',
            'Sehwag2021Proxy_R18',
            'Wu2020Adversarial',
            'Augustin2020Adversarial',
            'Engstrom2019Robustness',
            'Rice2020Overfitting',
            'Rony2019Decoupling',
            'Ding2020MMA'
        ]

        results = []

        for model_name in model_list:
            score, accuracy = great_score(x_test, y_test, model_name)
            results.append(score)
            print(f"Model: {model_name}")
            print(f"- GREAT Score: {score:.4f}")
            print(f"- Accuracy: {accuracy:.4f}")
            print()

        # Predefined accuracy values
        accuracies = [87.20, 85.60, 90.60, 90.00, 86.20, 89.20, 86.40, 86.60, 87.60, 86.40, 88.60, 84.60, 85.20, 82.20, 81.80, 79.20, 77.60]
        
        # Predefined ranking
        rankings = [17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

        print("Correlation with accuracies:")
        print(stats.spearmanr(results, accuracies))
        
        print("\nCorrelation with rankings:")
        print(stats.spearmanr(results, rankings))

        return results

    except Exception as e:
        print(f"An error occurred during the execution of test_great_score: {str(e)}")
        return None

if __name__ == "__main__":
    test_great_score()