# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2025
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

import json
import os

import numpy as np
import math

import pytest

from art.evaluations import get_retention_score_image, get_retention_score_text



def generate_synthetic_images(prompt, num_images=1):
    """
    Placeholder function for generating synthetic images using Stable Diffusion

    :param prompt: Image generation prompt
    :param num_images: Number of images to generate
    :return: List of generated image paths
    """
    # TODO: Implement actual Stable Diffusion image generation in production
    # This is just a placeholder returning mock image paths
    return [f"synthetic_images/image_{i}.png" for i in range(num_images)]


def paraphrase_text(prompts):
    """
    Placeholder function for text paraphrasing using DiffuseQ

    :param prompts: List of original prompts
    :return: List of paraphrased prompts
    """
    # TODO: Implement actual DiffuseQ text paraphrasing in production
    # This is just a placeholder returning mock paraphrased text
    return [f"paraphrased_{prompt}" for prompt in prompts]


def calculate_retention_score(output_file):
    """
    Calculate the final retention score

    :param output_file: Path to output file
    :return: retention score and standard deviation
    """
    with open(output_file, "r") as f:
        json_list = list(f)[1:]  # Skip first line containing config info

    differences = np.ones(len(json_list), dtype=float)
    differences = math.sqrt(math.pi / 2) * differences  # Initialize to maximum value

    for i, json_str in enumerate(json_list):
        result = json.loads(json_str)
        if "metrics" in result:
            metrics = result["metrics"]
            if "score" in metrics:  # Llama judge score
                score = metrics["score"]
                if score is not None:  # Ensure valid score
                    # Convert score from 1-5 range to 0-1 range
                    normalized_score = (score - 1) / 4
                    if normalized_score <= 0.5:
                        differences[i] = math.sqrt(math.pi / 2) * (1 - 2 * normalized_score)
                    else:
                        differences[i] = 0

    retention_score = np.mean(differences)
    score_std = np.std(differences)

    return retention_score, score_std


def test_get_score_text(
):
    """
    General function for getting scores

    :param question_file: Path to question file
    :param answer_list_files: List of paths to answer files
    :param rule_file: Path to rule file
    :param output_file: Path to output file
    :param mode: Evaluation mode ('text' or 'image')
    :param cfg_path: Path to MiniGPT config file
    :param context_file: Path to context file (for image mode)
    :param max_tokens: Maximum number of tokens
    :return: (retention_score, score_std)
    """
    question_file = None
    answer_list_files = None
    rule_file = None
    output_file = None
    cfg_path = None
    max_tokens = 1024

    if not cfg_path:
        cfg_path = "path/to/minigpt4_config.yaml"

    # Read original prompts from question file
    with open(os.path.expanduser(question_file)) as f:
        questions = [json.loads(line)["text"] for line in f]

    # Note: Should call actual DiffuseQ model for text paraphrasing
    paraphrased_prompts = paraphrase_text(questions)

    get_retention_score_text(
        question_file=question_file,
        answer_list_files=answer_list_files,
        rule_file=rule_file,
        output_file=output_file,
        cfg_path=cfg_path,
        gpu_id=0,
        max_tokens=max_tokens,
    )

    # Calculate final retention score
    retention_score, score_std = calculate_retention_score(output_file)
    print(f"Retention Score: {retention_score:.4f} (std: {score_std:.4f})")

    return retention_score, score_std



def test_get_score_image(
):
    """
    General function for getting scores

    :param question_file: Path to question file
    :param answer_list_files: List of paths to answer files
    :param rule_file: Path to rule file
    :param output_file: Path to output file
    :param mode: Evaluation mode ('text' or 'image')
    :param cfg_path: Path to MiniGPT config file
    :param context_file: Path to context file (for image mode)
    :param max_tokens: Maximum number of tokens
    :return: (retention_score, score_std)
    """
    question_file = None
    answer_list_files = None
    rule_file = None
    output_file = None
    cfg_path = None
    context_file = None
    max_tokens = 1024

    if not cfg_path:
        cfg_path = "path/to/minigpt4_config.yaml"

    # Read original prompts from question file
    with open(os.path.expanduser(question_file)) as f:
        questions = [json.loads(line)["text"] for line in f]

    # Note: Should call actual Stable Diffusion model for image generation
    synthetic_images = []
    for prompt in questions:
        images = generate_synthetic_images(prompt)
        synthetic_images.extend(images)

    get_retention_score_image(
        question_file=question_file,
        answer_list_files=answer_list_files,
        rule_file=rule_file,
        output_file=output_file,
        context_file=context_file,
        cfg_path=cfg_path,
        max_tokens=max_tokens,
    )

    # Calculate final retention score
    retention_score, score_std = calculate_retention_score(output_file)
    print(f"Retention Score: {retention_score:.4f} (std: {score_std:.4f})")

    return retention_score, score_std
