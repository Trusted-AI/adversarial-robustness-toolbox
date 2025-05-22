import json
import os
import torch
import numpy as np
import math
from retention_image_score import get_image_score
from retention_text_score import get_text_score


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


def get_score(
    question_file,
    answer_list_files,
    rule_file,
    output_file,
    mode="text",
    cfg_path=None,
    context_file=None,
    max_tokens=1024,
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
    if not cfg_path:
        cfg_path = "path/to/minigpt4_config.yaml"

    # Read original prompts from question file
    with open(os.path.expanduser(question_file)) as f:
        questions = [json.loads(line)["text"] for line in f]

    if mode == "image":
        # Note: Should call actual Stable Diffusion model for image generation
        synthetic_images = []
        for prompt in questions:
            images = generate_synthetic_images(prompt)
            synthetic_images.extend(images)

        get_image_score(
            question_file=question_file,
            answer_list_files=answer_list_files,
            rule_file=rule_file,
            output_file=output_file,
            context_file=context_file,
            cfg_path=cfg_path,
            max_tokens=max_tokens,
        )

    elif mode == "text":
        # Note: Should call actual DiffuseQ model for text paraphrasing
        paraphrased_prompts = paraphrase_text(questions)

        get_text_score(
            question_file=question_file,
            answer_list_files=answer_list_files,
            rule_file=rule_file,
            output_file=output_file,
            cfg_path=cfg_path,
            gpu_id=0,
            max_tokens=max_tokens,
        )

    else:
        raise ValueError("Mode must be either 'text' or 'image'")

    # Calculate final retention score
    retention_score, score_std = calculate_retention_score(output_file)
    print(f"Retention Score: {retention_score:.4f} (std: {score_std:.4f})")

    return retention_score, score_std


def parse_score(review):
    """
    Parse scores into float list, return [-1, -1] if parsing fails
    :param review: Review text
    :return: Score list
    """
    try:
        score_pair = review.split("\n")[0]
        score_pair = score_pair.replace(",", " ")
        sp = score_pair.split(" ")
        if len(sp) == 2:
            return [float(sp[0]), float(sp[1])]
        else:
            print("error", review)
            return [-1, -1]
    except Exception as e:
        print(e)
        print("error", review)
        return [-1, -1]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Calculate retention score for text or image")
    parser.add_argument("--question_file", type=str, required=True, help="Path to question file")
    parser.add_argument("--answer_files", nargs="+", required=True, help="Paths to answer files")
    parser.add_argument("--rule_file", type=str, required=True, help="Path to rule file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output file")
    parser.add_argument("--mode", type=str, choices=["text", "image"], default="text", help="Evaluation mode")
    parser.add_argument("--cfg_path", type=str, help="Path to MiniGPT config file")
    parser.add_argument("--context_file", type=str, help="Path to context file (required for image mode)")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Maximum number of tokens")

    args = parser.parse_args()

    # Validate context_file requirement for image mode
    if args.mode == "image" and not args.context_file:
        parser.error("--context_file is required when mode is 'image'")

    # Calculate retention score
    retention_score, std = get_score(
        question_file=args.question_file,
        answer_list_files=args.answer_files,
        rule_file=args.rule_file,
        output_file=args.output_file,
        mode=args.mode,
        cfg_path=args.cfg_path,
        context_file=args.context_file,
        max_tokens=args.max_tokens,
    )

    print(f"\nFinal Results:")
    print(f"Mode: {args.mode}")
    print(f"Retention Score: {retention_score:.4f}")
    print(f"Standard Deviation: {std:.4f}")
