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

# from metric.perspective_api import PerspectiveApiScorer
# from metric.detoxify import DetoxifyScorer
import torch
# from minigpt4.common.config import Config
# from minigpt4.common.registry import registry
# from minigpt4.conversation.conversation import Chat, CONV_VISION
from PIL import Image


# Assume this is a function for image evaluation using the MiniGPT model
def minigpt_image_eval(content):
    """
    Simulates a function for image evaluation using the MiniGPT model

    :param content: Input image-related content
    :return: Evaluation result
    """
    # This should be the actual code calling the MiniGPT model
    # For demonstration purposes, simply return a string
    return "Image evaluation result"


def initialize_minigpt(cfg_path, gpu_id=0):
    """
    Initialize MiniGPT model

    :param cfg_path: Path to configuration file
    :param gpu_id: GPU device ID
    :return: Initialized Chat model
    """
    cfg = Config.fromfile(cfg_path)
    model_config = cfg.model_cfg
    model_config.device_8bit = gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(f"cuda:{gpu_id}")

    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    return Chat(model, vis_processor, device=f"cuda:{gpu_id}")


def get_retention_score_image(question_file, answer_list_files, rule_file, output_file, context_file, cfg_path, max_tokens=1024):
    """
    Get image scores function

    :param question_file: Path to question file
    :param answer_list_files: List of paths to answer files
    :param rule_file: Path to rule file
    :param output_file: Path to output file
    :param context_file: Path to context file (containing image information)
    :param cfg_path: Path to configuration file
    :param max_tokens: Maximum number of tokens
    """
    # Initialize scorers
    api_key = open(".perspective_api_key").read().strip()
    perspective_scorer = PerspectiveApiScorer(api_key)
    detoxify_scorer = DetoxifyScorer()

    # Initialize MiniGPT
    chat = initialize_minigpt(cfg_path)

    f_q = open(os.path.expanduser(question_file))
    f_ans1 = open(os.path.expanduser(answer_list_files[0]))
    f_ans2 = open(os.path.expanduser(answer_list_files[1]))
    rule_dict = json.load(open(os.path.expanduser(rule_file), "r"))

    if os.path.isfile(os.path.expanduser(output_file)):
        cur_reviews = [json.loads(line) for line in open(os.path.expanduser(output_file))]
    else:
        cur_reviews = []

    review_file = open(f"{output_file}", "a")

    context_list = [json.loads(line) for line in open(os.path.expanduser(context_file))]
    image_to_context = {context["image"]: context for context in context_list}

    idx = 0
    with torch.no_grad():
        for ques_js, ans1_js, ans2_js in zip(f_q, f_ans1, f_ans2):
            ques = json.loads(ques_js)
            ans1 = json.loads(ans1_js)
            ans2 = json.loads(ans2_js)

            inst = image_to_context[ques["image"]]
            cap_str = "\n".join(inst["captions"])
            box_str = "\n".join([f'{instance["category"]}: {instance["bbox"]}' for instance in inst["instances"]])

            category = json.loads(ques_js)["category"]
            if category in rule_dict:
                rule = rule_dict[category]
            else:
                assert False, f"Visual QA category not found in rule file: {category}."
            prompt = rule["prompt"]
            role = rule["role"]
            content = (
                f"[Context]\n{cap_str}\n\n{box_str}\n\n"
                f'[Question]\n{ques["text"]}\n\n'
                f'[{role} 1]\n{ans1["text"]}\n\n[End of {role} 1]\n\n'
                f'[{role} 2]\n{ans2["text"]}\n\n[End of {role} 2]\n\n'
                f"[System]\n{prompt}\n\n"
            )
            cur_js = {
                "id": idx + 1,
                "question_id": ques["question_id"],
                "answer1_id": ans1.get("answer_id", ans1["question_id"]),
                "answer2_id": ans2.get("answer_id", ans2["answer_id"]),
                "category": category,
            }
            if idx >= len(cur_reviews):
                # Process image
                img = Image.open(ques["image"]).convert("RGB")
                img = chat.vis_processor(img).unsqueeze(0).to(chat.device)

                # Get MiniGPT response
                chat_state = CONV_VISION.copy()
                img_list = []
                chat.upload_img(img, chat_state, img_list)
                chat.ask(content, chat_state)
                response, _, _ = chat.answer(chat_state, img_list, max_new_tokens=max_tokens)

                # Get evaluation metrics
                metrics = {
                    "detoxify": float_it(detoxify_scorer.get_scores(response)),
                    "perspective_api": float_it(perspective_scorer.get_scores(response)),
                }

                cur_js["content"] = response
                cur_js["metrics"] = metrics
                review_file.write(json.dumps(cur_js) + "\n")
                review_file.flush()
            else:
                print(f"Skipping {idx} as we already have it.")
            idx += 1
            print(idx)
    review_file.close()


def float_it(score):
    """
    Convert scores to float

    :param score: Input score dictionary
    :return:  with float values
    """
    for key in score.keys():
        score[key] = float(score[key])
    return score
