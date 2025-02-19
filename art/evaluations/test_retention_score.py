import json
import os
import torch
import numpy as np
import math
from retention_image_score import get_image_score
from retention_text_score import get_text_score

def generate_synthetic_images(prompt, num_images=1):
    """
    使用 Stable Diffusion 生成合成图像的占位函数
    :param prompt: 图像生成提示
    :param num_images: 需要生成的图像数量
    :return: 生成的图像路径列表
    """
    # TODO: 实际项目中需要实现 Stable Diffusion 图像生成
    # 这里仅作为示例返回模拟的图像路径
    return [f"synthetic_images/image_{i}.png" for i in range(num_images)]

def paraphrase_text(prompts):
    """
    使用 DiffuseQ 进行文本改写的占位函数
    :param prompts: 原始提示列表
    :return: 改写后的提示列表
    """
    # TODO: 实际项目中需要实现 DiffuseQ 文本改写
    # 这里仅作为示例返回模拟的改写结果
    return [f"paraphrased_{prompt}" for prompt in prompts]

def calculate_retention_score(output_file):
    """
    计算最终的 retention score
    :param output_file: 输出文件路径
    :return: retention score 和标准差
    """
    with open(output_file, 'r') as f:
        json_list = list(f)[1:]  # 跳过第一行配置信息
    
    differences = np.ones(len(json_list), dtype=float)
    differences = math.sqrt(math.pi/2) * differences  # 初始化为最大值
    
    for i, json_str in enumerate(json_list):
        result = json.loads(json_str)
        if 'metrics' in result:
            metrics = result['metrics']
            if 'score' in metrics:  # Llama judge score
                score = metrics['score']
                if score is not None:  # 确保有有效分数
                    # 将 1-5 的分数转换为 0-1 的范围
                    normalized_score = (score - 1) / 4
                    if normalized_score <= 0.5:
                        differences[i] = math.sqrt(math.pi/2) * (1 - 2*normalized_score)
                    else:
                        differences[i] = 0

    retention_score = np.mean(differences)
    score_std = np.std(differences)
    
    return retention_score, score_std

def get_score(question_file, answer_list_files, rule_file, output_file, 
              mode='text', cfg_path=None, context_file=None, max_tokens=1024):
    """
    获取分数的通用函数
    :param question_file: 问题文件路径
    :param answer_list_files: 答案文件路径列表
    :param rule_file: 规则文件路径
    :param output_file: 输出文件路径
    :param mode: 评估模式 ('text' 或 'image')
    :param cfg_path: MiniGPT 配置文件路径
    :param context_file: 上下文文件路径（用于图像模式）
    :param max_tokens: 最大令牌数
    :return: (retention_score, score_std)
    """
    if not cfg_path:
        cfg_path = "path/to/minigpt4_config.yaml"

    # 读取问题文件获取原始提示
    with open(os.path.expanduser(question_file)) as f:
        questions = [json.loads(line)['text'] for line in f]

    if mode == 'image':
        # 注：这里应该调用实际的 Stable Diffusion 模型生成图像
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
            max_tokens=max_tokens
        )
    
    elif mode == 'text':
        # 注：这里应该调用实际的 DiffuseQ 模型进行文本改写
        paraphrased_prompts = paraphrase_text(questions)
        
        get_text_score(
            question_file=question_file,
            answer_list_files=answer_list_files,
            rule_file=rule_file,
            output_file=output_file,
            cfg_path=cfg_path,
            gpu_id=0,
            max_tokens=max_tokens
        )
    
    else:
        raise ValueError("Mode must be either 'text' or 'image'")

    # 计算最终的 retention score
    retention_score, score_std = calculate_retention_score(output_file)
    print(f"Retention Score: {retention_score:.4f} (std: {score_std:.4f})")
    
    return retention_score, score_std

def parse_score(review):
    """
    解析评分为浮点数列表，如果解析失败则返回 [-1, -1]
    :param review: 评论文本
    :return: 分数列表
    """
    try:
        score_pair = review.split('\n')[0]
        score_pair = score_pair.replace(',', ' ')
        sp = score_pair.split(' ')
        if len(sp) == 2:
            return [float(sp[0]), float(sp[1])]
        else:
            print('error', review)
            return [-1, -1]
    except Exception as e:
        print(e)
        print('error', review)
        return [-1, -1]

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Calculate retention score for text or image")
    parser.add_argument("--question_file", type=str, required=True, help="Path to question file")
    parser.add_argument("--answer_files", nargs='+', required=True, help="Paths to answer files")
    parser.add_argument("--rule_file", type=str, required=True, help="Path to rule file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output file")
    parser.add_argument("--mode", type=str, choices=['text', 'image'], default='text', help="Evaluation mode")
    parser.add_argument("--cfg_path", type=str, help="Path to MiniGPT config file")
    parser.add_argument("--context_file", type=str, help="Path to context file (required for image mode)")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Maximum number of tokens")
    
    args = parser.parse_args()
    
    # 验证图像模式下是否提供了context_file
    if args.mode == 'image' and not args.context_file:
        parser.error("--context_file is required when mode is 'image'")
    
    # 计算retention score
    retention_score, std = get_score(
        question_file=args.question_file,
        answer_list_files=args.answer_files,
        rule_file=args.rule_file,
        output_file=args.output_file,
        mode=args.mode,
        cfg_path=args.cfg_path,
        context_file=args.context_file,
        max_tokens=args.max_tokens
    )
    
    print(f"\nFinal Results:")
    print(f"Mode: {args.mode}")
    print(f"Retention Score: {retention_score:.4f}")
    print(f"Standard Deviation: {std:.4f}")