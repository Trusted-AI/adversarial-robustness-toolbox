import json
import os
from metric.perspective_api import PerspectiveApiScorer
from metric.detoxify import DetoxifyScorer
import torch
from minigpt4.common.config import Config
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION
from PIL import Image

# 假设这是调用minigpt模型进行图像评价的函数
def minigpt_image_eval(content):
    """
    模拟minigpt对图像相关内容进行评价的函数
    :param content: 输入的图像相关内容
    :return: 评价结果
    """
    # 这里应该是实际调用minigpt模型的代码
    # 为了示例，简单返回一个字符串
    return "Image evaluation result"

def initialize_minigpt(cfg_path, gpu_id=0):
    """初始化 MiniGPT 模型"""
    cfg = Config.fromfile(cfg_path)
    model_config = cfg.model_cfg
    model_config.device_8bit = gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(f'cuda:{gpu_id}')
    
    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    return Chat(model, vis_processor, device=f'cuda:{gpu_id}')

def get_image_score(question_file, answer_list_files, rule_file, output_file, context_file, cfg_path, max_tokens=1024):
    """
    获取图像分数的函数
    :param question_file: 问题文件路径
    :param answer_list_files: 答案文件路径列表
    :param rule_file: 规则文件路径
    :param output_file: 输出文件路径
    :param context_file: 上下文文件路径（包含图像信息）
    :param cfg_path: 配置文件路径
    :param max_tokens: 最大令牌数
    """
    # 初始化评分器
    api_key = open(".perspective_api_key").read().strip()
    perspective_scorer = PerspectiveApiScorer(api_key)
    detoxify_scorer = DetoxifyScorer()
    
    # 初始化 MiniGPT
    chat = initialize_minigpt(cfg_path)
    
    f_q = open(os.path.expanduser(question_file))
    f_ans1 = open(os.path.expanduser(answer_list_files[0]))
    f_ans2 = open(os.path.expanduser(answer_list_files[1]))
    rule_dict = json.load(open(os.path.expanduser(rule_file), 'r'))

    if os.path.isfile(os.path.expanduser(output_file)):
        cur_reviews = [json.loads(line) for line in open(os.path.expanduser(output_file))]
    else:
        cur_reviews = []

    review_file = open(f'{output_file}', 'a')

    context_list = [json.loads(line) for line in open(os.path.expanduser(context_file))]
    image_to_context = {context['image']: context for context in context_list}

    idx = 0
    with torch.no_grad():
        for ques_js, ans1_js, ans2_js in zip(f_q, f_ans1, f_ans2):
            ques = json.loads(ques_js)
            ans1 = json.loads(ans1_js)
            ans2 = json.loads(ans2_js)

            inst = image_to_context[ques['image']]
            cap_str = '\n'.join(inst['captions'])
            box_str = '\n'.join([f'{instance["category"]}: {instance["bbox"]}' for instance in inst['instances']])

            category = json.loads(ques_js)['category']
            if category in rule_dict:
                rule = rule_dict[category]
            else:
                assert False, f"Visual QA category not found in rule file: {category}."
            prompt = rule['prompt']
            role = rule['role']
            content = (f'[Context]\n{cap_str}\n\n{box_str}\n\n'
                       f'[Question]\n{ques["text"]}\n\n'
                       f'[{role} 1]\n{ans1["text"]}\n\n[End of {role} 1]\n\n'
                       f'[{role} 2]\n{ans2["text"]}\n\n[End of {role} 2]\n\n'
                       f'[System]\n{prompt}\n\n')
            cur_js = {
                'id': idx + 1,
                'question_id': ques['question_id'],
                'answer1_id': ans1.get('answer_id', ans1['question_id']),
                'answer2_id': ans2.get('answer_id', ans2['answer_id']),
                'category': category
            }
            if idx >= len(cur_reviews):
                # 处理图像
                img = Image.open(ques['image']).convert('RGB')
                img = chat.vis_processor(img).unsqueeze(0).to(chat.device)
                
                # 获取 MiniGPT 响应
                chat_state = CONV_VISION.copy()
                img_list = []
                chat.upload_img(img, chat_state, img_list)
                chat.ask(content, chat_state)
                response, _, _ = chat.answer(chat_state, img_list, max_new_tokens=max_tokens)
                
                # 获取评估指标
                metrics = {
                    'detoxify': float_it(detoxify_scorer.get_scores(response)),
                    'perspective_api': float_it(perspective_scorer.get_scores(response))
                }
                
                cur_js['content'] = response
                cur_js['metrics'] = metrics
                review_file.write(json.dumps(cur_js) + '\n')
                review_file.flush()
            else:
                print(f'Skipping {idx} as we already have it.')
            idx += 1
            print(idx)
    review_file.close()

def float_it(score):
    """将评分转换为浮点数"""
    for key in score.keys():
        score[key] = float(score[key])
    return score