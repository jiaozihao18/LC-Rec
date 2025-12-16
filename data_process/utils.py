import html
import json
import os
import pickle
import re
import time

import torch
# import gensim
from transformers import AutoModel, AutoTokenizer
import collections
from openai import OpenAI, APIError, APIConnectionError, RateLimitError, AuthenticationError


def _is_vllm_service(base_url):
    """
    检测是否为vLLM服务（本地部署）
    
    Args:
        base_url: API基础URL
    
    Returns:
        bool: 如果是vLLM服务返回True，否则返回False
    """
    if base_url is None:
        return False
    vllm_indicators = ['localhost', '127.0.0.1', '0.0.0.0']
    base_url_lower = base_url.lower()
    return any(indicator in base_url_lower for indicator in vllm_indicators)


def get_res_batch(model_name, prompt_list, api_info, response_format=None, system_message=None, use_json_object=False, guided_decoding_backend="outlines"):
    """
    批量调用大模型API，支持JSON Schema和JSON Object结构化输出
    支持OpenAI兼容API和vLLM本地部署的结构化输出
    
    Args:
        model_name: 模型名称
        prompt_list: prompt列表
        api_info: API配置信息，包含：
            - api_key: API密钥
            - base_url: (可选) API基础URL，默认为dashscope北京地域
                - 如果是vLLM服务，通常为 "http://localhost:8000/v1" 或 "http://0.0.0.0:8000/v1"
            - region: (可选) 地域，'beijing' 或 'singapore'
            - use_vllm: (可选) 显式指定是否使用vLLM，如果不指定则根据base_url自动检测
        response_format: (可选) Pydantic模型类，用于JSON Schema结构化输出。如果提供，将使用parse方法或guided_json
        system_message: (可选) 系统消息，如果为None，使用默认消息
        use_json_object: (可选) 是否使用JSON Object模式，如果为True，将返回JSON字符串
        guided_decoding_backend: (可选) vLLM引导解码后端，可选值: "outlines", "lm-format-enforcer", "xgrammar"，默认为"outlines"
    
    Returns:
        output_list: 输出列表。如果使用response_format（JSON Schema），返回解析后的Pydantic对象列表；
                     如果使用use_json_object，返回JSON字符串列表；否则返回普通文本字符串列表
    """
    # 获取API配置
    api_key = api_info.get("api_key") or api_info.get("api_key_list", [""])[0]
    if not api_key:
        raise ValueError("api_info must contain 'api_key'")
    
    # 确定base_url
    base_url = api_info.get("base_url")
    if base_url is None:
        region = api_info.get("region", "beijing")
        base_url = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1" if region == "singapore" else "https://dashscope.aliyuncs.com/compatible-mode/v1"
    
    # 检测是否为vLLM服务
    use_vllm = api_info.get("use_vllm", None)
    if use_vllm is None:
        use_vllm = _is_vllm_service(base_url)
    
    # 创建客户端
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    # 准备系统消息
    if system_message is None:
        system_message = "You are a helpful assistant."
    
    # 构建messages列表
    messages_list = [
        [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
        for prompt in prompt_list
    ]
    
    # 调用API
    output_list = []
    for messages in messages_list:
        try:
            if response_format:
                # 使用JSON Schema结构化输出
                if use_vllm:
                    # vLLM使用guided_json通过extra_body传递
                    json_schema = response_format.model_json_schema()
                    
                    # 方法1: 使用beta.parse方法（推荐，自动解析）
                    try:
                        completion = client.beta.chat.completions.parse(
                            model=model_name,
                            messages=messages,
                            temperature=0.4,
                            response_format=response_format,
                            extra_body={"guided_decoding_backend": guided_decoding_backend}
                        )
                        # 获取解析后的Pydantic对象
                        parsed = completion.choices[0].message.parsed
                        output_list.append(parsed)
                    except Exception as parse_error:
                        # 如果beta.parse方法失败，回退到使用create方法配合guided_json
                        print(f"Warning: beta.parse failed, using guided_json fallback: {parse_error}")
                        completion = client.chat.completions.create(
                            model=model_name,
                            messages=messages,
                            temperature=0.4,
                            extra_body={
                                "guided_json": json_schema,
                                "guided_decoding_backend": guided_decoding_backend
                            }
                        )
                        # 手动解析JSON字符串
                        json_str = completion.choices[0].message.content.strip()
                        try:
                            json_data = json.loads(json_str)
                            parsed = response_format(**json_data)
                            output_list.append(parsed)
                        except (json.JSONDecodeError, Exception) as e:
                            print(f"Failed to parse JSON response: {e}, content: {json_str}")
                            output_list.append(None)
                else:
                    # OpenAI兼容API使用parse方法
                    completion = client.chat.completions.parse(
                        model=model_name,
                        messages=messages,
                        temperature=0.4,
                        response_format=response_format
                    )
                    # 获取解析后的Pydantic对象
                    parsed = completion.choices[0].message.parsed
                    output_list.append(parsed)
            elif use_json_object:
                # 使用JSON Object模式
                if use_vllm:
                    # vLLM可以使用guided_json配合简单的JSON schema
                    # 或者直接使用response_format
                    completion = client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        temperature=0.4,
                        response_format={"type": "json_object"},
                        extra_body={"guided_decoding_backend": guided_decoding_backend} if guided_decoding_backend else {}
                    )
                else:
                    completion = client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        temperature=0.4,
                        response_format={"type": "json_object"}
                    )
                output = completion.choices[0].message.content.strip()
                output_list.append(output)
            else:
                # 普通文本输出
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.4,
                    max_tokens=1024
                )
                output = completion.choices[0].message.content.strip()
                output_list.append(output)
        except Exception as e:
            print(f"API call failed: {e}")
            # 如果使用结构化输出但失败，返回None
            output_list.append(None)
    
    return output_list




def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def set_device(gpu_id):
    if gpu_id == -1:
        return torch.device('cpu')
    else:
        return torch.device(
            'cuda:' + str(gpu_id) if torch.cuda.is_available() else 'cpu')

def load_plm(model_path='bert-base-uncased'):

    tokenizer = AutoTokenizer.from_pretrained(model_path,)

    print("Load Model:", model_path)

    model = AutoModel.from_pretrained(model_path,low_cpu_mem_usage=True,)
    return tokenizer, model

def load_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data

def clean_text(raw_text):
    if isinstance(raw_text, list):
        new_raw_text=[]
        for raw in raw_text:
            raw = html.unescape(raw)
            raw = re.sub(r'</?\w+[^>]*>', '', raw)
            raw = re.sub(r'["\n\r]*', '', raw)
            new_raw_text.append(raw.strip())
        cleaned_text = ' '.join(new_raw_text)
    else:
        if isinstance(raw_text, dict):
            cleaned_text = str(raw_text)[1:-1].strip()
        else:
            cleaned_text = raw_text.strip()
        cleaned_text = html.unescape(cleaned_text)
        cleaned_text = re.sub(r'</?\w+[^>]*>', '', cleaned_text)
        cleaned_text = re.sub(r'["\n\r]*', '', cleaned_text)
    index = -1
    while -index < len(cleaned_text) and cleaned_text[index] == '.':
        index -= 1
    index += 1
    if index == 0:
        cleaned_text = cleaned_text + '.'
    else:
        cleaned_text = cleaned_text[:index] + '.'
    if len(cleaned_text) >= 2000:
        cleaned_text = ''
    return cleaned_text

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def make_inters_in_order(inters):
    user2inters, new_inters = collections.defaultdict(list), list()
    for inter in inters:
        user, item, rating, timestamp = inter
        user2inters[user].append((user, item, rating, timestamp))
    for user in user2inters:
        user_inters = user2inters[user]
        user_inters.sort(key=lambda d: d[3])
        for inter in user_inters:
            new_inters.append(inter)
    return new_inters

def write_json_file(dic, file):
    print('Writing json file: ',file)
    with open(file, 'w') as fp:
        json.dump(dic, fp, indent=4)

def write_remap_index(unit2index, file):
    print('Writing remap file: ',file)
    with open(file, 'w') as fp:
        for unit in unit2index:
            fp.write(unit + '\t' + str(unit2index[unit]) + '\n')


unified_user_analysis_prompt = """Analyze user preferences and intentions based on purchase history and item interaction in {dataset_full_name} category.

Purchase History:
{history_items}

Target Item:
Title: {item_title}
Description: {item_description}
{review_section}

Extract:
1. User Preferences (from purchase history, third person):
   - General: Overall preferences summary
   - Long-term: Characteristics across all purchases
   - Short-term: Recent purchase patterns

2. User-Related Intention (first person, user's needs/motivations):
   - What the user wants or needs, not the item itself

3. Item Characteristics (third person, objective item features):
   - Item's attributes, features, and what makes it attractive

Output JSON format:
{{
  "general_preference": "...",
  "long_term_preference": "...",
  "short_term_preference": "...",
  "user_related_intention": "...",
  "item_related_intention": "..."
}}

Notes:
- Preferences: third person, concise, general (no specific items)
- User-related intention: first person, user's needs (exclude item title)
- Item-related intention: third person, objective item characteristics"""


# remove 'Magazine', 'Gift', 'Music', 'Kindle'
amazon18_dataset_list = [
    'Appliances', 'Beauty',
    'Fashion', 'Software', 'Luxury', 'Scientific',  'Pantry',
    'Instruments', 'Arts', 'Games', 'Office', 'Garden',
    'Food', 'Cell', 'CDs', 'Automotive', 'Toys',
    'Pet', 'Tools', 'Kindle', 'Sports', 'Movies',
    'Electronics', 'Home', 'Clothing', 'Books'
]

amazon18_dataset2fullname = {
    'Beauty': 'All_Beauty',
    'Fashion': 'AMAZON_FASHION',
    'Appliances': 'Appliances',
    'Arts': 'Arts_Crafts_and_Sewing',
    'Automotive': 'Automotive',
    'Books': 'Books',
    'CDs': 'CDs_and_Vinyl',
    'Cell': 'Cell_Phones_and_Accessories',
    'Clothing': 'Clothing_Shoes_and_Jewelry',
    'Music': 'Digital_Music',
    'Electronics': 'Electronics',
    'Gift': 'Gift_Cards',
    'Food': 'Grocery_and_Gourmet_Food',
    'Home': 'Home_and_Kitchen',
    'Scientific': 'Industrial_and_Scientific',
    'Kindle': 'Kindle_Store',
    'Luxury': 'Luxury_Beauty',
    'Magazine': 'Magazine_Subscriptions',
    'Movies': 'Movies_and_TV',
    'Instruments': 'Musical_Instruments',
    'Office': 'Office_Products',
    'Garden': 'Patio_Lawn_and_Garden',
    'Pet': 'Pet_Supplies',
    'Pantry': 'Prime_Pantry',
    'Software': 'Software',
    'Sports': 'Sports_and_Outdoors',
    'Tools': 'Tools_and_Home_Improvement',
    'Toys': 'Toys_and_Games',
    'Games': 'Video_Games'
}

amazon14_dataset_list = [
    'Beauty','Toys','Sports'
]

amazon14_dataset2fullname = {
    'Beauty': 'Beauty',
    'Sports': 'Sports_and_Outdoors',
    'Toys': 'Toys_and_Games',
}

# c1. c2. c3. c4.
amazon_text_feature1 = ['title', 'category', 'brand']

# re-order
amazon_text_feature1_ro1 = ['brand', 'main_cat', 'category', 'title']

# remove
amazon_text_feature1_re1 = ['title']

amazon_text_feature2 = ['title']

amazon_text_feature3 = ['description']

amazon_text_feature4 = ['description', 'main_cat', 'category', 'brand']

amazon_text_feature5 = ['title', 'description']


