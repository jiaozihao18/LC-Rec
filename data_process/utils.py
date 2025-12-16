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


def get_res_batch(model_name, prompt_list, max_tokens, api_info, use_json_format=False, system_message=None):
    """
    批量调用大模型API
    
    Args:
        model_name: 模型名称
        prompt_list: prompt列表
        max_tokens: 最大token数
        api_info: API配置信息，包含：
            - api_key: API密钥
            - base_url: (可选) API基础URL，默认为dashscope北京地域
            - region: (可选) 地域，'beijing' 或 'singapore'
        use_json_format: 是否使用JSON格式输出
        system_message: (可选) 系统消息，如果为None且use_json_format=True，会自动添加JSON提示
    
    Returns:
        output_list: 输出列表
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
    
    # 创建客户端
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    # 准备系统消息（必须包含"JSON"关键词才能使用json_object格式）
    if system_message is None:
        system_message = "Please respond in JSON format. Your response must be valid JSON." if use_json_format else "You are a helpful assistant."
    elif use_json_format and "json" not in system_message.lower():
        system_message += " Please respond in JSON format."
    
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
        completion_params = {
            "model": model_name,
            "messages": messages,
            "temperature": 0.4,
            "max_tokens": max_tokens,
        }
        if use_json_format:
            completion_params["response_format"] = {"type": "json_object"}
        
        completion = client.chat.completions.create(**completion_params)
        output = completion.choices[0].message.content.strip()
        output_list.append(output)
    
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


unified_user_analysis_prompt = """You are analyzing a user's preferences and intentions based on their purchase history and interactions with items in the {dataset_full_name} category.

User's Purchase History (in chronological order):
{history_items}

Target Item for Analysis:
- Title: {item_title}
- Description: {item_description}
- Brand: {item_brand}
- Categories: {item_categories}
{review_section}

Please analyze and extract the following information:

1. **User Preferences** (based on purchase history):
   - General preference: A brief third-person summary of the user's overall preferences
   - Long-term preference: Inherent characteristics reflected across all purchases
   - Short-term preference: Recent preferences reflected in recent purchases

2. **User Intentions** (based on the target item and review):
   - User-related intention: The user's personal preferences and needs inferred from their interaction with this item
   - Item-related intention: The characteristics and features of this item that attracted the user

Please provide your analysis in the following JSON format:
{{
  "general_preference": "...",
  "long_term_preference": "...",
  "short_term_preference": "...",
  "user_related_intention": "...",
  "item_related_intention": "..."
}}

Important notes:
- All preferences should be in third person, concise, and general (avoid listing specific items)
- User-related intention should not include the item title
- If review is not available, base intentions only on item features
- Be specific but concise in your analysis"""


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


