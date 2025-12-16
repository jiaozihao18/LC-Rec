
import argparse
import os
import json
import re
from utils import get_res_batch, load_json, unified_user_analysis_prompt, amazon18_dataset2fullname, write_json_file


def format_history_items(item_list, item2feature, max_his_len):
    """格式化历史商品信息"""
    formatted_items = []
    for j, item_id in enumerate(item_list):
        item_feat = item2feature[str(item_id)]
        item_str = f"{j+1}. Title: {item_feat['title']}"
        if item_feat.get('description'):
            item_str += f"\n   Description: {item_feat['description'][:200]}..."  # 截断过长的描述
        if item_feat.get('brand'):
            item_str += f"\n   Brand: {item_feat['brand']}"
        formatted_items.append(item_str)
    
    if max_his_len > 0 and len(formatted_items) > max_his_len:
        formatted_items = formatted_items[-max_his_len:]
    
    return "\n".join(formatted_items)


# 默认值常量
DEFAULT_PREFERENCES = {
    "general_preference": "The user enjoys high-quality items.",
    "long_term_preference": "The user prefers high-quality items.",
    "short_term_preference": "The user has been focusing on quality items recently.",
    "user_related_intention": "I enjoy high-quality items.",
    "item_related_intention": "High-quality item with good features."
}


def parse_json_response(response_text):
    """解析JSON格式的响应（由于使用JSON格式输出，解析应该更简单）"""
    response_text = response_text.strip()
    
    # 查找JSON对象
    start_idx = response_text.find('{')
    end_idx = response_text.rfind('}')
    
    if start_idx == -1 or end_idx == -1:
        raise ValueError("No JSON object found in response")
    
    json_str = response_text[start_idx:end_idx+1]
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        # 如果解析失败，尝试修复常见问题
        json_str = re.sub(r'//.*?\n', '', json_str)  # 移除注释
        json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
        try:
            return json.loads(json_str)
        except:
            raise ValueError(f"Failed to parse JSON: {e}")




def generate_user_data(args, inters, item2feature, reviews, api_info, mode='train'):
    """
    生成用户意图数据（train或test模式）
    mode: 'train' 使用倒数第三个item, 'test' 使用最后一个item
    """
    dataset_full_name = amazon18_dataset2fullname[args.dataset]
    dataset_full_name = dataset_full_name.replace("_", " ").lower()
    
    prompt_list = []
    user_data_list = []  # 存储(user, item, history)用于后续处理
    
    for user, item_list in inters.items():
        user = int(user)
        
        # 根据模式选择目标item
        if mode == 'train':
            if len(item_list) < 3:
                continue  # 跳过历史不足的用户
            target_item = int(item_list[-3])
            history = item_list[:-3]
        else:  # test
            if len(item_list) < 1:
                continue
            target_item = int(item_list[-1])
            history = item_list[:-1]
        
        # 获取目标商品特征
        item_feat = item2feature[str(target_item)]
        item_title = item_feat.get('title', '')
        item_description = item_feat.get('description', 'N/A')
        item_brand = item_feat.get('brand', 'N/A')
        item_categories = item_feat.get('categories', 'N/A')
        
        # 获取review（如果存在）
        review = reviews.get(str((user, target_item)), {}).get('review', '')
        review_section = f"\nUser Review: {review}" if review else "\nNote: No user review available for this item."
        
        # 对于preference，使用去掉最后3个的历史
        preference_history = item_list[:-3] if len(item_list) >= 3 else []
        preference_history_items = format_history_items(preference_history, item2feature, args.max_his_len) if preference_history else "No purchase history available."
        
        # 构建prompt
        prompt = unified_user_analysis_prompt.format(
            dataset_full_name=dataset_full_name,
            history_items=preference_history_items,
            item_title=item_title,
            item_description=item_description,
            item_brand=item_brand,
            item_categories=item_categories,
            review_section=review_section
        )
        
        prompt_list.append(prompt)
        user_data_list.append((user, target_item, history))
    
    # 批量调用大模型
    results = {}
    st = 0
    while st < len(prompt_list):
        print(f"Processing {mode} data: {st}/{len(prompt_list)}")
        
        res = get_res_batch(args.model_name, prompt_list[st:st+args.batchsize], args.max_tokens, api_info, use_json_format=True)
        
        for i, answer in enumerate(res):
            user, target_item, history = user_data_list[st + i]
            
            # 解析响应
            if not answer:
                print(f"Empty response for user {user}, using defaults")
                parsed_data = DEFAULT_PREFERENCES.copy()
            else:
                try:
                    parsed_data = parse_json_response(answer)
                    # 确保所有字段都存在，使用默认值填充缺失字段
                    for key, default_value in DEFAULT_PREFERENCES.items():
                        if key not in parsed_data:
                            parsed_data[key] = default_value
                except Exception as e:
                    print(f"Failed to parse JSON for user {user}: {e}")
                    print(f"Response: {answer[:200]}...")
                    parsed_data = DEFAULT_PREFERENCES.copy()
            
            results[user] = {
                "item": target_item,
                "inters": history,
                **{k: parsed_data[k] for k in DEFAULT_PREFERENCES.keys()}
            }
        
        st += args.batchsize
    
    return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Instruments', help='Instruments / Arts / Games')
    parser.add_argument('--root', type=str, default='')
    parser.add_argument('--api_info', type=str, default='./api_info.json')
    parser.add_argument('--model_name', type=str, default='text-davinci-003')
    parser.add_argument('--max_tokens', type=int, default=1024)  # 增加token数以支持JSON输出
    parser.add_argument('--batchsize', type=int, default=16)
    parser.add_argument('--max_his_len', type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    args.root = os.path.join(args.root, args.dataset)

    api_info = load_json(args.api_info)
    # 兼容旧的api_key_list格式
    if "api_key" not in api_info and "api_key_list" in api_info:
        api_key_list = api_info["api_key_list"]
        api_info["api_key"] = api_key_list[0] if isinstance(api_key_list, list) else api_key_list
    if "api_key" not in api_info:
        raise ValueError("api_info must contain 'api_key'")

    # 加载数据
    inter_path = os.path.join(args.root, f'{args.dataset}.inter.json')
    inters = load_json(inter_path)

    item2feature_path = os.path.join(args.root, f'{args.dataset}.item.json')
    item2feature = load_json(item2feature_path)

    reviews_path = os.path.join(args.root, f'{args.dataset}.review.json')
    if os.path.exists(reviews_path):
        reviews = load_json(reviews_path)
    else:
        print(f"Warning: {reviews_path} not found, using empty reviews")
        reviews = {}

    # 生成训练集和测试集数据（每个都包含preference和intention）
    print("Generating train data...")
    train_data = generate_user_data(args, inters, item2feature, reviews, api_info, mode='train')
    
    print("Generating test data...")
    test_data = generate_user_data(args, inters, item2feature, reviews, api_info, mode='test')

    # 构建最终的用户字典
    user_dict = {
        "user_explicit_preference": {},
        "user_vague_intention": {
            "train": {},
            "test": {}
        }
    }

    # 处理训练集数据（preference从train数据中提取，但每个用户只保存一次）
    for user, data in train_data.items():
        user_dict["user_explicit_preference"][user] = [
            data["general_preference"],
            data["long_term_preference"],
            data["short_term_preference"]
        ]
        user_dict["user_vague_intention"]["train"][user] = {
            "item": data["item"],
            "inters": data["inters"],
            "querys": [data["user_related_intention"], data["item_related_intention"]]
        }

    # 处理测试集数据（preference已经在train中设置，这里只设置intention）
    for user, data in test_data.items():
        # 如果用户不在训练集中，也设置preference
        if user not in user_dict["user_explicit_preference"]:
            user_dict["user_explicit_preference"][user] = [
                data["general_preference"],
                data["long_term_preference"],
                data["short_term_preference"]
            ]
        
        user_dict["user_vague_intention"]["test"][user] = {
            "item": data["item"],
            "inters": data["inters"],
            "querys": [data["user_related_intention"], data["item_related_intention"]]
        }

    # 直接输出最终文件
    output_file = os.path.join(args.root, f'{args.dataset}.user.json')
    write_json_file(user_dict, output_file)
    print(f"Successfully generated {output_file}")
