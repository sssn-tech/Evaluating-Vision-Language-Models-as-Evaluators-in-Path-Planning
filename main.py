import json
import pandas as pd

# 读取 JSON 数据
with open("PathEval/2D/test_dataset_2D.json", "r", encoding="utf-8") as f:
    data = json.load(f)

rows = []
for item in data:
    row = {
        "id": item["id"],
        "image": item["image"],
        "scenario": item["scenario"],
        "ground_truth": item["ground truth"]
    }
    # 将 path 1 和 path 2 的描述指标合并并加前缀
    for key, value in item["path 1 descriptors"].items():
        row[f"path1_{key}"] = value
    for key, value in item["path 2 descriptors"].items():
        row[f"path2_{key}"] = value
    rows.append(row)

df = pd.DataFrame(rows)
df = df.sample(frac=1).reset_index(drop=True) # 随机打乱
"""
RangeIndex: 1050 entries, 0 to 1049
Data columns (total 18 columns):
 #   Column                   Non-Null Count  Dtype  
---  ------                   --------------  -----  
 0   id                       1050 non-null   int64  
 1   image                    1050 non-null   object 
 2   scenario                 1050 non-null   object 
 3   ground_truth             1050 non-null   object 
 4   path1_Minimum clearance  1050 non-null   float64
 5   path1_Maximum clearance  1050 non-null   float64
 6   path1_Average clearance  1050 non-null   float64
 7   path1_Path length        1050 non-null   float64
 8   path1_Smoothness         1050 non-null   float64
 9   path1_Sharp turns        1050 non-null   int64  
 10  path1_Maximum angle      1050 non-null   float64
 11  path2_Minimum clearance  1050 non-null   float64
 12  path2_Maximum clearance  1050 non-null   float64
 13  path2_Average clearance  1050 non-null   float64
 14  path2_Path length        1050 non-null   float64
 15  path2_Smoothness         1050 non-null   float64
 16  path2_Sharp turns        1050 non-null   int64  
 17  path2_Maximum angle      1050 non-null   float64
dtypes: float64(12), int64(3), object(3)
"""


from openai import OpenAI
import base64
import os

# 从环境变量读取API key
client = OpenAI(
    api_key=os.environ.get("DOUBAO_API_KEY"),  
    base_url="https://ark.cn-beijing.volces.com/api/v3"
)

import base64

def query_LLM(img_path, prompt):
    # 处理图片路径为None的情况
    if img_path is None:
        image_data = None
    else:
        img_path = './PathEval/2D/' + img_path
        with open(img_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

    system_prompt = """
        You are a helpful agent. Your answer must strictly follow the formatting requirements the user claim.
    """

    user_prompt = prompt + """\n\n 
        Your answer must strictly follow the formatting requirements: answer (1 or 0), followed by a comma, followed by a paragraph to describe the reason. There's no any other content. For example: '1,Because...'
    """

    # 构建消息列表
    messages = [
        {"role": "system", "content": system_prompt}
    ]

    if image_data:
        user_message = {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}
            ]
        }
    else:
        user_message = {
            "role": "user",
            "content": user_prompt  # 没有图片时，直接使用文本内容
        }

    messages.append(user_message)

    # 调用模型
    response = client.chat.completions.create(
        model="doubao-1-5-thinking-vision-pro-250428",
        messages=messages,
        max_tokens=500
    )

    response_text = str(response.choices[0].message.content)
    # 提取答案和原因
    # 寻找第一个逗号的位置
    comma_index = response_text.index(',')
    ans = response_text[:comma_index].strip()
    reason = response_text[comma_index + 1:].strip()
    return ans, reason, response_text

def build_description(row):
    p1_min_d = row['path1_Minimum clearance']
    p1_max_d = row['path1_Maximum clearance']
    p1_avg_d = row['path1_Average clearance']
    p1_len = row['path1_Path length']
    p1_smooth = row['path1_Smoothness']
    p1_turns = row['path1_Sharp turns']
    p1_max_angle = row['path1_Maximum angle']
    p2_min_d = row['path2_Minimum clearance']
    p2_max_d = row['path2_Maximum clearance']
    p2_avg_d = row['path2_Average clearance']
    p2_len = row['path2_Path length']
    p2_smooth = row['path2_Smoothness']
    p2_turns = row['path2_Sharp turns']
    p2_max_angle = row['path2_Maximum angle']

    res = f"""
        These are descriptions for the scenario:

        Path 1 has a minimum clearance of {p1_min_d:.2f} meters, a maximum clearance of {p1_max_d:.2f} meters, 
        and an average clearance of {p1_avg_d:.2f} meters. In comparison, Path 2 has a minimum clearance of 
        {p2_min_d:.2f} meters, a maximum clearance of {p2_max_d:.2f} meters, and an average clearance of 
        {p2_avg_d:.2f} meters.

        Regarding smoothness, Path 1 has a smoothness value of {p1_smooth:.2f}, while Path 2 has a smoothness 
        of {p2_smooth:.2f}. This value represents the total angular change along the path — lower values imply 
        smoother paths.

        Path 1 includes {p1_turns} sharp turns (greater than 90 degrees), with the sharpest turn being 
        {p1_max_angle:.2f} degrees. Path 2 has {p2_turns} sharp turns, with a maximum turn angle of 
        {p2_max_angle:.2f} degrees.

        In terms of distance, Path 1 has a total length of {p1_len:.2f} meters, while Path 2 is {p2_len:.2f} meters long.
    """

    return res

def test_cases(df, num_cases=20):
    """
    测试给定 DataFrame 中的多个案例,每个案例在三种模式下(Vision Only, Vision With Desc., Desc. Only)进行评估。

    参数:
        df (pd.DataFrame): 包含测试数据的 DataFrame。
        num_cases (int): 要测试的案例数量，默认最多测试 20 个。
    """
    modes = ['Vision Only', 'Vision With Desc.', 'Desc. Only']

    for i, row in df.iterrows():
        if i >= num_cases:
            break

        ID = row['id']
        img_path = row['image']
        scenario = row['scenario']
        ground_truth = row['ground_truth']
        desc = build_description(row)
                # ----------------------------------------------
        print(f"\n==================== CASE {i+1} / {num_cases} ====================")
        print(f"ID: {ID}")
        print(f"Ground Truth: {ground_truth}")

        prompt, desc = '', ''
        print(f'- Scenario: {scenario}')
        for mode in modes:
            # 设置 prompt 和图像路径
            if mode == 'Vision Only':
                prompt = scenario
                current_img_path = img_path
            elif mode == 'Vision With Desc.':
                desc = build_description(row=row)
                prompt = scenario + desc
                current_img_path = img_path
            elif mode == 'Desc. Only':
                desc = build_description(row=row)
                prompt = scenario + desc
                current_img_path = None

            # 模拟调用 LLM
            ans, reason, raw_ans = query_LLM(img_path=current_img_path, prompt=prompt)

            print('\n----------------------------------------------')
            print(f'- Mode: {mode}')
            print(f'- Image: {current_img_path}')
            print(f'- Description: {desc}')
            print(f'- LLM chose: {ans}')
            print(f'- LLM reason: {reason}')

            if int(ans) == int(ground_truth[-1]):
                print(f'- Correct (Ground Truth: {ground_truth})')
            else:
                print(f'-  Wrong  (Ground Truth: {ground_truth})')


# 示例调用（假设你已经有一个 DataFrame df）
test_cases(df, num_cases=10)
   
