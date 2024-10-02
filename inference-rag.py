import os
import torch
import json
import re
import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams
from loguru import logger
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd

# 初始化日志
logger.add("logs/app_{time:YYYY-MM-DD}.log", level="INFO", rotation="00:00", retention="10 days", compression="zip")

# 设置GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用 GPU 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using GPU: {os.environ['CUDA_VISIBLE_DEVICES']}")
logger.info(f"Model will run on device: {device}")

# 替换为您本地模型的路径
MODEL_PATH = "/app/model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
llm = LLM(MODEL_PATH, max_model_len=2048, quantization="gptq")

# 定义推理参数
# sampling_params = SamplingParams(
#     best_of=2,  
#     use_beam_search=True,  
#     max_tokens=2048,  
#     temperature=0  
# )
sampling_params = SamplingParams(
    temperature=0,  # Slightly increase randomness
    top_k=2,  # Limit vocabulary range
    max_tokens=2048,  # Limit the number of generated tokens
)
# 读取 CSV 文件，假设文件名为 'inference_data.csv'，并且包含 'inference' 列
csv_file = 'inference_data.csv'  
df = pd.read_csv(csv_file, encoding='gbk')

# 筛选出 'answer_match' 列为 1 的行
df_filtered = df[df['answer_match'] == 1]

# 将 'problem'、'question'、'options' 和 'gpt_inference' 列的内容拼接为一个字符串，仅对筛选后的行进行操作
df_filtered['combined_text'] = df_filtered.apply(lambda row: f"材料: {row['problem']} \n问题描述: {row['question']} \n选项: {row['options']} \n推理过程: {row['gpt_inference']}", axis=1)

# 提取拼接后的 'combined_text' 列作为推理文本
inference_texts = df_filtered['combined_text'].tolist()

# 使用 bge-small-zh-v1.5 模型计算文本嵌入
embedding_model = SentenceTransformer('/app/model_rag')
inference_embeddings = embedding_model.encode(inference_texts, convert_to_tensor=True)

# 构建 FAISS 索引
embedding_dimension = inference_embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dimension)
index.add(inference_embeddings.cpu().numpy())

# 定义一个函数来检索与查询相关的推理文本
def retrieve_inferences(query, top_k=1):
    query_embedding = embedding_model.encode([query], convert_to_tensor=True).cpu().numpy()
    distances, indices = index.search(query_embedding, top_k)
    retrieved_examples = [f'参考示例{idx+1}:\n'+inference_texts[idx] for idx in indices[0]]
    retrieved_examples = "\n".join(retrieved_examples)
    return retrieved_examples

tail_message="""
    **注意事项**：
    1. 请严格按照问题要求回答，不能修改或篡改问题中的任何条件或表述。如果问题中有特定要求（如“通过中转点”），请严格遵守所有限定条件。
    2. 推理过程遵循闭世界假设，未观测到的事实或无法推断的事实视为假。只能依据材料中的信息进行推理，不得使用材料外的知识（如语言常识、常识推理等）。
    3. 请严格按以下 JSON 格式回复：{{"inference":"你的逻辑推理部分","answer":"A"}}。确保 JSON 格式完全合法，任何不符合该格式的回复视为错误。
    """
# 执行推理
def api_retry(query, max_retries=5, retry_delay=5):
    attempts = 0
    outputs = None  
    while attempts < max_retries:
        try:
            outputs = llm.generate([query], sampling_params, use_tqdm=False)
            return outputs[0].outputs[0].text
        except Exception as e:
            attempts += 1
            logger.error(f"推理失败: {e}. 正在重试 ({attempts}/{max_retries})")
            time.sleep(retry_delay)
    logger.error("推理多次失败，跳过此条输入")
    return None



def get_prompt(problem, question, options):
    """生成推理问题的 prompt"""
    options_str = '\n'.join(f"{'ABCDEFG'[i]}. {o}" for i, o in enumerate(options))
#     prompt = f"""<|system|>你是一个逻辑推理专家，擅长解决逻辑推理问题。以下是一个逻辑推理的题目，形式为单项选择题。所有的问题都是（close-world assumption）闭世界假设，即未观测事实都为假。请逐步分析问题并在最后一行输出答案，最后一行的格式为"答案是：A"。注意，你必须选出一个选项作为正确答案，任何不符合上述格式要求的回复都视为错误。题目如下：<|endoftext|>
# <|user|>
# ### 题目:
# {problem}

# ### 问题:
# {question}
# {options_str}
# <|endoftext|>\n<|assistant|>\n

# """
    prompt = f"""
    请按以下步骤逐步回答问题：
    
    一、仔细阅读以下材料。
    二、基于材料内容，逐步进行逻辑推理。
    三、根据推理结果选择最合适的选项。
    
    *材料*：{problem}
    *问题*：{question}
    *选项*：{options_str}
    """
    return prompt



def most_frequent_char(char1, char2, char3):
    """找出出现频率最高的字符"""
    frequency = {char1: 0, char2: 0, char3: 0}
    frequency[char1] += 1
    frequency[char2] += 1
    frequency[char3] += 1
    return max(frequency, key=frequency.get)


def process_datas(datas):
    """处理数据并进行推理"""
    results = []
    # 初始化计数器
    count = 0
    correct = 0
    for data in tqdm(datas, desc="Submitting tasks", total=len(datas)):
        problem = data['problem']
        for id, question in enumerate(data['questions']):
            prompt = get_prompt(problem, question['question'], question['options'])
            retrieved_examples = retrieve_inferences(prompt, top_k=1)
            merge_message=prompt + "\n以下是类似的材料及对应推理过程（仅供参考）：\n" + retrieved_examples+'\n'+tail_message
            messages = [
                {"role": "system", "content": "你是一名逻辑推理专家。你的任务是分析逻辑问题并尽可能给出正确答案"},
                {"role": "user", "content": merge_message}
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            # 三次推理以得到最常见的答案
            # res1, res2, res3 = api_retry(prompt), api_retry(prompt), api_retry(prompt)
            # extract_response1, extract_response2, extract_response3 = extract(res1), extract(res2), extract(res3)
            while True:
                try:
                    res1 = api_retry(text)
                    print('-'*80+'\n'+res1)
                    # print(prompt)
                    if not res1:
                        logger.error("未获得有效推理结果，跳过此问题")
                        continue
                    content = res1.replace('\n', '')

            
                    # 使用更可靠的方法提取 JSON 格式的数据
                    # 正则表达式模式，使用非贪婪匹配，同时捕获两个字段的值
                    pattern = r'"inference".*?"(.*?)".*?"answer".*?"(.*?)"}'
                    
                    # 进行匹配
                    match1 = re.search(pattern, content, re.DOTALL)
                    if match1:
                        inference = match1.group(1)
                        ans = match1.group(2)
                    # print('-'*80 + '\n', content)
                    # if match:
                    #     gpt_data = json.loads(match.group())
                    # else:
                    #     logger.error("无法从输出中提取有效的JSON数据")
                    #     break
                    # inference = gpt_data.get('inference')
                    # ans = gpt_data.get('answer')
                    
            
                    if ans:
                        print('答案为:'+ans)
                        # count += 1
                        # if data['questions'][id]['answer'] == ans:
                        #     correct += 1
                        #     if count % 5 == 0:
                        #         print(f'已回答{count}题，正确{correct}题，正确率{correct/count:.2%}')
                        data['questions'][id]['answer'] = ans
                        # data['questions'][id]['inference'] = inference
                    break
                except Exception as e:
                    logger.error(f"解析推理结果时出错: {e}")
                    continue
        results.append(data)               
    return results

def main(ifn, ofn):
    """主处理流程"""
    if os.path.exists(ofn):
        logger.info(f"Output file {ofn} already exists.")

    data = []
    logger.info(f"Loading data from {ifn}...")
    with open(ifn, encoding='utf-8') as reader:
        for line in reader:
            sample = json.loads(line)
            data.append(sample)
    
    if len(data) == 0:
        logger.error(f"No data loaded from {ifn}.")
    else:
        logger.info(f"Successfully loaded {len(data)} samples from {ifn}.")

    return_list = process_datas(data)
    with open(ofn, 'w', encoding='utf-8') as writer:
            for sample in return_list:
                writer.write(json.dumps(sample, ensure_ascii=False))
                writer.write('\n')
    csv_data=pd.DataFrame(return_list)
    csv_data.to_csv('./result.csv',index=False)
    logger.info("All tasks finished!")
    print("All tasks finished!")

count, correct = 0, 0
main('/tcdata/round2_test_data.jsonl', '/app/results.jsonl')
