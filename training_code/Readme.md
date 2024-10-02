# 训练代码说明

## 概述

此文档详细说明了之前用于微调模型的训练代码和训练过程，包括所使用的数据、环境配置、训练步骤和注意事项，最新提交并未对模型进行微调。

## 1. 训练数据

- **数据集名称**：`inference_data.csv`
- **数据来源**：由赛事官方提供数据处理后得到
- **数据预处理**：
  - 将训练数据按 每个背景材料对应一个问题+对应选项+对应回答作为一份训练样本
  - 根据找到的模型在instruction中配置special_tokens

## 2. 环境配置

- **操作系统**： `Linux` 
- **硬件要求**：
  - CPU： `Intel(R) Xeon(R) CPU E5-2699A v4`
  - GPU： `NVIDIA V100 32GB`
  - 内存： `32GB RAM`
- **软件依赖**：
  - **Python版本**： `Python 3.11`
  - **必要的库及版本**：
    - `torch==2.1.2`
    - `transformers==4.44.0`
- **环境搭建**：
  - 提供`requirements.txt`文件，按如下bash命令安装依赖项：

    ```bash
    pip install -r requirements.txt
    ```

## 3. 训练步骤

### 3.1 数据准备

- **数据路径**：说 `/app/round1_train_data.jsonl`
- **数据格式**：按比赛官方训练数据修改并通过gpt4o分析，每行包含 `{"id": "...", "problem": "...",...}`

### 3.2 模型初始化

- **预训练模型**：Qwen2.5-32B-Instruct-GPTQ-Int4
- **模型地址**：https://modelscope.cn/models/qwen/Qwen2.5-32B-Instruct-GPTQ-Int4
- **模型初始化方法**：从预训练模型加载

### 3.2 训练参数

- **超参数设置**：
  - 学习率（Learning Rate）： `1e-4`
  - 批次大小（Batch Size）： `4`
  - 训练轮数（Epochs）： `1`
  - 优化器（Optimizer）： 无
  - 损失函数（Loss Function）： `CrossEntropyLoss`
- **微调参数**：`
        "self_attn.q_proj",    # 自注意力机制的 Query 权重
        "self_attn.k_proj",    # 自注意力机制的 Key 权重
        "self_attn.v_proj",    # 自注意力机制的 Value 权重
        "self_attn.o_proj",    # 自注意力机制的 Output 权重
        "mlp.gate_proj",       # 前馈网络的 gate 投影层
        "mlp.up_proj",         # 前馈网络的上升投影层
        "mlp.down_proj"        # 前馈网络的下降投影层

### 3.4 训练过程

- **运行训练脚本**：

  /app/training_code/Qwen2.5微调.ipynb

## 4. 实际提交模型改进方案
      1.利用inference_data.csv作为向量数据库提供相似材料及对应分析过程
      2.通过优化top_k、temperature、max_token等参数来生成sequence
      3.优化提示词、格式化输出保证闭世界逻辑、给出答案前进行逻辑分析及准确提取答案
      4.vllm推理加速
      5.模型的选取测试
      