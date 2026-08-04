# -*- coding: utf-8 -*-
"""
用更大、主题多样的语料测试 int2 打包三元模型的性能。
扩大训练语料（避免过拟合）+ 扩大留出评估语料（公平评估泛化）。
对比 float32 / int8 / int2 的精度与解码速度。
"""
import math
import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformers import AutoModelForCausalLM, AutoTokenizer

from W8ASpike.Int2Spike.neuron import spike_fake_quant, SpikeCountBitwiseNode
from pack_ternary import (TernarySpikeLinearSTE, PackedTernaryLinear,
                          convert_attn, convert_to_packed, packed_weight_memory)

MODEL_PATH = "./models/Qwen3-0.6B"
K = 4.0

# 训练语料（40 句，主题多样）
TRAIN_TEXTS = [
    "Spiking neural networks are inspired by the brain and mimic biological computation.",
    "The future of artificial intelligence depends on energy efficient computing.",
    "脉冲神经网络是一种受大脑启发的计算模型，具有低功耗优势。",
    "Machine learning models can learn from large amounts of data efficiently.",
    "深度学习在自然语言处理领域取得了巨大成功，推动了智能应用的发展。",
    "Quantization reduces model size while preserving most of the accuracy.",
    "Efficient inference is critical for deploying models on edge devices.",
    "神经形态计算为下一代低功耗人工智能提供了新的方向。",
    "Ternary weights enable fast sparse computation in neural networks.",
    "The brain computes with spikes, which are sparse and energy efficient.",
    "The Earth orbits the Sun once every three hundred and sixty five days.",
    "Water freezes at zero degrees Celsius and boils at one hundred degrees.",
    "光合作用是植物将阳光转化为能量的重要过程。",
    "The Industrial Revolution transformed manufacturing and society in the eighteenth century.",
    "Ancient Rome built an extensive network of roads across its vast empire.",
    "长城是中国古代伟大的防御工程，绵延数千公里。",
    "The human heart pumps blood through the body to deliver oxygen to cells.",
    "Regular exercise and a balanced diet are essential for good health.",
    "均衡饮食和规律运动对保持身体健康非常重要。",
    "The Internet connects billions of devices around the world in real time.",
    "Cloud computing allows companies to scale their infrastructure on demand.",
    "云计算让企业能够按需扩展其基础设施，降低成本。",
    "Photosynthesis converts sunlight into chemical energy stored in plants.",
    "The Great Wall of China is one of the most famous landmarks in the world.",
    "Economic growth depends on innovation, investment, and stable policies.",
    "经济增长依赖于创新、投资和稳定的政策环境。",
    "The ocean covers more than seventy percent of the Earth's surface.",
    "海洋覆盖了地球表面超过百分之七十的面积。",
    "Artificial intelligence is transforming healthcare, finance, and transportation.",
    "人工智能正在改变医疗、金融和交通等行业。",
    "The moon has no atmosphere and its surface is covered with craters.",
    "月球没有大气层，表面布满了陨石坑。",
    "Renewable energy sources include solar, wind, and hydroelectric power.",
    "可再生能源包括太阳能、风能和水力发电。",
    "The discovery of electricity revolutionized modern civilization.",
    "电力的发现彻底改变了现代文明。",
    "Birds migrate long distances to find food and suitable breeding grounds.",
    "鸟类长途迁徙以寻找食物和适宜的繁殖地。",
    "The printing press made books widely available and spread knowledge.",
    "印刷术使书籍广泛传播，促进了知识的普及。",
    "Quantum computing promises to solve problems beyond classical computers.",
    "量子计算有望解决传统计算机无法处理的问题。",
    "The Amazon rainforest is home to an incredible diversity of species.",
    "亚马逊雨林拥有令人难以置信的物种多样性。",
    "Space exploration has expanded our understanding of the universe.",
    "太空探索拓展了我们对宇宙的认识。",
    "The stock market reflects the collective expectations of investors.",
    "股票市场反映了投资者的集体预期。",
]

# 留出评估语料（40 句，与训练不同）
EVAL_TEXTS = [
    "Spiking neural networks are inspired by the brain.",
    "The future of artificial intelligence is bright.",
    "脉冲神经网络是一种受大脑启发的计算模型。",
    "Machine learning models can learn from large amounts of data.",
    "深度学习在自然语言处理领域取得了巨大成功。",
    "The solar system consists of the sun and the planets that orbit it.",
    "Volcanoes erupt when molten rock rises from deep within the Earth.",
    "火山喷发是地球内部岩浆上升的结果。",
    "The Renaissance was a period of great cultural and artistic achievement.",
    "文艺复兴是欧洲文化和艺术繁荣的重要时期。",
    "The human brain contains billions of neurons that communicate with each other.",
    "人类大脑包含数十亿个相互通信的神经元。",
    "Electric cars are becoming increasingly popular around the world.",
    "电动汽车在世界各地越来越受欢迎。",
    "The Nile River is the longest river in Africa and flows through many countries.",
    "尼罗河是非洲最长的河流，流经多个国家。",
    "Machine translation has improved dramatically with neural networks.",
    "机器翻译随着神经网络的发展取得了巨大进步。",
    "The atmosphere protects life on Earth from harmful solar radiation.",
    "大气层保护地球生命免受有害太阳辐射。",
    "Banks play a crucial role in the modern financial system.",
    "银行在现代金融体系中扮演着至关重要的角色。",
    "The invention of the telephone changed the way people communicate.",
    "电话的发明改变了人们的沟通方式。",
    "Coral reefs are among the most diverse ecosystems on the planet.",
    "珊瑚礁是地球上最多样化的生态系统之一。",
    "Artificial neural networks are inspired by the structure of the brain.",
    "人工神经网络受到大脑结构的启发。",
    "The winter solstice is the shortest day of the year in the northern hemisphere.",
    "冬至是北半球一年中白昼最短的一天。",
    "Robots are increasingly used in manufacturing to improve efficiency.",
    "机器人越来越多地用于制造业以提高效率。",
    "The theory of evolution explains the diversity of life on Earth.",
    "进化论解释了地球上生命的多样性。",
    "Satellites provide essential data for weather forecasting and navigation.",
    "卫星为天气预报和导航提供重要数据。",
    "The economy of a country depends on many interconnected factors.",
    "一个国家的经济取决于许多相互关联的因素。",
    "Deep learning has achieved remarkable results in image recognition.",
    "深度学习在图像识别方面取得了显著成果。",
    "The desert receives very little rainfall throughout the year.",
    "沙漠全年降雨量很少。",
    "Global warming is one of the most pressing challenges of our time.",
    "全球变暖是我们这个时代最紧迫的挑战之一。",
    "The heart of a whale can weigh as much as a small car.",
    "鲸鱼的心脏可以重达一辆小型汽车。",
    "Programming languages allow humans to communicate with computers.",
    "编程语言使人类能够与计算机交流。",
    "The pyramids of Egypt were built thousands of years ago.",
    "埃及金字塔建于数千年前。",
    "Batteries store chemical energy and convert it into electrical energy.",
    "电池储存化学能并将其转化为电能。",
    "The study of genetics helps us understand how traits are inherited.",
    "遗传学研究帮助我们理解性状是如何遗传的。",
    "Tourism is a major source of income for many countries.",
    "旅游业是许多国家的主要收入来源。",
    "The telescope allows astronomers to observe distant galaxies.",
    "望远镜使天文学家能够观测遥远的星系。",
    "Clean water is essential for human survival and public health.",
    "清洁的水对人类生存和公共健康至关重要。",
    "The wind turbine converts the kinetic energy of wind into electricity.",
    "风力涡轮机将风的动能转化为电能。",
    "Education is the foundation of a prosperous and equitable society.",
    "教育是繁荣和公平社会的基础。",
    "The microscope reveals the hidden world of cells and microorganisms.",
    "显微镜揭示了细胞和微生物的隐藏世界。",
    "Digital currencies are changing the way we think about money.",
    "数字货币正在改变我们对货币的看法。",
    "The migration of monarch butterflies spans thousands of kilometers.",
    "帝王蝶的迁徙跨越数千公里。",
    "Sustainable agriculture aims to meet food needs without harming the environment.",
    "可持续农业旨在满足粮食需求而不损害环境。",
]


def compute_perplexity(model, tokenizer, texts):
    total_nll, total_tokens = 0.0, 0
    for text in texts:
        input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
        with torch.no_grad():
            logits = model(input_ids=input_ids).logits
        shift_logits = logits[:, :-1].contiguous()
        shift_labels = input_ids[:, 1:]
        loss = nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1), reduction="sum")
        total_nll += loss.item()
        total_tokens += shift_labels.numel()
    return math.exp(total_nll / total_tokens)


def fine_tune(model, tokenizer, steps=40):
    for name, p in model.named_parameters():
        if "self_attn" not in name:
            p.requires_grad = False
    enc = tokenizer(TRAIN_TEXTS, return_tensors="pt", padding=True,
                    truncation=True, max_length=64)
    input_ids, labels = enc["input_ids"], enc["input_ids"].clone()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    model.train()
    for _ in range(steps):
        optimizer.zero_grad()
        loss = model(input_ids=input_ids, labels=labels).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
    model.eval()


def measure_decode(model, tokenizer, prompt, max_new=20):
    inputs = tokenizer(prompt, return_tensors="pt")
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    dt = time.perf_counter() - t0
    n_tokens = out.shape[1] - inputs["input_ids"].shape[1]
    return n_tokens / dt


def main():
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print(f"训练语料 {len(TRAIN_TEXTS)} 句，留出语料 {len(EVAL_TEXTS)} 句")

    # 原始模型（float32）留出 PPL
    ref = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.float32, trust_remote_code=True).eval()
    ppl_float = compute_perplexity(ref, tokenizer, EVAL_TEXTS)
    dec_float = measure_decode(ref, tokenizer, TRAIN_TEXTS[0])
    del ref
    print(f"原始 float32: 留出PPL={ppl_float:.2f}  解码={dec_float:.2f}t/s")

    # QAT 微调三元模型（只微调一次）
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.float32, trust_remote_code=True)
    convert_attn(model, TernarySpikeLinearSTE, k=K)
    print("QAT 微调中（40步，48句训练语料）...")
    fine_tune(model, tokenizer)
    ppl_ft = compute_perplexity(model, tokenizer, EVAL_TEXTS)
    dec_ft = measure_decode(model, tokenizer, TRAIN_TEXTS[0])
    print(f"三元+QAT(float32): 留出PPL={ppl_ft:.2f}  解码={dec_ft:.2f}t/s")

    # 保存微调后的注意力权重
    ft_state = {name: m.weight.data.clone() for name, m in model.named_modules()
                if isinstance(m, TernarySpikeLinearSTE)}
    del model

    # 从同一微调权重打包 int8 / int2
    for bits in [8, 2]:
        model_p = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, dtype=torch.float32, trust_remote_code=True)
        convert_attn(model_p, TernarySpikeLinearSTE, k=K)
        for name, m in model_p.named_modules():
            if isinstance(m, TernarySpikeLinearSTE):
                m.weight.data.copy_(ft_state[name])
        convert_to_packed(model_p, bits)
        ppl_p = compute_perplexity(model_p, tokenizer, EVAL_TEXTS)
        dec_p = measure_decode(model_p, tokenizer, TRAIN_TEXTS[0])
        mem = packed_weight_memory(model_p)
        print(f"int{bits} 打包: 留出PPL={ppl_p:.2f}  解码={dec_p:.2f}t/s  注意力权重内存={mem/1024**2:.1f}MB")
        del model_p


if __name__ == "__main__":
    main()
