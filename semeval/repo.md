# SemEval 2026 Task 10: Psycholinguistic Conspiracy Marker Extraction and Detection
## 任务概述
本仓库包含SemEval 2026第10任务"心理语言学阴谋标记提取与检测"的代码实现。该任务包含两个子任务：

子任务1: 阴谋标记提取 (Conspiracy marker extraction)

子任务2: 阴谋检测分类 (Conspiracy detection classification)

# 当前进展

## 基准结果
子任务1: 当前准确率为0.16（与官方基准一致）

子任务2: 当前准确率为0.78（与官方基准一致）

## 实验方法尝试
### 子任务1:
目前还未尝试子任务1的新训练方法
### 子任务2:
已尝试的替代方法及其结果：

方法A: 准确率 0.76

方法B: 准确率 0.74

方法B: 加入 R-Drop + Label Smoothing + Weighted Loss + 清洗 + 数据增强 准确率 0.80

## 待探索方向
目前正在研究以下方法来提升准确率：

预训练语言模型微调 (BERT, RoBERTa, DeBERTa)

数据增强技术

注意力机制优化
