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
### 子任务1:目前最高准确率0.21
已尝试的替代方法及其结果：（其中方法A ~ D 是在基础模型上修改的）

方法A:  加入 PAD 忽略 + class weight（类别权重）+ warmup（学习率预热） + max_length=256（更长序列）   准确率 0.19

方法B:  利用方法A训练的模型，修改推理代码，加入Softmax + 置信度过滤 并调整部分参数   准确率 0.21

方法C:  在方法A的方法上增加基于5个类别的简单同义词替换 准确率 0.20

方法D:  使用token classification方法，加上优化推理  准确率 0.21

评估阶段：

distilbert模型，使用token classification方法                           准确率 0.21

把模型更换为Roberta-base                                               准确率 0.19

把模型更换为deberta-v3-small                                           准确率 0.18

### 子任务2:目前最高准确率0.81
已尝试的替代方法及其结果：（其中方法A ~ E 是在基础模型上修改的）

方法A: 准确率 0.76

方法B: 准确率 0.74

方法C: 加入 R-Drop + Label Smoothing + Weighted Loss + 清洗 + 数据增强 准确率 0.80

方法D: 加入 R-Drop + 交叉验证 + 清洗 + 数据增强 准确率 0.80

方法E: 利用方法D中保存的最佳模型，修改推理代码，加入Softmax + 微调阈值 准确率 0.81

评估阶段：baseline 准确率 0.73

在取得最高准确率的代码上更换验证集，没有其他改动                          准确率 0.74

把模型更换为Roberta-base                                               准确率 0.73

修改推理代码，logit ensemble + threshold tuning（threshold=0.45）       准确率 0.74

把模型更换为deberta-v3-small                                            准确率 0.72

继续使用Roberta-base训练，但是在训练集中添加150条生成的类似数据            准确率 0.73

继续使用Roberta-base训练，删除数据清理部分，修改learning_rate为1e-5，训练使用官方提供数据集，验证集为生成数据集        准确率 0.73

添加optuna超参数方法                                                      准确率 0.74

## 待探索方向
目前正在研究以下方法来提升准确率：

预训练语言模型微调 (BERT, RoBERTa, DeBERTa)

数据增强技术

注意力机制优化
