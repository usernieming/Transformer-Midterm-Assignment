

# 基于Transformer 实现文本建模实验 (Mid-Term Assignment)

本仓库包含了《大模型基础与应用》课程期中作业的完整实现。我们从底层构建了 Transformer Encoder 架构，并在 **Tiny Shakespeare** 数据集上完成了字符级语言建模任务。

## 1. 提交内容摘要
- **核心模块手工实现**：从底层矩阵运算开始实现 Multi-head Self-attention、Position-wise FFN、残差连接 (Residual)、层归一化 (Layer Norm) 以及正余弦位置编码 (Positional Encoding)。
- **训练优化技巧**：集成了 AdamW 优化器、梯度裁剪 (Gradient Clipping) 以及模型参数量统计功能。
- **消融实验 (Ablation Study)**：系统对比了“位置编码”对 Transformer 模型收敛性能的影响。
- **可视化结果**：```results/loss_curve.png ```|```results/loss_no_pos.png```
## 2. 项目目录结构
```text
期中作业/
├── src/                # 核心源代码目录
│   ├── model.py        # Transformer 架构实现 (Encoder)
│   ├── dataset.py      # 数据加载与处理 pipeline
│   └── test.py         # 维度校验与单元测试脚本
├── results/            # 实验产出目录
│   ├── loss_with_pos.png  # (实验A) 包含位置编码的收敛曲线
│   ├── loss_no_pos.png    # (实验B) 移除位置编码后的收敛曲线
│   └── model.pth          # 训练完成的模型权重文件
├── train.py            # 统一训练入口脚本
├── README.md           # 项目说明文档 (当前文件)
└── requirements.txt    # 依赖声明文件

```

## 3. 环境准备与安装

使用 Python 3.10+ 环境。通过以下命令安装必要依赖：

```bash
pip install torch matplotlib requests numpy

```



## 4. 实验复现指南 (Reproducibility)

本项目已固定随机种子以确保结果可重现。

### 4.1 基础实验 (Base Model)

1. 确保 `src/model.py` 中的 `pos_encoding` 代码处于启用状态。
2. 运行训练：

```bash
python train.py

```

训练结束后，Loss 曲线将保存至 `results/loss_with_pos.png`。

### 4.2 消融实验 (Ablation Study)

1. 在 `src/model.py` 的 `forward` 方法中注释掉位置编码相加的代码行。
2. 在 `train.py` 中将保存文件名修改为 `loss_no_pos.png` 后再次运行。

## 5. 实验设置与超参数 (Hyperparameters)

配置严格遵循作业表 3 的建议设置：

| 超参数 (Parameter) | 取值 (Value) |
| --- | --- |
| 嵌入维度 (d_model) | 128 |
| 多头数量 (Heads) | 4 |
| 前馈层维度 (d_ff) | 512 |
| 层数 (Layers) | 2 |
| 学习率 (LR) | 3e-4 |
| 优化器 (Optimizer) | AdamW |
| 序列长度 (Seq Len) | 64 |

## 6. 实验结论分析

通过对比发现，包含位置编码的模型收敛更加稳定且 Loss 更低（约 2.5 左右）。 移除位置编码后模型性能显著下降，证明了自注意力机制本身不具备捕获序列位置信息的能力，必须依赖显式的位置注入。

---

## 附录：requirements.txt 内容声明

为了符合开源规范，本项目运行所需的库列表如下：

```text
torch>=2.0.0
matplotlib
requests
numpy

