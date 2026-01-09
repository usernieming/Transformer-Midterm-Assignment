import os
import sys
import torch

# 解决 OMP 报错的关键：允许重复加载库
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 确保导入路径正确
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from model import TransformerEncoder

    print("✅ 成功找到 TransformerEncoder 类！")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit()


def test():
    # 使用作业要求的超参数 [cite: 80]
    vocab_size = 1000
    d_model = 128
    n_layers = 2
    h = 4

    model = TransformerEncoder(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, h=h)

    # 统计模型参数 (进阶加分项 )
    total_params = model.count_parameters()
    print(f"模型总参数量: {total_params:,}")

    # 模拟输入 (batch_size=32, seq_len=64) [cite: 80]
    dummy_input = torch.randint(0, vocab_size, (32, 64))

    # 前向传播
    output = model(dummy_input)

    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状: {output.shape}")

    if output.shape == (32, 64, vocab_size):
        print("🎉 维度校验成功！准备进入训练阶段。")


if __name__ == "__main__":
    test()