import os

# 必须在 import torch 之前设置环境变量
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import sys

# 确保能导入 src 里的模块
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from src.model import TransformerEncoder
from src.dataset import TinyShakespeareDataset

# 硬件检测
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"正在使用设备: {device}")

# 按照作业表 3 设置超参数 [cite: 79, 80]
LR = 3e-4
BATCH_SIZE = 32
SEQ_LEN = 64
EPOCHS = 1000

# 初始化数据和模型 [cite: 10]
ds = TinyShakespeareDataset(seq_len=SEQ_LEN)
model = TransformerEncoder(
    vocab_size=ds.vocab_size,
    d_model=128,
    n_layers=2,
    h=4
).to(device)

print(f"模型总参数量: {model.count_parameters():,}")

# 进阶技巧：AdamW 优化器 [cite: 19]
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()
loss_history = []

print("🚀 正在启动训练...")
model.train()
for i in range(EPOCHS):
    x, y = ds.get_batch(BATCH_SIZE)
    x, y = x.to(device), y.to(device)

    logits = model(x)
    loss = criterion(logits.view(-1, ds.vocab_size), y.view(-1))

    optimizer.zero_grad()
    loss.backward()

    # 进阶技巧：梯度裁剪 [cite: 19]
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    optimizer.step()
    loss_history.append(loss.item())

    if i % 100 == 0:
        print(f"迭代 {i:4d} | Loss: {loss.item():.4f}")

# 确保 results 文件夹存在并保存曲线 [cite: 15, 19]
os.makedirs("results", exist_ok=True)
plt.figure(figsize=(10, 5))
plt.plot(loss_history)
plt.title("Training Loss Curve")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig("results/loss_curve.png")

# 保存模型 [cite: 19]
# 修改后的代码：手动区分文件名
save_name = "loss_no_pos"  # 当你注释掉位置编码时，把这里改写成这个名字
# save_name = "loss_with_pos" # 当你有位置编码时，用这个名字

plt.savefig(f"results/{save_name}.png")
torch.save(model.state_dict(), f"results/{save_name}.pth")
print(f"✅ 训练完成！结果已保存至 results/{save_name}.png")