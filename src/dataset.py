import torch
import requests
import os


class TinyShakespeareDataset:
    def __init__(self, seq_len=64):
        self.seq_len = seq_len
        self.url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        self.data_path = "input.txt"

        # 1. 下载数据
        if not os.path.exists(self.data_path):
            print("正在下载数据集...")
            data = requests.get(self.url).text
            with open(self.data_path, "w") as f:
                f.write(data)
        else:
            with open(self.data_path, "r") as f:
                data = f.read()

        # 2. 构建字符表 (Vocab)
        chars = sorted(list(set(data)))
        self.vocab_size = len(chars)
        print(f"字符集大小: {self.vocab_size}")

        # 映射表：字符 <-> 数字
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

        # 将整个文本转换为数字
        self.data_tensor = torch.tensor([self.stoi[c] for c in data], dtype=torch.long)

    def get_batch(self, batch_size):
        # 随机选择起始位置
        ix = torch.randint(len(self.data_tensor) - self.seq_len, (batch_size,))

        # 构建输入 x 和 目标 y (y 是 x 往后偏移一个字符)
        x = torch.stack([self.data_tensor[i: i + self.seq_len] for i in ix])
        y = torch.stack([self.data_tensor[i + 1: i + self.seq_len + 1] for i in ix])

        return x, y

    def decode(self, tokens):
        """将数字序列转回文字，方便后续观察模型生成的文本"""
        return ''.join([self.itos[int(t)] for t in tokens])