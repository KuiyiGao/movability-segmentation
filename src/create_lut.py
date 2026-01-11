"""
根据你的 MOVABILITY_MAPPING（182→4）生成 256×uint8 LUT，并保存为 lut_movability.npy
仅首次执行一次即可
"""
import numpy as np

# ----------- 1. 完整映射表（仅示例，需补全 0–181） ----------
MOVABILITY_MAPPING = {         # 0:不可移动,1:人可移动,2:机械可移动,3:高动态
    # 不可移动
    **{i: 0 for i in range(0, 10)},
    # 人可移动
    **{i: 1 for i in range(10, 19)},
    # 机械可移动
    **{i: 2 for i in range(19, 26)},
    # 高动态
    **{i: 3 for i in range(26, 33)},
    # 其余类别……自行补全直至 181
}

# ----------- 2. 构建 256 长 LUT ----------
lut = np.zeros(256, dtype=np.uint8)        # 默认全部映射到 0（不可移动）
for k, v in MOVABILITY_MAPPING.items():
    lut[k] = v

np.save("lut_movability.npy", lut)
print("保存完成：lut_movability.npy")