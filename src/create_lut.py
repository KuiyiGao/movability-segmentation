import numpy as np
MOVABILITY_MAPPING = {
    **{i: 0 for i in range(0, 10)},
    **{i: 1 for i in range(10, 19)},
    **{i: 2 for i in range(19, 26)},
    **{i: 3 for i in range(26, 33)},
}
lut = np.zeros(256, dtype=np.uint8)        
for k, v in MOVABILITY_MAPPING.items():
    lut[k] = v
np.save('src/lut_movability.npy', lut)
print('finished')