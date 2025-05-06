import numpy as np
import torch
from scipy.interpolate import interp1d

def backtrack(R):
    path = np.zeros((2, 1))
    p, q = [], []
    i, j = np.array(R.shape) - 1
    while i > 0 or j > 0:
        p.append(i)
        q.append(j)
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            min_neighbor = min(R[i - 1][j], R[i][j - 1], R[i - 1][j - 1])
            if min_neighbor == R[i - 1][j]:
                i -= 1
            elif min_neighbor == R[i][j - 1]:
                j -= 1
            else:
                i -= 1
                j -= 1
    p.append(0), q.append(0)
    p.reverse(), q.reverse()
    path = np.pad(path, ((0, 0), (0, len(p) - path.shape[1])), mode='constant')
    path[0, :] = np.array(p)
    path[1, :] = np.array(q)
    return path


def WarpData(y, path):
    xp = path[0, :]
    yp = path[1, :]

    interp_func = interp1d(xp, yp, kind='linear')

    warping_index = interp_func(np.arange(xp.min(), xp.max() + 1)).astype(np.int64)
    warping_index[0] = yp.min()
    warping_data = y[:, warping_index.T]

    return warping_data

# 生成最终扭曲数据
def WarpOut(y, path):
    # y=normal.unnormalize(y)
    # x为基准数据 y为待校准数据
    if torch.is_tensor(y):
        if y.is_cuda:
            y = y.cpu()
        y = y.numpy()
    if torch.is_tensor(path):
        if path.is_cuda:
            path = path.cpu()
        path = path.numpy()
    out = y * 0
    for i in range(y.shape[0]):
        op_path = backtrack(path[i, :, :])
        out[i, :, :] = WarpData(y[i, :, :], op_path)  # 转置是为了变成横向量

    out = torch.tensor(out).float()
    if torch.cuda.is_available():
        out = out.cuda()
    return out
