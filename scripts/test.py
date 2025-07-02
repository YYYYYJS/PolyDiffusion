import torch


def mix_sequence(x1, x2, S):
    BS, H, W, D = x1.size()
    _, H2, W2, _ = x2.size()
    x2 = x2.view(BS, -1, D)
    res = []

    k = 0
    for i in range(0, H, S):
        for j in range(0, H, S):
            temp = x1[:, i:i + S, j:j + S, :].reshape(BS, -1, D)
            t = x2[:, k:k + 1, :]
            res.append(torch.cat((temp, t), dim=1))
            k = k + 1

    res = torch.cat(res, dim=1)
    return res.view(BS, -1, D)


def separate_sequence(x):
    step = 17
    BS, L, D = x.size()
    x1 = []
    x2 = []

    for i in range(0, L, step):
        x1.append(x[:, i:i + step - 1])
        x2.append(x[:, i + step - 1:i + step])
    x1 = torch.cat(x1, dim=1)
    x2 = torch.cat(x2, dim=1)
    return x1, x2


def reorder(x, S):
    BS, L, D = x.size()
    H = int(L ** 0.5)
    R = S * S

    temp = torch.zeros((BS, H, H, D))

    k = 0
    for i in range(0, H, S):
        for j in range(0, H, S):
            xs = x[:, k:k + R].reshape(BS, S, S, D)
            temp[:, i:i + S, j:j + S, :] = xs
            k = k + R

    return temp


