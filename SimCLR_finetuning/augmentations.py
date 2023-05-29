import numpy as np
import torch


def DataTransform(sample):
    jitter_scale_ratio = 1.1
    jitter_ratio = 0.8
    max_seg = 8
    weak_aug = scaling(sample, jitter_scale_ratio)
    strong_aug = jitter(permutation(sample, max_segments=max_seg), jitter_ratio)

    return weak_aug, strong_aug


def jitter(x, sigma=0.8):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)


def scaling(x, sigma=1.1):
    # https://arxiv.org/pdf/1706.00527.pdf
    factor = np.random.normal(loc=2., scale=sigma, size=(x.shape[0], x.shape[2]))
    ai = []
    for i in range(x.shape[1]):
        xi = x[:, i, :]
        ai.append(np.multiply(xi, factor[:, :])[:, np.newaxis, :])
    return np.concatenate((ai), axis=1)


def permutation(x, max_segments=5, seg_mode="random"):
    x = np.array(x, dtype=object)
    orig_steps = np.arange(x.shape[2])
    orig_steps = np.array(orig_steps, dtype=object)

    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))
    num_segs = np.array(num_segs, dtype=object)

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[2] - 2, num_segs[i] - 1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
                splits = np.array(splits, dtype=object)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
                splits = np.array(splits,dtype=object)
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            ret[i] = pat[0,warp.astype('int64')]
        else:
            ret[i] = pat
    return ret
    # return torch.from_numpy(ret)



