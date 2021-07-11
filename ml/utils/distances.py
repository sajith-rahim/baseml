import numpy as np


def euclidean_distance(a, b):
    if isinstance(a, list) and isinstance(b, list):
        a = np.array(a)
        b = np.array(b)

    return np.sqrt(sum(np.square(a - b)))


def cosine_sim(a, b):
    if isinstance(a, list) and isinstance(b, list):
        a = np.array(a)
        b = np.array(b)

    return a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b))


def cosine_dist(a, b):
    return 1.0 - cosine_sim(a, b)

# u = [0, 0, 1, 6]
# v = [0, 0, 4, 2]
# print(cosine_sim(u,v))
