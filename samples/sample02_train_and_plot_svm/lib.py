import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm


def generate_features(
    n=100,
    pos_loc=(5, 0),
    pos_scale=(1.0, 1.0),
    neg_loc=(-5, 0),
    neg_scale=(1.0, 1.0),
):
    x0 = np.random.normal(loc=neg_loc, scale=neg_scale, size=(n, 2))
    y0 = np.zeros(n)

    x1 = np.random.normal(loc=pos_loc, scale=pos_scale, size=(n, 2))
    y1 = np.ones(n)

    x = np.r_[x0, x1]
    y = np.r_[y0, y1]

    return x, y


def train(x, y):
    clf = svm.LinearSVC(random_state=0)
    clf.fit(x, y)
    return clf


def plot(x, y=None, coef=None, intercept=None):
    fig, ax = plt.subplots()

    if y is None:
        ax.scatter(x[:, 0], x[:, 1], label="items")
    else:
        _x = x[np.where(y == 1.0)]
        ax.scatter(_x[:, 0], _x[:, 1], label="1")
        _x = x[np.where(y == 0.0)]
        ax.scatter(_x[:, 0], _x[:, 1], label="0")

    if coef is not None and intercept is not None:
        line = np.linspace(x[:, 0].min(), x[:, 0].max())
        ax.plot(line, -(line * coef[0] + intercept) / coef[1])

    ax.set_xlim(x[:, 0].min(), x[:, 0].max())
    ax.set_ylim(x[:, 1].min(), x[:, 1].max())
    ax.legend()
    ax.grid()

    return fig


class Items:
    def __init__(self) -> None:
        self.x, self.y = generate_features()

    def append(self, _x0, _x1, _y):
        self.x = np.r_[self.x, np.array([[_x0, _x1]])]
        self.y = np.r_[self.y, np.array([_y])]
