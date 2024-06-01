import matplotlib.pyplot as plt
import numpy as np
from prmlmy.util import cv_, norm2s, calc_range


def plot_decision_boundary(model, X_train, y_train=None, x1_range=None, x2_range=None, points=300,
                           title=None, pad_ratio=0.2, ax=None):
    ax = ax or plt
    x1_range = x1_range or calc_range(X_train[:, 0], pad_ratio=pad_ratio)
    x2_range = x2_range or calc_range(X_train[:, 1], pad_ratio=pad_ratio)
    if y_train is None:
        y_train = np.zeros(X_train.shape[0])

    x1s = np.linspace(x1_range[0], x1_range[1], num=points)
    x2s = np.linspace(x2_range[0], x2_range[1], num=points)
    x1, x2 = np.meshgrid(x1s, x2s)
    x = np.array([x1, x2]).reshape(2, -1).T
    y = model.predict(x).reshape(points, points)

    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
    ax.contourf(x1, x2, y, alpha=0.2)
    if title:
        ax.set_title(title)


def plot_decision_proba(model, X_train, y_train=None, x1_range=None, x2_range=None, points=300,
                        title=None, pad_ratio=0.2, ax=None):
    ax = ax or plt
    x1_range = x1_range or calc_range(X_train[:, 0], pad_ratio=pad_ratio)
    x2_range = x2_range or calc_range(X_train[:, 1], pad_ratio=pad_ratio)
    if y_train is None:
        y_train = np.zeros(X_train.shape[0])

    x1s = np.linspace(x1_range[0], x1_range[1], num=points)
    x2s = np.linspace(x2_range[0], x2_range[1], num=points)
    x1, x2 = np.meshgrid(x1s, x2s)
    x = np.array([x1, x2]).reshape(2, -1).T
    y = model.proba(x).reshape(points, points)

    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
    ax.contourf(x1, x2, y, np.linspace(0, 1, 5), alpha=0.2)
    if title:
        ax.set_title(title)


def get_figsize_default(ncols, nrows):
    width = ncols * 5 + 1
    height = nrows * 4 + 1
    return width, height


def grid_plot(rows, cols, plot_func, row_names=None, col_names=None, figsize=None, *args, **kwargs):
    row_names = row_names or [str(row) for row in rows]
    col_names = col_names or [str(col) for col in cols]
    figsize = figsize or get_figsize_default(len(cols), len(rows))
    fig, axs = plt.subplots(nrows=len(rows), ncols=len(cols), figsize=figsize)
    axs = axs.reshape(len(rows), len(cols))
    for row_axs, row, row_name in zip(axs, rows, row_names):
        for ax, col, col_name in zip(row_axs, cols, col_names):
            title = ":".join([row_name, col_name])
            plot_func(row, col, title, ax=ax, *args, **kwargs)
