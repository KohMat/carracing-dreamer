from typing import Union

import numpy as np
from visdom import Visdom


class Plot:
    def __init__(self, x_label="Epoch", y_label="Loss", visdom=None):
        self.impl = LinePlot(x_label=x_label, y_label=y_label, visdom=visdom)

    def add(
        self,
        epoch: Union[int, float, np.float32],
        value: Union[int, float, np.float32],
        window_name: str,
        legend_name: str,
    ):
        if epoch > 0:
            self.impl.add(epoch, value, window_name, legend_name)
        else:
            self.impl.replace(epoch, value, window_name, legend_name)


class LinePlot:
    def __init__(self, x_label="Epoch", y_label="Loss", visdom=None):
        if visdom is None:
            self.vis = Visdom()
        else:
            self.vis = visdom
        self.x_label = x_label
        self.y_label = y_label

    def add(
        self,
        x: Union[int, float, np.float32, np.ndarray],
        y: Union[int, float, np.float32, np.ndarray],
        window_name: str,
        legend_name: str,
    ):
        if isinstance(x, (int, float, np.float32)):
            x = np.array([x])
        if isinstance(y, (int, float, np.float32)):
            y = np.array([y])
        update = "append"
        self.vis.line(
            X=x,
            Y=y,
            win=window_name,
            name=str(legend_name),
            update=update,
            opts=dict(
                title=window_name, xlabel=self.x_label, ylabel=self.y_label
            ),
        )

    def replace(
        self,
        x: Union[int, float, np.float32, np.ndarray],
        y: Union[int, float, np.float32, np.ndarray],
        window_name: str,
        legend_name: str,
    ):
        if isinstance(x, (int, float, np.float32)):
            x = np.array([x])
        if isinstance(y, (int, float, np.float32)):
            y = np.array([y])
        update = "replace"
        self.vis.line(
            X=x,
            Y=y,
            win=window_name,
            name=str(legend_name),
            update=update,
            opts=dict(
                title=window_name, xlabel=self.x_label, ylabel=self.y_label
            ),
        )

    def reset(self, window_name: str, legend_name: str):
        self.vis.line(
            X=None, Y=None, win=window_name, name=legend_name, update="remove"
        )
