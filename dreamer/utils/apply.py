from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
import torch


def apply(
    data: Union[
        float,
        np.ndarray,
        List[np.ndarray],
        Tuple[np.ndarray],
        Dict[Any, np.ndarray],
        torch.Tensor,
    ],
    func: Callable,
):
    if isinstance(data, float):
        return func(data)

    elif isinstance(data, np.ndarray):
        return func(data)

    elif isinstance(data, list):
        tensors = list()
        for datum in data:
            tensors.append(func(datum))
        return tensors

    elif isinstance(data, tuple):
        data_type = type(data)  # for named tuple
        tensors = list()
        for datum in data:
            tensors.append(func(datum))
        return data_type(*tensors)

    elif isinstance(data, dict):
        tensors = dict()
        for k in data.keys():
            tensors[k] = func(data[k])
        return tensors

    elif isinstance(data, torch.Tensor):
        return func(data)

    else:
        raise ValueError(
            f"The given value type: {type(data)} is not supported"
        )
