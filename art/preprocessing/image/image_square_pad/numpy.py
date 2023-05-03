# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2023
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
This module implements square padding for images and object detection bounding boxes.
"""
import logging
from typing import Dict, List, Any, Optional, TYPE_CHECKING, Tuple, Union

import numpy as np
from tqdm.auto import tqdm

from art.preprocessing.preprocessing import Preprocessor

if TYPE_CHECKING:
    from art.utils import CLIP_VALUES_TYPE

logger = logging.getLogger(__name__)


class ImageSquarePad(Preprocessor):
    """
    This module implements square padding for images and object detection bounding boxes.
    """

    params = ["channels_first", "label_type", "pad_mode", "pad_kwargs", "clip_values", "verbose"]

    label_types = ["classification", "object_detection"]

    def __init__(
        self,
        channels_first: bool = False,
        label_type: str = "classification",
        pad_mode: str = "constant",
        pad_kwargs: Optional[Dict[str, Any]] = None,
        clip_values: Optional["CLIP_VALUES_TYPE"] = None,
        apply_fit: bool = True,
        apply_predict: bool = False,
        verbose: bool = False,
    ):
        """
        Create an instance of SquarePad.

        :param height: The height of the resized image.
        :param width: The width of the resized image.
        :param channels_first: Set channels first or last.
        :param label_type: String defining the label type. Currently supported: `classification`, `object_detection`
        :param pad_mode: String defining the padding method. Supported by options in the `np.pad` function.
        :param pad_kwargs: A dictionary of additional keyword arguments used by the `np.pad` function.
        :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
               for features.
        :param apply_fit: True if applied during fitting/training.
        :param apply_predict: True if applied during predicting.
        :param verbose: Show progress bars.
        """
        super().__init__(is_fitted=True, apply_fit=apply_fit, apply_predict=apply_predict)
        self.channels_first = channels_first
        self.label_type = label_type
        self.pad_mode = pad_mode
        self.pad_kwargs = pad_kwargs if pad_kwargs is not None else {}
        self.clip_values = clip_values
        self.verbose = verbose
        self._check_params()

    def __call__(  # type: ignore
        self, x: Union[np.ndarray, List[np.ndarray]], y: Optional[Union[np.ndarray, List[Dict[str, np.ndarray]]]] = None
    ) -> Tuple[Union[np.ndarray, List[np.ndarray]], Optional[Union[np.ndarray, List[Dict[str, np.ndarray]]]]]:
        """
        Square pad `x` and adjust bounding boxes for labels `y` accordingly.

        :param x: Input samples.
        :param y: Label of the samples `x`.
        :return: Transformed samples and labels.
        """
        x_preprocess = []
        y_preprocess: Optional[Union[np.ndarray, List[Dict[str, np.ndarray]]]]
        if y is not None and self.label_type == "object_detection":
            y_preprocess = []
        else:
            y_preprocess = y

        for i, x_i in enumerate(tqdm(x, desc="ImageSquarePad", disable=not self.verbose)):
            if self.channels_first:
                x_i = np.transpose(x_i, (1, 2, 0))

            # Calculate padding
            height, width, _ = x_i.shape
            if height > width:
                pad_top = 0
                pad_bottom = 0
                pad_left = int(np.floor((height - width) / 2))
                pad_right = int(np.ceil((height - width) / 2))
            else:
                pad_top = int(np.floor((width - height) / 2))
                pad_bottom = int(np.ceil((width - height) / 2))
                pad_left = 0
                pad_right = 0

            # Pad image to square size
            padding = [[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]]
            x_pad = np.pad(x_i, padding, mode=self.pad_mode, **self.pad_kwargs)  # type: ignore

            if self.channels_first:
                x_pad = np.transpose(x_pad, (2, 0, 1))

            if self.clip_values is not None:
                x_pad = np.clip(x_pad, self.clip_values[0], self.clip_values[1])

            x_preprocess.append(x_pad)

            if y is not None and self.label_type == "object_detection":
                # Copy labels
                y_pad = {k: v.copy() for k, v in y[i].items()}

                # Shift bounding boxes
                y_pad["boxes"][:, 0] += pad_left
                y_pad["boxes"][:, 1] += pad_top
                y_pad["boxes"][:, 2] += pad_left
                y_pad["boxes"][:, 3] += pad_top

                y_preprocess.append(y_pad)  # type: ignore

        if isinstance(x, np.ndarray):
            return np.stack(x_preprocess, axis=0), y_preprocess

        return x_preprocess, y_preprocess

    def _check_params(self) -> None:
        if self.clip_values is not None:
            if len(self.clip_values) != 2:
                raise ValueError("`clip_values` should be a tuple of 2 floats containing the allowed data range.")

            if self.clip_values[0] >= self.clip_values[1]:
                raise ValueError("Invalid `clip_values`: min >= max.")
