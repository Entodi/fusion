"""
MIT License

Copyright (c) [2019] [Philip Bachman]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""
# Modified work Copyright 2020 Alex Fedorov


import torch
from torch import Tensor

from fusion.criterion.mi_estimator.critic import ABaseCritic


class SeparableCritic(ABaseCritic):
    def __call__(self, x: Tensor, y: Tensor) -> Tensor:
        self._check(x, y)
        s = torch.mm(x, y.t())
        return s

    @staticmethod
    def _check(x: Tensor, y: Tensor):
        assert len(x.size()) == 2
        assert len(y.size()) == 2
        assert x.size(1) == y.size(1)


class ScaledDotProduct(SeparableCritic):
    def __call__(self, x: Tensor, y: Tensor) -> Tensor:
        s = super().__call__(x, y)
        dim_l = x.size(1)
        s = s / dim_l ** 0.5
        return s


class CosineSimilarity(SeparableCritic):
    def __init__(self, temperature: float = 1.0):
        self._temperature = temperature

    def __call__(self, x: Tensor, y: Tensor) -> Tensor:
        s = super().__call__(x, y)
        x_norm = torch.norm(x, dim=1)
        y_norm = torch.norm(y, dim=1)
        s = s / x_norm
        s = s / y_norm
        s = s / self._temperature
        return s
