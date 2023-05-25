"""
BSD 3-Clause License

Copyright (c) 2018, Devon Hjelm
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
# Modified work Copyright 2020 Alex Fedorov

import abc
import math

import torch
import torch.nn.functional as F
from torch import Tensor


def log_sum_exp(x: Tensor, axis: int = 0):
    """Log sum exp function

    Args:
        x: Input.
        axis: Axis over which to perform sum.
        
    Returns:
        torch.Tensor: log sum exp
    """
    x_max = torch.max(x, axis)[0]
    y = torch.log((torch.exp(x - x_max)).sum(axis)) + x_max
    return y


class ABaseMeasure(abc.ABC):
    def __init__(self, average: bool = False):
        self._average = average

    @abc.abstractmethod
    def get_positive_expectation(self, p: Tensor) -> Tensor:
        pass

    @abc.abstractmethod
    def get_negative_expectation(self, q: Tensor) -> Tensor:
        pass

    def _if_average(self, e: Tensor) -> Tensor:
        return e.mean() if self._average else e


class GanMeasure(ABaseMeasure):
    def get_positive_expectation(self, p: Tensor) -> Tensor:
        Ep = -F.softplus(-p)
        return self._if_average(Ep)

    def get_negative_expectation(self, q: Tensor) -> Tensor:
        Eq = F.softplus(-q) + q
        return self._if_average(Eq)


class JsdMeasure(ABaseMeasure):
    def get_positive_expectation(self, p: Tensor) -> Tensor:
        Ep = math.log(2.0) - F.softplus(-p)
        return self._if_average(Ep)

    def get_negative_expectation(self, q: Tensor) -> Tensor:
        Eq = F.softplus(-q) + q - math.log(2.0)
        return self._if_average(Eq)


class X2Measure(ABaseMeasure):
    def get_positive_expectation(self, p: Tensor) -> Tensor:
        Ep = p ** 2
        return self._if_average(Ep)

    def get_negative_expectation(self, q: Tensor) -> Tensor:
        Eq = -0.5 * ((torch.sqrt(q ** 2) + 1.0) ** 2)
        return self._if_average(Eq)


class KLMeasure(ABaseMeasure):
    def get_positive_expectation(self, p: Tensor) -> Tensor:
        Ep = p
        return self._if_average(Ep)

    def get_negative_expectation(self, q: Tensor) -> Tensor:
        Eq = torch.exp(q - 1.0)
        return self._if_average(Eq)


class RKLMeasure(ABaseMeasure):
    def get_positive_expectation(self, p: Tensor) -> Tensor:
        Ep = -torch.exp(-p)
        return self._if_average(Ep)

    def get_negative_expectation(self, q: Tensor) -> Tensor:
        Eq = q - 1.0
        return self._if_average(Eq)


class DVMeasure(ABaseMeasure):
    def get_positive_expectation(self, p: Tensor) -> Tensor:
        Ep = p
        return self._if_average(Ep)

    def get_negative_expectation(self, q: Tensor) -> Tensor:
        Eq = log_sum_exp(q, 0) - math.log(q.size(0))
        return self._if_average(Eq)


class H2Measure(ABaseMeasure):
    def get_positive_expectation(self, p: Tensor) -> Tensor:
        Ep = 1.0 - torch.exp(-p)
        return self._if_average(Ep)

    def get_negative_expectation(self, q: Tensor) -> Tensor:
        Eq = torch.exp(q) - 1.0
        return self._if_average(Eq)


class W1Measure(ABaseMeasure):
    def get_positive_expectation(self, p: Tensor) -> Tensor:
        Ep = p
        return self._if_average(Ep)

    def get_negative_expectation(self, q: Tensor) -> Tensor:
        Eq = q
        return self._if_average(Eq)
