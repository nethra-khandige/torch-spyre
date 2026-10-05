# Copyright 2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests that dynamic= calls mark_dynamic bare, with no min=/max= (N6).

Deliberately eager-only -- no torch.compile here. Compiling a bare
symbolic dim with no for_each_tile loop around it is a known-unsafe
configuration on this backend today (see
new_dir/docs/update_for_vivek.md's N6 finding): without a declared bound,
compilation falls through to a legacy addressing path that dispatches to
hardware instead of refusing. These tests only need to confirm what
mark_dynamic was called with, which doesn't require tracing at all.
"""

import torch
from torch._dynamo.decorators import _DimRange
from torch.testing._internal.common_utils import run_tests, TestCase

import torch_spyre  # noqa: F401

DEVICE = torch.device("spyre")


class TestMarkDynamicBare(TestCase):
    def test_dim_is_marked_dynamic(self) -> None:
        x = torch.rand(560, 1024, dtype=torch.float16)
        x_dev = x.to(DEVICE, dynamic={0: {"min": 70, "max": 630, "granularity": 70}})
        self.assertIn(0, x_dev._dynamo_dynamic_indices)

    def test_mark_dynamic_called_with_no_bound(self) -> None:
        # The actual regression this guards: passing min=/max= to
        # mark_dynamic installs a StrictMinMaxConstraint that collides with
        # the granularity check added downstream (ConstraintViolationError).
        # A bare call records _DimRange(0, None, None); min=70, max=630
        # would record _DimRange(0, 70, 630) instead.
        x = torch.rand(560, 1024, dtype=torch.float16)
        x_dev = x.to(DEVICE, dynamic={0: {"min": 70, "max": 630, "granularity": 70}})
        self.assertIn(_DimRange(0, None, None), x_dev._dynamo_dynamic_range)


if __name__ == "__main__":
    run_tests()
