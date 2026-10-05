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

"""Tests for the dynamic= declaration validation (N4).

Five rules from PR4326_runtime_spec_v2_for_nethra.md, all enforced at
.to() time before anything is allocated: min/max/granularity are positive
ints, min <= max, max % granularity == 0 (the declared ceiling, distinct
from the pre-existing check on today's real size), min >= the serving-
plugin floor, and max / granularity <= max_buckets. Also covers the
negative-dim canonicalization folded into the same validation pass.
"""

import torch
from torch.testing._internal.common_utils import run_tests, TestCase

import torch_spyre  # noqa: F401
from torch_spyre._C import get_reserved_dims
from torch_spyre._inductor import config as inductor_config

DEVICE = torch.device("spyre")


class TestDynamicDeclarationValidation(TestCase):
    def setUp(self) -> None:
        # Real size at dim 0 is 560, so every valid param set below must
        # keep 560 within [min, max] and 560 % granularity == 0.
        self.x = torch.rand(560, 1024, dtype=torch.float16)

    def _assert_rejected(self, spec, match) -> None:
        with self.assertRaisesRegex(ValueError, match):
            self.x.to(DEVICE, dynamic={0: spec})

    def test_non_int_min_rejected(self) -> None:
        self._assert_rejected(
            {"min": 56.0, "max": 616, "granularity": 8}, "positive int"
        )

    def test_zero_granularity_rejected(self) -> None:
        self._assert_rejected({"min": 56, "max": 616, "granularity": 0}, "positive int")

    def test_negative_max_rejected(self) -> None:
        self._assert_rejected(
            {"min": 56, "max": -616, "granularity": 8}, "positive int"
        )

    def test_min_exceeds_max_rejected(self) -> None:
        self._assert_rejected({"min": 700, "max": 616, "granularity": 8}, "exceeds max")

    def test_max_not_multiple_of_granularity_rejected(self) -> None:
        # The declared ceiling itself must divide evenly -- a different
        # rule from the real-size check below.
        self._assert_rejected(
            {"min": 56, "max": 620, "granularity": 8},
            "not a multiple of granularity",
        )

    def test_min_below_floor_rejected(self) -> None:
        self._assert_rejected(
            {"min": 4, "max": 616, "granularity": 4},
            "minimum supported reservation floor",
        )

    def test_too_many_buckets_rejected(self) -> None:
        over_cap = (inductor_config.max_buckets + 1) * 8
        self._assert_rejected(
            {"min": 8, "max": over_cap, "granularity": 8},
            "exceeding the cap",
        )

    def test_real_size_outside_declared_range_rejected(self) -> None:
        # Pre-existing check, not one of the five declaration rules: the
        # *current* tensor's size must itself be admissible.
        self._assert_rejected(
            {"min": 64, "max": 512, "granularity": 64},  # 560 > 512
            "is not within",
        )

    def test_real_size_off_granularity_rejected(self) -> None:
        # 600 / 100 = 6 buckets, well under the cap, so this clears every
        # declaration rule and only fails on 560 % 100 != 0.
        self._assert_rejected(
            {"min": 100, "max": 600, "granularity": 100},  # 560 % 100 != 0
            "not a multiple of granularity",
        )

    def test_negative_dim_canonicalized(self) -> None:
        # dim=-2 on a 2-D tensor means dim 0. (dim=-1 would resolve to the
        # innermost/stick dim, which is a separate, deliberate restriction
        # -- not what this test is covering.)
        x_dev = self.x.to(
            DEVICE, dynamic={-2: {"min": 70, "max": 630, "granularity": 70}}
        )
        self.assertEqual(
            get_reserved_dims(x_dev),
            {0: {"min": 70, "max": 630, "granularity": 70}},
        )


if __name__ == "__main__":
    run_tests()
