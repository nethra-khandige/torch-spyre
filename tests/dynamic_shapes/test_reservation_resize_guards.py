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

"""Tests for the reserved_dims map and the resize_ pinning guard (N2, N7).

Covers the on-tensor storage (SpyreTensorImpl::reserved_dims) from the
runtime side: the buffer is actually sized for `max`, a resize within
[min, max] on a granularity multiple never reallocates, and a resize
outside any of the three bounds refuses loudly rather than silently
reallocating or running with a stale tail.
"""

import torch
from torch.testing._internal.common_utils import run_tests, TestCase

import torch_spyre  # noqa: F401

DEVICE = torch.device("spyre")

# 9 buckets (630 / 70), well under N4's max_buckets cap of 32.
DIM_MIN = 70
DIM_MAX = 630
GRANULARITY = 70


class TestReservationResizeGuard(TestCase):
    def setUp(self) -> None:
        self.x = torch.rand(560, 1024, dtype=torch.float16)
        self.x_dev = self.x.to(
            DEVICE,
            dynamic={0: {"min": DIM_MIN, "max": DIM_MAX, "granularity": GRANULARITY}},
        )

    def test_storage_padded_to_max_not_real_size(self) -> None:
        # device_size[0] at allocation time is DIM_MAX, so storage bytes
        # reflect the ceiling, not the 560 rows actually copied in.
        expected_min_bytes = DIM_MAX * 1024 * self.x_dev.element_size()
        self.assertGreaterEqual(
            self.x_dev.untyped_storage().nbytes(), expected_min_bytes
        )

    def test_logical_size_stays_real(self) -> None:
        # The split this whole feature depends on: the buffer is padded,
        # but what the tensor reports as its own size is not.
        self.assertEqual(self.x_dev.size(0), 560)

    def test_resize_within_bounds_does_not_reallocate(self) -> None:
        before = self.x_dev.untyped_storage().data_ptr()
        nbytes_before = self.x_dev.untyped_storage().nbytes()
        self.x_dev.resize_((140, 1024))  # in [70, 630], multiple of 70
        self.assertEqual(self.x_dev.untyped_storage().data_ptr(), before)
        self.assertEqual(self.x_dev.untyped_storage().nbytes(), nbytes_before)

    def test_resize_below_min_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "outside the reservation"):
            self.x_dev.resize_((40, 1024))  # < DIM_MIN

    def test_resize_above_max_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "outside the reservation"):
            self.x_dev.resize_((700, 1024))  # > DIM_MAX

    def test_resize_off_granularity_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "multiple of"):
            self.x_dev.resize_((141, 1024))  # not a multiple of 70

    def test_resize_changing_a_non_reserved_dim_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "may only change the reserved dim"):
            self.x_dev.resize_((140, 512))


if __name__ == "__main__":
    run_tests()
