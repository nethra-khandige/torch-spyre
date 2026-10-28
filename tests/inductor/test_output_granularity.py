# Copyright 2025 The Torch-Spyre Authors.
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

"""output_granularity: granularity of a symbolic output size.

Cases mirror the formula table in cross_kernel/docs/output_contract_scaling.md.
Input s0 in [128, 512], granularity 64; s1 in [96, 384], granularity 96.
"""

import unittest

import sympy
from torch.utils._sympy.functions import FloorDiv

from torch_spyre._inductor.reserved_dims_contract import output_granularity

s0 = sympy.Symbol("s0", integer=True, positive=True)
s1 = sympy.Symbol("s1", integer=True, positive=True)
CONTRACTS = {s0: (128, 512, 64), s1: (96, 384, 96)}


class TestOutputGranularity(unittest.TestCase):
    def check(self, expr, expected):
        self.assertEqual(output_granularity(expr, CONTRACTS), expected)

    def test_tiled_reduce(self):
        self.check(128 * FloorDiv(s0, 64), 128)

    def test_cat_self(self):
        self.check(2 * s0, 128)

    def test_pad_16_rows(self):
        self.check(s0 + 16, 16)

    def test_drop_first_row(self):
        self.check(s0 - 1, 1)

    def test_every_other_row(self):
        self.check(FloorDiv(s0 + 1, 2), 32)

    def test_every_third_row(self):
        self.check(FloorDiv(s0 + 2, 3), 1)

    def test_cat_two_inputs(self):
        self.check(s0 + s1, 32)

    def test_unchanged_size(self):
        self.check(s0, 64)

    def test_symbol_without_contract_is_one(self):
        self.assertEqual(output_granularity(2 * s0, {}), 1)

    def test_too_many_sizes_is_one(self):
        self.assertEqual(output_granularity(2 * s0, {s0: (1, 100000, 1)}), 1)

    def test_static_size(self):
        self.assertEqual(output_granularity(sympy.Integer(256), CONTRACTS), 256)

    def test_min_not_multiple_of_granularity(self):
        # min 100, G 64 -> first legal size is 128, so s0 + 16 stays 16.
        self.assertEqual(output_granularity(s0 + 16, {s0: (100, 512, 64)}), 16)


if __name__ == "__main__":
    unittest.main()
