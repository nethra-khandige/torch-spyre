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

"""EXPERIMENTAL: unit tests for symbolic DimHint.split_count support.

Covers the small, mechanical, additive patch applied to
propagate_hints.py's DimHint and wsr/coarse_tile_hints.py's _hints_levels
so that a symbolic split_count (e.g. floor(R/G), standing in for
num_tiles = R // G in the phase-1 dynamic-batch design) survives instead
of crashing with a TypeError.

This does NOT test the full symbolic-loop-unfolding feature end to end --
nothing today actually produces a symbolic split_count from a real
mark_dynamic-annotated torch.compile() call (WSR still bails out on
symbolic dims). See docs/wsr-notes.md section 9 and
docs/hbm-allocation-tile-movement-report.md for the full design and the
remaining gap list. This file only tests the one layer patched so far:
_hints_levels tolerating a symbolic split_count handed to it directly.

Uses MagicMock stand-ins for ComputedBuffer (same pattern as
TestHintsLevels in test_coarse_tiling.py) so no torch.compile or Spyre
hardware is required -- only DimHint + _hints_levels.
"""

import unittest
from unittest.mock import MagicMock

import sympy

from torch._inductor.ir import ComputedBuffer

from torch_spyre._inductor.propagate_hints import DimHint
from torch_spyre._inductor.wsr.coarse_tile_hints import _hints_levels


def _make_op(hints):
    """Return a fake ComputedBuffer with the given DimHint list.

    hints: list of (hint_id, split_count, loop_var) tuples. split_count may
    be a plain int (the ordinary, static-shape case) or a sympy.Expr with
    free symbols (the EXPERIMENTAL symbolic case).
    """
    op = MagicMock(spec=ComputedBuffer)
    op.get_name.return_value = "buf0"
    op.dim_hints = [
        DimHint(
            dim_names=[f"dim{i}"],
            split_count=sc,
            loop_var=lv,
            is_reduction=False,
            hint_id=hid,
        )
        for i, (hid, sc, lv) in enumerate(hints)
    ]
    return op


class TestSymbolicSplitCount(unittest.TestCase):
    """split_count = floor(R/G): a stand-in for the real dynamic-batch
    formula (R = mark_dynamic-bound runtime batch size, G = spyre_hint
    tile size), used here without depending on the WSR path that would
    eventually compute it for real.
    """

    def test_symbolic_split_count_not_dropped_as_size1(self):
        """A symbolic split_count must not be treated as the size-1 no-op
        case, and must survive being appended to levels.

        Regression coverage: sympy.Integer(floor(R/64)) raises TypeError --
        confirmed directly (see conversation) -- so the original
        `levels.append((h.hint_id, sympy.Integer(h.split_count)))` line
        would have crashed even in this single-op, no-merge case.
        """
        c0 = sympy.Symbol("c0")
        r = sympy.Symbol("R", positive=True, integer=True)
        symbolic_count = sympy.floor(r / 64)  # stand-in for num_tiles = R // G
        op = _make_op([(0, symbolic_count, c0)])

        levels = _hints_levels([op])

        self.assertEqual(len(levels), 1)
        hint_id, count = levels[0]
        self.assertEqual(hint_id, 0)
        self.assertEqual(count, symbolic_count)

    def test_symbolic_split_count_survives_merge_across_ops(self):
        """The actual crash scenario: op0's hint for hint_id 0 is the
        degenerate broadcast case (split_count=1, loop_var=None); op1's
        hint for the SAME hint_id is symbolic. This forces the
        `prev.split_count == 1 and h.split_count > 1` comparison in the
        best-hint merge loop -- confirmed to raise TypeError
        ("cannot determine truth value of Relational") before the fix,
        since `symbolic_expr > 1` returns an unevaluated Relational whose
        bool() is undecidable.
        """
        r = sympy.Symbol("R", positive=True, integer=True)
        symbolic_count = sympy.floor(r / 64)
        c0 = sympy.Symbol("c0")
        op0 = _make_op([(0, 1, None)])  # broadcast at op0 for hint_id 0
        op1 = _make_op([(0, symbolic_count, c0)])  # symbolic tile at op1

        levels = _hints_levels([op0, op1])

        self.assertEqual(len(levels), 1)
        hint_id, count = levels[0]
        self.assertEqual(hint_id, 0)
        self.assertEqual(count, symbolic_count)

    def test_symbolic_and_concrete_hints_side_by_side(self):
        """A symbolic hint (dynamic batch dim) alongside an ordinary
        concrete hint (static dim) on the same op -- the expected common
        case for phase-1 dynamic batch size, where only batch is symbolic.
        """
        c0, c1 = sympy.Symbol("c0"), sympy.Symbol("c1")
        r = sympy.Symbol("R", positive=True, integer=True)
        symbolic_count = sympy.floor(r / 64)
        op = _make_op([(0, symbolic_count, c0), (1, 4, c1)])

        levels = _hints_levels([op])

        self.assertEqual(len(levels), 2)
        self.assertEqual(levels[0], (0, symbolic_count))
        self.assertEqual(levels[1], (1, sympy.Integer(4)))

    def test_concrete_split_count_unchanged(self):
        """Sanity check: the ordinary static-shape path is byte-for-byte
        unaffected by the sympify() change (sympify(int) == Integer(int)).
        """
        c0 = sympy.Symbol("c0")
        op = _make_op([(0, 4, c0)])

        levels = _hints_levels([op])

        self.assertEqual(levels, [(0, sympy.Integer(4))])


if __name__ == "__main__":
    unittest.main()
