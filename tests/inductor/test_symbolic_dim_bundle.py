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

"""Unit tests for bundle.py dimension-symbol support (mark_dynamic path).

These tests cover the 8 changes made to bundle.py:
  1. symbol_kinds + sym_idx_to_dim_origin built unconditionally
  2. Dedup dimension symbols by pytorch_sym + build dim_param_names
  3. Function signature includes input_arg<index, granularity=G, max_value=M> params
  4. sym_canonical wired with canonical + duplicate dim symbol indices
  5. arith.constant skipped for dimension symbols (continue branch)
  6. _extract_symbol_ids scans dimToSymbolMapping_ in addition to scheduleTree_
  7. _resolve_sym consults sym_canonical without gating on symbolic_args
  8. sdsc_execute emission keyed on ``if symbol_ids:`` not ``if use_symbols:``

compile_op_spec is mocked throughout so no Spyre hardware or torch.compile
infrastructure is required.
"""

import os
import tempfile
from unittest.mock import patch

from torch._inductor.test_case import TestCase as InductorTestCase

from torch_spyre._inductor.codegen.bundle import (
    _extract_symbol_ids,
    generate_bundle,
)
from torch_spyre._inductor.codegen.compute_ops import SymbolKind
from torch_spyre._inductor.op_spec import OpSpec


# ---------------------------------------------------------------------------
# Helpers shared across test classes
# ---------------------------------------------------------------------------


def _make_sdsc_json(
    sdsc_idx: int = 0,
    *,
    dim_sym_ids: dict[str, list[int]] | None = None,
    hbm_sym_ids_per_core: dict[str, int] | None = None,
    num_cores: int = 1,
) -> dict:
    """Minimal SDSC JSON for testing _extract_symbol_ids and generate_bundle.

    Args:
        sdsc_idx: Used to form the top-level JSON key, e.g. "0_fused_test".
        dim_sym_ids: Mapping of dim label → list of symbol IDs for
            dimToSymbolMapping_, e.g. {"mb": [-1]}.
        hbm_sym_ids_per_core: Mapping of core key → symbol ID for
            startAddressCoreCorelet_.data_, e.g. {"[0, 0, 0]": -2}.
        num_cores: Value for numCoresUsed_.
    """
    schedule_tree: list = []
    if hbm_sym_ids_per_core:
        schedule_tree = [
            {
                "component_": "hbm",
                "startAddressCoreCorelet_": {"data_": hbm_sym_ids_per_core},
            }
        ]
    return {
        f"{sdsc_idx}_fused_test": {
            "numCoresUsed_": num_cores,
            "dscs_": [
                {
                    "op": {
                        "dimToSymbolMapping_": dim_sym_ids or {},
                        "scheduleTree_": schedule_tree,
                    }
                }
            ],
        }
    }


def _minimal_op_spec() -> OpSpec:
    """A stub OpSpec whose content is irrelevant (compile_op_spec is mocked)."""
    return OpSpec(
        op="gelu",
        is_reduction=False,
        iteration_space={},
        args=[],
        op_info={},
    )


# ---------------------------------------------------------------------------
# Tests for _extract_symbol_ids (Change 6)
# ---------------------------------------------------------------------------


class TestExtractSymbolIds(InductorTestCase):
    """Pure unit tests for _extract_symbol_ids — the extended parser that
    scans dimToSymbolMapping_ before scheduleTree_."""

    def test_dim_only(self):
        """Dimension symbol ID from dimToSymbolMapping_ is returned."""
        sdsc_json = _make_sdsc_json(dim_sym_ids={"mb": [-1]})
        self.assertEqual(_extract_symbol_ids(sdsc_json), [-1])

    def test_hbm_only(self):
        """HBM address symbol ID from scheduleTree_ is returned."""
        sdsc_json = _make_sdsc_json(hbm_sym_ids_per_core={"[0, 0, 0]": -2})
        self.assertEqual(_extract_symbol_ids(sdsc_json), [-2])

    def test_dim_before_hbm_ordering(self):
        """Dimension IDs appear before HBM address IDs in the output list."""
        sdsc_json = _make_sdsc_json(
            dim_sym_ids={"mb": [-1]},
            hbm_sym_ids_per_core={"[0, 0, 0]": -2},
        )
        self.assertEqual(_extract_symbol_ids(sdsc_json), [-1, -2])

    def test_positive_values_excluded(self):
        """Positive values (concrete HBM addresses) are not collected."""
        sdsc_json = _make_sdsc_json(hbm_sym_ids_per_core={"[0, 0, 0]": 0x400000000})
        self.assertEqual(_extract_symbol_ids(sdsc_json), [])

    def test_deduplication(self):
        """The same symbol ID appearing in both locations is collected once."""
        sdsc_json = {
            "0_fused_test": {
                "numCoresUsed_": 1,
                "dscs_": [
                    {
                        "op": {
                            "dimToSymbolMapping_": {"mb": [-1]},
                            "scheduleTree_": [
                                {
                                    "component_": "hbm",
                                    "startAddressCoreCorelet_": {
                                        "data_": {"[0, 0, 0]": -1}
                                    },
                                }
                            ],
                        }
                    }
                ],
            }
        }
        self.assertEqual(_extract_symbol_ids(sdsc_json), [-1])

    def test_empty_json(self):
        """Empty SDSC JSON returns an empty list without errors."""
        self.assertEqual(_extract_symbol_ids({}), [])

    def test_multiple_dim_labels(self):
        """Multiple entries in dimToSymbolMapping_ are all collected."""
        sdsc_json = _make_sdsc_json(dim_sym_ids={"mb": [-1], "nb": [-2]})
        self.assertEqual(_extract_symbol_ids(sdsc_json), [-1, -2])


# ---------------------------------------------------------------------------
# Tests for generate_bundle with dimension symbols (Changes 1–5, 7–8)
# ---------------------------------------------------------------------------


class TestGenerateBundleDimensionSymbols(InductorTestCase):
    """Integration tests for generate_bundle that verify the bundle.mlir
    content produced when mark_dynamic dimension symbols are present.

    compile_op_spec is mocked so these tests run without Spyre hardware.
    """

    def setUp(self):
        super().setUp()
        self._tmpdir = tempfile.TemporaryDirectory()
        self.output_dir = self._tmpdir.name

    def tearDown(self):
        self._tmpdir.cleanup()
        super().tearDown()

    def _read_bundle(self) -> str:
        with open(os.path.join(self.output_dir, "bundle.mlir")) as f:
            return f.read()

    def _run_bundle(self, compiled_entries, op_specs=None):
        """Call generate_bundle with mocked compile_op_spec.

        compiled_entries: list of (sdsc_json, local_sym_values, affine_strides,
            local_symbol_kinds) tuples, one per OpSpec.
        op_specs: list of OpSpec stubs (defaults to one per entry).
        """
        if op_specs is None:
            op_specs = [_minimal_op_spec() for _ in compiled_entries]

        side_effects = list(compiled_entries)
        with patch(
            "torch_spyre._inductor.codegen.bundle.compile_op_spec",
            side_effect=side_effects,
        ):
            generate_bundle(
                "test",
                self.output_dir,
                op_specs,
                unroll_loops=False,
                symbolic_args=False,  # HBM addresses NOT symbolized
            )
        return self._read_bundle()

    # ------------------------------------------------------------------
    # Change 1 + 3: function signature gains input_arg param for dim sym
    # ------------------------------------------------------------------

    def test_single_dim_sym_function_signature(self):
        """@sdsc_bundle gains an input_arg<index, granularity=G, max_value=M>
        parameter for a dimension symbol even when symbolic_args=False."""
        dim_kind = SymbolKind.dimension(granularity=56, max_value=616, pytorch_sym="s0")
        entry = (_make_sdsc_json(dim_sym_ids={"mb": [-1]}), [0], [], [dim_kind])

        bundle = self._run_bundle([entry])

        self.assertIn(
            "%sym_0_1_base: !sdscbundle.input_arg<index, granularity=56, max_value=616>",
            bundle,
        )

    # ------------------------------------------------------------------
    # Change 3: input_arg_extract op produces plain index SSA name
    # ------------------------------------------------------------------

    def test_single_dim_sym_extract_op(self):
        """input_arg_extract converts %sym_0_1_base to plain index %sym_0_1."""
        dim_kind = SymbolKind.dimension(granularity=56, max_value=616, pytorch_sym="s0")
        entry = (_make_sdsc_json(dim_sym_ids={"mb": [-1]}), [0], [], [dim_kind])

        bundle = self._run_bundle([entry])

        self.assertIn(
            "%sym_0_1 = sdscbundle.input_arg_extract value from %sym_0_1_base"
            " : !sdscbundle.input_arg<index, granularity=56, max_value=616> -> index",
            bundle,
        )

    # ------------------------------------------------------------------
    # Changes 7 + 8: sdsc_execute lists the extracted SSA value as operand
    # ------------------------------------------------------------------

    def test_single_dim_sym_sdsc_execute_operand(self):
        """sdsc_execute receives %sym_0_1 as its operand and -1 as symbol_id."""
        dim_kind = SymbolKind.dimension(granularity=56, max_value=616, pytorch_sym="s0")
        entry = (_make_sdsc_json(dim_sym_ids={"mb": [-1]}), [0], [], [dim_kind])

        bundle = self._run_bundle([entry])

        self.assertIn('sdscbundle.sdsc_execute (%sym_0_1)', bundle)
        self.assertIn('"symbol_ids"=[-1]', bundle)

    # ------------------------------------------------------------------
    # Change 5: no arith.constant is emitted for a dimension symbol
    # ------------------------------------------------------------------

    def test_no_arith_constant_for_dim_sym(self):
        """Dimension symbols are replaced by function params — no arith.constant."""
        dim_kind = SymbolKind.dimension(granularity=56, max_value=616, pytorch_sym="s0")
        entry = (_make_sdsc_json(dim_sym_ids={"mb": [-1]}), [0], [], [dim_kind])

        bundle = self._run_bundle([entry])

        self.assertNotIn("arith.constant", bundle)

    # ------------------------------------------------------------------
    # Change 2 + 4: duplicate pytorch_sym yields a single canonical param
    # ------------------------------------------------------------------

    def test_duplicate_pytorch_sym_single_param(self):
        """Two SDSCs sharing the same pytorch_sym produce one function param.

        Both sdsc_execute calls must reference the canonical SSA name %sym_0_1.
        """
        dim_kind_0 = SymbolKind.dimension(
            granularity=56, max_value=616, pytorch_sym="s0"
        )
        dim_kind_1 = SymbolKind.dimension(
            granularity=56, max_value=616, pytorch_sym="s0"
        )
        # SDSC 0 owns global sym_id -1; SDSC 1 (offset=1) owns global sym_id -2.
        entry_0 = (
            _make_sdsc_json(sdsc_idx=0, dim_sym_ids={"mb": [-1]}),
            [0],
            [],
            [dim_kind_0],
        )
        entry_1 = (
            _make_sdsc_json(sdsc_idx=1, dim_sym_ids={"mb": [-2]}),
            [0],
            [],
            [dim_kind_1],
        )

        bundle = self._run_bundle([entry_0, entry_1])

        # Exactly one !sdscbundle.input_arg param (in signature) and one extract op.
        self.assertEqual(
            bundle.count(
                "!sdscbundle.input_arg<index, granularity=56, max_value=616>"
            ),
            2,  # 1 in param list + 1 in input_arg_extract
        )
        # Both SDSC calls resolve to the canonical %sym_0_1.
        self.assertEqual(bundle.count("sdscbundle.sdsc_execute (%sym_0_1)"), 2)

    # ------------------------------------------------------------------
    # Change 2: distinct pytorch_syms produce separate params
    # ------------------------------------------------------------------

    def test_two_distinct_pytorch_syms_two_params(self):
        """Two different pytorch_syms produce two independent function params."""
        dim_s0 = SymbolKind.dimension(granularity=56, max_value=616, pytorch_sym="s0")
        dim_s1 = SymbolKind.dimension(granularity=32, max_value=256, pytorch_sym="s1")
        entry = (
            _make_sdsc_json(dim_sym_ids={"mb": [-1], "nb": [-2]}),
            [0, 0],
            [],
            [dim_s0, dim_s1],
        )

        bundle = self._run_bundle([entry])

        self.assertIn(
            "%sym_0_1_base: !sdscbundle.input_arg<index, granularity=56, max_value=616>",
            bundle,
        )
        self.assertIn(
            "%sym_0_2_base: !sdscbundle.input_arg<index, granularity=32, max_value=256>",
            bundle,
        )
        self.assertIn("%sym_0_1 = sdscbundle.input_arg_extract", bundle)
        self.assertIn("%sym_0_2 = sdscbundle.input_arg_extract", bundle)
        self.assertIn("sdscbundle.sdsc_execute (%sym_0_1, %sym_0_2)", bundle)

    # ------------------------------------------------------------------
    # Change 8 (negative): no dim sym → no operands, no params
    # ------------------------------------------------------------------

    def test_no_dim_sym_empty_signature(self):
        """Without dimension symbols @sdsc_bundle() has no params, no extract ops,
        and sdsc_execute has no operands."""
        entry = (_make_sdsc_json(), [], [], [])

        bundle = self._run_bundle([entry])

        self.assertIn("func.func @sdsc_bundle()", bundle)
        self.assertNotIn("input_arg", bundle)
        self.assertIn("sdscbundle.sdsc_execute ()", bundle)
