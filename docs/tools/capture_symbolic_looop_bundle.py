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
"""EXPERIMENTAL: demonstrates a symbolic (runtime-valued) loop trip count
surviving into generated bundle.mlir.

This bypasses the entire real pipeline (WSR, work division, torch.compile) --
nothing today actually produces a symbolic LoopSpec.count end to end (see
docs/wsr-notes.md section 9 for the full gap list). Instead this hand-builds
a LoopSpec directly and calls generate_bundle() on it, the same style
tests/inductor/test_symbolic_dim_bundle.py uses to unit-test bundle.py in
isolation, with compile_op_spec mocked so no Spyre hardware / backend
compiler is required.

Usage:
    python3 docs/tools/capture_symbolic_loop_bundle.py
"""

import os
import sys
import tempfile
from unittest.mock import patch

import sympy

# import torch FIRST, standalone, before anything under torch_spyre.* --
# torch_spyre/__init__.py's own first line is `import torch`, which triggers
# PyTorch's backend-autoload machinery for torch_spyre itself; importing a
# torch_spyre submodule before torch has been imported on its own trips a
# circular import (torch_spyre mid-import when autoload tries to use it).
import torch  # noqa: E402,F401

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "tests", "inductor")
)

from torch_spyre._inductor.codegen.bundle import generate_bundle  # noqa: E402
from torch_spyre._inductor.op_spec import LoopSpec, OpSpec  # noqa: E402

from test_symbolic_dim_bundle import _make_sdsc_json  # noqa: E402


def _minimal_op_spec() -> OpSpec:
    """A stub OpSpec whose content is irrelevant (compile_op_spec is mocked)."""
    return OpSpec(
        op="add",
        is_reduction=False,
        iteration_space={},
        args=[],
        op_info={},
    )


def main() -> None:
    # The symbolic loop count: stand-in for "R / G" (runtime batch size
    # divided by the fixed tile granularity) from the Option-B design.
    # Nothing here computes it from a real mark_dynamic bound -- it's a
    # bare sympy.Symbol to prove the codegen layer accepts *some* non-integer
    # Expr as LoopSpec.count without raising NotImplementedError.
    num_tiles = sympy.Symbol("num_tiles")

    inner_op = _minimal_op_spec()
    loop = LoopSpec(count=num_tiles, body=[inner_op])

    # One compiled entry per OpSpec in body, in the same order generate_bundle
    # will walk them: (sdsc_json, symbol_values, affine_strides, symbol_kinds).
    # No dimension/kernel symbols on the op itself -- keep the demo focused on
    # the loop-bound symbol alone.
    compiled_entry = (_make_sdsc_json(dim_sym_ids={}), [], [], [])

    with tempfile.TemporaryDirectory() as output_dir:
        with patch(
            "torch_spyre._inductor.codegen.bundle.compile_op_spec",
            return_value=compiled_entry,
        ):
            generate_bundle("symbolic_loop_demo", output_dir, [loop], use_symbols=False)

        with open(os.path.join(output_dir, "bundle.mlir")) as f:
            print(f.read())


if __name__ == "__main__":
    main()
