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

from typing import Sequence, Union

import sympy

from torch._inductor.utils import IndentedBuffer
from torch._inductor.utils import (
    get_kernel_metadata,
    get_fused_kernel_name,
    sympy_product,
)
from torch._inductor.scheduler import (
    BaseScheduling,
    BaseSchedulerNode,
    FusedSchedulerNode,
    SchedulerNode,
)
from torch._inductor.virtualized import V
from torch._inductor.codecache import code_hash
from torch.utils._ordered_set import OrderedSet

from .spyre_kernel import SpyreKernel
from .pass_utils import iteration_space


def _compute_max_input_sizes(kernel) -> list[list[int]]:
    """
    Return max shape for each kernel arg

    List of ints: per-dim overrides for an input with at least one symbolic dim;
      -1 marks a static dimension, a positive int is the ShapeEnv upper bound.
    """
    result: list[list[int]] = []
    shape_env = V.graph.sizevars.shape_env
    input_names = {name for name, targ in kernel.spyre_kernel_args if targ.is_input}

    for name in kernel.args.python_argdefs()[1]:
        if name not in input_names:
            result.append([])
            continue

        buf = V.graph.get_buffer(name)
        if buf is None:
            result.append([])
            continue

        layout = buf.get_layout()
        max_shape: list[int] = []
        has_symbolic = False
        for s in layout.size:
            if isinstance(s, (int, sympy.Integer)):
                max_shape.append(-1)
            elif hasattr(s, "free_symbols") and s.free_symbols:
                has_symbolic = True
                if shape_env is not None and hasattr(shape_env, "bound_sympy"):
                    vr = shape_env.bound_sympy(s)
                    # Use int() only when the bound is a concrete sympy.Integer.
                    # if isinstance(vr.upper, sympy.Integer):
                    if (
                        hasattr(vr, "upper")
                        and isinstance(vr.upper, sympy.Integer)
                        and int(vr.upper) > 0
                    ):
                        max_val = int(vr.upper)
                    else:
                        max_val = V.graph.sizevars.size_hint(s)
                else:
                    max_val = V.graph.sizevars.size_hint(s)
                max_shape.append(max_val)
            else:
                max_shape.append(-1)

        result.append(max_shape if has_symbolic else [])

    return result


class SuperDSCScheduling(BaseScheduling):
    def group_fn(self, sizes):
        """
        Process the iteration sizes in case a transformation needs to be applied.
        """
        return tuple(V.graph.sizevars.simplify(sympy_product(s)) for s in sizes)

    def flush(self):
        """
        Flush the generated kernel and python wrapper code to the source code file.
        """
        # Overrides superclass method that raises NotImplementedError.
        pass

    def can_buffer_be_removed_through_fusion(
        self, name: str, fused_node_names: OrderedSet[str]
    ) -> bool:
        """
        Spyre currently needs intermediate buffers to be allocated even if only used within a single Kernel.
        TODO: Revisit this as part of https://github.com/torch-spyre/torch-spyre/issues/1266
        """
        return False

    def can_fuse_vertical(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        """
        Check whether node1 and node2 can be vertically fused or not.
        """
        # TODO: Revisit this as part of https://github.com/torch-spyre/torch-spyre/issues/826
        return False

    def can_fuse_horizontal(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        """
        Check whether node1 and node2 can be horizontally fused or not.
        """
        # TODO: Revisit this as part of https://github.com/torch-spyre/torch-spyre/issues/826
        return False

    def generate_node_schedule(self, nodes: Sequence[SchedulerNode]):
        node_schedule: list[SchedulerNode] = []
        done = OrderedSet[BaseSchedulerNode]()
        for node in nodes:
            if node in done:
                continue
            done.add(node)
            if isinstance(node, SchedulerNode):
                node_schedule.append(node)
            else:
                raise RuntimeError(f"Unexpected node type: {type(node)}")
        return node_schedule

    def codegen_node(self, node: Union[FusedSchedulerNode, SchedulerNode]) -> None:
        """
        Generate a kernel given a list of pre-fused nodes.
        """
        assert self.scheduler
        nodes = [
            node
            for node in node.get_nodes()
            if node.get_name() not in self.scheduler.removed_ops
        ]
        if len(nodes) == 0:
            return

        node_schedule = self.generate_node_schedule(nodes)
        kernel = SpyreKernel()
        with kernel:
            for node in node_schedule:
                var_ranges = iteration_space(node)
                vars = list(var_ranges.keys())
                index_vars = [
                    vars[: len(node._body.iter_vars)],
                    vars[len(node._body.iter_vars) :],
                ]
                node.codegen(index_vars)
        # TODO: compute max shapes here (before codegen_kernel) and pass into
        # codegen_kernel → create_op_spec so mb_/out_ in the SDSC use max values.
        with V.set_kernel_handler(kernel):
            src_code = kernel.codegen_kernel()
        kernel_name = self.define_kernel(src_code, node_schedule, kernel)
        kernel.kernel_name = kernel_name
        kernel.code_hash = code_hash(src_code)

        with V.set_kernel_handler(kernel):
            for node in node_schedule:
                node.mark_run()

        self.codegen_comment(node_schedule, kernel_name)
        kernel.call_kernel(kernel.kernel_name)

        V.graph.removed_buffers |= kernel.removed_buffers
        V.graph.inplaced_to_remove |= kernel.inplaced_to_remove

        self.free_buffers_in_scheduler()

    def define_kernel(self, src_code, node_schedule, kernel):
        """
        Codegen kernel definition to go in output wrapper code
        """
        wrapper = V.graph.wrapper_code
        if src_code in wrapper.src_to_kernel:
            kernel_name = wrapper.src_to_kernel[src_code]
        else:
            fused_name = get_fused_kernel_name(node_schedule, "original_aten")
            kernel_name = "_".join(["sdsc", fused_name, wrapper.next_kernel_suffix()])
            wrapper.src_to_kernel[src_code] = kernel_name
            # Compute max host shapes for each input -- needed for symbolic shapes
            max_input_sizes = _compute_max_input_sizes(kernel)
            buf = IndentedBuffer()
            buf.writeline(f"async_compile.sdsc('{kernel_name}',")
            with buf.indent():
                buf.splice(f"{src_code}")
                buf.writeline(f", max_input_sizes={max_input_sizes!r}")
            buf.writeline(")")
            origins, detailed_origins = get_kernel_metadata(node_schedule, wrapper)
            metadata_comment = f"{origins}\n{detailed_origins}"
            wrapper.define_kernel(kernel_name, buf.getvalue(), metadata_comment)

        return kernel_name
