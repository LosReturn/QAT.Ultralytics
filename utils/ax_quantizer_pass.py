import copy
import dataclasses
import itertools
import operator
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING
import math
import torch
import torch.nn.functional as F
from torch.ao.quantization.fx._decomposed import quantized_decomposed_lib  # noqa: F401
from torch.ao.quantization.pt2e.export_utils import _WrapperModule
from torch.ao.quantization.quantizer import (
    DerivedQuantizationSpec,
    EdgeOrNode,
    QuantizationSpecBase,
    SharedQuantizationSpec,
)
from torch.fx import Graph, GraphModule, Node
from torch.fx.subgraph_rewriter import replace_pattern_with_filters, ReplacedPatterns
from torch.ao.quantization.fake_quantize import FusedMovingAvgObsFakeQuantize, FakeQuantizeBase, _is_per_channel, _is_symmetric_quant, _is_per_tensor
from utils.ax_fakequant import SimpFakeQuantize, SimpPerChannelFakeQuantize

import torch
import torch.nn as nn
import torch.fx as fx
from torch.fx.passes.infra.pass_base import PassBase, PassResult
import functools
import logging

import torch.nn as nn
import torch.nn.functional as F
from torch.fx import symbolic_trace, GraphModule, Node
from torch._ops import ops as aten_ops # 用于识别 aten 操作符

def replace_lsqfakequantize_with_simple_fakequantize(gm:GraphModule):
    nodes_to_remove = []
    for node in list(gm.graph.nodes):
        # Identify the target nodes: call_module nodes whose target starts with '_activation_post_process_'
        if node.op == 'call_module' and node.target.startswith('activation_post_process'):

            original_ap_module_name = node.target
            original_ap_module = getattr(gm, original_ap_module_name)

            if not isinstance(original_ap_module, FakeQuantizeBase):
                continue

            if isinstance(original_ap_module, FusedMovingAvgObsFakeQuantize):
                # print(f"  Skipping replacement for {node.name} (target: {node.target}) as it's already a MockFakeQuantize.")
                continue # Move to the next node in the graph
            # 1. Create the new module instance
            new_ap_module_name = f"{original_ap_module_name}_custom"

            quant_min = original_ap_module.activation_post_process.quant_min
            quant_max = original_ap_module.activation_post_process.quant_max
            scale,zp = original_ap_module.calculate_qparams()
            is_per_channel = _is_per_channel(original_ap_module.activation_post_process.qscheme)
            ch_axis = original_ap_module.ch_axis
            is_symmetric=_is_symmetric_quant(original_ap_module.activation_post_process.qscheme)
            dtype=original_ap_module.activation_post_process.dtype
            qscheme=original_ap_module.activation_post_process.qscheme
            with torch.no_grad():
                if is_per_channel:
                    new_ap_module = SimpPerChannelFakeQuantize(quant_max=quant_max,quant_min=quant_min,ch_axis=ch_axis,is_symmetric_quant=is_symmetric,dtype=dtype,qscheme=qscheme) # Set your custom parameter
                else:
                    new_ap_module = SimpFakeQuantize(quant_max=quant_max,quant_min=quant_min,is_symmetric_quant=is_symmetric,dtype=dtype,qscheme=qscheme) # Set your custom parameter
            device = scale.device
            new_ap_module.scale = scale
            new_ap_module.zero_point = zp.to(torch.int)
            new_ap_module.scale.to(device)
            new_ap_module.zero_point.to(device)
            # 3. Add the new module to the qat_model's submodules
            # This makes the new module accessible by name in the graph
            new_ap_module.to(scale.device)
            gm.add_module(new_ap_module_name, new_ap_module)

            # 4. Create the new node in the graph
            # Insert it at the same position as the original node
            # `inserting_before(node)` is generally good for replacement
            with gm.graph.inserting_before(node):
                # The new node calls the newly added module
                # It takes the same arguments as the original node
                new_node = gm.graph.call_module(new_ap_module_name, node.args, node.kwargs)
                new_node.meta = node.meta # Copy metadata if important for export/debug

            # 5. Redirect all uses of the old node to the new node
            # This is the core of "keeping same position" in the data flow
            node.replace_all_uses_with(new_node)

            # 6. Mark the original node for removal
            nodes_to_remove.append(node)

    # 7. Remove the old nodes from the graph
    for node in nodes_to_remove:
        gm.graph.erase_node(node)

    # 8. Recompile the modified graph
    gm.recompile()
