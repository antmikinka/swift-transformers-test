import argparse
import numpy as np
import torch
import torch.nn as nn
import coremltools as ct
from transformers import AutoTokenizer, AutoModelForCausalLM

# When using float16, all predicted logits are 0. To be debugge


from coremltools.optimize.torch.palettization import (
    DKMPalettizer,
    DKMPalettizerConfig,
    ModuleDKMPalettizerConfig,
)


def selector(op):
    return op.op_type != "l2_norm"
    

    
compute_precision = ct.transform.FP16ComputePrecision(op_selector=selector)


#compute_precision = ct.precision.FLOAT16
#compute_precision = ct.precision.FLOAT32
#compute_precision = ct.transform.FP16ComputePrecision(op_selector)

#compute_units = ct.ComputeUnit.CPU_ONLY
compute_units = ct.ComputeUnit.CPU_AND_NE





# Fixed sequence length
shape = (1, 128)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_id",
    required=True,
    help="OpenELM checkpoint from the Hub. Example: 'apple/OpenELM-1_1B-Instruct'",
)
parser.add_argument(
    "--output_dir",
    required=True,
    help="Parent folder to save the converted Core ML model",
)
args = parser.parse_args()

model_id = args.model_id
basename = model_id.split("/")[-1]
outpath = f"{args.output_dir}/{basename}-{shape[1]}-{compute_precision}.mlpackage"

print(model_id)
print(outpath)

# OpenELM uses the Llama tokenizer, see https://huggingface.co/apple/OpenELM-270M-Instruct/blob/main/generate_openelm.py#L21.
# It also uses custom code.

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
model.eval()

### palettization stuff ###
# model is already defined as noted above
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
data = create_data()


# Prepare model for palettization
config = DKMPalettizerConfig(global_config=ModuleDKMPalettizerConfig(n_bits=6, weight_threshold = 512, quantize_activations=True))
# the following modules are available for palettization
# torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.Linear, torch.nn.LayerNorm, torch.nn.Embedding, and torch.nn.MultiheadAttention
palettizer = DKMPalettizer(model, config)

prepared_model = palettizer.prepare()

# Fine-tune the model for a few epochs after this.
for inputs, labels in data:
    output = model(inputs)
    loss = loss_fn(output, labels)
    loss.backward()
    optimizer.step()
    palettizer.step()

# prepare for conversion
finalized_model = palettizer.finalize(inplace=True)

# then trace and then ct convert

### ending palettization stuff ###







## palettization may need to be after the code below. idk if it will work but lets try.

inputs = {
    "input_ids": np.random.randint(0, tokenizer.vocab_size, shape),
}

with torch.no_grad():
    t_inputs = {k: torch.tensor(v, dtype=torch.int32) for k, v in inputs.items()}
    outputs = finalized_model(**t_inputs, use_cache=False)

class Wrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.finalized_model = finalized_model
        
    def forward(self, *args, **kwargs):
        input_ids = args[0]
        return self.finalized_model(
            input_ids=input_ids,
            return_dict=False,
            use_cache=False,
            **kwargs
        )








to_jit = Wrapper(finalized_model)
jit_inputs = list(t_inputs.values())
jitted_model = torch.jit.trace(to_jit, jit_inputs)
jitted_model.eval();

with torch.no_grad():
    output_jit = jitted_model(*jit_inputs)

assert torch.allclose(output_jit[0], outputs["logits"])

## Core ML conversion

coreml_input_types = [ct.TensorType(
    name="input_ids",
    shape=ct.Shape(shape=shape),
    dtype=np.int32,
)]
#coreml_output_types = [ct.TensorType(name=name) for name in outputs.keys()]
coreml_output_types = [ct.TensorType(name=name, dtype=np.float32) for name in outputs.keys()]

# Conversion fails with `Conversion for torch.repeat_interleave with non-zero dim has not been implemented`.
# We hack a special case shortcut when the first dim is `1`.

from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil import register_torch_op
from coremltools.converters.mil.frontend.torch.torch_op_registry import _TORCH_OPS_REGISTRY
from coremltools.converters.mil.frontend.torch.ops import _get_inputs

del _TORCH_OPS_REGISTRY["repeat_interleave"]

@register_torch_op
def repeat_interleave(context, node):
    """
    Copy from https://github.com/apple/coremltools/blob/0bef2d6aabd122527cf86cc983c08fb16a4041b5/coremltools/converters/mil/frontend/torch/ops.py#L5174
    plus special case for dim=1 and bs=1
    """
    x, repeats, dim, _ = _get_inputs(context, node, expected=4)

    special_case = dim.val == 1 and x.shape[0] == 1
    if special_case:
        x = mb.reshape(x=x, shape=(x.shape[1:]))

    repeats_val = repeats.val
    if isinstance(repeats_val, np.ndarray):
        repeats_val0 = np.expand_dims(repeats_val, 0).reshape(-1)[0]
        if np.any(repeats_val != repeats_val0):
            raise NotImplementedError(
                "Conversion for torch.repeat_interleave with Tensor repeats has not been implemented"
            )
        repeats_val = repeats_val0

    # This would operate on the flattened input tensor
    if dim is None:
        x = mb.reshape(x=x, shape=(-1,))
    else:
        if dim.val != 0 and not special_case:
            raise NotImplementedError(
                "Conversion for torch.repeat_interleave with non-zero dim has not been implemented"
            )

    """
    on a high level:
         x
         | tile in dim 0
         v
        [x, x, ...]
         | reshape to split the repeats
         v
        [[x],
         [x],
         ...]
         | transpose(1, 0)
         V
        [x^T, x^T, ...]
         | flatten
         V
        result
    """

    reps = [1] * x.rank
    reps[0] = repeats_val
    x_tiled = mb.tile(x=x, reps=reps)

    split_reps = [repeats_val] + list(x.shape)
    x_reshaped = mb.reshape(x=x_tiled, shape=list(split_reps))

    perm = [*range(x.rank + 1)]
    perm[0] = 1
    perm[1] = 0
    x_transposed = mb.transpose(x=x_reshaped, perm=perm)

    result_shape = list(x.shape)
    result_shape[0] = -1
    if special_case:
        result_shape = [1] + result_shape
    result = mb.reshape(x=x_transposed, shape=result_shape, name=node.name)

    context.add(result)





eps = 1e-6
###
def stable_l2_norm(x, eps):
    max_val = x.abs().max(axis=-1, keepdim=True).values
    max_val = torch.clamp(max_val, min=eps)
    xscaled = x / max_val
    scaled_norm = torch.acos(xscaled)
    return x / torch.clamp(scaled_norm, min=eps), max_val
###
class CustomRMSNorm(nn.Module):
    def __init__(self, weight, eps):
        super().__init__()
        self.weight = weight
        self.eps = eps
    
    def forward(self, x):
        x, max_val = stable_l2_norm(x, self.eps)
        return x * (x.size(-1) ** 0.5 / max_val) * self.weight

###
model.transformer.norm = CustomRMSNorm(model.transformer.norm.weight, model.transformer.norm.eps)

for layer in model.transformer.layers:
    layer.attn.q_norm = CustomRMSNorm(layer.attn.q_norm.weight, layer.attn.q_norm.eps)
    layer.attn.k_norm = CustomRMSNorm(layer.attn.k_norm.weight, layer.attn.k_norm.eps)
    layer.ffn_norm = CustomRMSNorm(layer.ffn_norm.weight, layer.ffn_norm.eps)
    layer.attn_norm = CustomRMSNorm(layer.attn_norm.weight, layer.attn_norm.eps)





coreml_model = ct.convert(
    jitted_model,
    convert_to="mlprogram",
    minimum_deployment_target=ct.target.macOS14,
    inputs=coreml_input_types,
    outputs=coreml_output_types,
    compute_precision=compute_precision,
    compute_units=compute_units,
    pass_pipeline=ct.PassPipeline.DEFAULT_PALETTIZATION, # palettization
)


import sys
if sys.platform == "darwin":
    coreml_outputs = coreml_model.predict(t_inputs)
    print(f"Converted, max diff for random inputs: {abs(coreml_outputs['logits'] - outputs['logits'].numpy()).max()}")

# Override tokenizer
model_name = "pcuenq/test-llama-tokenizer"

architecture = model.config.model_type
'''
user_defined_metadata = {
    "co.huggingface.exporters.name": model_name,
    "co.huggingface.exporters.task": "text-generation",
    "co.huggingface.exporters.architecture": architecture,
    "co.huggingface.exporters.framework": "pytorch",
    "co.huggingface.exporters.precision": compute_precision,
}
'''

# Assuming `compute_precision` is used to apply float16 precision selectively
precision_description = "FP16 for all ops except l2_norm"

user_defined_metadata = {
    "co.huggingface.exporters.name": model_name,
    "co.huggingface.exporters.task": "text-generation",
    "co.huggingface.exporters.architecture": architecture,
    "co.huggingface.exporters.framework": "pytorch",
    "co.huggingface.exporters.precision": precision_description,
}



spec = coreml_model._spec
spec.description.metadata.userDefined.update(user_defined_metadata)

coreml_model.save(outpath)
card = f"""
This repository contains a Core ML conversion of [{model_id}](https://hf.co/{model_id}) with the following characteristics:

    - Sequence length: {shape[-1]}, fixed.
    - Precision: {precision_description}.

Please, check the [original model card](https://hf.co/{model_id}) for additional details on the model.
"""
with open(f"{args.output_dir}/README.md", "w") as f:
    f.write(card)
