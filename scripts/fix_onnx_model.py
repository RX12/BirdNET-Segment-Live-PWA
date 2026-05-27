"""
Fix the bird_mixit_4source.onnx model by adding a Slice node before the 
final Sub to trim the expanded input to match the Sum output shape.

The issue: tf2onnx conversion of the Conv-TasNet model produces a Sub node that 
tries to subtract Sum:0 (reconstructed sources) from ExpandDims_2:0 (original input),
but the causal convolutions cause the reconstructed output to be shorter than the input.
This makes the model fail with "Attempting to broadcast an axis by a dimension other than 1".

Fix: Insert a dynamic Shape->Slice chain so ExpandDims_2 is trimmed to match Sum's length.
"""
import onnx
from onnx import helper, TensorProto, numpy_helper
import numpy as np

model_path = "public/models/bird_mixit_4source.onnx"
output_path = "public/models/bird_mixit_4source.onnx"

print(f"Loading model from {model_path}...")
model = onnx.load(model_path)
graph = model.graph

# Find the final Sub node
sub_node = None
sub_idx = None
for i, node in enumerate(graph.node):
    if node.op_type == "Sub" and node.name == "sub":
        sub_node = node
        sub_idx = i
        break

if sub_node is None:
    print("ERROR: Could not find the 'sub' node")
    exit(1)

print(f"Found Sub node at index {sub_idx}")
print(f"  Inputs: {list(sub_node.input)}")
print(f"  Outputs: {list(sub_node.output)}")

expand_dims_name = sub_node.input[0]  # ExpandDims_2:0 (original input, longer)
sum_name = sub_node.input[1]          # Sum:0 (reconstructed, shorter)

# We need to:
# 1. Get the shape of Sum:0 along the last axis
# 2. Slice ExpandDims_2:0 along the last axis to match
# 3. Use the sliced version in the Sub node

# Create nodes to dynamically slice ExpandDims_2 to match Sum's shape

# Node 1: Get the shape of Sum
shape_node = helper.make_node(
    "Shape",
    inputs=[sum_name],
    outputs=["_fix_sum_shape"],
    name="_fix_shape_of_sum"
)

# Node 2: Create constants for Slice
# We want to slice the last dimension of ExpandDims_2 from 0 to Sum's last dim length
# Starts = [0] for last axis
starts_const = numpy_helper.from_array(np.array([0], dtype=np.int64), name="_fix_slice_starts")
# Axes = [-1] (last axis)
axes_const = numpy_helper.from_array(np.array([-1], dtype=np.int64), name="_fix_slice_axes")
# Steps = [1]
steps_const = numpy_helper.from_array(np.array([1], dtype=np.int64), name="_fix_slice_steps")

# Node 3: Gather the last element of Sum's shape to get the length
# Use [-1] index to get last dim
gather_idx_const = numpy_helper.from_array(np.array(-1, dtype=np.int64), name="_fix_gather_idx")

gather_node = helper.make_node(
    "Gather",
    inputs=["_fix_sum_shape", "_fix_gather_idx"],
    outputs=["_fix_sum_last_dim_scalar"],
    name="_fix_gather_last_dim",
    axis=0
)

# Node 4: Unsqueeze the scalar to [1] shape for Slice's ends parameter
unsqueeze_axes_const = numpy_helper.from_array(np.array([0], dtype=np.int64), name="_fix_unsqueeze_axes")
unsqueeze_node = helper.make_node(
    "Unsqueeze",
    inputs=["_fix_sum_last_dim_scalar", "_fix_unsqueeze_axes"],
    outputs=["_fix_slice_ends"],
    name="_fix_unsqueeze_end"
)

# Node 5: Slice ExpandDims_2 along last axis
slice_node = helper.make_node(
    "Slice",
    inputs=[expand_dims_name, "_fix_slice_starts", "_fix_slice_ends", "_fix_slice_axes", "_fix_slice_steps"],
    outputs=["_fix_trimmed_input"],
    name="_fix_slice_input"
)

# Add initializers
graph.initializer.append(starts_const)
graph.initializer.append(axes_const)
graph.initializer.append(steps_const)
graph.initializer.append(gather_idx_const)
graph.initializer.append(unsqueeze_axes_const)

# Insert the new nodes just before the Sub node
graph.node.insert(sub_idx, slice_node)
graph.node.insert(sub_idx, unsqueeze_node)
graph.node.insert(sub_idx, gather_node)
graph.node.insert(sub_idx, shape_node)

# Update the Sub node's first input to use the trimmed version
# (sub_idx shifted by 4 due to insertions)
graph.node[sub_idx + 4].input[0] = "_fix_trimmed_input"

print(f"Inserted 4 fix nodes before Sub node")
print(f"Sub node now uses inputs: {list(graph.node[sub_idx + 4].input)}")

# Validate and save
print("Checking model...")
try:
    onnx.checker.check_model(model)
    print("Model validation passed!")
except Exception as e:
    print(f"Model validation warning (may still work): {e}")

print(f"Saving fixed model to {output_path}...")
onnx.save(model, output_path)
print("Done!")

# Test the fixed model
print("\n=== TESTING FIXED MODEL ===")
try:
    import onnxruntime as ort
    sess = ort.InferenceSession(output_path, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    
    for length in [22050, 44100, 66150, 132300, 198450]:
        try:
            test_input = np.random.randn(1, 1, length).astype(np.float32)
            results = sess.run(None, {input_name: test_input})
            shapes = [r.shape for r in results]
            print(f"  Input={length} -> Output shapes: {shapes} ✓")
        except Exception as e:
            print(f"  Input={length} -> ERROR: {e}")
except ImportError:
    print("  onnxruntime not installed, skipping test")
except Exception as e:
    print(f"  Test error: {e}")
