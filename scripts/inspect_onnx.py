"""Inspect ONNX model input/output shapes and find the Sub node's expected dimensions."""
import sys
try:
    import onnx
except ImportError:
    print("ERROR: onnx not installed. Run: pip install onnx")
    sys.exit(1)

model_path = "public/models/bird_mixit_4source.onnx"
model = onnx.load(model_path)
graph = model.graph

print("=== INPUTS ===")
for inp in graph.input:
    shape = [d.dim_value if d.dim_value else d.dim_param for d in inp.type.tensor_type.shape.dim]
    print(f"  {inp.name}: shape={shape}, dtype={inp.type.tensor_type.elem_type}")

print("\n=== OUTPUTS ===")
for out in graph.output:
    shape = [d.dim_value if d.dim_value else d.dim_param for d in out.type.tensor_type.shape.dim]
    print(f"  {out.name}: shape={shape}, dtype={out.type.tensor_type.elem_type}")

# Find the Sub node
print("\n=== SUB NODE(S) ===")
for node in graph.node:
    if node.op_type == "Sub":
        print(f"  Name: {node.name}")
        print(f"  Inputs: {list(node.input)}")
        print(f"  Outputs: {list(node.output)}")

# Try to run the model with onnxruntime to inspect actual shapes
print("\n=== RUNTIME TEST ===")
try:
    import onnxruntime as ort
    import numpy as np
    
    sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    input_shape = sess.get_inputs()[0].shape
    print(f"  Input name: {input_name}, shape: {input_shape}")
    
    for out in sess.get_outputs():
        print(f"  Output name: {out.name}, shape: {out.shape}")
    
    # Test with different input lengths
    for length in [66150, 66304, 132300, 198450, 22050]:
        try:
            test_input = np.random.randn(1, 1, length).astype(np.float32)
            results = sess.run(None, {input_name: test_input})
            for i, r in enumerate(results):
                print(f"  Input={length} -> Output[{i}] shape={r.shape}")
        except Exception as e:
            print(f"  Input={length} -> ERROR: {e}")
except ImportError:
    print("  onnxruntime not installed, skipping runtime test")
except Exception as e:
    print(f"  Runtime test error: {e}")
