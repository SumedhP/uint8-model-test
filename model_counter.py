import onnx
from collections import Counter
import numpy as np

def count_operations(onnx_model_path):
    # Load the ONNX model
    model = onnx.load(onnx_model_path)
    # Get all nodes in the graph
    nodes = model.graph.node
    # Get all initializers (parameters like weights and biases)
    initializers = model.graph.initializer
    
    # Count the types of operations
    op_counts = Counter(node.op_type for node in nodes)
    
    # Calculate the total number of parameters
    total_parameters = sum(
        np.prod(onnx.numpy_helper.to_array(init).shape) for init in initializers
    )
    
    # Sort the operations alphabetically
    op_counts = dict(sorted(op_counts.items()))
    
    # Print the counts
    for op_type, count in op_counts.items():
        print(f"{op_type}: {count}")
    
    # Print the total number of parameters
    print("\nTotal Parameters:", total_parameters)

    print()

if __name__ == "__main__":
    print("Operations in the generic model:")
    count_operations("Generic_model.onnx")
    
    print("Operations in the new model:")
    count_operations("HUST_model.onnx")
    
    print("Operations in the new model preprocessed:")
    count_operations("HUST_model_preprocessed.onnx")
