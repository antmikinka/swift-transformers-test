import coremltools as ct
import os
import argparse
import json

def load_model(model_path, weights_dir):
    try:
        model = ct.models.MLModel(model_path)
        spec = model._spec
        model = ct.models.MLModel(spec, weights_dir=weights_dir)
        return model, spec
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found.")
        return None, None

def get_precision(weights):
    if hasattr(weights, 'floatValue'):
        return "float32"
    elif hasattr(weights, 'float16Value'):
        return "float16"
    elif hasattr(weights, 'intValue'):
        return "int32"
    elif hasattr(weights, 'rawValue'):
        return "raw byte array (custom precision)"
    else:
        return "unknown"

def print_model_description(spec):
    print("Model Description:")
    metadata = spec.description.metadata
    if metadata:
        print(f"Model Name: {metadata.shortDescription}")
        print(f"Author: {metadata.author}")
        print(f"License: {metadata.license}")
        print(f"Version: {metadata.versionString}")

def print_model_io(spec):
    print("\nModel Inputs:")
    for input_desc in spec.description.input:
        print(f"Name: {input_desc.name}, Type: {input_desc.type.WhichOneof('Type')}")

    print("\nModel Outputs:")
    for output_desc in spec.description.output:
        print(f"Name: {output_desc.name}, Type: {output_desc.type.WhichOneof('Type')}")

def inspect_layers(spec):
    layer_types = set()

    print("\nLayers and their Precision:")
    for layer in spec.neuralNetwork.layers:
        layer_type = layer.WhichOneof('layer')
        layer_types.add(layer_type)

        print(f"Layer: {layer.name}")
        if hasattr(layer, 'weights'):
            precision = get_precision(layer.weights)
            print(f"Type: {layer_type}, Precision: {precision}")
        else:
            print(f"Type: {layer_type}, Precision: unknown")

    print("\nUnique Layer Types in the Model:")
    for layer_type in layer_types:
        print(layer_type)

def load_weights(weights_path):
    if os.path.exists(weights_path):
        with open(weights_path, "rb") as f:
            weights_data = f.read()
            print("\nweights.bin file loaded, size:", len(weights_data), "bytes")
    else:
        print("\nweights.bin file not found")

def get_weights_metadata(model):
    try:
        weight_metadata_dict = ct.optimize.coreml.get_weights_metadata(model)
    except AttributeError:
        print("Error: get_weights_metadata function not found in coremltools.utils")
        weight_metadata_dict = {}

    print("\nWeights Metadata:")
    #print(weight_metadata_dict)
    for weight_name, weight_metadata in weight_metadata_dict.items():
        print(f"Weight Name: {weight_name}")
        print(f" - val: {weight_metadata.val}")
        print(f" - Sparsity: {weight_metadata.sparsity}")
        print(f" - Unique Values: {weight_metadata.unique_values}")
        print(f" - Child Ops: {weight_metadata.child_ops}")

def load_manifest(manifest_path):
    with open(manifest_path, "r") as f:
        manifest_data = json.load(f)
        return manifest_data

def print_manifest_info(manifest_data):
    print("\nManifest Information:")
    #print(f"Model Name: {manifest_data['name']}")
    #print(f"Model Version: {manifest_data['version']}")
    #print(f"Author: {manifest_data['author']}")
    #print(f"License: {manifest_data['license']}")
    #print(f"Description: {manifest_data['description']}")

def main():
    parser = argparse.ArgumentParser(description="Inspect a Core ML model.")
    parser.add_argument("--mlmodel_path", help="Path to the .mlmodel file")
    parser.add_argument("--weights_dir", help="Directory containing the weights file")
    parser.add_argument("--manifest_path", help="Path to the Manifest.json file")
    args = parser.parse_args()

    if not args.mlmodel_path or not args.weights_dir or not args.manifest_path:
        print("Error: Please provide --mlmodel_path, --weights_dir, and --manifest_path arguments.")
        return

    model_path = args.mlmodel_path
    weights_dir = args.weights_dir
    manifest_path = args.manifest_path

    model, spec = load_model(model_path, weights_dir)
    manifest_data = load_manifest(manifest_path)

    if model and spec:
        print_model_description(spec)
        print_model_io(spec)
        inspect_layers(spec)
        load_weights(os.path.join(weights_dir, "weight.bin"))
        get_weights_metadata(model)
        print_manifest_info(manifest_data)

if __name__ == "__main__":
    main()
    
 #python layer-iteration.py --mlmodel_path "/Volumes/Macintosh HD/Users/anthonymikinka/corenet/mlx_examples/open_elm/OpenELM-270M-Instruct-128-FP16ComputePrecisionv2.mlpackage" --weights_dir "/Volumes/Macintosh HD/Users/anthonymikinka/corenet/mlx_examples/open_elm/OpenELM-270M-Instruct-128-FP16ComputePrecisionv2.mlpackage/Data/com.apple.CoreML/weights" --manifest_path "/Volumes/Macintosh HD/Users/anthonymikinka/corenet/mlx_examples/open_elm/OpenELM-270M-Instruct-128-FP16ComputePrecisionv2.mlpackage/Manifest.json"