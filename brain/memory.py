import json
import math
import os

MEMORY_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "brain_memory.json"
)


def save_brain(brain, filename=MEMORY_FILE):
    data = {
        "version": 2,
        "architecture": {
            "inputs": brain.input_count,
            "hidden_layers": brain.hidden_layers,
            "outputs": brain.output_count
        },
        "weights": brain.weights,
        "biases": brain.biases
    }

    temporary_file = filename + ".tmp"

    with open(temporary_file, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)

    os.replace(temporary_file, filename)
    print(f"Brain saved to: {filename}")


def _validate_parameters(weights, biases, layer_sizes):
    if len(weights) != len(layer_sizes) - 1:
        raise ValueError("Invalid number of weight layers")

    if len(biases) != len(layer_sizes) - 1:
        raise ValueError("Invalid number of bias layers")

    for layer_index, (layer_weights, layer_biases) in enumerate(
        zip(weights, biases)
    ):
        source_size = layer_sizes[layer_index]
        destination_size = layer_sizes[layer_index + 1]

        if len(layer_weights) != destination_size:
            raise ValueError("Invalid number of neurons in weight layer")

        if len(layer_biases) != destination_size:
            raise ValueError("Invalid number of neuron biases")

        for neuron_weights in layer_weights:
            if len(neuron_weights) != source_size:
                raise ValueError("Invalid weight dimensions")

            for value in neuron_weights:
                if not isinstance(value, (int, float)):
                    raise ValueError("Weights must be numeric")
                if not math.isfinite(value):
                    raise ValueError("Weights must be finite")

        for value in layer_biases:
            if not isinstance(value, (int, float)):
                raise ValueError("Biases must be numeric")
            if not math.isfinite(value):
                raise ValueError("Biases must be finite")


def load_brain(filename=MEMORY_FILE):
    if not os.path.exists(filename):
        return None

    with open(filename, "r", encoding="utf-8") as file:
        data = json.load(file)

    version = data.get("version")
    architecture = data.get("architecture", {})

    if version == 1:
        inputs = architecture["inputs"]
        hidden_layers = [architecture["hidden"]]
        outputs = architecture["outputs"]

        weights = [
            data["weights_hidden"],
            data["weights_output"]
        ]

        biases = [
            data["biases_hidden"],
            data["biases_output"]
        ]

    elif version == 2:
        inputs = architecture["inputs"]
        hidden_layers = architecture["hidden_layers"]
        outputs = architecture["outputs"]
        weights = data["weights"]
        biases = data["biases"]

    else:
        raise ValueError("Unsupported brain memory version")

    if (
        not isinstance(inputs, int)
        or isinstance(inputs, bool)
        or inputs < 1
        or not isinstance(outputs, int)
        or isinstance(outputs, bool)
        or outputs < 1
        or not isinstance(hidden_layers, list)
        or not hidden_layers
        or any(
            not isinstance(size, int)
            or isinstance(size, bool)
            or size < 1
            for size in hidden_layers
        )
    ):
        raise ValueError("Invalid brain architecture")

    from network import NeuralNetwork

    brain = NeuralNetwork(
        inputs=inputs,
        hidden_layers=hidden_layers,
        outputs=outputs
    )

    layer_sizes = [inputs] + hidden_layers + [outputs]
    _validate_parameters(weights, biases, layer_sizes)

    brain.weights = weights
    brain.biases = biases

    return brain