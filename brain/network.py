import math
import random

from memory import save_brain, load_brain


def sigmoid(value):
    value = max(-60.0, min(60.0, value))
    return 1.0 / (1.0 + math.exp(-value))


class NeuralNetwork:
    def __init__(
        self,
        inputs=2,
        hidden=4,
        outputs=1,
        seed=42,
        hidden_layers=None
    ):
        if hidden_layers is None:
            if isinstance(hidden, int):
                hidden_layers = [hidden]
            else:
                hidden_layers = list(hidden)
        else:
            hidden_layers = list(hidden_layers)

        if (
            not isinstance(inputs, int)
            or isinstance(inputs, bool)
            or inputs < 1
        ):
            raise ValueError("inputs must be a positive integer")

        if (
            not isinstance(outputs, int)
            or isinstance(outputs, bool)
            or outputs < 1
        ):
            raise ValueError("outputs must be a positive integer")

        if not hidden_layers or any(
            not isinstance(size, int)
            or isinstance(size, bool)
            or size < 1
            for size in hidden_layers
        ):
            raise ValueError(
                "hidden_layers must contain positive integers"
            )

        self.input_count = inputs
        self.hidden_layers = hidden_layers
        self.hidden_count = hidden_layers[0]
        self.output_count = outputs

        self.layer_sizes = [inputs] + hidden_layers + [outputs]

        rng = random.Random(seed)

        self.weights = []
        self.biases = []

        for source_size, destination_size in zip(
            self.layer_sizes[:-1],
            self.layer_sizes[1:]
        ):
            self.weights.append([
                [
                    rng.uniform(-1.0, 1.0)
                    for _ in range(source_size)
                ]
                for _ in range(destination_size)
            ])

            self.biases.append([
                rng.uniform(-1.0, 1.0)
                for _ in range(destination_size)
            ])

    def forward(self, inputs):
        if len(inputs) != self.input_count:
            raise ValueError(
                f"Expected {self.input_count} inputs, "
                f"received {len(inputs)}"
            )

        activations = [list(inputs)]

        for layer_weights, layer_biases in zip(
            self.weights,
            self.biases
        ):
            previous = activations[-1]
            current = []

            for neuron_weights, bias in zip(
                layer_weights,
                layer_biases
            ):
                total = bias + sum(
                    weight * value
                    for weight, value in zip(
                        neuron_weights,
                        previous
                    )
                )
                current.append(sigmoid(total))

            activations.append(current)

        return activations

    def predict(self, inputs):
        activations = self.forward(inputs)
        return activations[-1]

    def train(self, dataset, epochs=20000, learning_rate=0.5):
        if not dataset:
            raise ValueError("Training dataset cannot be empty")

        if epochs < 1:
            raise ValueError("epochs must be positive")

        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        for epoch in range(epochs):
            total_loss = 0.0

            for inputs, targets in dataset:
                if len(targets) != self.output_count:
                    raise ValueError(
                        "Target count does not match output layer"
                    )

                activations = self.forward(inputs)
                outputs = activations[-1]

                # Calculate output-layer errors.
                deltas = [None] * len(self.weights)

                output_deltas = []

                for output, target in zip(outputs, targets):
                    error = target - output
                    total_loss += error * error

                    output_deltas.append(
                        error * output * (1.0 - output)
                    )

                deltas[-1] = output_deltas

                # Calculate hidden-layer errors backwards.
                for layer_index in range(
                    len(self.weights) - 2,
                    -1,
                    -1
                ):
                    layer_deltas = []
                    next_weights = self.weights[layer_index + 1]
                    next_deltas = deltas[layer_index + 1]

                    for neuron_index, activation in enumerate(
                        activations[layer_index + 1]
                    ):
                        error_signal = sum(
                            next_deltas[next_neuron]
                            * next_weights[next_neuron][neuron_index]
                            for next_neuron in range(
                                len(next_deltas)
                            )
                        )

                        layer_deltas.append(
                            activation
                            * (1.0 - activation)
                            * error_signal
                        )

                    deltas[layer_index] = layer_deltas

                # Update every layer after calculating all deltas.
                for layer_index, layer_weights in enumerate(
                    self.weights
                ):
                    previous = activations[layer_index]
                    layer_deltas = deltas[layer_index]

                    for neuron_index, neuron_weights in enumerate(
                        layer_weights
                    ):
                        for weight_index in range(
                            len(neuron_weights)
                        ):
                            neuron_weights[weight_index] += (
                                learning_rate
                                * layer_deltas[neuron_index]
                                * previous[weight_index]
                            )

                        self.biases[layer_index][neuron_index] += (
                            learning_rate * layer_deltas[neuron_index]
                        )

            if (epoch + 1) % 2000 == 0:
                average_loss = total_loss / len(dataset)

                print(
                    f"Epoch {epoch + 1:5d}/{epochs} "
                    f"| Loss: {average_loss:.6f}"
                )

    def describe(self):
        return " -> ".join(str(size) for size in self.layer_sizes)


def main():
    dataset = [
        ([0.0, 0.0], [0.0]),
        ([0.0, 1.0], [1.0]),
        ([1.0, 0.0], [1.0]),
        ([1.0, 1.0], [0.0]),
    ]

    brain = load_brain()

    if brain is None:
        print("No saved brain found. Creating a new brain.")
        brain = NeuralNetwork(
            inputs=2,
            hidden_layers=[8, 4],
            outputs=1
        )
    else:
        print("Saved brain loaded successfully!")

    print("\nARTIFICIAL BRAIN v0.4")
    print("Architecture:", brain.describe())
    print("Training pattern: XOR")
    print("\nTraining...\n")

    brain.train(
        dataset,
        epochs=20000,
        learning_rate=0.5
    )

    print("\nSaving learned parameters...")
    save_brain(brain)

    print("\nTesting the brain:\n")

    correct = 0

    for inputs, targets in dataset:
        prediction = brain.predict(inputs)[0]
        predicted_class = 1 if prediction >= 0.5 else 0
        expected_class = int(targets[0])

        if predicted_class == expected_class:
            correct += 1

        print(
            f"{inputs} -> {prediction:.4f} "
            f"| Predicted: {predicted_class} "
            f"| Expected: {expected_class}"
        )

    print(f"\nTraining accuracy: {correct}/{len(dataset)}")


if __name__ == "__main__":
    main()