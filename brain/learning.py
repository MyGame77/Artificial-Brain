from neuron import Neuron


def train(neuron, dataset, epochs=20, learning_rate=0.1):
    for epoch in range(epochs):
        errors = 0

        for inputs, expected in dataset:
            prediction = neuron.predict(inputs)
            error = expected - prediction

            if error != 0:
                errors += 1

                for i in range(len(neuron.weights)):
                    neuron.weights[i] += learning_rate * error * inputs[i]

                neuron.bias += learning_rate * error

        print(f"Epoch {epoch + 1}: errors={errors}")

        if errors == 0:
            print("Learning complete!")
            break


if __name__ == "__main__":
    dataset = [
        ([0, 0], 0),
        ([0, 1], 0),
        ([1, 0], 0),
        ([1, 1], 1),
    ]

    brain = Neuron(2)

    print("Before training:", brain.weights, brain.bias)

    train(brain, dataset)

    print("\nAfter training:", brain.weights, brain.bias)
    print("\nTesting learned pattern:")

    for inputs, expected in dataset:
        print(
            f"{inputs} -> {brain.predict(inputs)} "
            f"(expected {expected})"
        )