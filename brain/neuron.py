class Neuron:
    def __init__(self, input_count):
        self.weights = [0.0] * input_count
        self.bias = 0.0

    def activate(self, inputs):
        total = sum(
            weight * value
            for weight, value in zip(self.weights, inputs)
        )
        return total + self.bias

    def predict(self, inputs):
        return 1 if self.activate(inputs) >= 0 else 0