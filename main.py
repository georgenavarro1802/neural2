import csv

from numpy import exp, array, random, dot


class NeuronLayer:

    def __init__(self, number_of_neurons, number_of_inputs_per_neuron):
        self.synaptic_weights = 2 * random.random((number_of_inputs_per_neuron, number_of_neurons)) - 1


class NeuralNetwork:

    def __init__(self, layer1, layer2):
        self.layer1 = layer1
        self.layer2 = layer2

    # The Sigmoid function, which describes an S shaped curve.
    # We pass the weighted sum of the inputs through this function to
    # normalise them between 0 and 1.
    @staticmethod
    def __sigmoid(x):
        return 1 / (1 + exp(-x))

    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    # It indicates how confident we are about the existing weight.
    @staticmethod
    def __sigmoid_derivative(x):
        return x * (1 - x)

    # We train the neural network through a process of trial and error.
    # Adjusting the synaptic weights each time.
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in xrange(number_of_training_iterations):
            # Pass the training set through our neural network
            output_from_layer_1, output_from_layer_2 = self.think(training_set_inputs)

            # Calculate the error for layer 2 (The difference between the desired output
            # and the predicted output).
            layer2_error = training_set_outputs - output_from_layer_2
            layer2_delta = layer2_error * self.__sigmoid_derivative(output_from_layer_2)

            # Calculate the error for layer 1 (By looking at the weights in layer 1,
            # we can determine by how much layer 1 contributed to the error in layer 2).
            layer1_error = layer2_delta.dot(self.layer2.synaptic_weights.T)
            layer1_delta = layer1_error * self.__sigmoid_derivative(output_from_layer_1)

            # Calculate how much to adjust the weights by
            layer1_adjustment = training_set_inputs.T.dot(layer1_delta)
            layer2_adjustment = output_from_layer_1.T.dot(layer2_delta)

            # Adjust the weights.
            self.layer1.synaptic_weights += layer1_adjustment
            self.layer2.synaptic_weights += layer2_adjustment

    # The neural network thinks.
    def think(self, inputs):
        output_from_layer1 = self.__sigmoid(dot(inputs, self.layer1.synaptic_weights))
        output_from_layer2 = self.__sigmoid(dot(output_from_layer1, self.layer2.synaptic_weights))
        return output_from_layer1, output_from_layer2

    # The neural network prints its weights
    def print_weights(self):
        print "    Layer 1 (4 Neurons, each with 5 inputs): "
        print self.layer1.synaptic_weights
        print "    Layer 2 (1 Neuron, with 4 inputs):"
        print self.layer2.synaptic_weights

    @staticmethod
    def print_results(output):
        if output[0] > 0.5:
            result = "### The result could be: 1 based on sigmoid function and trainings: {0}".format(output)
        elif output[0] < 0.5:
            result = "### The result could be: 0 on sigmoid function and trainings: {0}".format(output)
        else:
            result = "### Try result the neural with more iterations, because it does not determine the result"
        # console formatting
        print "#" * 40
        print "#" * 40
        print result
        print "#" * 40
        print "#" * 40


def conv(s):
    try:
        s = int(s)
    except ValueError:
        pass
    return s


if __name__ == "__main__":

    # Read from csv file and fill list of inputs and ouputs
    # inputs are the first 5 columns and the last columns is the output
    inputs_l = []
    outputs_l = []
    with open('data.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            # Inputs list
            inputs_l.append([conv(x) for x in row[:5]])
            # Output list
            outputs_l.append(conv(row[5]))

    # unknown input
    new_input = array([0, 0, 0, 0, 1])

    # Seed the random number generator
    random.seed(1)

    # Create layer 1 (4 neurons, each with 3 inputs)
    layer1 = NeuronLayer(4, 5)

    # Create layer 2 (a single neuron with 4 inputs)
    layer2 = NeuronLayer(1, 4)

    # Combine the layers to create a neural network
    neural_network = NeuralNetwork(layer1, layer2)

    print "Stage 1) Random starting synaptic weights: "
    neural_network.print_weights()

    # The training set. We have csv examples, each consisting of 5 input values and 1 output value.
    training_set_inputs = array(inputs_l)
    training_set_outputs = array([outputs_l]).T

    # Train the neural network using the training set.
    # Do it 60,000 times and make small adjustments each time.
    neural_network.train(training_set_inputs, training_set_outputs, 60000)

    print "Stage 2) New synaptic weights after training: "
    neural_network.print_weights()

    # Test the neural network with a new situation.
    print "Stage 3) Considering a new situation {} -> ?: ".format(new_input)
    hidden_state, output = neural_network.think(new_input)

    # Print possibles results
    neural_network.print_results(output)
