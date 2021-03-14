from MyCNNClassifier import MyCNNClassifier


class PolicyEstimator(object):
    def __init__(self, statement_size, action_size, device_in='cuda'):
        self.n_inputs = statement_size
        self.n_outputs = action_size
        self.device_in = device_in

        # Define Conv network
        self.network = MyCNNClassifier(1, action_size).to(self.device_in)
        print(self.network)

    def predict(self, input_0):
        action_probs = self.network.forward(input_0.to(self.device_in))
        return action_probs