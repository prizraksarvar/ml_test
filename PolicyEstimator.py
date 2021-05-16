from MyCNNClassifier import MyCNNClassifier


class PolicyEstimator(object):
    def __init__(self, statement_size, action_size, device_in='cuda'):
        self.n_inputs = statement_size
        self.n_outputs = action_size
        self.device_in = device_in

        # Define Conv network
        self.network = MyCNNClassifier(1, action_size).to(self.device_in)
        self.step = 1
        print(self.network)

    def predict(self, input_0):
        if self.step == 1:
            action_probs = self.network.forward1(input_0.to(self.device_in))
        if self.step == 2:
            action_probs = self.network.forward2(input_0.to(self.device_in))
        if self.step == 3:
            action_probs = self.network.forward3(input_0.to(self.device_in))
        if self.step == 4:
            action_probs = self.network.forward4(input_0.to(self.device_in))
        if self.step == 5:
            action_probs = self.network.forward(input_0.to(self.device_in))
        return action_probs

    def next_step(self):
        self.step = self.step + 1
        if self.step > 5:
            self.step = 5

