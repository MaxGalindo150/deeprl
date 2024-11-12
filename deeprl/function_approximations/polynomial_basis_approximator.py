import torch
from sklearn.preprocessing import PolynomialFeatures
from deeprl.function_approximations.base_approximator import BaseApproximator

class PolynomialBasisApproximator(BaseApproximator):
    """
    Polynomial basis function approximator using scikit-learn.
    
    :param degree: The degree of the polynomial.
    :param env: The environment to interact with.
    :param kwargs: Additional keyword arguments for the PolynomialFeatures class.
    """
    def __init__(self, degree, env=None, **kwargs):
        self.poly = PolynomialFeatures(degree=degree)
        if env is None:
            raise ValueError("An environment instance is required to initialize the approximator.")
        dummy_state = torch.zeros(env.observation_space.shape[0]).reshape(1, -1).numpy()
        self.poly.fit_transform(dummy_state)
        self.weights = torch.zeros((self.poly.n_output_features_, env.action_space.n), dtype=torch.float32)

    def compute_features(self, state):
        """
        Compute polynomial features using scikit-learn's PolynomialFeatures.
        :param state: A scalar state or a batch of states.
        :return: A feature vector as a PyTorch tensor.
        """
        state = torch.tensor(state).reshape(1, -1).numpy()
        features = self.poly.fit_transform(state)
        return torch.tensor(features, dtype=torch.float32)

    def predict(self, state):
        features = self.compute_features(state)
        return torch.matmul(features, self.weights)

    def update(self, state, target, alpha=0.1):
        features = self.compute_features(state)  # Shape: [1, n_output_features]
        prediction = self.predict(state)        # Shape: [n_actions]
        error = target - prediction             # Shape: [n_actions]
        
        # Update weights for each action independently
        for action in range(self.weights.shape[1]):  # Loop over actions
            self.weights[:, action] += alpha * error[action] * features.squeeze()

