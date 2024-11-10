import torch
from sklearn.kernel_approximation import RBFSampler
from deeprl.function_approximations.base_approximator import BaseApproximator

class RBFBasisApproximator(BaseApproximator):
    """
    Radial basis function (RBF) approximator using scikit-learn's RBFSampler.

    :param gamma: The inverse of the RBF kernel's standard deviation.
    :param n_components: The number of components for the RBF transformation.
    :param env: The environment to extract observation and action space dimensions.
    """
    def __init__(self, gamma=1.0, n_components=100, env=None):
        self.rbf = RBFSampler(gamma=gamma, n_components=n_components, random_state=42)

        if env is None:
            raise ValueError("An environment instance is required to initialize the approximator.")
        
        # Initialize weights for state features and actions
        self.weights = torch.zeros((n_components, env.action_space.n), dtype=torch.float32)

        # Fit RBF sampler to the observation space dimensionality
        dummy_state = torch.zeros(env.observation_space.shape[0]).reshape(1, -1).numpy()
        self.rbf.fit(dummy_state)

    def compute_features(self, state):
        """
        Compute RBF features using scikit-learn's RBFSampler.

        :param state: A scalar state or a batch of states.
        :return: A feature vector as a PyTorch tensor.
        """
        state = torch.tensor(state).reshape(1, -1).numpy()  # Ensure state is 2D
        features = self.rbf.transform(state)
        return torch.tensor(features, dtype=torch.float32)

    def predict(self, state, action=None):
        """
        Predict the value for a given state (and optionally action) using RBF features.

        :param state: The input state.
        :param action: The action index (for multi-action environments).
        :return: Predicted value(s).
        """
        features = self.compute_features(state)
        if action is None:
            return torch.matmul(features, self.weights)  # Predict all actions
        return torch.dot(features.squeeze(), self.weights[:, action])

    def update(self, state, target, action, alpha=0.1):
        """
        Update the approximator weights using gradient descent.

        :param state: The input state.
        :param target: The target value.
        :param action: The action index corresponding to the update.
        :param alpha: The learning rate for weight updates.
        """
        features = self.compute_features(state).squeeze()  # Compute RBF features
        prediction = self.predict(state, action)  # Predict Q-value
        error = target - prediction  # Compute error
        self.weights[:, action] += alpha * error * features  # Update weights
