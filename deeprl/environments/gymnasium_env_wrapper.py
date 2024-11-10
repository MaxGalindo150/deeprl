import gymnasium as gym

class GymnasiumEnvWrapper:
    """
    Wrapper for Gymnasium environments that allows easy integration with DeepRL.
    
    :param env: Name of the environment in Gymnasium or an instance of a Gymnasium environment.
    :type env: str or gymnasium.Env
    :param kwargs: Additional arguments to create the environment if `env` is a string.
    """

    def __init__(self, env, **kwargs):
        """
        Initializes the wrapper with the Gymnasium environment.
        
        :param env: Name of the environment in Gymnasium or an instance of a Gymnasium environment.
        :type env: str or gymnasium.Env
        :param kwargs: Additional arguments to create the environment if `env` is a string.
        """
        self.env = gym.make(env, **kwargs) if isinstance(env, str) else env
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.spec = self.env.spec  
        self.env_kwargs = kwargs

    def reset(self):
        """
        Resets the environment and returns the initial state.
        """
        state, _ = self.env.reset()
        return state

    def step(self, action):
        """
        Executes an action in the environment.
        
        :param action: Action to execute.
        :return: Next state, reward, done flag, truncated flag, and additional info.
        """
        next_state, reward, done, truncated, info = self.env.step(action)
        return next_state, reward, done, truncated, info

    def render(self):
        """
        Renders the environment.
        
        :param mode: Render mode.
        """
        return self.env.render()

    def close(self):
        """
        Closes the environment.
        """
        self.env.close()
    
    def get_underlying_env(self):
        """
        Returns the underlying environment which might have additional attributes like `P`.
        If there is no underlying environment, returns the current environment.
        """
        env = self.env
        while hasattr(env, 'env'):
            env = env.env
        return env