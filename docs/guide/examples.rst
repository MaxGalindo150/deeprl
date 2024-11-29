.. _examples:

Examples
========

These examples are only to demostrate the useo of the library and its functions, they are not meant to be used as a reference for the algorithms and the trainded agents may not solve the environments. Optimized hyperparameters and training procedures are not used in these examples.

- `Full Tutorial <https://yutudeeigencore>`_
- `All Notebooks <https://githubdeeigencore.com>`_
 
Basic Usage: Training, Saving, Loading
--------------------------------------

In the following example, we will train, save and load a DQN model on the Lunar Lander environment.

.. code-block:: python

    import gymnasium as gym

    from deeprl import DQN
    from deeprl.common.evaluation import evaluate_policy

    env = gym.make("LunarLander-v2")
    model = DQN("MlpPolicy", env, verbose=1)

    # Evaluate the model before training

    # Random Agent, before training
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=10,
        deterministic=True,
    )
    
    print("Mean reward before training:")
    print(f"{mean_reward:.2f} +/- {std_reward:.2f}")

    # Train the agent
    model.learn(total_timesteps=100_000)
    # Save the agent
    model.save("dqn_lunar")
    del model # delete trained model to demonstrate loading

    model = DQN.load("dqn_lunar") # Load the trained agent
    
    # Evaluate the trained agent
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=10,
        deterministic=True,
    )

    print("Mean reward after training:")
    print(f"mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")

    obs = env.reset()
    for i in range(1000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()

.. note::
  LunarLander requires the python package ``box2d``.
  You can install it using ``apt install swig`` and then ``pip install box2d box2d-kengz``

.. warning::
  ``load`` method re-creates the model from scratch and should be called on the Algorithm without instantiating it first,
  e.g. ``model = DQN.load("dqn_lunar", env=env)`` instead of ``model = DQN(env=env)`` followed by  ``model.load("dqn_lunar")``. The latter **will not work** as ``load`` is not an in-place operation.
  If you want to load parameters without re-creating the model, e.g. to evaluate the same model
  with multiple different sets of parameters, consider using ``set_parameters`` instead.

Multiprocessing: Unleashing the Power of Vectorized Environments
----------------------------------------------------------------

.. warning::
  Multiprocessing is not supported so far.

Callbacks: Monitoring Training
------------------------------

.. note::
  Please refer to the `Callback section <callbacks>`_ for more details.

You can define a custon callback function that will be called inside the agent.
This is useful when you want to monitor training, for instance display live learning curves in Tensorboard or save the best agent.
If your callback returns ``False``, training is aborted early.

.. code-block:: python

  import os

  import gymnasium as gym
  import numpy as np
  import matplotlib.pyplot as plt

  from deeprl import TD3
  from deeprl.common import results_plotter
  from deeprl.common.monitor import Monitor
  from deeprl.common.results_plotter import load_results, ts2xy, plot_results
  from deeprl.common.noise import NormalActionNoise
  from deeprl.common.callbacks import BaseCallback


  class SaveOnBestTrainingRewardCallback(BaseCallback):
      """
      Callback for saving a model (the check is done every ``check_freq`` steps)
      based on the training reward (in practice, we recommend using ``EvalCallback``).

      :param check_freq:
      :param log_dir: Path to the folder where the model will be saved.
        It must contains the file created by the ``Monitor`` wrapper.
      :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
      """
      def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
          super().__init__(verbose)
          self.check_freq = check_freq
          self.log_dir = log_dir
          self.save_path = os.path.join(log_dir, "best_model")
          self.best_mean_reward = -np.inf

      def _init_callback(self) -> None:
          # Create folder if needed
          if self.save_path is not None:
              os.makedirs(self.save_path, exist_ok=True)

      def _on_step(self) -> bool:
          if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose >= 1:
                  print(f"Num timesteps: {self.num_timesteps}")
                  print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose >= 1:
                      print(f"Saving new best model to {self.save_path}")
                    self.model.save(self.save_path)

          return True

  # Create log dir
  log_dir = "tmp/"
  os.makedirs(log_dir, exist_ok=True)

  # Create and wrap the environment
  env = gym.make("LunarLanderContinuous-v2")
  env = Monitor(env, log_dir)

  # Add some action noise for exploration
  n_actions = env.action_space.shape[-1]
  action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
  # Because we use parameter noise, we should use a MlpPolicy with layer normalization
  model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=0)
  # Create the callback: check every 1000 steps
  callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
  # Train the agent
  timesteps = 1e5
  model.learn(total_timesteps=int(timesteps), callback=callback)

  plot_results([log_dir], timesteps, results_plotter.X_TIMESTEPS, "TD3 LunarLander")
  plt.show()
