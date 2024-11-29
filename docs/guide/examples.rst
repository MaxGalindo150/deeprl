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


Dict Observations
-----------------

You can use environments with dictionary observation spaces. This is useful in the case where one can't directly
concatenate observations such as an image from a camera combined with a vector of servo sensor data (e.g., rotation angles).
DeepRL provides ``SimpleMultiObsEnv`` as an example of this kind of setting.
The environment is a simple grid world, but the observations for each cell come in the form of dictionaries.
These dictionaries are randomly initialized on the creation of the environment and contain a vector observation and an image observation.

.. code-block:: python

  from deeprl import PPO
  from deeprl.common.envs import SimpleMultiObsEnv


  # Stable Baselines provides SimpleMultiObsEnv as an example environment with Dict observations
  env = SimpleMultiObsEnv(random_start=False)

  model = PPO("MultiInputPolicy", env, verbose=1)
  model.learn(total_timesteps=100_000)


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


Callbacks: Evaluate Agent Performance
-------------------------------------
To periodically evaluate an agent's performance on a separate test environment, use ``EvalCallback``.
You can control the evaluation frequency with ``eval_freq`` to monitor your agent's progress during training.

.. code-block:: python

  import os
  import gymnasium as gym

  from deeprl import SAC
  from deeprl.common.callbacks import EvalCallback
  from deeprl.common.env_util import make_vec_env

  env_id = "Pendulum-v1"
  n_training_envs = 1
  n_eval_envs = 5

  # Create log dir where evaluation results will be saved
  eval_log_dir = "./eval_logs/"
  os.makedirs(eval_log_dir, exist_ok=True)

  # Initialize a vectorized training environment with default parameters
  train_env = make_vec_env(env_id, n_envs=n_training_envs, seed=0)

  # Separate evaluation env, with different parameters passed via env_kwargs
  # Eval environments can be vectorized to speed up evaluation.
  eval_env = make_vec_env(env_id, n_envs=n_eval_envs, seed=0,
                          env_kwargs={'g':0.7})

  # Create callback that evaluates agent for 5 episodes every 500 training environment steps.
  # When using multiple training environments, agent will be evaluated every
  # eval_freq calls to train_env.step(), thus it will be evaluated every
  # (eval_freq * n_envs) training steps. See EvalCallback doc for more information.
  eval_callback = EvalCallback(eval_env, best_model_save_path=eval_log_dir,
                                log_path=eval_log_dir, eval_freq=max(500 // n_training_envs, 1),
                                n_eval_episodes=5, deterministic=True,
                                render=False)

  model = SAC("MlpPolicy", train_env)
  model.learn(5000, callback=eval_callback)


Atari Games
-----------

.. .. figure:: ../_static/img/breakout.gif

  Trained A2C agent on Breakout

.. .. figure:: https://cdn-images-1.medium.com/max/960/1*UHYJE7lF8IDZS_U5SsAFUQ.gif

 Pong Environment


Training a RL agent on Atari games is straightforward thanks to ``make_atari_env`` helper function.
It will do `all the preprocessing <https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/>`_
and multiprocessing for you. To install the Atari environments, run the command ``pip install gymnasium[atari,accept-rom-license]`` to install the Atari environments and ROMs, or install Stable Baselines3 with ``pip install deeprl[extra]`` to install this and other optional dependencies.

.. .. image:: ../_static/img/colab-badge.svg
..    :target: https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/atari_games.ipynb
..

.. code-block:: python

  from deeprl.common.env_util import make_atari_env
  from deeprl.common.vec_env import VecFrameStack
  from deeprl import A2C

  # There already exists an environment generator
  # that will make and wrap atari environments correctly.
  # Here we are also multi-worker training (n_envs=4 => 4 environments)
  vec_env = make_atari_env("PongNoFrameskip-v4", n_envs=4, seed=0)
  # Frame-stacking with 4 frames
  vec_env = VecFrameStack(vec_env, n_stack=4)

  model = A2C("CnnPolicy", vec_env, verbose=1)
  model.learn(total_timesteps=25_000)

  obs = vec_env.reset()
  while True:
      action, _states = model.predict(obs, deterministic=False)
      obs, rewards, dones, info = vec_env.step(action)
      vec_env.render("human")

