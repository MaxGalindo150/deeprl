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

