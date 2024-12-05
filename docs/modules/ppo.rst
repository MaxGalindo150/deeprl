.. _ppo2:

.. automodule:: deeprl.deep.ppo2

The `Proximal Policy Optimization <https://arxiv.org/abs/1707.06347>`_ algorithm combines ideas from A2C (having multiple workers)
and TRPO (it uses a trust region to improve the actor).

The main idea is that after an update, the new policy should be not too far from the old policy.
For that, ppo uses clipping to avoid too large update.

.. note::

  PPO contains several modifications from the original algorithm not documented
  by OpenAI: advantages are normalized and value function can be also clipped.


Notes
-----

- Original paper: https://arxiv.org/abs/1707.06347
- Clear explanation of PPO on Arxiv Insights channel: https://www.youtube.com/watch?v=5P7I-xPq8u8
- OpenAI blog post: https://openai.com/research/openai-baselines-ppo
- Spinning Up guide: https://spinningup.openai.com/en/latest/algorithms/ppo.html
- 37 implementation details blog: https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/

Can I use?
----------

.. note::

  A recurrent version of PPO is available in our contrib repo: https://sb3-contrib.readthedocs.io/en/master/modules/ppo_recurrent.html

  However we advise users to start with simple frame-stacking as a simpler, faster
  and usually competitive alternative, more info in our report: https://wandb.ai/sb3/no-vel-envs/reports/PPO-vs-RecurrentPPO-aka-PPO-LSTM-on-environments-with-masked-velocity--VmlldzoxOTI4NjE4
  See also `Procgen paper appendix Fig 11. <https://arxiv.org/abs/1912.01588>`_.
  In practice, you can stack multiple observations using ``VecFrameStack``.


-  Recurrent policies: ❌
-  Multi processing: ✔️
-  Gym spaces:


============= ====== ===========
Space         Action Observation
============= ====== ===========
Discrete      ✔️      ✔️
Box           ✔️      ✔️
MultiDiscrete ✔️      ✔️
MultiBinary   ✔️      ✔️
Dict          ❌     ✔️
============= ====== ===========

Example
-------

This example is only to demonstrate the use of the library and its functions, and the trained agents may not solve the environments.
Train a PPO agent on ``CartPole-v1`` using 4 environments.

.. code-block:: python

  import gymnasium as gym

  from deeprl import PPO
  from deeprl.common.env_util import make_vec_env

  # Parallel environments
  vec_env = make_vec_env("CartPole-v1", n_envs=4)

  model = PPO("MlpPolicy", vec_env, verbose=1)
  model.learn(total_timesteps=25000)
  model.save("ppo_cartpole")

  del model # remove to demonstrate saving and loading

  model = PPO.load("ppo_cartpole")

  obs = vec_env.reset()
  while True:
      action, _states = model.predict(obs)
      obs, rewards, dones, info = vec_env.step(action)
      vec_env.render("human")
