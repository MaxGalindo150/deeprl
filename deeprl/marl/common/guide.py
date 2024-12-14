import os
import pandas as pd
import time

def test_marl_monitor():
    from pettingzoo.mpe import simple_spread_v3
    from deeprl.marl.common.marl_monitor import MARLMonitor
    
    log_file = "marl_"
    env = simple_spread_v3.parallel_env(max_cycles=25)
    monitored_env = MARLMonitor(env, filename=log_file, allow_early_resets=True)
    
    observations = monitored_env.reset()
    done = {agent: False for agent in monitored_env.env.agents}
    actions = {agent: monitored_env.env.action_space(agent).sample() for agent in monitored_env.env.agents}

    while not all(done.values()):
        next_obs, rewards, terminations, truncations, infos = monitored_env.step(actions)
        actions = {
            agent: monitored_env.env.action_space(agent).sample()
            for agent in monitored_env.env.agents
            if not terminations.get(agent, False) and not truncations.get(agent, False)
        }
        done = {agent: terminations.get(agent, False) or truncations.get(agent, False) for agent in monitored_env.env.agents}

    monitored_env.close()

    # Verifica que el archivo de log exista y tenga datos
    time.sleep(2)
    assert os.path.exists("marl_monitor_test.csv.monitor.csv"), "Log file not created"
    df = pd.read_csv("marl_monitor_test.csv.monitor.csv")
    assert not df.empty, "Log file is empty"
    assert "agent_id" in df.columns, "Log file missing 'agent_id' column"
    assert "reward" in df.columns, "Log file missing 'reward' column"
    assert "length" in df.columns, "Log file missing 'length' column"
    print("All tests passed!")

def test_monitor():
    import gymnasium as gym
    from deeprl.common.monitor import Monitor
    
    env = gym.make("CartPole-v1", render_mode="rgb_array")

    env = Monitor(env, filename="cartpole_dqn")
    
    observations = env.reset()
    done = False
    
    while not done:
        action = env.action_space.sample()
        observations, reward, done, truncated, info = env.step(action)
        
    env.close()
    
    



test_marl_monitor()
#test_monitor()
