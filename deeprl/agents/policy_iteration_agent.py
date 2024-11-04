import numpy as np
from deeprl.agents.base_agent import Agent

class PolicyIterationAgent(Agent):
    """
    Agente que implementa el algoritmo de Iteración de Política.
    """

    def __init__(self, env, gamma=0.99, theta=1e-6):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.policy = np.zeros(self.env.observation_space.n, dtype=int)
        self.value_table = np.zeros(self.env.observation_space.n)

    def policy_evaluation(self):
        """
        Evalúa la política actual.
        """
        while True:
            delta = 0
            for state in range(self.env.observation_space.n):
                v = self.value_table[state]
                action = self.policy[state]
                self.value_table[state] = sum([
                    prob * (reward + self.gamma * self.value_table[next_state])
                    for prob, next_state, reward, done in self.env.P[state][action]
                ])
                delta = max(delta, abs(v - self.value_table[state]))
            if delta < self.theta:
                break

    def policy_improvement(self):
        """
        Mejora la política basándose en la función de valor actual.
        """
        policy_stable = True
        for state in range(self.env.observation_space.n):
            old_action = self.policy[state]
            q_values = self.compute_q_values(state)
            self.policy[state] = np.argmax(q_values)
            if old_action != self.policy[state]:
                policy_stable = False
        return policy_stable

    def compute_q_values(self, state):
        """
        Calcula los valores Q para todas las acciones en un estado dado.
        """
        q_values = np.zeros(self.env.action_space.n)
        for action in range(self.env.action_space.n):
            for prob, next_state, reward, done in self.env.P[state][action]:
                q_values[action] += prob * (reward + self.gamma * self.value_table[next_state])
        return q_values

    def policy_iteration(self):
        """
        Ejecuta el algoritmo de iteración de política.
        """
        while True:
            self.policy_evaluation()
            if self.policy_improvement():
                break

    def act(self, state):
        """
        Selecciona la acción basada en la política actual.
        """
        return self.policy[state]

    def learn(self):
        """
        Ejecuta el proceso de aprendizaje (iteración de política).
        """
        self.policy_iteration()
