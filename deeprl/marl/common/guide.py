import pettingzoo
from pettingzoo.mpe import simple_spread_v3

# Verifica si el entorno creado es una instancia de AECEnv
print(isinstance(simple_spread_v3.parallel_env(), pettingzoo.ParallelEnv))
