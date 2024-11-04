from abc import ABC, abstractmethod

class BaseEnvironment(ABC):
    """
    Clase base abstracta para entornos personalizados compatibles con DeepRL.
    """

    @abstractmethod
    def reset(self):
        """
        Reinicia el entorno y devuelve el estado inicial.
        :return: Estado inicial del entorno.
        """
        pass

    @abstractmethod
    def step(self, action):
        """
        Ejecuta una acción en el entorno.
        
        :param action: Acción a ejecutar.
        :return: Estado siguiente, recompensa, indicador de finalización, y otra información.
        """
        pass

    @abstractmethod
    def render(self, mode="human"):
        """
        Renderiza el entorno.
        
        :param mode: Modo de renderizado.
        """
        pass

    def close(self):
        """
        Cierra el entorno.
        """
        pass
