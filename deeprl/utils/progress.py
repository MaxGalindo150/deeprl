def print_progress(episode, total_reward, avg_reward, steps, epsilon, header=False):
    """
    Muestra el progreso del entrenamiento en formato de tabla.
    
    :param episode: Número del episodio actual.
    :param total_reward: Recompensa total obtenida en el episodio.
    :param avg_reward: Promedio de recompensa hasta el episodio actual.
    :param steps: Número de pasos tomados en el episodio.
    :param header: Si es True, imprime el encabezado de la tabla.
    """
    if header:
        print("=" * 70)
        print(f"|| {'Episode':^10} || {'Reward':^10} || {'Avg Reward':^12} || {'Steps':^8} || {'Epsilon':^8} ||")
        print("=" * 70)
    print(f"|| {episode:^10} || {total_reward:^10.2f} || {avg_reward:^12.2f} || {steps:^8} || {epsilon:^8.2f} ||")