import torch
import numpy as np
import multiprocessing as mp
from core.env_wrapper import ChessWrapper
from core.mcts_gumbel import GumbelMCTS
from core.network import ChessGumbelNet


def _play_single_game(config, network_state_dict):
    """Jugar una partida completa usando CPU."""
    # Crear red local en CPU
    network = ChessGumbelNet(config)
    network.load_state_dict(network_state_dict)
    network.to("cpu")
    network.eval()

    # Forzar config a usar CPU para este worker
    config_cpu = type(config)()
    config_cpu.__dict__.update(config.__dict__)
    config_cpu.device = "cpu"

    env = ChessWrapper(config_cpu)
    mcts = GumbelMCTS(config_cpu)

    state = env.reset()
    game_history = []
    done = False
    moves_count = 0

    while not done:
        legal_actions = env.get_legal_actions()

        if not legal_actions:
            break

        best_action, _, counts = mcts.run_search(state, network, legal_actions)

        policy_target = np.zeros(config.action_space_size, dtype=np.float32)
        total_visits = sum(counts.values())

        if total_visits > 0:
            for action_idx, visit_count in counts.items():
                policy_target[action_idx] = visit_count / total_visits
        else:
            for action_idx in legal_actions:
                policy_target[action_idx] = 1.0 / len(legal_actions)

        # Guardar sample pre-procesado (listo para el buffer)
        state_compressed = state.numpy().astype(np.float16)
        game_history.append({
            "state": state_compressed,
            "policy": policy_target,
            "player_color": env.board.turn
        })

        state, reward, done = env.step(best_action)
        moves_count += 1

        max_moves = getattr(config, 'max_moves_per_game', 150)
        if moves_count > max_moves:
            done = True
            reward = 0

    # Calcular value targets basados en resultado
    samples = []
    for sample in game_history:
        value_target = reward * (1 if sample["player_color"] else -1)
        samples.append({
            "state": sample["state"],
            "policy": sample["policy"],
            "value": value_target
        })

    return samples, moves_count


def _worker_process(config, network_state_dict, result_queue, num_games):
    """Proceso worker que juega múltiples partidas."""
    total_moves = 0
    all_samples = []

    for _ in range(num_games):
        samples, moves = _play_single_game(config, network_state_dict)
        all_samples.extend(samples)
        total_moves += moves

    result_queue.put((all_samples, total_moves))


def parallel_self_play(config, network, replay_buffer, num_workers=None, games_per_worker=1):
    """
    Ejecutar self-play en paralelo usando múltiples workers en CPU.

    Args:
        config: Configuración
        network: Red neuronal (en GPU)
        replay_buffer: Buffer compartido
        num_workers: Número de procesos paralelos
        games_per_worker: Partidas por worker

    Returns:
        total_moves: Total de movimientos generados
    """
    if num_workers is None:
        num_workers = config.num_self_play_workers

    # Copiar pesos de la red a CPU para compartir con workers
    network_state_dict = {k: v.cpu() for k, v in network.state_dict().items()}

    result_queue = mp.Queue()
    processes = []

    # Lanzar workers
    for _ in range(num_workers):
        p = mp.Process(
            target=_worker_process,
            args=(config, network_state_dict, result_queue, games_per_worker)
        )
        p.start()
        processes.append(p)

    # Recolectar resultados
    total_moves = 0
    for _ in range(num_workers):
        samples, moves = result_queue.get()
        replay_buffer.add_samples(samples)
        total_moves += moves

    # Esperar a que terminen todos
    for p in processes:
        p.join()

    return total_moves
