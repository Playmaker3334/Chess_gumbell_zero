import torch
import numpy as np
import multiprocessing as mp
from core.env_wrapper import ChessWrapper
from core.mcts_gumbel import GumbelMCTS
from core.network import ChessGumbelNet


def _play_single_game(config, network_state_dict, should_log=False, game_id=0, training_step=0):
    network = ChessGumbelNet(config)
    network.load_state_dict(network_state_dict)
    network.to("cpu")
    network.eval()

    config_cpu = type(config)()
    config_cpu.__dict__.update(config.__dict__)
    config_cpu.device = "cpu"

    env = ChessWrapper(config_cpu)
    mcts = GumbelMCTS(config_cpu)

    state = env.reset()
    game_history = []
    move_log = []
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

        state_compressed = state.numpy().astype(np.float16)
        game_history.append({
            "state": state_compressed,
            "policy": policy_target,
            "player_color": env.board.turn
        })

        if should_log:
            move_uci = env.index_lookup.get(best_action)
            if move_uci:
                move_log.append(move_uci)

        state, reward, done = env.step(best_action)
        moves_count += 1

        max_moves = getattr(config, 'max_moves_per_game', 150)
        if moves_count > max_moves:
            done = True
            reward = 0

    samples = []
    for sample in game_history:
        value_target = reward * (1 if sample["player_color"] else -1)
        samples.append({
            "state": sample["state"],
            "policy": sample["policy"],
            "value": value_target
        })

    log_data = None
    if should_log:
        log_data = {
            "game_id": game_id,
            "training_step": training_step,
            "moves": move_log,
            "result": reward
        }

    return samples, moves_count, log_data


def _worker_process(config, network_state_dict, result_queue, num_games, log_game_idx=-1, game_id_start=0, training_step=0):
    total_moves = 0
    all_samples = []
    log_data = None

    for i in range(num_games):
        should_log = (i == log_game_idx)
        game_id = game_id_start + i
        
        samples, moves, game_log = _play_single_game(
            config, 
            network_state_dict, 
            should_log=should_log,
            game_id=game_id,
            training_step=training_step
        )
        all_samples.extend(samples)
        total_moves += moves
        
        if game_log is not None:
            log_data = game_log

    result_queue.put((all_samples, total_moves, log_data))


def parallel_self_play(config, network, replay_buffer, num_workers=None, games_per_worker=1, game_logger=None, game_counter=0, training_step=0):
    if num_workers is None:
        num_workers = config.num_self_play_workers

    network_state_dict = {k: v.cpu() for k, v in network.state_dict().items()}

    result_queue = mp.Queue()
    processes = []

    log_worker_idx = -1
    log_game_idx = -1
    
    if game_logger is not None and config.log_games:
        total_games = num_workers * games_per_worker
        if game_counter % config.log_every_n_games < total_games:
            log_worker_idx = (game_counter % config.log_every_n_games) // games_per_worker
            log_game_idx = (game_counter % config.log_every_n_games) % games_per_worker

    for worker_id in range(num_workers):
        game_id_start = game_counter + worker_id * games_per_worker
        worker_log_idx = log_game_idx if worker_id == log_worker_idx else -1
        
        p = mp.Process(
            target=_worker_process,
            args=(config, network_state_dict, result_queue, games_per_worker, worker_log_idx, game_id_start, training_step)
        )
        p.start()
        processes.append(p)

    total_moves = 0
    for _ in range(num_workers):
        samples, moves, log_data = result_queue.get()
        replay_buffer.add_samples(samples)
        total_moves += moves
        
        if log_data is not None and game_logger is not None:
            game_logger.start_game(log_data["game_id"], log_data["training_step"])
            for move_uci in log_data["moves"]:
                game_logger.log_move(move_uci)
            game_logger.end_game(log_data["result"])

    for p in processes:
        p.join()

    return total_moves
