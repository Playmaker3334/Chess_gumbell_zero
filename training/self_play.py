import torch
import numpy as np
from core.env_wrapper import ChessWrapper
from core.mcts_gumbel import GumbelMCTS


def self_play_worker(config, network, replay_buffer, num_games=1, game_logger=None, game_counter=0, training_step=0):
    env = ChessWrapper(config)
    mcts = GumbelMCTS(config)
    
    network.eval()
    
    total_moves = 0
    
    for game_idx in range(num_games):
        state = env.reset()
        game_history = [] 
        done = False
        moves_count = 0
        
        should_log = (game_logger is not None and 
                      config.log_games and 
                      (game_counter + game_idx) % config.log_every_n_games == 0)
        
        if should_log:
            game_logger.start_game(game_counter + game_idx, training_step)
        
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
            
            game_history.append((state.clone(), torch.tensor(policy_target), env.board.turn))
            
            if should_log:
                game_logger.log_move_by_index(best_action, env.index_lookup)
            
            state, reward, done = env.step(best_action)
            moves_count += 1
            
            max_moves = getattr(config, 'max_moves_per_game', 150)
            if moves_count > max_moves:
                done = True
                reward = 0

        replay_buffer.save_game(game_history, reward)
        total_moves += moves_count
        
        if should_log:
            game_logger.end_game(reward)
        
    return total_moves
