import torch
import numpy as np
from core.env_wrapper import ChessWrapper
from core.mcts_gumbel import GumbelMCTS

def self_play_worker(config, network, replay_buffer, num_games=1):
    env = ChessWrapper(config)
    mcts = GumbelMCTS(config)
    
    network.eval()
    
    for _ in range(num_games):
        state = env.reset()
        game_history = [] 
        done = False
        moves_count = 0
        
        while not done:
            legal_actions = env.get_legal_actions()
            
            if not legal_actions:
                break
                
            # Ejecutar Gumbel Search
            # counts: diccionario {accion_idx: numero_visitas}
            best_action, _, counts = mcts.run_search(state, network, legal_actions)
            
            # --- CORRECCIÓN MATEMÁTICA CRÍTICA ---
            # Antes pasábamos 'improved_policy' (Q-values) lo que causaba NaN.
            # Ahora construimos una distribución de probabilidad basada en Visitas.
            # Esto garantiza valores entre 0 y 1.
            
            policy_target = np.zeros(config.action_space_size, dtype=np.float32)
            total_visits = sum(counts.values())
            
            if total_visits > 0:
                for action_idx, visit_count in counts.items():
                    policy_target[action_idx] = visit_count / total_visits
            else:
                # Fallback por si acaso (distribución uniforme sobre legales)
                for action_idx in legal_actions:
                    policy_target[action_idx] = 1.0 / len(legal_actions)
            
            # Guardamos el tensor de probabilidad correcto
            game_history.append((state.clone(), torch.tensor(policy_target), env.board.turn))
            # -------------------------------------
            
            state, reward, done = env.step(best_action)
            moves_count += 1
            
            # Limite de movimientos para evitar bucles infinitos
            if moves_count > 150:
                done = True
                reward = 0 # Tablas

        replay_buffer.save_game(game_history, reward)
        
    return moves_count

