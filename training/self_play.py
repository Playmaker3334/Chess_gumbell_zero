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
        game_history = [] # (state, policy, player_color)
        done = False
        
        moves_count = 0
        
        while not done:
            legal_actions = env.get_legal_actions()
            
            # Si no hay movimientos legales (mate o ahogado)
            if not legal_actions:
                break
                
            # Ejecutar Gumbel Search
            best_action, improved_policy, _ = mcts.run_search(state, network, legal_actions)
            
            # Guardar estado actual para entrenamiento
            # Nota: Guardamos la policy completa (size 4672), GumbelMCTS ya nos la da formateada
            game_history.append((state.clone(), improved_policy, env.board.turn))
            
            # Aplicar acción al entorno
            state, reward, done = env.step(best_action)
            moves_count += 1
            
            # Limite para evitar partidas infinitas al inicio
            if moves_count > 200:
                done = True
                reward = 0 # Tablas por aburrimiento

        # Guardar partida en el buffer
        # reward es relativo al último jugador. Ajustar lógica en replay_buffer
        replay_buffer.save_game(game_history, reward)
        
    return moves_count