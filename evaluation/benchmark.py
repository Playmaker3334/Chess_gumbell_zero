import torch
import numpy as np
from core.env_wrapper import ChessWrapper
from core.network import ChessGumbelNet
from core.mcts_gumbel import GumbelMCTS

class RandomAgent:
    def select_action(self, legal_actions):
        return np.random.choice(legal_actions)

class Benchmarker:
    def __init__(self, config, model_path=None):
        self.config = config
        self.env = ChessWrapper(config)
        
        if model_path:
            self.network = ChessGumbelNet(config)
            self.network.load_state_dict(torch.load(model_path, map_location=config.device))
            self.network.to(config.device)
            self.network.eval()
            self.mcts = GumbelMCTS(config)
            self.agent_type = "Gumbel"
        else:
            self.agent_type = "Random"

    def play_vs_random(self, num_games=10):
        wins, losses, draws = 0, 0, 0
        random_opponent = RandomAgent()

        print(f"Starting Benchmark: {self.agent_type} vs Random ({num_games} games)")

        for i in range(num_games):
            self.env.reset()
            state = self.env.get_tensor()
            done = False
            
            # Alternar colores: Pares juega Modelo como White
            model_is_white = (i % 2 == 0)
            
            while not done:
                legal_actions = self.env.get_legal_actions()
                if not legal_actions: break

                is_white_turn = self.env.board.turn
                
                if (is_white_turn and model_is_white) or (not is_white_turn and not model_is_white):
                    # Turno del Modelo
                    if self.agent_type == "Gumbel":
                        action, _, _ = self.mcts.run_search(state, self.network, legal_actions)
                    else:
                        action = random_opponent.select_action(legal_actions)
                else:
                    # Turno del Random
                    action = random_opponent.select_action(legal_actions)

                state, reward, done = self.env.step(action)

            # Resultado desde perspectiva de White: 1 gana White, -1 gana Black, 0 Draw
            # Si model_is_white y reward=1 -> Win
            # Si model_is_white y reward=-1 -> Loss
            
            final_score = 0
            result = self.env.board.result()
            if result == "1-0": final_score = 1
            elif result == "0-1": final_score = -1
            
            if final_score == 0:
                draws += 1
            elif (model_is_white and final_score == 1) or (not model_is_white and final_score == -1):
                wins += 1
            else:
                losses += 1

            print(f"Game {i+1}: Result {result} (Model White: {model_is_white})")

        return wins, losses, draws