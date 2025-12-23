import torch
import chess
from core.env_wrapper import ChessWrapper
from core.network import ChessGumbelNet
from core.mcts_gumbel import GumbelMCTS

class PuzzleSolver:
    def __init__(self, config, model_path):
        self.config = config
        self.network = ChessGumbelNet(config)
        self.network.load_state_dict(torch.load(model_path, map_location=config.device))
        self.network.to(config.device)
        self.network.eval()
        self.mcts = GumbelMCTS(config)
        self.env = ChessWrapper(config)

    def solve(self, puzzles):
        """
        puzzles: list of tuples (fen, solution_san)
        Example: [("r1bqkbnr/pppp1ppp/2n5/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 3 3", "Qxf7#")]
        """
        correct = 0
        total = len(puzzles)
        
        print(f"Starting Puzzle Evaluation on {total} positions...")
        
        for fen, solution_san in puzzles:
            self.env.board.set_fen(fen)
            state_tensor = self.env.get_tensor()
            legal_actions = self.env.get_legal_actions()
            
            best_action_idx, _, _ = self.mcts.run_search(state_tensor, self.network, legal_actions)
            
            best_move_uci = self.env.index_lookup.get(best_action_idx)
            best_move_san = self.env.board.san(chess.Move.from_uci(best_move_uci))
            
            is_correct = (best_move_san == solution_san)
            if is_correct:
                correct += 1
                print(f"[PASS] FEN: {fen[:20]}... | Model: {best_move_san}")
            else:
                print(f"[FAIL] FEN: {fen[:20]}... | Model: {best_move_san} | Sol: {solution_san}")
                
        accuracy = (correct / total) * 100
        print(f"Puzzle Accuracy: {accuracy:.2f}%")
        return accuracy

def get_basic_mate_in_one():
    return [
        ("r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 4 4", "Qxf7#"),
        ("6k1/5ppp/8/8/8/8/8/4R1K1 w - - 0 1", "Re8#"),
        ("rnbqkbnr/ppppp2p/5p2/6p1/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2", "Qh5#")
    ]