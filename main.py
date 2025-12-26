import os
import torch
import argparse
import multiprocessing as mp
from config import Config
from core.network import ChessGumbelNet
from data.replay_buffer import ReplayBuffer
from data.dataset import ChessDataset
from torch.utils.data import DataLoader
from training.self_play import self_play_worker
from training.parallel_self_play import parallel_self_play
from training.trainer import GumbelTrainer
from evaluation.benchmark import Benchmarker
from interpretability.game_logger import GameLogger
from interpretability.gif_logger import GifLogger


class CombinedLogger:
    """Wrapper que combina GameLogger (txt) y GifLogger (gif)."""
    
    def __init__(self, text_logger, gif_logger):
        self.text_logger = text_logger
        self.gif_logger = gif_logger
    
    def start_game(self, game_id: int, training_step: int):
        if self.text_logger:
            self.text_logger.start_game(game_id, training_step)
        if self.gif_logger:
            self.gif_logger.start_game(game_id, training_step)
    
    def log_move(self, move_uci: str):
        if self.text_logger:
            self.text_logger.log_move(move_uci)
        if self.gif_logger:
            self.gif_logger.log_move(move_uci)
    
    def log_move_by_index(self, action_idx: int, index_lookup: dict):
        if self.text_logger:
            self.text_logger.log_move_by_index(action_idx, index_lookup)
        if self.gif_logger:
            self.gif_logger.log_move_by_index(action_idx, index_lookup)
    
    def end_game(self, result: float, reason=None):
        if self.text_logger:
            self.text_logger.end_game(result, reason)
        if self.gif_logger:
            self.gif_logger.end_game(result, reason)
    
    def close(self):
        if self.text_logger:
            self.text_logger.close()
        if self.gif_logger:
            self.gif_logger.close()


def main():
    parser = argparse.ArgumentParser(description="Gumbel MuZero Chess")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval", "play"])
    parser.add_argument("--model_path", type=str, default="checkpoints/latest.pt")
    parser.add_argument("--parallel", action="store_true", help="Usar self-play paralelo")
    parser.add_argument("--no-log", action="store_true", help="Desactivar logging de partidas")
    parser.add_argument("--no-gif", action="store_true", help="Desactivar generacion de GIFs")
    args = parser.parse_args()

    config = Config()
    
    if args.no_log:
        config.log_games = False

    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")
    if not os.path.exists("data"):
        os.makedirs("data")

    if args.mode == "train":
        network = ChessGumbelNet(config).to(config.device)
        if os.path.exists(args.model_path):
            print(f"Loading model from {args.model_path}")
            network.load_state_dict(torch.load(args.model_path, map_location=config.device))

        replay_buffer = ReplayBuffer(config)
        trainer = GumbelTrainer(config, network)
        
        game_logger = None
        if config.log_games:
            text_logger = GameLogger(log_dir=config.log_dir)
            
            gif_logger = None
            if not args.no_gif:
                gif_logger = GifLogger(output_dir="gifs", gif_every_n_games=config.log_every_n_games)
                print(f"GIF logging enabled. GIFs will be saved to: gifs/")
            
            game_logger = CombinedLogger(text_logger, gif_logger)
            print(f"Game logging enabled. Logs will be saved to: {config.log_dir}/")

        if os.path.exists("data/buffer.pkl"):
            replay_buffer.load_buffer("data/buffer.pkl")

        dataset = ChessDataset(replay_buffer, config.batch_size)
        dataloader = DataLoader(dataset, batch_size=None, batch_sampler=None)
        data_iter = iter(dataloader)

        use_parallel = args.parallel
        if use_parallel:
            print(f"Starting Training Loop (Parallel: {config.num_self_play_workers} workers)...")
        else:
            print("Starting Training Loop (Sequential)...")

        game_counter = 0

        for step in range(config.training_steps):
            if use_parallel:
                num_moves = parallel_self_play(
                    config, network, replay_buffer,
                    num_workers=config.num_self_play_workers,
                    games_per_worker=config.games_per_worker,
                    game_logger=game_logger,
                    game_counter=game_counter,
                    training_step=step
                )
                games_played = config.num_self_play_workers * config.games_per_worker
            else:
                num_moves = self_play_worker(
                    config, network, replay_buffer, 
                    num_games=1,
                    game_logger=game_logger,
                    game_counter=game_counter,
                    training_step=step
                )
                games_played = 1

            game_counter += games_played
            print(f"Step {step}: Generated {num_moves} moves ({games_played} games). Buffer size: {len(replay_buffer)}")

            if len(replay_buffer) > config.batch_size:
                for _ in range(30):
                    batch = next(data_iter)
                    logs = trainer.train_step(batch)

                if step % 10 == 0:
                    print(f"Loss: {logs['total_loss']:.4f} (Pol: {logs['policy_loss']:.4f}, Val: {logs['value_loss']:.4f})")

            if step > 0 and step % config.checkpoint_interval == 0:
                torch.save(network.state_dict(), f"checkpoints/step_{step}.pt")
                torch.save(network.state_dict(), "checkpoints/latest.pt")
                replay_buffer.save_buffer("data/buffer.pkl")

                bench = Benchmarker(config, "checkpoints/latest.pt")
                w, l, d = bench.play_vs_random(num_games=5)
                print(f"Benchmark: {w}W - {l}L - {d}D")

        if game_logger is not None:
            game_logger.close()
            print(f"Training complete. Game logs saved to: {config.log_dir}/")

    elif args.mode == "eval":
        from evaluation.puzzles import PuzzleSolver, get_basic_mate_in_one
        solver = PuzzleSolver(config, args.model_path)
        solver.solve(get_basic_mate_in_one())


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()