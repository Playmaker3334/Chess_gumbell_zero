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

def main():
    parser = argparse.ArgumentParser(description="Gumbel MuZero Chess")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval", "play"])
    parser.add_argument("--model_path", type=str, default="checkpoints/latest.pt")
    parser.add_argument("--parallel", action="store_true", help="Usar self-play paralelo")
    args = parser.parse_args()

    config = Config()

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

        # Cargar buffer previo si existe
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

        for step in range(config.training_steps):
            # 1. Self Play (Generar datos)
            if use_parallel:
                num_moves = parallel_self_play(
                    config, network, replay_buffer,
                    num_workers=config.num_self_play_workers,
                    games_per_worker=config.games_per_worker
                )
                games_played = config.num_self_play_workers * config.games_per_worker
            else:
                num_moves = self_play_worker(config, network, replay_buffer, num_games=1)
                games_played = 1

            print(f"Step {step}: Generated {num_moves} moves ({games_played} games). Buffer size: {len(replay_buffer)}")

            # 2. Training (Ajustar pesos)
            if len(replay_buffer) > config.batch_size:
                for _ in range(30):  # Train N steps per game
                    batch = next(data_iter)
                    logs = trainer.train_step(batch)

                if step % 10 == 0:
                    print(f"Loss: {logs['total_loss']:.4f} (Pol: {logs['policy_loss']:.4f}, Val: {logs['value_loss']:.4f})")

            # 3. Checkpointing & Eval
            if step > 0 and step % config.checkpoint_interval == 0:
                torch.save(network.state_dict(), f"checkpoints/step_{step}.pt")
                torch.save(network.state_dict(), "checkpoints/latest.pt")
                replay_buffer.save_buffer("data/buffer.pkl")

                # Run benchmark
                bench = Benchmarker(config, "checkpoints/latest.pt")
                w, l, d = bench.play_vs_random(num_games=5)
                print(f"Benchmark: {w}W - {l}L - {d}D")

    elif args.mode == "eval":
        from evaluation.puzzles import PuzzleSolver, get_basic_mate_in_one
        solver = PuzzleSolver(config, args.model_path)
        solver.solve(get_basic_mate_in_one())

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
