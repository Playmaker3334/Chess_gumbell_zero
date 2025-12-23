import os
import torch
import argparse
from config import Config
from core.network import ChessGumbelNet
from data.replay_buffer import ReplayBuffer
from data.dataset import ChessDataset
from torch.utils.data import DataLoader
from training.self_play import self_play_worker
from training.trainer import GumbelTrainer
from evaluation.benchmark import Benchmarker

def main():
    parser = argparse.ArgumentParser(description="Gumbel MuZero Chess")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval", "play"])
    parser.add_argument("--model_path", type=str, default="checkpoints/latest.pt")
    args = parser.parse_args()

    config = Config()
    
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")
    
    if args.mode == "train":
        network = ChessGumbelNet(config).to(config.device)
        if os.path.exists(args.model_path):
            print(f"Loading model from {args.model_path}")
            network.load_state_dict(torch.load(args.model_path))

        replay_buffer = ReplayBuffer(config)
        trainer = GumbelTrainer(config, network)
        
        # Cargar buffer previo si existe
        if os.path.exists("data/buffer.pkl"):
            replay_buffer.load_buffer("data/buffer.pkl")

        dataset = ChessDataset(replay_buffer, config.batch_size)
        dataloader = DataLoader(dataset, batch_size=None, batch_sampler=None)
        data_iter = iter(dataloader)

        print("Starting Training Loop...")
        for step in range(config.training_steps):
            # 1. Self Play (Generar datos)
            # En implementacion real, esto corre en procesos paralelos
            num_moves = self_play_worker(config, network, replay_buffer, num_games=1)
            print(f"Step {step}: Generated {num_moves} moves. Buffer size: {len(replay_buffer)}")

            # 2. Training (Ajustar pesos)
            if len(replay_buffer) > config.batch_size:
                for _ in range(10): # Train N steps per game
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
    main()