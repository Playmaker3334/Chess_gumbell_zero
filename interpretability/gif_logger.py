import os
import chess
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
import numpy as np
from PIL import Image
from io import BytesIO
from typing import List, Optional
from datetime import datetime


PIECE_UNICODE = {
    'K': '♔', 'Q': '♕', 'R': '♖', 'B': '♗', 'N': '♘', 'P': '♙',
    'k': '♚', 'q': '♛', 'r': '♜', 'b': '♝', 'n': '♞', 'p': '♟'
}

LIGHT_SQUARE = '#F0D9B5'
DARK_SQUARE = '#B58863'
HIGHLIGHT_FROM = '#AAD4AA'
HIGHLIGHT_TO = '#7FBF7F'


class GifLogger:
    """
    Genera GIFs animados de partidas de ajedrez durante el entrenamiento.
    Compatible con Kaggle (usa matplotlib + PIL, sin pygame/tkinter).
    """
    
    def __init__(self, output_dir: str = "gifs", gif_every_n_games: int = 10,
                 frame_duration: int = 700, board_size: int = 6):
        self.output_dir = output_dir
        self.gif_every_n_games = gif_every_n_games
        self.frame_duration = frame_duration
        self.board_size = board_size
        
        self.current_game_id = 0
        self.training_step = 0
        self.frames: List[Image.Image] = []
        self.board: Optional[chess.Board] = None
        self.last_move: Optional[chess.Move] = None
        self.move_number = 0
        self.is_recording = False
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def _render_board_to_image(self, title: str = "", subtitle: str = "") -> Image.Image:
        fig, ax = plt.subplots(figsize=(self.board_size, self.board_size + 0.8))
        ax.set_xlim(-0.5, 7.5)
        ax.set_ylim(-0.8, 8.0)
        ax.set_aspect('equal')
        ax.axis('off')
        fig.patch.set_facecolor('#FFFFFF')
        
        for row in range(8):
            for col in range(8):
                is_light = (row + col) % 2 == 1
                color = LIGHT_SQUARE if is_light else DARK_SQUARE
                
                if self.last_move is not None:
                    from_sq = self.last_move.from_square
                    to_sq = self.last_move.to_square
                    from_row, from_col = chess.square_rank(from_sq), chess.square_file(from_sq)
                    to_row, to_col = chess.square_rank(to_sq), chess.square_file(to_sq)
                    
                    if row == from_row and col == from_col:
                        color = HIGHLIGHT_FROM
                    elif row == to_row and col == to_col:
                        color = HIGHLIGHT_TO
                
                rect = patches.Rectangle((col - 0.5, row - 0.5), 1, 1,
                                          linewidth=0, facecolor=color)
                ax.add_patch(rect)
        
        border = patches.Rectangle((-0.5, -0.5), 8, 8, linewidth=2,
                                    edgecolor='#333333', facecolor='none')
        ax.add_patch(border)
        
        if self.board is not None:
            for square in chess.SQUARES:
                piece = self.board.piece_at(square)
                if piece:
                    row = chess.square_rank(square)
                    col = chess.square_file(square)
                    symbol = PIECE_UNICODE.get(piece.symbol(), '?')
                    
                    if piece.color == chess.WHITE:
                        ax.text(col, row, symbol, fontsize=36, ha='center', va='center',
                                color='white', fontweight='bold',
                                path_effects=[
                                    path_effects.Stroke(linewidth=2, foreground='black'),
                                    path_effects.Normal()
                                ])
                    else:
                        ax.text(col, row, symbol, fontsize=36, ha='center', va='center',
                                color='#222222', fontweight='bold')
        
        files = 'abcdefgh'
        for i in range(8):
            ax.text(i, -0.65, files[i], fontsize=11, ha='center', va='top',
                    color='#555555', fontweight='bold')
            ax.text(-0.4, i, str(i + 1), fontsize=11, ha='right', va='center',
                    color='#555555', fontweight='bold')
        
        if title:
            ax.text(3.5, 7.7, title, fontsize=13, ha='center', va='bottom',
                    fontweight='bold', color='#222222')
        
        if subtitle:
            ax.text(3.5, 7.35, subtitle, fontsize=10, ha='center', va='bottom',
                    color='#666666')
        
        plt.tight_layout(pad=0.5)
        
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close(fig)
        buf.seek(0)
        image = Image.open(buf).convert('RGB')
        
        return image
    
    def start_game(self, game_id: int, training_step: int):
        self.current_game_id = game_id
        self.training_step = training_step
        self.frames = []
        self.board = chess.Board()
        self.last_move = None
        self.move_number = 0
        self.is_recording = True
        
        title = f"Game #{game_id} | Step {training_step}"
        subtitle = "White to move"
        frame = self._render_board_to_image(title, subtitle)
        self.frames.append(frame)
    
    def log_move(self, move_uci: str):
        if not self.is_recording or self.board is None:
            return
        
        try:
            move = chess.Move.from_uci(move_uci)
            move_san = self.board.san(move)
            
            self.board.push(move)
            self.last_move = move
            self.move_number += 1
            
            title = f"Game #{self.current_game_id} | Move {self.move_number}: {move_san}"
            
            if self.board.is_game_over():
                if self.board.is_checkmate():
                    subtitle = "Checkmate!"
                elif self.board.is_stalemate():
                    subtitle = "Stalemate"
                else:
                    subtitle = "Game Over"
            else:
                subtitle = "White to move" if self.board.turn == chess.WHITE else "Black to move"
            
            frame = self._render_board_to_image(title, subtitle)
            self.frames.append(frame)
            
        except Exception as e:
            print(f"GifLogger: Error en movimiento {move_uci}: {e}")
    
    def log_move_by_index(self, action_idx: int, index_lookup: dict):
        move_uci = index_lookup.get(action_idx)
        if move_uci:
            self.log_move(move_uci)
    
    def end_game(self, result: float, reason: Optional[str] = None):
        if not self.is_recording or not self.frames:
            self.is_recording = False
            return None
        
        if result == 1:
            result_str = "1-0 White wins"
        elif result == -1:
            result_str = "0-1 Black wins"
        else:
            result_str = "½-½ Draw"
        
        if reason is None:
            if self.board and self.board.is_checkmate():
                reason = "Checkmate"
            elif self.board and self.board.is_stalemate():
                reason = "Stalemate"
            elif self.board and self.board.is_insufficient_material():
                reason = "Insufficient material"
            elif self.board and self.board.is_fifty_moves():
                reason = "50-move rule"
            elif self.board and self.board.is_repetition():
                reason = "Repetition"
            else:
                reason = "Game over"
        
        title = f"Game #{self.current_game_id} | {result_str}"
        subtitle = reason
        final_frame = self._render_board_to_image(title, subtitle)
        
        for _ in range(4):
            self.frames.append(final_frame)
        
        filename = f"game_{self.current_game_id:04d}_step_{self.training_step:06d}.gif"
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            self.frames[0].save(
                filepath,
                save_all=True,
                append_images=self.frames[1:],
                duration=self.frame_duration,
                loop=0,
                optimize=True
            )
            print(f"GIF guardado: {filepath} ({len(self.frames)} frames, {self.move_number} movimientos)")
            self.is_recording = False
            return filepath
        except Exception as e:
            print(f"GifLogger: Error guardando GIF: {e}")
            self.is_recording = False
            return None
    
    def close(self):
        self.frames = []
        self.board = None
        self.is_recording = False