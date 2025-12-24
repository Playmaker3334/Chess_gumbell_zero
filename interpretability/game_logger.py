import os
import chess
from datetime import datetime
from typing import Optional


class GameLogger:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        self.current_game_id = 0
        self.current_step = 0
        self.current_file = None
        self.move_number = 0
        self.is_white_turn = True
        self.board = None
        
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
    
    def start_game(self, game_id: int, training_step: int):
        self.current_game_id = game_id
        self.current_step = training_step
        self.move_number = 1
        self.is_white_turn = True
        self.board = chess.Board()
        
        filename = f"game_{game_id:04d}_step_{training_step:06d}.txt"
        filepath = os.path.join(self.log_dir, filename)
        self.current_file = open(filepath, "w", encoding="utf-8")
        
        self._write_header()
        self._write_board_state("Posicion inicial")
    
    def _write_header(self):
        separator = "=" * 64
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        self.current_file.write(f"{separator}\n")
        self.current_file.write(f"PARTIDA #{self.current_game_id} | Training Step: {self.current_step}\n")
        self.current_file.write(f"Fecha: {timestamp}\n")
        self.current_file.write(f"{separator}\n\n")
        self.current_file.flush()
    
    def _get_board_visual(self) -> str:
        if self.board is None:
            return ""
        
        lines = []
        lines.append("    a   b   c   d   e   f   g   h")
        lines.append("  +---+---+---+---+---+---+---+---+")
        
        piece_symbols = {
            "P": "P", "N": "N", "B": "B", "R": "R", "Q": "Q", "K": "K",
            "p": "p", "n": "n", "b": "b", "r": "r", "q": "q", "k": "k"
        }
        
        for rank in range(7, -1, -1):
            row = f"{rank + 1} |"
            for file in range(8):
                square = chess.square(file, rank)
                piece = self.board.piece_at(square)
                if piece:
                    symbol = piece_symbols.get(piece.symbol(), "?")
                    row += f" {symbol} |"
                else:
                    row += "   |"
            lines.append(row)
            lines.append("  +---+---+---+---+---+---+---+---+")
        
        lines.append("    a   b   c   d   e   f   g   h")
        
        return "\n".join(lines)
    
    def _write_board_state(self, caption: str = ""):
        if self.board is None:
            return
        
        if caption:
            self.current_file.write(f"{caption}\n")
        self.current_file.write(self._get_board_visual())
        self.current_file.write("\n\n")
        self.current_file.flush()
    
    def log_move(self, move_uci: str):
        if self.current_file is None or self.board is None:
            return
        
        try:
            move = chess.Move.from_uci(move_uci)
            move_san = self.board.san(move)
        except:
            move_san = move_uci
        
        separator_small = "-" * 40
        
        if self.is_white_turn:
            turn_str = "Blancas"
        else:
            turn_str = "Negras"
        
        self.current_file.write(f"{separator_small}\n")
        self.current_file.write(f"Jugada {self.move_number} | Turno: {turn_str}\n")
        self.current_file.write(f"Movimiento: {move_san} ({move_uci})\n")
        self.current_file.write(f"{separator_small}\n\n")
        
        try:
            self.board.push(move)
        except:
            pass
        
        self._write_board_state()
        
        if not self.is_white_turn:
            self.move_number += 1
        
        self.is_white_turn = not self.is_white_turn
    
    def log_move_by_index(self, action_idx: int, index_lookup: dict):
        move_uci = index_lookup.get(action_idx)
        if move_uci:
            self.log_move(move_uci)
    
    def end_game(self, result: float, reason: Optional[str] = None):
        if self.current_file is None:
            return
        
        separator = "=" * 64
        
        if result == 1:
            result_str = "Victoria Blancas (1-0)"
        elif result == -1:
            result_str = "Victoria Negras (0-1)"
        else:
            result_str = "Tablas (1/2-1/2)"
        
        total_moves = (self.move_number - 1) * 2
        if not self.is_white_turn:
            total_moves += 1
        
        if reason is None:
            if self.board and self.board.is_checkmate():
                reason = "Jaque mate"
            elif self.board and self.board.is_stalemate():
                reason = "Ahogado"
            elif self.board and self.board.is_insufficient_material():
                reason = "Material insuficiente"
            elif self.board and self.board.is_fifty_moves():
                reason = "Regla de 50 movimientos"
            elif self.board and self.board.is_repetition():
                reason = "Repeticion triple"
            else:
                reason = "Fin de partida"
        
        self.current_file.write(f"{separator}\n")
        self.current_file.write(f"RESULTADO: {result_str}\n")
        self.current_file.write(f"Movimientos totales: {total_moves}\n")
        self.current_file.write(f"Razon: {reason}\n")
        self.current_file.write(f"{separator}\n")
        
        self.current_file.close()
        self.current_file = None
        self.board = None
    
    def close(self):
        if self.current_file is not None:
            self.current_file.close()
            self.current_file = None
