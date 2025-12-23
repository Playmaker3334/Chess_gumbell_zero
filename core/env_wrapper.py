%%writefile Chess_gumbell_zero/core/env_wrapper.py
import chess
import numpy as np
import torch

class ChessWrapper:
    def __init__(self, config):
        self.board = chess.Board()
        self.config = config
        self.move_lookup = {move: i for i, move in enumerate(self._generate_legal_moves_vocab())}
        self.index_lookup = {i: move for move, i in self.move_lookup.items()}

    def reset(self):
        self.board.reset()
        return self.get_tensor()

    def step(self, action_idx):
        move_uci = self.index_lookup.get(action_idx)
        move = chess.Move.from_uci(move_uci)
        if move in self.board.legal_moves:
            self.board.push(move)
            done = self.board.is_game_over()
            reward = 0
            if done:
                result = self.board.result()
                if result == "1-0": reward = 1
                elif result == "0-1": reward = -1
            return self.get_tensor(), reward, done
        else:
            return self.get_tensor(), -1, True

    def get_legal_actions(self):
        legal_moves = []
        for move in self.board.legal_moves:
            uci = move.uci()
            if uci in self.move_lookup:
                legal_moves.append(self.move_lookup[uci])
        return legal_moves

    def get_tensor(self):
        history_planes = 8
        planes = np.zeros((119, 8, 8), dtype=np.float32)
        
        current_turn = self.board.turn
        moves_popped = [] # Lista para guardar movimientos y restaurarlos luego
        
        # 1. Viajar al pasado y llenar planos
        for i in range(history_planes):
            if i > 0:
                try:
                    move = self.board.pop() # Sacamos el movimiento
                    moves_popped.append(move) # Lo guardamos
                except IndexError:
                    break # Si no hay mas historial, paramos
            
            offset = i * 14
            
            for square in chess.SQUARES:
                row, col = divmod(square, 8)
                piece = self.board.piece_at(square)
                if piece:
                    rank = 0
                    if piece.color != current_turn:
                        rank = 6
                    
                    piece_type_idx = piece.piece_type - 1
                    plane_idx = offset + rank + piece_type_idx
                    planes[plane_idx][7-row][col] = 1

            if self.board.is_repetition(2):
                planes[offset + 12][:][:] = 1
            if self.board.is_repetition(3):
                planes[offset + 13][:][:] = 1

        # 2. Restaurar el tablero (Push de vuelta los movimientos guardados)
        # Debemos hacerlo en orden inverso (el ultimo que sacamos es el ultimo en volver)
        for move in reversed(moves_popped):
            self.board.push(move)

        # 3. Planos globales (Castling, turn, etc)
        aux_offset = 112
        if self.board.turn == chess.BLACK:
            planes[aux_offset][:][:] = 1
        
        planes[aux_offset + 1][:][:] = self.board.fullmove_number / 100.0
        
        if self.board.has_kingside_castling_rights(chess.WHITE): planes[aux_offset + 2][:][:] = 1
        if self.board.has_queenside_castling_rights(chess.WHITE): planes[aux_offset + 3][:][:] = 1
        if self.board.has_kingside_castling_rights(chess.BLACK): planes[aux_offset + 4][:][:] = 1
        if self.board.has_queenside_castling_rights(chess.BLACK): planes[aux_offset + 5][:][:] = 1
        
        planes[aux_offset + 6][:][:] = self.board.halfmove_clock / 100.0

        return torch.tensor(planes).unsqueeze(0)

    def _generate_legal_moves_vocab(self):
        moves = []
        dirs = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
        knight_moves = [(2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)]
        
        for square in range(64):
            src_row, src_col = divmod(square, 8)
            
            for dr, dc in dirs:
                for dist in range(1, 8):
                    dst_row, dst_col = src_row + dr * dist, src_col + dc * dist
                    if 0 <= dst_row < 8 and 0 <= dst_col < 8:
                        moves.append(self._coord_to_uci(src_row, src_col, dst_row, dst_col))
                    else:
                        break
                        
            for dr, dc in knight_moves:
                dst_row, dst_col = src_row + dr, src_col + dc
                if 0 <= dst_row < 8 and 0 <= dst_col < 8:
                    moves.append(self._coord_to_uci(src_row, src_col, dst_row, dst_col))
                    
            if src_row == 6: 
                for dc in [-1, 0, 1]:
                    dst_row = 7
                    dst_col = src_col + dc
                    if 0 <= dst_col < 8:
                        base = self._coord_to_uci(src_row, src_col, dst_row, dst_col)
                        for promo in ['n', 'b', 'r']: 
                            moves.append(base + promo)
        
        return sorted(list(set(moves)))

    def _coord_to_uci(self, r1, c1, r2, c2):
        files = "abcdefgh"
        return f"{files[c1]}{r1+1}{files[c2]}{r2+1}"
