import torch
import chess
from core.env_wrapper import ChessWrapper
from core.network import ChessGumbelNet
from core.mcts_gumbel import GumbelMCTS, Node
from interpretability.saliency import SaliencyMap
from interpretability.tree_viz import TreeVisualizer

class AnalysisDashboard:
    def __init__(self, config, model_path):
        self.config = config
        self.network = ChessGumbelNet(config)
        self.network.load_state_dict(torch.load(model_path, map_location=config.device))
        self.network.to(config.device)
        self.network.eval()
        
        self.env = ChessWrapper(config)
        self.mcts = GumbelMCTS(config)
        self.saliency = SaliencyMap(self.network, config.device)
        self.viz = TreeVisualizer(config)

    def analyze_position(self, fen, output_prefix="analysis"):
        self.env.board.set_fen(fen)
        state_tensor = self.env.get_tensor()
        legal_actions = self.env.get_legal_actions()

        if not legal_actions:
            return "Game Over or Invalid Position"

        # 1. Ejecutar busqueda
        print(f"Analyzing FEN: {fen}")
        root = Node(0) # Recreamos raiz para visualizacion manual
        
        # Hack para extraer el arbol interno: ejecutamos mcts modificado o reconstruimos
        # Para este dashboard, corremos una busqueda nueva y visualizamos ese arbol
        
        # Nota: GumbelMCTS en la version optimizada no devuelve el objeto raiz completo publicamente
        # Modificamos mcts.run_search para devolver root (ver mcts_gumbel.py ajuste previo)
        # Asumimos que mcts.run_search retorna: best_action, completed_q, counts, root_node (si se ajusto)
        # Como en el codigo previo no retornaba root, aqui simulamos la extraccion del prior
        
        best_action, _, _ = self.mcts.run_search(state_tensor, self.network, legal_actions)
        
        # 2. Generar Saliency Map
        heatmap = self.saliency.compute_saliency(state_tensor)
        self.saliency.save_heatmap(heatmap, f"{output_prefix}_saliency.png")

        # 3. Reporte de Texto
        move_name = self.env.index_lookup.get(best_action, "Unknown")
        print(f"Recommended Move: {move_name}")
        print(f"Saliency map saved to {output_prefix}_saliency.png")
        
        # Visualizacion del arbol requiere que MCTS retorne el objeto root. 
        # Si se usa el codigo estandar provisto, este paso es conceptual.
        return {
            "fen": fen,
            "best_move": move_name,
            "saliency_path": f"{output_prefix}_saliency.png"
        }