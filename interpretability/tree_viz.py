import graphviz
from core.env_wrapper import ChessWrapper

class TreeVisualizer:
    def __init__(self, config):
        self.config = config
        self.wrapper = ChessWrapper(config) 

    def visualize(self, root_node, filename="search_tree", format="png"):
        dot = graphviz.Digraph(comment='Gumbel Search Tree')
        dot.attr(rankdir='TB')

        def traverse(node, node_id, parent_id=None, action_taken=None):
            label = f"N:{node.visit_count}\nQ:{node.q_value:.2f}\nP:{node.prior:.2f}"
            
            color = "white"
            if node.q_value > 0.5: color = "#ffcccc" 
            elif node.q_value < -0.5: color = "#ccccff"
            
            dot.node(node_id, label, style="filled", fillcolor=color, shape="box")

            if parent_id is not None:
                move_uci = self.wrapper.index_lookup.get(action_taken, str(action_taken))
                edge_label = f"{move_uci}"
                dot.edge(parent_id, node_id, label=edge_label)

            for action, child in node.children.items():
                child_id = f"{node_id}_{action}"
                traverse(child, child_id, node_id, action)

        traverse(root_node, "root")
        dot.render(filename, format=format, cleanup=True)
        return f"{filename}.{format}"