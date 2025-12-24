import math
import numpy as np
import torch


class Node:
    def __init__(self, prior):
        self.visit_count = 0
        self.value_sum = 0
        self.children = {}
        self.prior = prior
        self.q_value = 0

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count


class GumbelMCTS:
    def __init__(self, config):
        self.config = config
        self.c_visit = config.c_visit
        self.c_scale = config.c_scale

    def run_search(self, root_state, network, legal_actions):
        if not legal_actions:
            return None, None, {}

        legal_actions_set = set(legal_actions)

        network.eval()
        with torch.no_grad():
            root_input = root_state.to(self.config.device)
            policy_logits, root_value = network(root_input)
            policy_logits = policy_logits.cpu().numpy()[0]

        root = Node(0)

        legal_mask = np.full(policy_logits.shape, -np.inf)
        legal_mask[legal_actions] = 0
        masked_logits = policy_logits + legal_mask

        gumbel_noise = np.random.gumbel(size=masked_logits.shape)
        perturbed_logits = masked_logits + gumbel_noise

        m = min(self.config.num_sampled_actions, len(legal_actions))
        top_k_indices = np.argsort(perturbed_logits)[-m:]
        
        top_k_indices = [a for a in top_k_indices if a in legal_actions_set]
        
        if not top_k_indices:
            top_k_indices = legal_actions[:m]

        for action in top_k_indices:
            root.children[action] = Node(policy_logits[action])

        self._sequential_halving(root, root_state, network, list(top_k_indices))

        counts = {a: child.visit_count for a, child in root.children.items()}
        completed_q = self._compute_completed_q(root, policy_logits)

        best_action = max(root.children.items(), key=lambda x: x[1].q_value)[0]
        
        if best_action not in legal_actions_set:
            best_action = legal_actions[0]

        return best_action, completed_q, counts

    def _sequential_halving(self, root, root_state, network, candidates):
        n = self.config.num_simulations
        m = len(candidates)

        if m == 0:
            return

        while m > 1:
            log_m = max(1, int(np.log2(m)))
            visits_per_action = max(1, n // (m * log_m))

            for action in candidates:
                for _ in range(visits_per_action):
                    self._simulate(root, action, root_state, network)

            scores = []
            for action in candidates:
                node = root.children[action]
                max_visits = max(c.visit_count for c in root.children.values()) if root.children else 1
                sigma = (self.c_visit + max_visits) * self.c_scale
                score = node.q_value + node.prior + sigma
                scores.append((score, action))

            scores.sort(reverse=True)
            m = max(1, m // 2)
            candidates = [a for s, a in scores[:m]]

    def _simulate(self, root, action, state_tensor, network):
        node = root.children[action]

        if not node.expanded():
            with torch.no_grad():
                state_input = state_tensor.to(self.config.device)
                _, value = network(state_input)
                leaf_value = value.item()
        else:
            leaf_value = node.value()

        node.value_sum += leaf_value
        node.visit_count += 1
        node.q_value = node.value_sum / node.visit_count

    def _compute_completed_q(self, root, logits):
        q_values = np.copy(logits)
        root_v = 0

        for action, node in root.children.items():
            if node.visit_count > 0:
                q_values[action] = node.q_value
                root_v += node.prior * node.q_value
            else:
                q_values[action] = root_v

        return q_values
