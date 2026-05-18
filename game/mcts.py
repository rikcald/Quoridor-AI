import math

import numpy as np

from game_logic_Ai import TOTAL_ACTIONS


class MCTSNode:
    """
    One node in the Monte Carlo Tree Search.

    Important convention:
    `value_sum` is stored from the perspective of the player to move in THIS node.
    That is why, when a parent selects among children, it uses `-child.mean_value()`.
    """

    def __init__(self, state, parent=None, action_taken=None, prior=0.0):
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        # Each node owns its own child dictionary.
        # e.g. using {} here inside __init__ avoids the shared-mutable-default bug.
        self.children = {}

    def is_expanded(self):
        return len(self.children) > 0

    def mean_value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def q_value_for_parent(self):
        """
        Convert this node's mean value into the parent's perspective.

        e.g. if this child node looks great for the opponent (+0.8 from their view),
        then it should look bad for the parent (-0.8).
        """
        return -self.mean_value()

    def expand(self, policy_probs):
        """
        Create one child per legal action using the policy prior.

        `policy_probs` is expected to already be masked and normalized over
        legal actions only.
        """
        valid_actions = self.state.get_valid_actions()

        for action in valid_actions:
            child_state = self.state.next_state(int(action))
            self.children[int(action)] = MCTSNode(
                state=child_state,
                parent=self,
                action_taken=int(action),
                prior=float(policy_probs[int(action)]),
            )

    def select_child(self, c_puct):
        """
        Pick the child maximizing the PUCT score:
        Q(s, a) + U(s, a)
        """
        best_score = float("-inf")
        best_child = None

        sqrt_parent_visits = math.sqrt(max(1, self.visit_count))

        for child in self.children.values():
            q_score = child.q_value_for_parent()
            u_score = (
                c_puct * child.prior * sqrt_parent_visits / (1 + child.visit_count)
            )
            score = q_score + u_score

            if score > best_score:
                best_score = score
                best_child = child

        return best_child


class MCTS:
    """
    Readable first implementation of AlphaZero-style search for Quoridor.

    Responsibilities:
    - use the policy-value network to evaluate leaf nodes
    - expand legal actions with priors
    - run PUCT selection
    - backpropagate values with alternating sign
    - return a policy target from root visit counts
    """

    def __init__(
        self,
        agent,
        num_simulations=50,
        c_puct=1.5,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
        add_dirichlet_noise=True,
    ):
        self.agent = agent
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.add_dirichlet_noise = add_dirichlet_noise

    def run(self, root_state):
        """
        Run MCTS from a copy of the given root state and return the root node.
        """
        root = MCTSNode(state=root_state.clone())

        for simulation_idx in range(self.num_simulations):
            node = root
            search_path = [node]

            # Selection: follow the tree until we hit an unexpanded node or a terminal node.
            while node.is_expanded() and not node.state.is_terminal():
                node = node.select_child(self.c_puct)
                search_path.append(node)

            if node.state.is_terminal():
                # Terminal values are exact, not predicted by the network.
                current_player = node.state.get_current_player()
                leaf_value = float(node.state.get_outcome_for_player(current_player))
            else:
                policy_probs, leaf_value = self._evaluate(node.state)

                # Add Dirichlet noise only once, at the root expansion.
                if simulation_idx == 0 and node is root and self.add_dirichlet_noise:
                    policy_probs = self._add_dirichlet_noise(
                        policy_probs, node.state.get_valid_actions()
                    )

                node.expand(policy_probs)

            self._backpropagate(search_path, leaf_value)

        return root

    def get_action_probs(self, root, temperature=1.0):
        """
        Convert root visit counts into the AlphaZero target policy pi.

        Output shape is always (TOTAL_ACTIONS,), with 0 for actions not expanded
        from the root.
        """
        action_probs = np.zeros(TOTAL_ACTIONS, dtype=np.float32)

        if not root.children:
            return action_probs

        visit_counts = np.zeros(TOTAL_ACTIONS, dtype=np.float32)
        for action, child in root.children.items():
            visit_counts[action] = child.visit_count

        if temperature <= 1e-8:
            best_action = int(np.argmax(visit_counts))
            action_probs[best_action] = 1.0
            return action_probs

        # e.g. with temperature 1.0, pi is proportional to visit counts;
        # with lower temperatures it becomes sharper around the best moves.
        tempered_counts = visit_counts ** (1.0 / temperature)
        total = float(tempered_counts.sum())

        if total <= 0:
            valid_actions = list(root.children.keys())
            action_probs[valid_actions] = 1.0 / len(valid_actions)
            return action_probs

        action_probs = tempered_counts / total
        return action_probs.astype(np.float32)

    def select_action(self, root, temperature=1.0):
        """
        Sample one action from the root visit distribution.
        """
        action_probs = self.get_action_probs(root, temperature=temperature)
        return int(np.random.choice(np.arange(TOTAL_ACTIONS), p=action_probs))

    def _evaluate(self, state):
        """
        Evaluate one leaf state with the policy-value network.

        Returns:
        - masked and normalized legal-action policy
        - scalar value from the current player's perspective
        """
        canonical_state = state.get_canonical_state()
        raw_policy, value = self.agent.predict(canonical_state)

        if hasattr(raw_policy, "detach"):
            raw_policy = raw_policy.detach().cpu().numpy()
        else:
            raw_policy = np.asarray(raw_policy, dtype=np.float32)

        valid_actions = state.get_valid_actions()
        masked_policy = self.agent.mask_and_normalize_policy(raw_policy, valid_actions)
        return masked_policy, float(value)

    def _add_dirichlet_noise(self, policy_probs, valid_actions):
        """
        Add AlphaZero-style exploration noise at the root.

        e.g. this prevents self-play from always repeating the same opening just
        because the current network already likes it slightly more than the rest.
        """
        noisy_policy = np.array(policy_probs, dtype=np.float32, copy=True)
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(valid_actions)).astype(
            np.float32
        )

        noisy_policy[valid_actions] = (1.0 - self.dirichlet_epsilon) * noisy_policy[
            valid_actions
        ] + self.dirichlet_epsilon * noise
        noisy_policy = self.agent.mask_and_normalize_policy(noisy_policy, valid_actions)
        return noisy_policy

    def _backpropagate(self, search_path, leaf_value):
        """
        Back up the leaf value while alternating the sign at every ply.

        e.g. if a leaf is +0.6 for the side to move there, then for its parent
        that same continuation is worth -0.6.
        """
        value = leaf_value

        for node in reversed(search_path):
            node.visit_count += 1
            node.value_sum += value
            value = -value
