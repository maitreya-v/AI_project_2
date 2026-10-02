"""Regression tests for the Project 2 multi-agent search implementations."""

import unittest

import layout
from pacman import GameState
from multiAgents import AlphaBetaAgent, MinimaxAgent


class TreeState:
    """Small deterministic game tree used to verify multi-ghost turn order."""

    def __init__(self, node, tree, scores, counter, num_agents=3):
        self.node = node
        self.tree = tree
        self.scores = scores
        self.counter = counter
        self.num_agents = num_agents

    def getNumAgents(self):
        return self.num_agents

    def getLegalActions(self, agentIndex=0):
        branches = self.tree.get(self.node, ())
        if branches and branches[0][0] != agentIndex:
            raise AssertionError(
                "node %s expected agent %s, received %s"
                % (self.node, branches[0][0], agentIndex)
            )
        return [action for _, action, _ in branches]

    def generateSuccessor(self, agentIndex, action):
        self.counter[0] += 1
        for expected_agent, candidate, child in self.tree.get(self.node, ()):
            if expected_agent == agentIndex and candidate == action:
                return TreeState(
                    child, self.tree, self.scores, self.counter, self.num_agents
                )
        raise AssertionError("invalid transition")

    def getScore(self):
        return self.scores[self.node]

    def isWin(self):
        return False

    def isLose(self):
        return False


def make_three_agent_tree():
    """Return a depth-one tree where action A has value 4 and B has value 3."""
    tree = {
        "root": [(0, "A", "A1"), (0, "B", "B1")],
        "A1": [(1, "x", "A1x"), (1, "y", "A1y")],
        "B1": [(1, "x", "B1x"), (1, "y", "B1y")],
        "A1x": [(2, "m", "A5"), (2, "n", "A7")],
        "A1y": [(2, "m", "A4"), (2, "n", "A6")],
        "B1x": [(2, "m", "B3"), (2, "n", "B9")],
        "B1y": [(2, "m", "B8"), (2, "n", "B10")],
    }
    scores = {
        "A5": 5,
        "A7": 7,
        "A4": 4,
        "A6": 6,
        "B3": 3,
        "B9": 9,
        "B8": 8,
        "B10": 10,
    }
    counter = [0]
    return TreeState("root", tree, scores, counter), counter


class MultiAgentSearchTests(unittest.TestCase):
    def test_minimax_handles_multiple_ghost_layers(self):
        state, counter = make_three_agent_tree()
        agent = MinimaxAgent(depth="1")

        self.assertEqual("A", agent.getAction(state))
        self.assertEqual(4, agent._lastSearchValue)
        self.assertEqual(14, counter[0])
        self.assertEqual(counter[0], agent._expandedNodes)
        self.assertEqual(7, agent._expandedStates)

    def test_alpha_beta_matches_minimax(self):
        minimax_state, _ = make_three_agent_tree()
        alpha_state, _ = make_three_agent_tree()
        minimax = MinimaxAgent(depth="1")
        alpha_beta = AlphaBetaAgent(depth="1")

        self.assertEqual(minimax.getAction(minimax_state), alpha_beta.getAction(alpha_state))
        self.assertEqual(minimax._lastSearchValue, alpha_beta._lastSearchValue)
        self.assertLessEqual(alpha_beta._expandedNodes, minimax._expandedNodes)

    def test_reference_values_on_minimax_classic(self):
        board = layout.getLayout("minimaxClassic")
        expected_values = (9, 8, 7, -492)

        for depth, expected in enumerate(expected_values, start=1):
            values = []
            for agent_class in (MinimaxAgent, AlphaBetaAgent):
                state = GameState()
                state.initialize(board, board.getNumGhosts())
                agent = agent_class(depth=str(depth))
                agent.getAction(state)
                values.append(agent._lastSearchValue)

            self.assertEqual([float(expected), float(expected)], values)

    def test_reference_expansion_counts(self):
        board = layout.getLayout("minimaxClassic")
        expected = {
            MinimaxAgent: ((22, 33), (155, 211), (726, 1160), (4178, 5916)),
            AlphaBetaAgent: ((14, 19), (136, 179), (592, 886), (3239, 4463)),
        }
        for agent_class, counts in expected.items():
            for depth, (expanded, generated) in enumerate(counts, start=1):
                state = GameState()
                state.initialize(board, board.getNumGhosts())
                agent = agent_class(depth=str(depth))
                agent.getAction(state)
                self.assertEqual((expanded, generated),
                                 (agent._expandedStates, agent._expandedNodes))

    def test_alpha_beta_keeps_equal_valued_branches(self):
        tree = {
            "root": [(0, "A", "Aghost"), (0, "B", "Bghost")],
            "Aghost": [(1, "x", "A5")],
            "Bghost": [(1, "x", "B5"), (1, "y", "B1")],
        }
        scores = {"A5": 5, "B5": 5, "B1": 1}
        counter = [0]
        state = TreeState("root", tree, scores, counter, num_agents=2)
        agent = AlphaBetaAgent(depth="1")

        self.assertEqual("A", agent.getAction(state))
        self.assertEqual(5, agent._lastSearchValue)
        self.assertEqual(5, counter[0])
        self.assertEqual(5, agent._expandedNodes)


if __name__ == "__main__":
    unittest.main()
