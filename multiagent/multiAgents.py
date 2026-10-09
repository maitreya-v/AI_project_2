# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        self._expandedStates = 0
        self._expandedNodes = 0
        self._lastSearchValue = None

        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        if not legalMoves:
            return Directions.STOP

        self._expandedStates = 1

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        # Count immediate successors for reproducible measurements.
        self._expandedNodes = len(legalMoves)
        self._lastSearchValue = bestScore

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        if successorGameState.isWin():
            return float("inf")
        if successorGameState.isLose():
            return float("-inf")

        score = successorGameState.getScore()

        # Food should create a steady gradient toward progress.  The built-in
        # score already rewards a pellet, while the terms below distinguish
        # otherwise equal moves and discourage wandering with food remaining.
        foodPositions = newFood.asList()
        if foodPositions:
            closestFood = min(manhattanDistance(newPos, food) for food in foodPositions)
            score += 12.0 / (closestFood + 1.0)
            score -= 4.0 * len(foodPositions)

        if newFood.count() < currentGameState.getFood().count():
            score += 20.0

        # Capsules are valuable escape routes, especially on crowded boards.
        score -= 6.0 * len(successorGameState.getCapsules())

        # Active ghosts are treated as immediate threats.  Scared ghosts are
        # opportunities only when Pac-Man can reach them before their timer
        # expires, which prevents reckless chases across the board.
        for ghostState, scaredTime in zip(newGhostStates, newScaredTimes):
            ghostPosition = ghostState.getPosition()
            if ghostPosition is None:
                continue

            distance = manhattanDistance(newPos, ghostPosition)
            if scaredTime > 0:
                if distance <= scaredTime:
                    score += 30.0 / (distance + 1.0)
            else:
                if distance == 0:
                    return float("-inf")
                if distance == 1:
                    score -= 500.0
                elif distance == 2:
                    score -= 80.0
                else:
                    score -= 3.0 / distance

        if action == Directions.STOP:
            score -= 15.0

        return score

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxAgent and AlphaBetaAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)
        self._expandedStates = 0
        self._expandedNodes = 0
        self._lastSearchValue = None

    @staticmethod
    def _nextTurn(agentIndex, depth, numAgents):
        """Return the following agent and completed Pac-Man ply count.

        A project depth is one complete round: Pac-Man followed by every
        ghost.  Consequently, depth advances only when play wraps back to
        agent zero.  This also handles a legal one-agent game correctly.
        """
        nextAgent = (agentIndex + 1) % numAgents
        nextDepth = depth + 1 if nextAgent == 0 else depth
        return nextAgent, nextDepth

    def _isCutoff(self, gameState, depth):
        """Return whether a state must be evaluated rather than expanded."""
        return depth == self.depth or gameState.isWin() or gameState.isLose()

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        self._expandedStates = 0
        self._expandedNodes = 0
        numAgents = gameState.getNumAgents()

        def minimaxValue(state, agentIndex, depth):
            if self._isCutoff(state, depth):
                return self.evaluationFunction(state)

            self._expandedStates += 1
            legalActions = state.getLegalActions(agentIndex)
            if not legalActions:
                return self.evaluationFunction(state)

            nextAgent, nextDepth = self._nextTurn(agentIndex, depth, numAgents)

            if agentIndex == 0:
                value = float("-inf")
                for action in legalActions:
                    self._expandedNodes += 1
                    successor = state.generateSuccessor(agentIndex, action)
                    value = max(value, minimaxValue(successor, nextAgent, nextDepth))
                return value

            value = float("inf")
            for action in legalActions:
                self._expandedNodes += 1
                successor = state.generateSuccessor(agentIndex, action)
                value = min(value, minimaxValue(successor, nextAgent, nextDepth))
            return value

        self._expandedStates += 1
        legalActions = gameState.getLegalActions(0)
        if not legalActions:
            self._lastSearchValue = self.evaluationFunction(gameState)
            return Directions.STOP

        bestAction = legalActions[0]
        bestValue = float("-inf")
        nextAgent, nextDepth = self._nextTurn(0, 0, numAgents)

        for action in legalActions:
            self._expandedNodes += 1
            successor = gameState.generateSuccessor(0, action)
            value = minimaxValue(successor, nextAgent, nextDepth)
            if value > bestValue:
                bestValue = value
                bestAction = action

        self._lastSearchValue = bestValue
        print("Pac-Man action", bestAction)
        return bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        self._expandedStates = 0
        self._expandedNodes = 0
        numAgents = gameState.getNumAgents()

        def alphaBetaValue(state, agentIndex, depth, alpha, beta):
            if self._isCutoff(state, depth):
                return self.evaluationFunction(state)

            self._expandedStates += 1
            legalActions = state.getLegalActions(agentIndex)
            if not legalActions:
                return self.evaluationFunction(state)

            nextAgent, nextDepth = self._nextTurn(agentIndex, depth, numAgents)

            if agentIndex == 0:
                value = float("-inf")
                for action in legalActions:
                    self._expandedNodes += 1
                    successor = state.generateSuccessor(agentIndex, action)
                    value = max(
                        value,
                        alphaBetaValue(successor, nextAgent, nextDepth, alpha, beta),
                    )
                    # Strict comparison deliberately avoids pruning on equality;
                    # the project autograder checks the exact expansion count.
                    if value > beta:
                        return value
                    alpha = max(alpha, value)
                return value

            value = float("inf")
            for action in legalActions:
                self._expandedNodes += 1
                successor = state.generateSuccessor(agentIndex, action)
                value = min(
                    value,
                    alphaBetaValue(successor, nextAgent, nextDepth, alpha, beta),
                )
                if value < alpha:
                    return value
                beta = min(beta, value)
            return value

        self._expandedStates += 1
        legalActions = gameState.getLegalActions(0)
        if not legalActions:
            self._lastSearchValue = self.evaluationFunction(gameState)
            return Directions.STOP

        bestAction = legalActions[0]
        bestValue = float("-inf")
        alpha = float("-inf")
        beta = float("inf")
        nextAgent, nextDepth = self._nextTurn(0, 0, numAgents)

        for action in legalActions:
            self._expandedNodes += 1
            successor = gameState.generateSuccessor(0, action)
            value = alphaBetaValue(successor, nextAgent, nextDepth, alpha, beta)
            if value > bestValue:
                bestValue = value
                bestAction = action
            alpha = max(alpha, bestValue)

        self._lastSearchValue = bestValue
        return bestAction

