# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
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


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
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

        # Generating the Next Game state
        successorGameState = currentGameState.generatePacmanSuccessor(action)

        newPos = successorGameState.getPacmanPosition()  # Pacman Position
        newFood = successorGameState.getFood()  # Food Left in the Game
        newGhostStates = successorGameState.getGhostStates()
        capsules = successorGameState.getCapsules()  # Remaining capsules un the game
        numberOfCapsules = len(capsules)
        # Sum of all the scared time if the ghost is in the scared condition.
        newCumulativeScaredTimes = 0
        for ghostState in newGhostStates:
            newCumulativeScaredTimes += ghostState.scaredTimer
        ghostPosition = successorGameState.getGhostPositions()
        # Getting the nearest distance from the pacman new position to the food, ghost and capsule.
        nearestFoodDistance = getNearestDistanceToEntity(newFood.asList(), newPos)
        nearestGhostDistance = getNearestDistanceToEntity(ghostPosition, newPos)
        nearestCapsuleDistance = getNearestDistanceToEntity(capsules, newPos)
        if nearestFoodDistance > 0:
            evalFunctionScore = .01 * nearestGhostDistance \
                                + 2.0 / nearestFoodDistance \
                                + successorGameState.getScore() \
                                + 6.9 * newCumulativeScaredTimes \
                                + 4.5 * successorGameState.getNumFood() \
                                + 4.5 * numberOfCapsules \
                                - 2 * nearestCapsuleDistance
        else:
            evalFunctionScore = .01 * nearestGhostDistance \
                                + successorGameState.getScore() \
                                + 6.9 * newCumulativeScaredTimes \
                                + 4.5 * successorGameState.getNumFood() \
                                + 4.5 * numberOfCapsules \
                                - 2 * nearestCapsuleDistance
        return evalFunctionScore


def getNearestDistanceToEntity(entitylocationList, currentPosition):
    """
    This method returns the minimum distance to entity from the current pacman position to any entity.
    :param entitylocationList: Gives the location of entities as a list to which we need to calculate the nearest
    distance.
    :type List of entity locations.
    :param currentPosition: Location of the pacman in current state
    :type currentPosition: Tuple of location.
    :return: Returns the minimum distance.
    :rtype: Number
    """
    totalDistanceToEntity = 0  # This is distance to the nearest entity.
    start = currentPosition  # At begin our start will be the current state.

    distanceDict = {}  # Distance dictionary {k,v}; k-> end point v-> distance in reaching the end point
    if len(entitylocationList) > 0:
        for entity in entitylocationList:  # We calculate the distance from start to all the Entities.
            end = entity
            if start == end:
                break

            # Getting the manhattan distance between the two points.
            manhatanDis = abs(start[0] - end[0]) + abs(start[1] - end[1])
            distanceDict[end] = manhatanDis
    else:
        return totalDistanceToEntity

    # Sorting the dictionary according to the min items
    if len(distanceDict.items()) > 0:
        minValue = min(distanceDict.items(), key=lambda x: x[1])
        return minValue[1]
    else:
        return totalDistanceToEntity


def scoreEvaluationFunction(currentGameState):
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
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
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
        """

        # Dictionary that stores the legal actions with the scores
        scoreActionDict = {}
        nodeIndex = self.index
        numberOfAgents = gameState.getNumAgents()

        # Total depth is a multiple of current depth and no of agents in the game.
        # For every node/state we process minimax we reduce the totalDepth by 1.
        totalDepth = self.depth * numberOfAgents

        # Getting the successors of the current states.
        successors = self.getSuccessors(gameState, nodeIndex)

        # For every successor process minimax.
        for successor in successors:
            score = self.processMinimax(successor, (nodeIndex + 1) % numberOfAgents, totalDepth - 1)
            scoreActionDict[score] = successor[0]

        # If the nodeIndex is 0 it is always Pacman and pacman acts as a max
        # Other than zero it is ghost. Ghosts act as a min.
        if nodeIndex == 0:
            keyIndex = max(scoreActionDict.keys())
        else:
            keyIndex = min(scoreActionDict.keys())

        finalAction = scoreActionDict[keyIndex]
        return finalAction

    def getSuccessors(self, gameState, nodeIndex):
        """
        Gets the list of tuple of legal action and next successor for the current state.
        :param gameState: the current game state
        :param nodeIndex: NodeIndex for the current state.
        :type nodeIndex: Number
        :return: List of tuple of legal action and next successor for the current state.
        :rtype: List of Tuple
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions(nodeIndex)

        # It is a list of tuples containing as [(action, successor),....]
        successors = []

        # Looping through all the legal action of the current state to get the successors.
        for action in legalMoves:
            successors.append((action, gameState.generateSuccessor(nodeIndex, action)))
        return successors

    def processMinimax(self, successor, nodeAgentIndex, currentDepth):
        """
        Processes the minimax of the current state and returns with a score.
        """
        # Check if Terminal reached
        if currentDepth == 0 or successor[1].isWin() or successor[1].isLose():
            return self.evaluationFunction(successor[1])

        # If the nodeIndex is 0 it is always Pacman and pacman acts as a max
        # Other than zero it is ghost. Ghosts act as a min.
        if nodeAgentIndex == 0:
            return self.maximize(currentDepth, nodeAgentIndex, successor)
        else:
            return self.minimize(currentDepth, nodeAgentIndex, successor)

    def minimize(self, currentDepth, nodeAgentIndex, successor):
        """
        Processes the minimize operation on the the current state and returns with a score.
        """
        nodeScore = [float("inf")]
        childrenNodes = self.getSuccessors(successor[1], nodeAgentIndex)
        for childNode in childrenNodes:
            nodeScore.append(self.processMinimax(childNode, (nodeAgentIndex + 1) % successor[1].getNumAgents(),
                                                 currentDepth - 1))
        bestScore = min(nodeScore)
        return bestScore

    def maximize(self, currentDepth, nodeAgentIndex, successor):
        """
        Processes the maximize operation on the current state and returns with a score.
        """
        nodeScore = [float("-inf")]
        childrenNodes = self. \
            getSuccessors(successor[1], nodeAgentIndex)
        for childNode in childrenNodes:
            nodeScore.append(
                self.processMinimax(childNode, (nodeAgentIndex + 1) % successor[1].getNumAgents(),
                                    currentDepth - 1))
        bestScore = max(nodeScore)
        return bestScore


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """

        # Dictionary that stores the legal actions with the scores
        scoreActionDict = {}
        nodeIndex = self.index
        numberOfAgents = gameState.getNumAgents()

        # Total depth is a multiple of current depth and no of agents in the game.
        # For every node/state we process minimax we reduce the totalDepth by 1.
        totalDepth = self.depth * numberOfAgents

        alpha = float("-inf")
        beta = float("inf")

        if nodeIndex == 0:
            score = float("-inf")
        else:
            score = float("-inf")

        # Looping over the legal moves and processing AlphaBeta on the generated successor
        for legalMove in gameState.getLegalActions(nodeIndex):
            # Success to the legal move.
            successor = (legalMove, gameState.generateSuccessor(nodeIndex, legalMove))

            # New score for the current state after processing alpha beta on all of its children.
            newscore = self.processAlphaBeta(successor,
                                             (nodeIndex + 1) % numberOfAgents,
                                             totalDepth - 1,
                                             alpha,
                                             beta)
            scoreActionDict[newscore] = successor[0]

            # Checking for pruning and alpha beta update, at the root node.
            if nodeIndex == 0:
                score = max(newscore, score)
                if score < alpha:
                    break
                alpha = max(alpha, score)
            else:
                score = min(newscore, score)
                if score > beta:
                    break
                beta = min(beta, score)

        if nodeIndex == 0:
            keyIndex = max(scoreActionDict.keys())
        else:
            keyIndex = min(scoreActionDict.keys())

        finalAction = scoreActionDict[keyIndex]
        return finalAction


    def processAlphaBeta(self, successor, nodeAgentIndex, currentDepth, alpha, beta):
        """
        Processes the alphabeta algorithm on the current state and returns with a score.
        """
        # Terminal Node Check
        if currentDepth == 0 or successor[1].isWin() or successor[1].isLose():
            return self.evaluationFunction(successor[1])
        if nodeAgentIndex == 0:
            return self.maximize(alpha, beta, currentDepth, nodeAgentIndex, successor)
        else:
            return self.minimize(alpha, beta, currentDepth, nodeAgentIndex, successor)

    def minimize(self, alpha, beta, currentDepth, nodeAgentIndex, successor):
        """
        Processes the minimize operation on the current state and returns with a score.
        We also have pruning operation if the currentNodeScore is less than the alpha.
        """
        currentNodeScore = float("inf")
        for childMove in successor[1].getLegalActions(nodeAgentIndex):
            childNode = (childMove, successor[1].generateSuccessor(nodeAgentIndex, childMove))
            childNodeScore = self.processAlphaBeta(childNode,
                                                   (nodeAgentIndex + 1) % successor[
                                                       1].getNumAgents(),
                                                   currentDepth - 1, alpha, beta)

            currentNodeScore = min(currentNodeScore, childNodeScore)
            if currentNodeScore < alpha:
                return currentNodeScore
            beta = min(beta, currentNodeScore)
        return currentNodeScore

    def maximize(self, alpha, beta, currentDepth, nodeAgentIndex, successor):
        """
        Processes the maximize operation on the current state and returns with a score.
        We also have pruning operation if the currentNodeScore is greater than the beta.
        """
        currentNodeScore = float("-inf")
        for childMove in successor[1].getLegalActions(nodeAgentIndex):
            childNode = (childMove, successor[1].generateSuccessor(nodeAgentIndex, childMove))
            childNodeScore = self.processAlphaBeta(childNode,
                                                   (nodeAgentIndex + 1) % successor[
                                                       1].getNumAgents(),
                                                   currentDepth - 1, alpha, beta)
            currentNodeScore = max(currentNodeScore, childNodeScore)
            if currentNodeScore > beta:
                return currentNodeScore
            alpha = max(alpha, currentNodeScore)
        return currentNodeScore


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        # Dictionary that stores the legal actions with the scores
        scoreActionDict = {}
        nodeIndex = self.index
        numberOfAgents = gameState.getNumAgents()

        # Total depth is a multiple of current depth and no of agents in the game.
        # For every node/state we process minimax we reduce the totalDepth by 1.
        totalDepth = self.depth * numberOfAgents

        alpha = float("-inf")
        beta = float("inf")

        if nodeIndex == 0:
            score = float("-inf")
        else:
            score = float("-inf")

        # Looping over the legal moves and processing Expectimax on the generated successor
        for legalMove in gameState.getLegalActions(nodeIndex):
            # Success to the legal move.
            successor = (legalMove, gameState.generateSuccessor(nodeIndex, legalMove))
            # New score for the current state after processing Expectimax on all of its children.
            newscore = self.processExpectimax(successor,
                                              (nodeIndex + 1) % numberOfAgents,
                                              totalDepth - 1,
                                              alpha,
                                              beta)
            scoreActionDict[newscore] = successor[0]

            # Checking for pruning and alpha beta update, at the root node.
            if nodeIndex == 0:
                score = max(newscore, score)
                if score < alpha:
                    break
                alpha = max(alpha, score)
            else:
                score = min(newscore, score)
                if score > beta:
                    break
                beta = min(beta, score)

        if nodeIndex == 0:
            keyIndex = max(scoreActionDict.keys())
        else:
            keyIndex = min(scoreActionDict.keys())

        finalAction = scoreActionDict[keyIndex]
        return finalAction

    def processExpectimax(self, successor, nodeAgentIndex, currentDepth, alpha, beta):
        """
        Processes the expectimax algorithm on the current state and returns with a score.
        """
        # Terminal Node Check
        if currentDepth == 0 or successor[1].isWin() or successor[1].isLose():
            return self.evaluationFunction(successor[1])
        if nodeAgentIndex == 0:
            return self.maximize(alpha, beta, currentDepth, nodeAgentIndex, successor)
        else:
            return self.expectimize(alpha, beta, currentDepth, nodeAgentIndex, successor)

    def expectimize(self, alpha, beta, currentDepth, nodeAgentIndex, successor):
        """
        Processes the minimize operation on the current state and returns with a score.
        There is no pruning here and the score returned is an average of child node scores.
        """
        nodeScoreList = []
        for childMove in successor[1].getLegalActions(nodeAgentIndex):
            childNode = (childMove, successor[1].generateSuccessor(nodeAgentIndex, childMove))
            childNodeScore = self.processExpectimax(childNode,
                                                    (nodeAgentIndex + 1) % successor[
                                                        1].getNumAgents(),
                                                    currentDepth - 1, alpha, beta)
            nodeScoreList.append(childNodeScore)
        #  Getting average of the NodeScoreList as this is an expectimax node.
        currentNodeScore = reduce(lambda x, y: x + y, nodeScoreList) / len(nodeScoreList)
        return currentNodeScore

    def maximize(self, alpha, beta, currentDepth, nodeAgentIndex, successor):
        """
        Processes the maximize operation on the current state and returns with a score.
        We also have pruning operation if the currentNodeScore is greater than the beta.
        """
        currentNodeScore = float("-inf")
        for childMove in successor[1].getLegalActions(nodeAgentIndex):
            childNode = (childMove, successor[1].generateSuccessor(nodeAgentIndex, childMove))
            childNodeScore = self.processExpectimax(childNode,
                                                    (nodeAgentIndex + 1) % successor[
                                                        1].getNumAgents(),
                                                    currentDepth - 1, alpha, beta)
            currentNodeScore = max(currentNodeScore, childNodeScore)
            if currentNodeScore > beta:
                return currentNodeScore
            alpha = max(alpha, currentNodeScore)
        return currentNodeScore


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    successorGameState = currentGameState
    newPos = successorGameState.getPacmanPosition()  # Pacman Position
    newFood = successorGameState.getFood()  # Food Left in the Game
    newGhostStates = successorGameState.getGhostStates()
    capsules = successorGameState.getCapsules()  # Remaining capsules un the game
    numberOfCapsules = len(capsules)
    # Sum of all the scared time if the ghost is in the scared condition.
    newCumulativeScaredTimes = 0
    for ghostState in newGhostStates:
        newCumulativeScaredTimes += ghostState.scaredTimer
    ghostPosition = successorGameState.getGhostPositions()
    # Getting the nearest distance from the pacman new position to the food, ghost and capsule.
    nearestFoodDistance = getNearestDistanceToEntity(newFood.asList(), newPos)
    nearestGhostDistance = getNearestDistanceToEntity(ghostPosition, newPos)
    nearestCapsuleDistance = getNearestDistanceToEntity(capsules, newPos)
    if nearestFoodDistance > 0:
        evalFunctionScore = .01 * nearestGhostDistance \
                            + 2.0 / nearestFoodDistance \
                            + successorGameState.getScore() \
                            + 6.9 * newCumulativeScaredTimes \
                            + 4.5 * successorGameState.getNumFood() \
                            + 4.5 * numberOfCapsules \
                            - 2 * nearestCapsuleDistance
    else:
        evalFunctionScore = .01 * nearestGhostDistance \
                            + successorGameState.getScore() \
                            + 6.9 * newCumulativeScaredTimes \
                            + 4.5 * successorGameState.getNumFood() \
                            + 4.5 * numberOfCapsules \
                            - 2 * nearestCapsuleDistance
    return evalFunctionScore


# Abbreviation
better = betterEvaluationFunction