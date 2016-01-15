# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""
import util


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    # start state of the problem
    startState = problem.getStartState()

    # if the start state is the goal state then we stop processing.
    if problem.isGoalState(startState):
        "Start State is the Goal State: No need to go further."
        return []
    else:

        # Final List of actions to be performed to reach the goal state.
        actionSequence = []

        # Init Fringe as  a stack
        fringe = util.Stack()

        # Push the start state to the fringe.
        fringe.push((startState, []))

        # Init Explored Set
        exploredSet = set()

        # This is used to check element presence in the fringe to prevent adding the same element.
        # States are added to as soon as we push to fringe.
        fringePushSet = set()

        while not fringe.isEmpty():

            # Pop Out First Element of the Fringe. Note as this is a stack, it wil follow LIFO
            fringeNode = fringe.pop()

            # fringeNodeState gives state of the first node in the fringe.
            fringeNodeState = fringeNode[0]

            # Add the state to explored set.
            exploredSet.add(fringeNodeState)

            # fringeNodeAction is the traversal path to that state.
            fringeNodeAction = fringeNode[1]
            actionSequence = fringeNodeAction

            # check whether the current state retrieved from the fringe is goal state.
            if problem.isGoalState(fringeNodeState):
                break
            else:  # Fetch children of the current state as the it is not the goal state.
                childNodes = problem.getSuccessors(fringeNodeState)
                # Loop through the childStates and add to Fringe only they are not in explored set.
                for child in childNodes:
                    nextStateDirection = child[1]
                    subActionList = list(actionSequence)
                    subActionList.append(nextStateDirection)

                    if child[0] not in exploredSet:
                        fringe.push((child[0], subActionList))
                        fringePushSet.add(child[0])
    print actionSequence
    return actionSequence


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""

    # start state of the problem
    startState = problem.getStartState()

    # if the start state is the goal state then we stop processing.
    if problem.isGoalState(startState):
        "Start State is the Goal State: No need to go further."
        return []
    else:
        # Init Fringe as a Queue
        fringe = util.Queue()

        # Push the start state to the fringe.
        fringe.push(startState)

        # Init Explored Set
        exploredSet = set()

        # This is used to check element presence in the fringe to prevent adding the same element.
        # States are added to as soon as we push to fringe.
        fringePushSet = set()

        # Contains the Directions with respect to nodes. [k,v] --> [nodes,directions]
        directionDictionary = {}

        # Contains the parent node w.r.t to the current node.
        previousNodeDictionary = {}

        # This is the final node list which is formed to reach the goal.
        finalNodeList = []

        # It contains final path to the goal.
        finalDirectionList = []

        # Contains the final goal state once achieved.
        goalState = []

        while not fringe.isEmpty():

            # Pop Out Last Element of the Fringe. Note as this is a Queue, it wil follow FIFO
            fringeNode = fringe.pop()

            # fringeNodeState gives state of the first node in the fringe.
            fringeNodeState = fringeNode

            # Add the state to explored set.
            exploredSet.add(fringeNodeState)

            # check whether the current state retrieved from the fringe is goal state.
            if problem.isGoalState(fringeNodeState):
                goalState = fringeNodeState
                break
            else:  # Fetch children of the  current state as the it is not the goal state.
                childNodes = problem.getSuccessors(fringeNodeState)
                # Loop through the childStates and add to Fringe only they are not in explored set.
                for child in childNodes:
                    if child[0] not in exploredSet and child[0] not in fringePushSet:
                        nextStateDirection = child[1]  # direction of the child state
                        fringe.push(child[0])  # put the child state on the fringe
                        fringePushSet.add(child[0])  # put the child state on the fringe push set
                        directionDictionary[
                            child[0]] = nextStateDirection  # add child state to the direction dictionary
                        previousNodeDictionary[
                            child[0]] = fringeNodeState  # add parent state to the previous Node dictionary

    # Creating final Node List.
    # We put in the gaol state first and then traverse back to the starting state.
    while True:
        finalNodeList.append(goalState)
        if previousNodeDictionary[goalState] == problem.getStartState():
            break
        else:
            goalState = previousNodeDictionary[goalState]

    # Creating final Direction List from the node list.
    for node in reversed(finalNodeList):
        finalDirectionList.append(directionDictionary[node])

    # displaySearchStats(finalDirectionList, finalNodeList, previousNodeDictionary)
    return finalDirectionList


def uniformCostSearch(problem):
    """Search the node of least total cost first."""

    # start state of the problem
    startState = problem.getStartState()

    # if the start state is the goal state then we stop processing.
    if problem.isGoalState(startState):
        "Start State is the Goal State: No need to go further."
        return []
    else:

        # Final List of actions to be performed to reach the goal state.
        actionSequence = []

        # Init Fringe as a Queue
        fringe = util.PriorityQueue()

        # Push the start state to the fringe. So a fringe node is like ((state,[exploredPath],pathCost),pathCost)
        fringe.push((startState, [], 0), 0)

        # This is used to check element presence in the fringe to prevent adding the same element.
        # States are added to as soon as we pop out of fringe.
        fringePopSet = set()

        while not fringe.isEmpty():

            # Pop Out Last Element of the Fringe. Note as this is a Queue, it wil follow FIFO
            fringeNode = fringe.pop()

            # fringeNodeState gives state of the first node in the fringe.
            fringeNodeState = fringeNode[0]

            # We only expand a state if it has not been expanded yet.
            if fringeNodeState in fringePopSet:
                continue
            else:
                fringePopSet.add(fringeNodeState)

            # fringeNodeAction is the traversal path to that state.
            fringeNodeAction = fringeNode[1]
            actionSequence = fringeNodeAction

            # Total path cost to the current state.
            fringeNodePathCost = fringeNode[2]

            # check whether the current state retrieved from the fringe is goal state.
            if problem.isGoalState(fringeNodeState):
                break
            else:  # Fetch children of the current state as the it is not the goal state.
                childNodes = problem.getSuccessors(fringeNodeState)
                # Loop through the childStates and add to Fringe only they are not in explored set.
                for child in childNodes:
                    nextStateDirection = child[1]
                    subActionList = list(actionSequence)
                    subActionList.append(nextStateDirection)

                    childPathCost = child[2]
                    totalChildNodePathCost = fringeNodePathCost + childPathCost

                    fringe.push((child[0], subActionList, totalChildNodePathCost), totalChildNodePathCost)

    print actionSequence
    return actionSequence


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""

    # start state of the problem
    startState = problem.getStartState()

    # if the start state is the goal state then we stop processing.
    if problem.isGoalState(startState):
        "Start State is the Goal State: No need to go further."
        return []
    else:
        # Init Fringe as a Priority Queue
        fringe = util.PriorityQueue()

        # Push the start state to the fringe.
        # Entry to the fringe will always be ((state,totalPathCost,previousNode),priority)
        fringe.push((startState, 0,), 1)

        # This is used to check element presence in the fringe to prevent adding the same element.
        # States are added to as soon as we pop out of fringe.
        fringePopSet = set()

        # Contains the Directions with respect to nodes. [k,v] --> [nodes,directions]
        directionDictionary = {}

        # Contains the parent node w.r.t to the current node.
        previousNodeDictionary = {}

        # This is the final node list which is formed to reach the goal.
        finalNodeList = []

        # It contains final path to the goal.
        finalDirectionList = []

        # Contains the final goal state once achieved.
        goalState = []

        while not fringe.isEmpty():

            # Pop Out Last Element of the Fringe. Note as this is a Queue, it wil follow FIFO
            fringeNode = fringe.pop()

            # fringeNodeState gives state of the first node in the fringe.
            fringeNodeState = fringeNode[0]

            # We only expand a state if it has not been expanded yet.
            if fringeNodeState in fringePopSet:
                continue
            else:
                # As the node is expanded, add it to fringe Pop Set
                fringePopSet.add(fringeNodeState)

                if fringeNodeState != problem.getStartState():
                    fringeParentNode = fringeNode[2]
                    fringeParentNodeDirection = fringeNode[3]

                    # add current state's parent to the previous Node dictionary
                    previousNodeDictionary[fringeNodeState] = fringeParentNode

                    # add current state's direction to the direction dictionary
                    directionDictionary[fringeNodeState] = fringeParentNodeDirection

            # Total path cost to the current state
            fringeNodePathCost = fringeNode[1]

            # check whether the current state retrieved from the fringe is goal state.
            if problem.isGoalState(fringeNodeState):
                goalState = fringeNodeState
                break
            else:  # Fetch children of the  current state as the it is not the goal state.
                childNodes = problem.getSuccessors(fringeNodeState)
                # Loop through the childStates and add to Fringe only they are not in explored set.
                for child in childNodes:

                    if child[0] not in previousNodeDictionary.keys() and child[0] not in fringePopSet:
                        childPathCost = child[2]
                        hueristicCost = heuristic(child[0], problem)
                        totalPathCost = fringeNodePathCost + childPathCost
                        totalCombinedCost = totalPathCost + hueristicCost
                        nextStateDirection = child[1]  # direction of the child state

                        # put the child state on the fringe
                        fringe.push((child[0], totalPathCost, fringeNodeState, nextStateDirection), totalCombinedCost)

    # Creating final Node List.
    # We put in the gaol state first and then traverse back to the starting state.
    while True:
        finalNodeList.append(goalState)
        if previousNodeDictionary[goalState] == problem.getStartState():
            break
        else:
            goalState = previousNodeDictionary[goalState]

    # Creating final Direction List from the node list.
    for node in reversed(finalNodeList):
        finalDirectionList.append(directionDictionary[node])

    # displaySearchStats(finalDirectionList, finalNodeList, previousNodeDictionary)
    return finalDirectionList


def displaySearchStats(finalDirectionList, finalNodeList, previousNodeDictionary):
    print "\n#########Previous Node Dict############3"
    for item in previousNodeDictionary.keys():
        print "\n Node: %s Previous Node: %s" % (item, previousNodeDictionary[item])
    print "Final Node List", finalNodeList
    print "Final Direction List", finalDirectionList


bfs = breadthFirstSearch


# Abbreviations
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()
