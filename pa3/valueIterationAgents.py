# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount=0.9, iterations=100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0

        # Looping through the iterations
        while iterations > 0:
            iterations -= 1

            # previousValues is the temporary dict to store the copy of self.values
            # in order to store the current updates values for the state
            # self.values is updated only after one iteration for all the possible states
            previousValues = self.values.copy()

            # Iterating over the possible states from the current state.
            for state in mdp.getStates():

                # Check if the current state is terminal state
                if mdp.isTerminal(state):
                    continue
                # qValueList holds the value qvalues for state action pair
                qValueList = []

                # Getting Qvalue for all possible actions for the current
                # state in the loop
                actions = self.mdp.getPossibleActions(state)
                for action in actions:
                    qValueList.append(self.computeQValueFromValues(state, action))

                previousValues[state] = max(qValueList)
            self.values = previousValues

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        # transitions is a list of tuple of (nextState,prob)
        transitions = self.mdp.getTransitionStatesAndProbs(state, action)
        qStateValue = 0.0

        for nextState, prob in transitions:
            currentReward = self.mdp.getReward(state, action, nextState)
            qStateValue += float(prob) * (
                float(currentReward) + float(self.discount) * float(self.values[nextState]))

        return qStateValue

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """

        # Getting all the possible actions of the state
        actions = self.mdp.getPossibleActions(state)

        # if there are no possible actions for the state, return nothing
        if len(actions) < 1:
            return
        # stateActionDict holds action values
        stateActionDict = util.Counter()
        for action in actions:
            stateActionDict[action] = self.computeQValueFromValues(state, action)
        return stateActionDict.argMax()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
