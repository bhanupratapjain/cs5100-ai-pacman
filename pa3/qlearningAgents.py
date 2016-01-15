# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random, util, math


class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """

    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        self.qStateValues = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        return self.qStateValues[state, action]

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        qValues = []
        actions = self.getLegalActions(state)
        for action in actions:
            qValues.append(self.qStateValues[state, action])
        return max(qValues)

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        actions = self.getLegalActions(state)
        if len(actions) < 1:
            return
        stateActionDict = util.Counter()
        for action in actions:
            stateActionDict[action] = self.qStateValues[(state, action)]
        return stateActionDict.argMax()

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Get all the legal actions from the state
        legalActions = self.getLegalActions(state)
        action = None

        if util.flipCoin(self.epsilon):
            action = random.choice(legalActions)
        else:
            action = self.getPolicy(state)

        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """

        learningRate = self.alpha
        discount = self.discount

        # cuurentStateQValue is the Qvalue for the current state action pair
        cuurentStateQValue = self.getQValue(state, action)

        # nextStateQValueList will hold the values from the next state action pair
        nextStateQValueList = []

        # Getting legal actions for the next state
        leagalActions = self.getLegalActions(nextState)
        for nextStateAciton in leagalActions:
            nextStateQValueList.append(self.getQValue(nextState, nextStateAciton))

        if len(nextStateQValueList) > 0:
            qSample = reward + float(discount * max(nextStateQValueList))
        else:
            qSample = reward

        newCurrentStatQValue = float(1 - learningRate) * cuurentStateQValue + float(learningRate * qSample)

        self.qStateValues[state, action] = newCurrentStatQValue

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self, state)
        self.doAction(state, action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """

    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        qvalue = 0.0

        # Getting features of the current state action pair
        features = self.featExtractor.getFeatures(state, action)
        for k in features.keys():
            qvalue += self.weights[k] * features[k]
        return qvalue

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        learningRate = self.alpha
        discount = self.discount

        # Getting features of the current state action pair
        features = self.featExtractor.getFeatures(state, action)

        # Getting current state Qvalue
        cuurentQStateValue = self.getQValue(state, action)

        # nextStateQValueList will hold the values from the next state action pair
        nextStateQValueList = []

        # Legal actions for the next state
        leagalActions = self.getLegalActions(nextState)
        for nextStateAciton in leagalActions:
            nextStateQValueList.append(self.getQValue(nextState, nextStateAciton))

        if len(nextStateQValueList) > 0:
            qSample = reward + float(discount * max(nextStateQValueList))
        else:
            qSample = reward

        difference = qSample - cuurentQStateValue

        for k in features.keys():
            newFeatureWeight = self.weights[k] + learningRate * difference * features[k]
            self.weights[k] = newFeatureWeight

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
