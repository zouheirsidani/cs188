# qlearningAgents.py
# ------------------
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


from os import stat
from sys import setprofile
from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

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
        
        "*** YOUR CODE HERE ***"
        self.qvalues = util.Counter() # A Counter is a dict with default 0
    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        if (state,action) not in self.qvalues.keys():
          self.qvalues[(state,action)]=0
        return self.qvalues[(state,action)]
        util.raiseNotDefined()


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        actions = self.getLegalActions(state)
        value = - math.inf
        for a in actions:
            value1 = self.getQValue(state,a)
            if value1 > value:
                value = value1
        return value
        util.raiseNotDefined()

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        actions = self.getLegalActions(state)
        action = ''
        choice = []
        value = - math.inf
        for a in actions:
            value1 = self.getQValue(state,a)
            if value1 > value:
                choice = [a]
                value = value1
            elif value1==value:
              choice.append(a)
        if not choice:
          return 'exit'
        action = random.choice(choice)
        return action
        util.raiseNotDefined()

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
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        if util.flipCoin(self.epsilon):
          action = random.choice(legalActions)
        else:
          action = self.computeActionFromQValues(state)
        return action
        util.raiseNotDefined()

        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        self.qvalues[(state,action)]  = (1-self.alpha)*self.getQValue(state,action)+self.alpha*(reward+self.discount*self.getQValue(nextState,self.computeActionFromQValues(nextState)))
        
        # util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
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
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
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
        "*** YOUR CODE HERE ***"
        
        # self.qvalues[(state,action)]=0
        if state == 'TERMINAL_STATE':
          return 0
        features = self.featExtractor.getFeatures(state,action)
        sum =0
        for state_action_tuple, value in features.items():
          sum = sum+self.weights[state_action_tuple]*value
        return sum




        # if (state,action) not in self.weights.keys():
        #   self.weights[(state,action)]=[0]*len(features)
      
        
        # for i in range(len(self.featExtractor.getFeatures(state,action))):
        #   if isinstance(self.featExtractor.getFeatures(state,action)[state,action], float) or isinstance(self.featExtractor.getFeatures(state,action)[state,action], int):
        #     self.qvalues[(state,action)]=self.qvalues[(state,action)]+self.weights[(state,action)][i]*self.featExtractor.getFeatures(state,action)[state,action]
        #   else:
        #     self.qvalues[(state,action)]=self.qvalues[(state,action)]+self.weights[(state,action)][i]*self.featExtractor.getFeatures(state,action)[state,action][i]
          
        # return self.qvalues[(state,action)]
        util.raiseNotDefined()

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        features = self.featExtractor.getFeatures(state,action)
        diff = reward + self.discount*self.getQValue(nextState,self.computeActionFromQValues(nextState))-self.getQValue(state,action)
        for state_action_tuple, value in features.items():
          self.weights[state_action_tuple]=self.weights[state_action_tuple]+self.alpha*diff*value
        # if (state,action) not in self.weights.keys():
        #   self.weights[(state,action)]=[0]*len(self.featExtractor.getFeatures(state,action))
        # diff = reward + self.discount*self.getQValue(nextState,self.computeActionFromQValues(nextState))-self.getQValue(state,action)
        # for i in range(len(self.featExtractor.getFeatures(state,action))):
        #   if isinstance(self.featExtractor.getFeatures(state,action)[state,action], float) or isinstance(self.featExtractor.getFeatures(state,action)[state,action], int):
        #     self.weights[(state,action)][i]=self.weights[(state,action)][i]+self.alpha*diff*self.featExtractor.getFeatures(state,action)[state,action]
        #   else:
        #     self.weights[(state,action)][i]=self.weights[(state,action)][i]+self.alpha*diff*self.featExtractor.getFeatures(state,action)[state,action][i]
        

        
        # util.raiseNotDefined()

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
