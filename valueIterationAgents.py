# valueIterationAgents.py
# -----------------------
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
    def __init__(self, mdp, discount = 0.9, iterations = 100):
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
        self.values = util.Counter() # A Counter is a dict with default 0

        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        #print(f"""iterations : {iterations}""")
        #print(f"""mdp state : {mdp.getStates()}""")
        #print(f"""discount : {discount}""")
        #print("")
        for i in range(iterations):
            new_values = util.Counter()
            for state in mdp.getStates():
                actions = mdp.getPossibleActions(state)
                #print(actions)
                if not actions:
                    new_values[state] = 0 #state sans action
                else:
                    new_values[state] = max(self.computeQValueFromValues(state, action) for action in actions) #maj de la valeur du state avec la max(qvalue) des actions possibles
                    #print(new_values[state])
            self.values = new_values
        
        



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
        "*** YOUR CODE HERE ***"
        mdp = self.mdp
        q_value = 0
        transitions = mdp.getTransitionStatesAndProbs(state, action) #liste de tuple avec le state et la proba d'y arriver
        for trans_state, prob in transitions:
            reward = mdp.getReward(state, action, trans_state)
            q_value += prob * (reward + self.discount * self.values[trans_state]) #maj de la qvalue selon l'equation de Bellman
        return q_value


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        mdp = self.mdp  
        if mdp.isTerminal(state):
            return None
        actions = mdp.getPossibleActions(state) #on recupere les actions possibles (il y en a au moins une car if isterminal return none)
        q_values = [(self.computeQValueFromValues(state,action), action) for action in actions] #on créé une liste de tuple qui stocke toutes les qvalues et leur action associée
        best_action = max(q_values)[1] #on recupere l'action avec la meilleure qvalue
        return best_action




    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
