import numpy as np

class Agent: 
    """
    Class to represent the agent
    
    """
    def __init__(self, n_states, n_actions,
                 alpha = 0.001, gamma = 0.9, eps_max = 1.0, eps_min = 0.01, d_eps=0.99995):

        # 
        self.n_states = n_states
        self.n_actions = n_actions
        # hyperparams
        self.alpha = alpha
        self.gamma = gamma
        # epsilon
        self.eps_max = eps_max
        self.eps_min = eps_min
        self.epsilon = eps_max
        self.d_eps = d_eps
        # Q
        self.Q = {}
        self.init_Q()

    def init_Q(self): 
        for state in range(self.n_states):
            for action in range(self.n_actions):
                self.Q[(state, action)] = 0
    
    def update_Q(self, state, new_state, action, reward):

        maxQ, _ = self.find_max_Q(new_state)
        currentQ = self.Q[(state, action)]

        dQ = self.alpha * (reward + self.gamma * maxQ - currentQ)
        
        self.Q[(state, action)] = currentQ + dQ 
        self.decrease_epsilon()

    def find_max_Q(self, state):
        allQ = []
        for action in range(self.n_actions):
            allQ.append(self.Q[(state, action)])
        maxQ = np.amax(allQ)
        return maxQ, allQ
    
    def find_best_action(self, state):
        maxQ, allQ = self.find_max_Q(state)
        best_action = np.argwhere(allQ==maxQ).flatten()

        # if there are many best equal actions
        if np.shape(best_action)[0] > 1:
            best_action = np.random.choice(best_action)
        else:
            best_action = best_action[0]
        
        return best_action
    
    def choose_action(self, state):

        if np.random.random() > self.epsilon:
            action = self.find_best_action(state)
        else:
            action = np.random.choice(range(self.n_actions))
        
        return action 
    
    def decrease_epsilon(self):
        if self.epsilon > self.eps_min:
            self.epsilon = self.epsilon*self.d_eps
        else:
            self.epsilon = self.eps_min
        
