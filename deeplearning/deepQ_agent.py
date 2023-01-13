import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 

class Network(nn.Module): 
    """
    Neural network for deep learning
    """

    def __init__(self, learning_rate, input_dims, output_dims):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(*input_dims, 128)
        self.fc2 = nn.Linear(128, output_dims)

        self.optimizer = optim.Adam(self.parameters(), lr = learning_rate)
        self.loss = nn.MSELoss()

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, data):
        """
        The forward function is one forward pass of the NN. 
        It predicts the Q-value of a given state
        In the case of the DeepQAgent. there are 4 actions in the action space, there should be 4 possible Q value outputs 
        """
        input = F.relu(self.fc1(data))
        output = self.fc2(input)

        return output 

class Agent: 
    """
    Class to represent the agent
    
    """
    def __init__(self, n_actions, input_dims,
                 learning_rate = 0.0001, gamma = 0.9, eps_max = 1.0, eps_min = 0.01, d_eps=0.99995):

        # action space 
        self.n_actions = n_actions
        self.input_dims = input_dims
        # hyperparams
        self.learning_rate = learning_rate
        self.gamma = gamma
        # epsilon
        self.eps_max = eps_max
        self.eps_min = eps_min
        self.epsilon = eps_max
        self.d_eps = d_eps
        # Q
        self.Q = Network(self.learning_rate, self.input_dims, self.n_actions)

    def choose_action(self, state):
        if np.random.random() > self.epsilon:
            action = self.find_best_action(state)
        else:
            action = np.random.choice(range(self.n_actions))    
        return action 
    
    def find_best_action(self, state):
        state = T.tensor(state).to(self.Q.device)
        actions = self.Q.forward(state)
        best_action = T.argmax(actions).item()

        return best_action
    
    def decrease_epsilon(self):
        if self.epsilon > self.eps_min:
            self.epsilon = self.epsilon*self.d_eps
        else:
            self.epsilon = self.eps_min
    
    def learn(self, state, new_state, action, reward):

        self.Q.optimizer.zero_grad()

        # convert to tensor objects for compatibility with pytorch
        states = T.tensor(state).to(self.Q.device)
        new_states = T.tensor(new_state).to(self.Q.device)
        actions = T.tensor(action).to(self.Q.device)
        rewards = T.tensor(reward).to(self.Q.device)
        
        # predict the possible q values from a given state, then select the action taken
        q_pred = self.Q.forward(states)[actions]
        # find the maximum q value of the next state
        q_new = self.Q.forward(new_states).max()
        # the target is the maximum possible q function 
        q_target = reward + self.gamma*q_new

        loss = self.Q.loss(q_target, q_pred).to(self.Q.device)
        
        loss.backward()
        self.Q.optimizer.step()

        self.decrease_epsilon()


        
