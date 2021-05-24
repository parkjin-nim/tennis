import numpy as np;
import torch;
from ddpg_agent import Agent, ReplayBuffer;

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size

class mAgent():
    def __init__(self, num_agents, state_size, action_size, random_seed):
        self.num_agents = num_agents;
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed);
        self.agents = [Agent(state_size, action_size, self.memory, BATCH_SIZE, random_seed) for agent_posit in range(num_agents)];
    
    def reset(self):
        [agent.reset() for agent in self.agents];
    
    def step(self, states, actions, rewards, next_states, dones):
        [self.agents[posit].step(states[posit], actions[posit], rewards[posit], next_states[posit], dones[posit]) for posit in range(self.num_agents)];
    
    def act(self, states):
        actions = [self.agents[posit].act(np.array([states[posit]])) for posit in range(self.num_agents)];
        return actions;
    
    def save(self):
        for posit in range(self.num_agents):
            torch.save(self.agents[posit].actor_local.state_dict(), 'checkpoint_actor_%d.pth'%(posit));
            torch.save(self.agents[posit].critic_local.state_dict(), 'checkpoint_critic_%d.pth'%(posit));
            
    def load(self):
        for posit in range(self.num_agents):
            self.agents[posit].actor_local.load_state_dict(torch.load('checkpoint_actor_%d.pth'%(posit), map_location='cpu'));
            self.agents[posit].critic_local.load_state_dict(torch.load('checkpoint_critic_%d.pth'%(posit), map_location='cpu'));


