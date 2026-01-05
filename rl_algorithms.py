from gridworld_env import GridWordEnv
import numpy as np
import matplotlib.pyplot as plt
class ValueIteration():
    def __init__(self):
        self.gridworld = GridWordEnv(gird_size=5, render=True)
        self.initstate = self.gridworld.reset()
        self.epsilon = 1e-5
        self.gamma = 0.8

    def iteration(self):

        v = np.zeros((5,5))
        pi = np.zeros((5,5), dtype=int)
        v_new = np.ones((5,5))
    
        while np.max(np.abs(v_new - v)) > self.epsilon:
            v = v_new.copy()
            for i in range(self.gridworld.gird_size):
                for j in range(self.gridworld.gird_size):
                    q_list = []
                    for action in range(4):
                        new_state, reward, done = self.gridworld.state_transition(np.array((i,j)), action)
                        q = reward + self.gamma * (0 if done else v[(new_state[0], new_state[1])])
                        q_list.append(q)

                    #greedy action selection
                    action_star = np.argmax(np.array(q_list))
                    # policy update
                    pi[(i,j)] = action_star

                    #value update
                    v_new[(i,j)] = np.max(np.array(q_list))
        print(v_new)
        return pi
    
    def run_policy(self, pi):
        
        done = False
        state = self.initstate
        while not done:
            state, reward, done, _ = self.gridworld.step(pi[(state[0],state[1])])
        plt.show()
        

class PolicyIteration():
    def __init__(self):
        self.gridworld = GridWordEnv(gird_size=5, render=True)
        self.initstate = self.gridworld.reset()
        self.epsilon = 1e-5
        self.gamma = 0.8

    def solve_policy_evalutaion(self, pi):
        first_flag = True
        v_pi = np.zeros((5,5))
        v = np.zeros((5,5))
        while np.max(np.abs(v_pi - v)) > self.epsilon or first_flag:
            first_flag = False     
            v = v_pi.copy()
            for i in range(self.gridworld.gird_size):
                for j in range(self.gridworld.gird_size):
                
                    new_state, reward, done = self.gridworld.state_transition(np.array((i,j)), pi[(i,j)])
                    #policy evaluation
                    
                    v_pi[(i,j)] = reward + self.gamma * (0 if done else v[(new_state[0],new_state[1])])

        return v_pi

    def iteration(self):
        pi = np.zeros((5,5), dtype=int)
        new_pi = np.zeros((5,5), dtype=int)
        first_flag = True
        while np.array_equal(new_pi, pi) == False or first_flag:
            first_flag = False
            
            pi = new_pi.copy()

            v_pi = self.solve_policy_evalutaion(pi)

            for i in range(self.gridworld.gird_size):
                for j in range(self.gridworld.gird_size):
                    q_list = []
                    #policy imporvement
                    for action in range(4):
                        new_state, reward, done = self.gridworld.state_transition(np.array((i,j)), action)
                        q_list.append(reward + self.gamma* (0 if done else v_pi[(new_state[0],new_state[1])]))
                    new_pi[(i,j)] = np.argmax(q_list)                  
    
        return new_pi
    
    def run_policy(self, pi):
        
        done = False
        state = self.initstate
        while not done:
            state, reward, done, _ = self.gridworld.step(pi[(state[0],state[1])])
        plt.show()

class MonteCarlo():
    def __init__(self):
        self.epsilon = 1e-5
        self.max_step = 200
        self.env = GridWordEnv(gird_size=5, render=True)
        self.env.reset()
        self.pi = np.zeros((5,5))
        


    def policy_evaluation(self, reward, new_state, done):
        v_pi = np.zeros((5,5))
        v    = np.zeros((5,5)) 
        first_flag = True
        while np.max(np.abs(v_pi - v)) > self.epsilon or first_flag:
            v = v_pi.copy()
            v_pi = reward + self.gamma * (0 if done else v[new_state[0], new_state[1]])
        
        return v_pi
            
            
    def generate_episdoe(self, pi):
        self.env.reset()
        step = 0
        while step < self.max_step:
            pi_action = pi[(self.env.state[0], self.env.state[1])]
            self.env.step(pi_action)
            step += 1


        pass




if __name__ == "__main__":
    # vi = ValueIteration()
    # pi = vi.iteration()
    # vi.run_policy(pi)

    p = PolicyIteration()
    pi = p.iteration()
    p.run_policy(pi)
