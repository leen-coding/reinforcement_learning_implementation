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
        self.max_step = 500
        self.env = GridWordEnv(gird_size=5, render=True)
        self.env.reset()
        self.env.render_enabled = False
        self.pi = np.zeros((5,5), dtype=int)
        self.gamma = 0.8
        self.N = np.zeros((5,5,4)) #每个状态，每个动作的次数

    def policy_evaluation_every_visit(self, q_pi, MDP_episode):
        #本质上还是policy evaluation, 这个v_pi是依赖pi的。
        # MC 这里可以直接估计q, 而不是估计v然后从v 推导q， 否则要做两遍遍历
        G = 0 
        for current_state, pi_action, reward, new_state in reversed(MDP_episode): #倒序更新G
            self.N[current_state[0],current_state[1], pi_action] =  self.N[current_state[0],current_state[1], pi_action] + 1
            G = reward + self.gamma * G
            # 增量式更新v_pi的平均值。w_k+1 = w_k - 1/k * (w_k - xk)
            q_pi[current_state[0], current_state[1], pi_action] = q_pi[current_state[0], current_state[1], pi_action] - (q_pi[current_state[0], current_state[1], pi_action] - G)/self.N[current_state[0],current_state[1], pi_action]
        return q_pi
            
            
    def generate_episdoe(self, q_pi, epsilon):
        self.env.reset()
        step = 0
        MDP_episode = []
        reward_sum = 0
        while step < self.max_step:
            current_state = self.env.state.copy()
            greedy_action = np.argmax(q_pi[current_state[0],current_state[1]])
            pi_action = self.epsilon_greedy(greedy_action, epsilon)
            new_state, reward, done, _ = self.env.step(pi_action)
            step += 1
            MDP_episode.append((current_state, pi_action, reward, new_state))
            reward_sum = reward_sum + reward
            if done: break
        return MDP_episode, reward_sum
    
    def epsilon_greedy(self, greedy_action, epsilon):
        probs = np.ones(4) * epsilon/4
        probs[greedy_action] = 1 - epsilon + epsilon/4
        return np.random.choice(4, p=probs)



    def train_mc(self):
        q_pi = np.zeros((5,5,4))
        epsilon = 0.1
        
        num_episode = 50000
        for iters in range(num_episode):
            MDP_episode,reward_sum = self.generate_episdoe(q_pi, epsilon)
            
            q_pi = self.policy_evaluation_every_visit(q_pi, MDP_episode)
            if iters % 1000 == 0:
                print(reward_sum)

        pi = np.argmax(q_pi, axis=2)
        return pi
    
    def run_policy(self, pi):
        done = False
        state = self.env.reset()
        self.env.render_enabled = True
        while not done:
            action = pi[state[0], state[1]]
            self.state, reward, done, _ = self.env.step(action)
            state = self.state
        plt.show()



if __name__ == "__main__":

    # vi = ValueIteration()
    # pi = vi.iteration()
    # vi.run_policy(pi)

    # p = PolicyIteration()
    # pi = p.iteration()
    # p.run_policy(pi)

    mt = MonteCarlo()
    pi = mt.train_mc()
    
    # print(pi)

    # pi = np.array([ [3, 3, 3, 3, 1],
    #                 [3, 0, 2, 1, 1],
    #                 [0, 0, 3, 3, 1],
    #                 [0, 3, 3, 3, 1],
    #                 [3, 3, 3, 3, 0]])
    
    mt.run_policy(pi)
