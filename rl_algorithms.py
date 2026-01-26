from gridworld_env import GridWordEnv
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class EnvParams:
    gamma: float = 0.8
    grid_size: int = 5
    epsilon: float = 1e-5


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

class TemporalDifference():
    def __init__(self):
        self.env = GridWordEnv(gird_size=5, render=True)
        self.gamma = 0.99
        self.max_step = 2000
        self.episode = 50
        self.state0 = self.env.reset()
        self.greedy_action = np.zeros((5,5))
        self.v_pi = np.zeros((5,5))
        self.alpha = 1
   
    def epsilon_greedy_action(self, epsilon, greedy_action) -> int: 
        probs = np.full(4, epsilon/4, dtype="float")
        probs[greedy_action] = 1 - epsilon + epsilon/4
        return np.random.choice(4, p = probs)

    def run_td(self):
        for i in range(self.episode):
            step_matrix = np.zeros((5,5))
            state = self.env.reset()
            next_state = state.copy()
            step = 0
            done = False
            while step < self.max_step and not done:
                step = step + 1
                step_matrix[state[0], state[1]] += 1
                state = next_state.copy()
                q_pi = []
                for j in range(4):
                    q_pi.append()
                # 如果是单纯的td 算法，到这里就没有办法进行了。没有办法更新策略Pi, 因为是model free,因此没办法遍历四个动作，找最优动作。这个问题的来源是，bellman eq解的是v pi, 如果要进行更新策略则需要q_pi, 而v_pi到q_pi这个过程是需要transition model的。所以引入sarsa.
                next_state, reward, done, _ = self.env.step(action)
                self.v_pi[state[0], state[1]] = self.v_pi[state[0], state[1]] - self.alpha/step_matrix[state[0], state[1]] * (self.v_pi[state[0], state[1]] - (reward + self.gamma * self.v_pi[next_state[0], next_state[1]]))

class SARSA():
    def __init__(self):
        self.episode = 5000
        self.q_pi = np.zeros((5,5,4))
        self.alpha = 0.1
        self.gamma = 0.9
        self.env = GridWordEnv(gird_size=5, render=True)
        self.env.render_enabled = False
        self.max_step = 2000


    def epsilon_greedy_action(self, epsilon, greedy_action) -> int: 
        probs = np.full(4, epsilon/4, dtype="float")
        probs[greedy_action] = 1 - epsilon + epsilon/4
        return np.random.choice(4, p = probs)

    def run_sarsa(self):
        for i in range(self.episode):
            reward_sum = 0 
            state = self.env.reset()
            next_state = state.copy()
            step = 0
            done = False
            greedy_action = np.argmax(self.q_pi[state[0],state[1]])
            action = self.epsilon_greedy_action(0.2, greedy_action)
            while step < self.max_step and not done:
                step = step + 1
                next_state, reward, done, _ = self.env.step(action)
                current_q =  self.q_pi[state[0],state[1], action]
                action_current_state = action

                greedy_action = np.argmax(self.q_pi[next_state[0],next_state[1]])
                action = self.epsilon_greedy_action(0.2, greedy_action)

                next_q = 0 if done else self.q_pi[next_state[0],next_state[1],action] 

                self.q_pi[state[0],state[1],action_current_state] = current_q - self.alpha * (current_q - (reward + self.gamma * next_q))
                state = next_state.copy()
                reward_sum = reward_sum + reward

            if i%50 == 0:
                print(reward_sum) 

    def eval_sarsa(self):
        state = self.env.reset()
        self.env.render_enabled = True
        done = False
        while not done:
            greedy_action = np.argmax(self.q_pi[state[0],state[1]])
            state, reward, done, _ = self.env.step(greedy_action)
            

                


class QLearning():
    def __init__(self):
        self.env = GridWordEnv(render=True)
        self.env.render_enabled = False
        self.max_step = 2000
        self.episode = 2000
        self.q_pi = np.zeros((5,5,4))
        self.alpha = 0.1
        self.gamma = 0.99 
    
    def epsilon_greedy(self, epsilon, greedy_action):
        probs = np.full(4, epsilon/4, dtype="float")
        probs[greedy_action] = 1 - epsilon + epsilon/4
        return np.random.choice(4, p=probs)

    def run_q_learning(self):

        for i in range(self.episode):
            step = 0
            done = False
            state = self.env.reset()
            reward_sum = 0
            while step < self.max_step and not done:
                
                greedy_action = np.argmax(self.q_pi[state[0], state[1]])
                # behavior policy 是epsilon greedy
                action = self.epsilon_greedy(0.2, greedy_action)

                next_state, reward, done, _ = self.env.step(action)

                this_q = self.q_pi[state[0], state[1], action]
                
                greedy_action_next_state = np.argmax(self.q_pi[next_state [0], next_state [1]])
     
                td_target = reward if done else reward + self.gamma* self.q_pi[next_state[0], next_state[1], greedy_action_next_state]

                td_error = this_q - td_target
                # target policy 是 greedy. 
                self.q_pi[state[0], state[1], action] = this_q - self.alpha * td_error
                # behavior policy 和 target policy 不同，所以q learning 是off-policy. 如果说把q learning 改成on-policy 那实际上和sarasa是一样的了。 
                state = next_state.copy()
                reward_sum = reward_sum + reward
            if i%50 == 0:
                print(f'reward of episode {i} is {reward_sum}')
            

    def eval(self):
        state = self.env.reset()
        self.env.render_enabled = True
        done = False
        while not done:

            action = np.argmax(self.q_pi[state[0], state[1]])
            state, reward, done, _ = self.env.step(action)


class SARSAwithFunction():

    def __init__(self):
        pass


    def second_order_fuction(self, w, state):
        x = state[0]
        y = state[1]

        f = 

        return 
                
                
                
                






if __name__ == "__main__":

    algorithm_name = "QLearning"

    if algorithm_name == "VI":

        vi = ValueIteration()
        pi = vi.iteration()
        vi.run_policy(pi)

    if algorithm_name == "PE":
        p = PolicyIteration()
        pi = p.iteration()
        p.run_policy(pi)

    if algorithm_name == "MC":
        mt = MonteCarlo()
        pi = mt.train_mc()
        mt.run_policy(pi)

    if algorithm_name == "SARSA":
        sa = SARSA()
        sa.run_sarsa()
        sa.eval_sarsa()

    if algorithm_name == "QLearning":
        ql = QLearning()
        ql.run_q_learning()
        ql.eval()
        
