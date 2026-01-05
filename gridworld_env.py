
import numpy as np
import matplotlib.pyplot as plt

class GridWordEnv:

    def __init__(self,gird_size=5, goal=None, render = False):
        self.render_enabled = render
        if render:
            self.fig, self.ax = plt.subplots()

        self.gird_size = gird_size
        self.goal = np.array(goal) if goal is not None else np.array([gird_size-1, gird_size-1])
        self.obstacles = [np.array([2,2]), np.array([1,3]), np.array([3,0]),np.array([1,2]),np.array([3,1])]
        self.action_dict: dict = {
        0:(-1,0), # up
        1:(1,0),  # down
        2:(0,-1), # left
        3:(0,1)   # right
    }
        self.reset()

    def state_transition(self, state: np.ndarray, action: int):
        reward = 0
        done = False
        new_state = state + np.array(self.action_dict[action])
        if np.any(new_state < 0) or np.any(new_state >= self.gird_size):
            reward = -1 
            new_state = state 
        if any(np.array_equal(new_state, obs) for obs in self.obstacles):
            reward = -1
        if np.array_equal(new_state, self.goal):
            reward = 1
            done = True
        return new_state, reward, done
    
    def reset(self):
        self.state = np.array([0,0])
        self.episdoe_return = 0
        return self.state

    def step(self, action: int):
        reward = 0
        done = False
        new_state = self.state + np.array(self.action_dict[action])
        if np.any(new_state < 0) or np.any(new_state >= self.gird_size):
            reward = -1
            new_state = self.state

        if any(np.array_equal(new_state, obs) for obs in self.obstacles):
            reward = -1

        self.state = new_state
        if np.array_equal(self.state, self.goal):
            reward = 1
            done = True

        if self.render_enabled:
            self.render()

        return self.state, reward, done, {}
    
    def render(self):
        self.ax.clear()
        grid = np.zeros((self.gird_size, self.gird_size))
        self.ax.imshow(grid, cmap='Greys', extent=(0, self.gird_size, 0, self.gird_size))
        self.ax.scatter(self.state[1] + 0.5, self.state[0] + 0.5, c='blue', s=200, label='Agent')
        self.ax.scatter(self.goal[1] + 0.5, self.goal[0] + 0.5, c='red', s=50, marker='*', label='Goal')
        for obs in self.obstacles:
            self.ax.scatter(obs[1] + 0.5, obs[0] + 0.5, c='black', s=100, label='Obstacle')
        self.ax.scatter(0.5, 0.5, c='green', s=50, label='Start')
        self.ax.set_xticks(np.arange(0, self.gird_size + 1, 1))
        self.ax.set_yticks(np.arange(0, self.gird_size + 1, 1))
        self.ax.grid(True)
        plt.pause(0.1)


if __name__ == "__main__":
    env = GridWordEnv(gird_size=5, render=True)
    state = env.reset()
    done = False
    while not done:
        action = np.random.choice(4)  # Random action for demonstration
        state, reward, done, _ = env.step(action)
    plt.show()




        

