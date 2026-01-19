import numpy as np

class GaussNoiseGenerator():

    def __init__(self, seed = 44):
        self.rng = np.random.default_rng(seed = seed)
        

    def __call__(self) -> float: 

        return float(self.rng.normal(loc = 0, scale= 2))

class RMexpample():

    def __init__(self):
        self.noise = GaussNoiseGenerator()
        self.tol = 1e-10

    def g0_obs(self, w):
        return w**3 - 2 + self.noise()
    
    def solve_rm(self):
        wk = 0
        wk_new = 0
        firstFlag = True
        step = 0
        while abs(wk - wk_new) > self.tol or firstFlag:
            step = step + 1 
            firstFlag = False
            wk = wk_new
            wk_new = wk - 1/step * self.g0_obs(wk)
            
        return wk
             
        
if  __name__ == "__main__":
    rm = RMexpample()
    wk = rm.solve_rm()
    print(wk)
    print(2**(1/3))
    

        