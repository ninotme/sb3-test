from hyperpolicy import HyperPolicy
from left_env import GoLeftEnv
import pgpe
import matplotlib.pyplot as plt 


eval = True

print("===================PGPE LEARN==============")
hp = HyperPolicy(2) 
theta_before = hp.resample()
rho_before  = hp.get_rho() 
env = GoLeftEnv(10)

performances = pgpe.learn(
    env=env, 
    pol=hp, 
    gamma=0.9,
    step_size=0.1, 
    batch_size=3, 
    task_horizon=10, 
    max_iterations=1000
    )

print("paramenters: ", hp.get_rho()) 
print("perfomances: = ", performances)

# print train perfomances

plt.plot(performances) 
#plt.ylim([-100, 10]) 
plt.show()
if eval: 
    
    print("EVALUATION OF THE MODEL") 

    theta = hp.resample() 
    rho = hp.get_rho()
    print("theta before training = ", theta_before)
    print("theta after learnning = ", theta) 
    
    print("rho before = ", rho_before) 
    print("tho after = ", rho) 
    
    o = env.reset()
    
    
    for i in range(120):
        a = hp.act(o) 
        o, r, trunc, term, _ = env.step(a)
        
        done = trunc or term
        env.render()
        
        if done: 
            env.reset() 
            
        
        
