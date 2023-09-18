from hyperpolicy import HyperPolicy
from left_env import GoLeftEnv
import pgpe

eval = True

hp = HyperPolicy(2) 
theta_before = hp.resample()
rho_before  = hp.get_rho() 
env = GoLeftEnv()

pgpe.learn(
    env=env, 
    pol=hp, 
    gamma=0.1,
    step_size=0.1, 
    batch_size=10, 
    task_horizon=300, 
    max_iterations=100
    )

print("paramenters: ", hp.get_rho()) 


if eval: 
    
    print("EVALUATION OF THE MODEL") 

    theta = hp.resample() 
    rho = hp.get_rho()
    print("theta before training = ", theta_before)
    print("theta after learnning = ", theta) 
    
    print("rho before = ", rho_before) 
    print("tho after = ", rho) 
    
    o = env.reset()
    for i in range(100):
        a = hp.act(o) 
        o, r, trunc, term, _ = env.step(a)
        
        done = trunc or term
        env.render()
        
        if done: 
            env.reset() 
            
        


        
