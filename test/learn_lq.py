from hyperpolicy_lq import GaussianPolicy
from lq import LQ
import pgpe_gym as pgpe
import matplotlib.pyplot as plt 
import numpy as np 

eval = False

print("===================PGPE LEARN==============")
hp = GaussianPolicy(1) 
theta_before = hp.resample()

rho_before  = hp.get_rho() 
env = LQ()


performances, rho_coll = pgpe.learn(
    env=env, 
    pol=hp, 
    gamma=0.4,
    step_size=0.9, 
    batch_size=10, 
    task_horizon=30, 
    max_iterations=1000
    )

 
for perf in performances: 
    print(perf) 
 

print("--------rho dynamics-------") 
for rho in rho_coll: 
        print("mu : ", rho) 
    
# print train performances
#rhos = np.array(rho_coll).flatten("F")
plt.plot(performances) 
plt.show()

with open('./output.txt', "w+") as file: 
    
    
    for i in range(len(performances)): 
        file.write("mu = ")
        file.write(str(rho_coll[i])) 
        file.write("\t\t\t") 
        file.write("perf = ")
        file.write(str(performances[i])) 
        file.write("\n") 


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
            
        
        if done: 
            env.reset() 
            
        
