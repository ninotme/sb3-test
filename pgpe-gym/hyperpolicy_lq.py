import numpy as np
from gymnasium  import spaces
from numpy.random import normal

forced_policy = False
verbose = False
debug = True


class GaussianPolicy:
    def __init__(self, param_number, init_mu=1, init_sigma=2.0):

        
        self.param_number = param_number

        # Inizializzazione della distribuzione dei parametri
        self.mu = np.full( self.param_number, init_mu)
        print("init_mu = ", self.mu) 
        self.sigma = np.full(  self.param_number, init_sigma) 
        print("init_sigma = ", self.sigma) 
        

        self.theta = []
        print("done initializing")

    # Called by algorithm
    #this should be overwritten by a policy class
    def act(self, state):
       # print("act::state: ", state)
        
        
        feat = self.theta[0]
        print("action taken: ", feat)
        return feat
        

    #calcolo del delta_rho
    def eval_gradient(self, actor_params, disc_returns, use_baseline=False):
        

        print("actor_params: ", actor_params, "len ", len(actor_params))
        
        
        if verbose:
            print("-------in eval_gradient()________") 
            print("disc_returns = ", disc_returns) 
            print("actor_params = ", actor_params) 
        
        grad_mu = disc_returns * (actor_params - self.mu) / (self.sigma ** 2) 
        grad_sigma = disc_returns * (( (actor_params - self.mu) ** 2) - self.sigma ** 2 ) / (self.sigma ** 3)
        
        grad_mu = np.mean(grad_mu) 
        grad_sigma = np.mean(grad_sigma) 
        
        
        # if debug=True we compute intermediate result for troubleshooting
        if debug: 
            N = len(disc_returns)
            sum_sigma = np.full(self.param_number, 0) 
            sum_mu = np.full(self.param_number, 0)

            total_mu_score = np.full(self.param_number, 0.0) 
            total_sigma_score = np.full(self.param_number, 0.0)
            
            for i in range(N):
                # param and return associated to episode i 
                theta = np.array(actor_params[i]) 
                disc_ret = disc_returns[i]
                score_mu = (theta - self.mu) /   self.sigma ** 2 
                    
                
                
                # OVERFLOW!?!?!?
                print("Checking for sigma overflow.....")
                term1 = (theta - self.mu) 
                print("theta - mu = ", term1) 
                
                numerator = term1 ** 2 - self.sigma ** 2
                print("numerator = ", numerator) 
                
                score_sigma = ((theta - self.mu)**2 - (self.sigma ** 2)) / (self.sigma ** 3)
                    
                
                print("===========score_mu: ", score_mu) 
                print("score_mu = ", score_mu)
                print("len(score_mu) = ", len(score_mu)) 
                
                print("multiply ", np.multiply(score_mu, disc_ret)) 
                #total_sigma_score += score_sigma * disc_ret
                print("==========score_sigma: ", score_sigma) 
                
                print("theta = ", theta)
                print("score_mu = ", score_mu)
                print("score_sigma = ", score_sigma)
                #print("mus = ", mus) 
                #print("sigmas = ", sigmas) 
                        
                
                
                # sum to the toatal score
                total_mu_score += np.multiply(score_mu[0], disc_ret) 
                total_sigma_score += np.multiply(score_sigma[0], disc_ret) 
                print("total_mu_score = ", total_mu_score) 
                print("toatal_sigma_score = ", total_sigma_score) 
                    
            _grad_mu = np.divide(total_mu_score, N)
            _grad_sigma = np.divide(total_sigma_score, N)
            print("grad_sigma = ", grad_sigma) 
            print("self.sigma = ", self.sigma)
        
         
            print('[', grad_mu, ', \n', grad_sigma, ']' )
            grad_mu = _grad_mu
            grad_sigma = _grad_sigma
        
        return [grad_mu, grad_sigma]

       

    def eval_params(self):
        return [ self.mu, self.sigma] 
    
    def set_params(self,rho):
        self.mu = np.array(rho[0]) 
        self.sigma = np.array(rho[1])
        

    def resample(self):
        #print("resampling theta...") 
        #print(self.mu, " ", self.sigma, ' ', self.param_number) 
        #self.theta = normal(self.mu, self.sigma, self.param_number)
        
        
        #self.theta = [ normal(self.mu[i], self.sigma[i]**2) for i in range(self.param_number) ] 
        
        #random.normal needs the standard deviation, not the variance ?(-_-) 
        
        #abs because we are are derivating the score function, so the standard deviation can be negative
        self.theta  =  normal(self.mu, abs(self.sigma) ) 
        print("sampled theta = ", self.theta) 
        
        return self.theta

    #internal methods
    def set_rho(self, mu, sigma):
        self.mu = np.array(mu)
        self.sigma = sigma
    def get_rho(self):
        return [self.mu, self.sigma]

    def set_theta(self, theta):
        self.theta = theta
    def get_theta(self):
        return self.theta


    ## Unused methods

    def renyi(self, pol):
        pass
    def eval_fisher(self):
        pass

#test con finta reward_batch [1000, 1000]
# il batch di  parametri della policy di test campionati dalla policy
# degli iperparamenteri sono 2:  [theta1, theta2]
if __name__ == '__main__':

    hp = HyperPolicy(2)

    # prelevo due campioni dalla attuale distribuzione degli iperparametri
    theta1 = hp.resample()
    theta2 = hp.resample()

    #test update
    actor_params = [theta1, theta2]
    disc_returns = [1000, 1000]

    # valutazione del gradiente
    grad  = hp.eval_gradient(actor_params, disc_returns)

    #update della distribuzione degli iperparamentri
    step_size = 0.0001
    delta_rho = np.multiply(grad, step_size)
    hp.set_params(hp.get_rho() + delta_rho)
    thetaf = hp.resample()
    print('update distribution: ', hp.get_rho()[0], hp.get_rho()[1]) 

