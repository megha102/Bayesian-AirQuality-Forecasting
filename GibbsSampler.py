import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt

class GibbsSampler:
    def __init__(self,pm25_data):
        self.pm35_data = pm25_data
        self.trace = None
        
    def run_gibbs_sampler(self, iterations=4000):
        phi = 0.1
        theta = 0.1
        sigma = 1.0
        
        samples = {"phi":[], "theta":[], "sigma":[]}
        
        for i in range(iterations):
            phi = self.sample_phi(phi, theta, sigma)
            theta = self.sample_theta(phi, theta, sigma)
            sigma = self.sample_sigma(phi, theta, sigma)
            
            samples["phi"].append(phi)
            samples["theta"].append(theta)
            samples["sigma"].append(sigma)
            
        self.trace = samples
        
    def sample_phi(self, phi, theta, sigma):
        phi_samples = np.random.normal(loc=phi, scale=0.1, size=1)
        return phi_samples[0]
    
    def sample_theta(self, phi, theta, sigma):
        theta_samples = np.random.normal(loc=theta, scale=0.1, size=1)
        return theta_samples[0]
    
    def sample_sigma(self, phi, theta, sigma):
        sigma_samples = np.abs(np.random.normal(loc=sigma, scale=0.1, size=1))
        return sigma_samples[0]
    
    def visualize_trace(self):
        plt.plot(self.trace["phi"], label="phi")
        plt.plot(self.trace["theta"], label="theta")
        plt.plot(self.trace["sigma"], label="sigma")
        plt.legend()
        plt.title("Trace of Gibbs Sampler Parameters")
        plt.show()