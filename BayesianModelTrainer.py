import pymc3 as pm
import matplotlib.pyplot as plt
import theano
import numpy as np
import pandas as pd

class BayesianModelTrainer:
    def __init__(self,pm25_data):
        self.pm25_data = pm25_data
        self.trace = None
        
    def specify_model(self):
        theano.config.openmp = True
        theano.config.optimizer = "fast_compile"
        with pm.Model() as m:
            
            #priors
            phi = pm.Normal("phi",mu=0, sd=1)
            theta = pm.Normal("theta", mu=0, sd=1)
            sigma = pm.HalfNormal("sigma",sd=1)
            
            #likelihood
            pm.AR("pm25_observation",rho=phi, sigma=sigma, observed=self.pm25_data["PM2.5"])
            
            for t in range(1,len(self.pm25_data)):
                pm.Deterministic(f'pm25_ar_{t}', phi * self.pm25_data["PM2.5"][t-1])
                
            pm.Deterministic("pm25_ma", theta * sigma)
            
        self.m = m
            
    def train_model(self,draws=4000, tune=2000):
        with self.m:
            self.trace = pm.sample(draws=draws, tune=tune)
            
    def visualize_model(self):
        pm.plot_posterior(self.trace, var_names=["phi", "theta", "sigma"])
        plt.show()
            
            