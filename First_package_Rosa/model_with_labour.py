# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 18:14:24 2019

@author: janro
"""

######################################################################################################################
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import math as m
import add_func as af
#import pandas as pd





###########################################################################################

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#CLASS DEFINITION

class life_cycle_2:
    """
    The life cycle model is here constructed.

    Consumers life T periods, retire after R.  Consumer may be educated or unecudated (i=0 or 1 for eductaed).
    Her productivity process is givan as:
    \ log p_{j,t,i} &= \kappa_i + \eta_{j,t,i} + \gamma_1 j -\gamma_2 j^{2} -\gamma_1 j^3 \\
    eta_{j+1,t+1} &=\rho_i eta_{j,t} + \epsilon_{j,t,i}\\
    \kappa_i &\sim N(0, \sigma_i)

    Where kappa_i are fixed effects, \eta_{j,t,i} is AR productivity process.
    Prarmetrs to calibrate a model are from Kindermann & Kruger (2017) and Kopecky (2017)


    She optimize savings (a_{j,t}) and consumption (c_{j,t}) during the life time,
    write to bugdet constraint given by wage w_t and productivity p_{j,t,i} thus solve a problem:

    max_{a_{j+1,t+1},c_{j,t}} u(c_{j,t})+E\sum_{k=0}\pi_{j+i,t+i} \beta^k u(c_{j+k,t+k})
    c_{j,t} +a_{j+1,t+1} <= ed_i w_t p_{j,t,i} \tilde{P}_t   + (1 +r_t)a_{j,t}
    where:
    \tilde{P}_t - aggregate shock
    \pi_{j+i,t+i} - probabilty of surviving to period t+i
    ed_i - education premium


    """

#######################################################CLASS CONSTRUCTOR, I WRITE ALL PARAMETERS HERE##########

    def __init__(self,
     T = 79, # number of periods of life
     R = 45, # retirement age
     alpha=0.4, # capital elasticity - used to to have suitable guess of w (optional, assuming closed econonmy what contradicts partial equilibirum)
     beta=0.98, #discount factor
     r = 0.02, # interst rate- since it is partial equilibrium
     sigma  = 2.0, #CRRA function parameter
     rho_edu = 0.98, # autororelation parameter in AR(1) productivity process
     sigma_AR_edu = 0.0180, #conditional variance  in AR(1) productivity process
     rho_unedu = 0.98, # autororelation parameter in AR(1) productivity process
     sigma_AR_unedu = 0.0346, #conditional variance  in AR(1) productivity process
     a_n =60, #number of points
     a_r = 5, #number of grid points at Rouwenhorst discretization
     n_sim = 3000, #number of simulations to get distribution of consumers
     n_sim_ag = 1, #number of simulations to get distribution of consumers
     grid_min =1e-4, #minimal assets d
     grid_max = 100.0, #maximal assets
     edu_premium = [1.0,1.34], #education premimum
     edu_prop = [0.7,0.3], #proportion of eductated
     ind_fe = [0.1517,  0.2061], #variation of the fixed effects for the different education groups
     n_fe =2, #number of fixed effect shocks
     age_eff_coef_1 = 0.0,#0.048,
     age_eff_coef_2 = 0.0, #-0.0008,
     age_eff_coef_3 = 0.0,#-6.46e-7,
     n_ag = 5, #number of grid points of for aggregated shocks
     mu_ag = 1.0, #expected value of the aggregate shock
     sigma_ag = 0.001, #variation of the aggregate shock
     lambda_pref = 5,
     chi = 1.67
     ):

        self.T, self.R, self.alpha, self.beta, self.r, self.sigma, self.rho_edu, self.sigma_AR_edu, self.rho_unedu, self.sigma_AR_unedu, self.a_n,\
        self.a_r,self.n_sim, self.grid_min, self.grid_max, self.edu_premium, self.edu_prop, self.ind_fe, self.n_fe, self.age_eff_coef_1, self.age_eff_coef_2, self.age_eff_coef_3, self.n_ag, \
        self.n_sim_ag, self.lambda_pref, self.chi \
        = T, R, alpha, beta, r, sigma, rho_edu, sigma_AR_edu, rho_unedu, sigma_AR_unedu,\
        a_n, a_r, n_sim, grid_min, grid_max, edu_premium, edu_prop, ind_fe, n_fe, age_eff_coef_1,\
         age_eff_coef_2, age_eff_coef_2, n_ag, n_sim_ag, lambda_pref,chi
        self.inv_sigma = 1/sigma
#grid definition
        self.grid = np.geomspace(self.grid_min, self.grid_max, num=self.a_n+1) #grid definition
# wage
        self.w = 1.0 #(1-alpha)*((1+r)/alpha)**(alpha/(alpha-1)) #wage guess

# life tables
        prob_dead = np.zeros(self.T+1)
        #prob_dead = np.genfromtxt('life_table.csv', delimiter=',')
        self.prob_surv = 1 - prob_dead
#Aggregate shock disretization
        [self.val_ag, self.Prob_ag] =  af.normal_discrete_1(mu_ag, sigma_ag,n_ag)
        self.val_ag = np.ones(n_ag)
#education premum and poprotion of educated

        self.edu_premium = edu_premium
        self.edu_prop = edu_prop
### productivity shocks for educated
        [val,P] =  af.approx_markov(rho_edu, m.sqrt(sigma_AR_edu), m=2, n=a_r) #take values of shocks and transition matrix for AR productivity shocks
        self.val_edu = val# [1.0,1.0,1.0,1.0,1.0]# values of the shock
        self.P_edu =  P #np.full((a_r,a_r),1/7.0)  #tranistion matrix
        self.ind_fe_edu = [-ind_fe[1], ind_fe[1] ] # values of the individual fixed effects
        self.P_ind_fe_edu = [0.59,0.41] # probability of the fixed effects


        self.edu_numb  = int(edu_prop[0]*n_sim) #proportion of the simulations of the educated consumers

### productivity shocks for uneducated
        [val,P] =  af.approx_markov(rho_unedu, m.sqrt(sigma_AR_unedu), m=2, n=a_r) #take values of shocks and transition matrix
        self.P_unedu =  P   #tranistion matrix
        self.val_unedu = val #[1.0,1.0,1.0,1.0,1.0]
        self.ind_fe_unedu = [-ind_fe[0], ind_fe[0] ] # values of the individual fixed effects
        self.P_ind_fe_unedu =   [0.59,0.41] # probability of the fixed effects

#pension definitions, here the same, minimal pension for all consumers
        self.pens = 0.4*self.w
# initail profductivity
        self.initial_prod = 2 #initial productivity
        self.initial_asset = self.grid[0] #initial assets
#grid and policy functions matrix definition

#policy function of assets and consumption for educated and uneducated
        self.pf_a_edu = np.zeros((self.T+1, self.a_n+1, self.a_r, n_fe, n_ag)) #saving policy function
        self.pf_c_edu = np.zeros((self.T+1,self.a_n+1,self.a_r,n_fe, n_ag)) #consumption policy unction
        self.pf_l_edu = np.zeros((self.T+1, self.a_n+1, self.a_r, n_fe, n_ag)) #labour policy function
        self.pf_a_unedu = np.zeros((self.T+1, self.a_n+1, self.a_r,n_fe, n_ag)) #saving policy function
        self.pf_c_unedu = np.zeros((self.T+1,self.a_n+1,self.a_r,n_fe, n_ag)) #consumption policy unction
        self.pf_l_unedu = np.zeros((self.T+1, self.a_n+1, self.a_r, n_fe, n_ag)) #labour policy function


# distribution of savings and consumption
        self.sav_distr =  np.zeros((n_sim, n_sim_ag, T+1)) #distribution of savings from simulation
        self.cons_distr = np.zeros((n_sim, n_sim_ag, T+1)) #distribution of consumption from simulation
        self.l_distr = np.zeros((n_sim, n_sim_ag, T+1))
        self.income_distr = np.zeros((n_sim, n_sim_ag, T+1,2))
# aditional table to control shock values
        self.zsim1 = np.zeros((n_sim, T+1))
# aditional table to check a budget constraint
        self.bc = np.zeros((n_sim, n_sim_ag, T+1))
# calibration parameters
        self.l_chosen = 0.5
        self.tau_l = 0.018
#pension system
        w = self.w
        self.median_income = 0.2*w
        self.median_income_old = 2*w
        self.final_pernament =  0.3*np.ones((self.a_r, n_fe,n_ag, 2))
        self.final_pernament_old = 2*w*np.ones((self.a_r, n_fe,n_ag,2))
        self.z_ag_hist = np.zeros((n_sim_ag, T+1))
        self.y_ss = 2*self.median_income
        self.pension_rescal = 1
######################################################################################################################
#####################################################################################################################
# some simple functions

    def utility(self,x): #calculate utility
         return x**(1-self.sigma)/(1-self.sigma)

    def marginal_u(self,x): #marginal utility
         if(x<1e-6):
              print("error")
         return x**(-self.sigma)

    def inv_marginal_u(self,x): #inverse of marginal utility
         if(x<1e-6):
              print("error - inv_marginal_u",x)
              return 1e-6**(-self.inv_sigma)
         return x**(-self.inv_sigma)
    def pension(self, f,p,q,edu):
        if(0.38*self.median_income>self.final_pernament[p,f,q,edu]):
            return 0.9*self.final_pernament[p,f,q,edu]
        elif((0.38*self.median_income<=self.final_pernament[p,f,q,edu]) and (1.59*self.median_income>self.final_pernament[p,f,q,edu])):
            return 0.9*0.38*self.median_income + 0.32*(self.final_pernament[p,f,q,edu] - 0.38*self.median_income)
        else:
            return 0.9*0.38*self.median_income + 0.32*((1.59 - 0.38)*self.median_income) + 0.15*(self.final_pernament[p,f,q,edu] - 1.59*self.median_income)
##########################################################################################################################
##########################################################################################################################
#Calculate the policy function for consumption and savings
    def policy_functions(self):
          """
          Find policy functions using endogenous grid method.
          """
          #grid definition
          end_grid = np.zeros(self.a_n+1) #endogenous grid definition
          end_grid_cons = np.zeros(self.a_n+1)
          pf_a = np.zeros((self.T+1, self.a_n+1, self.a_r,self.n_fe,self.n_ag)) #local table for the asset policy function
          pf_c = np.zeros((self.T+1,self.a_n+1,self.a_r,self.n_fe, self.n_ag)) #local table for the consumption policy function
          pf_l = np.zeros((self.T+1,self.a_n+1,self.a_r,self.n_fe, self.n_ag))

########################################POLICY FUNCTION FOR EDUCTAED#####################################################
          #iteration for last year
          for f in range(self.n_fe):
               for p in range(self.a_r):
                   for q in range(self.n_ag):
                       for i in range(self.a_n+1):
                           pf_c[self.T,i,p,f,q] = (1+self.r)*self.grid[i] + self.pension(f,p,q,1)
                           pf_a[self.T,i,p,f,q] = 0.0
                           pf_l[self.T,i,p,f,q] = 0.0
         #iteration for the retirees
          for j in range(self.T-1,self.R,-1):
               for f in range(self.n_fe):
                    for p in range(self.a_r):
                        for q in range(self.n_ag):
                            for i in range(self.a_n+1):

                                m_cons_sum =  self.marginal_u(pf_c[j+1,i,p,f,q])

                                cons = self.inv_marginal_u(self.prob_surv[j]*self.beta*(1+self.r)*m_cons_sum)
                                a = 1/(1+self.r)*(self.grid[i]+cons-self.pension(f,p,q,1))
                                a = np.maximum(self.grid_min,a)


                                end_grid[i] = a
                            pf_a[j,:,p,f,q] = np.maximum(np.interp(self.grid, end_grid, self.grid),self.grid_min) #interpolate on exogenous grid
                            pf_c[j,:,p,f,q] =(1+self.r)*self.grid+ self.pension(f,p,q,1) - pf_a[j,:,p,f,q]
                            pf_l[self.T,i,p,f,q] = 0.0
        #iteration for the workes
          for j in range(self.R,-1,-1):
               for f in range(self.n_fe):
                    for p in range(self.a_r):
                        for q in range(self.n_ag):
                            w = (1-self.tau_l)*self.edu_premium[1]*self.w*m.exp(self.ind_fe_edu[f]+self.val_edu[p]+self.age_eff_coef_1*j +self.age_eff_coef_2*(j)**2+ self.age_eff_coef_3*(j)**3)

                            for i in range(self.a_n+1):
                                m_cons_sum = 0
                                for i_p in range(self.a_r):
                                    for i_q in range(self.n_ag):

                                        m_cons_sum = m_cons_sum+ self.P_edu[p,i_p]*self.Prob_ag[i_q]*self.marginal_u(self.val_ag[i_q]*pf_c[j+1,i,i_p,f,i_q])
                                if j == self.R:
                                    m_cons_sum = self.marginal_u(pf_c[j+1,i,p,f,q])
                                cons = self.inv_marginal_u(self.prob_surv[j]*self.beta*(1+self.r)*m_cons_sum) #compute consumption

                                l = np.minimum(np.maximum( cons**(-self.sigma/self.chi)*(w*1/self.lambda_pref)**(1/self.chi),0.0),1.0)

                                a = 1/(1+self.r)*(self.grid[i]+cons-w*l)*self.val_ag[q] #compute endogenous grid values
                                a = np.maximum(self.grid_min,a)

                                end_grid[i] = a
                                end_grid_cons[i] = cons
                            pf_a[j,:,p,f,q] = np.maximum(np.interp(self.grid, end_grid, self.grid),self.grid_min) #interpolate on exogenous grid
                            pf_c[j,:,p,f,q] = np.maximum(np.interp(self.grid, end_grid, end_grid_cons),1e-9)

                            pf_l[j,:,p,f,q] =np.minimum(np.maximum(-1/w*((1+self.r)*self.grid/self.val_ag[q] - pf_a[j,:,p,f,q]- pf_c[j,:,p,f,q]),0.0),1.0) #find consumption policy function


          self.pf_a_edu = pf_a
          self.pf_c_edu = pf_c
          self.pf_l_edu = pf_l
########################################POLICY FUNCTION FOR UNEDUCTAED#####################################################

          pf_a = np.zeros((self.T+1, self.a_n+1, self.a_r,self.n_fe, self.n_ag)) #asset policy function
          pf_c = np.zeros((self.T+1,self.a_n+1,self.a_r,self.n_fe, self.n_ag)) #cnsumption policy function
          pf_l = np.zeros((self.T+1,self.a_n+1,self.a_r,self.n_fe, self.n_ag))
          #iteration for last year
          for f in range(self.n_fe):
               for p in range(self.a_r):
                   for q in range(self.n_ag):
                       for i in range(self.a_n+1):
                           pf_c[self.T,i,p,f,q] = (1+self.r)*self.grid[i] + self.pension(f,p,q,0)
                           pf_a[self.T,i,p,f,q] = 0.0
                           pf_l[self.T,i,p,f,q] = 0.0

          #iteration for the retirees
          for j in range(self.T-1,self.R,-1):
               for f in range(self.n_fe):
                    for p in range(self.a_r):
                        for q in range(self.n_ag):
                            for i in range(self.a_n+1):

                                m_cons_sum = self.marginal_u(pf_c[j+1,i,p,f,q])
                                cons = self.inv_marginal_u(self.prob_surv[j]*self.beta*(1+self.r)*m_cons_sum)
                                a = 1/(1+self.r)*(self.grid[i]+cons-self.pension(f,p,q,0))
                                a = np.maximum(self.grid_min,a)

                                end_grid[i] = a
                            pf_a[j,:,p,f,q] = np.maximum(np.interp(self.grid, end_grid, self.grid),self.grid_min) #interpolate on exogenous grid
                            pf_c[j,:,p,f,q] = (1+self.r)*self.grid + self.pension(f,p,q,0) - pf_a[j,:,p,f,q]
                            pf_l[self.T,i,p,f,q] = 0.0

        #iteration for the workes
          for j in range(self.R,-1,-1):
               for f in range(self.n_fe):
                    for p in range(self.a_r):
                        for q in range(self.n_ag):
                            w = (1-self.tau_l)*self.edu_premium[0]*self.w*m.exp(self.ind_fe_unedu[f]+self.val_unedu[p]+self.age_eff_coef_1*j +self.age_eff_coef_2*(j)**2+ self.age_eff_coef_3*(j)**3)
                            for i in range(self.a_n+1):
                                m_cons_sum = 0
                                for i_p in range(self.a_r):
                                    for i_q in range(self.n_ag):
                                        m_cons_sum = m_cons_sum + self.P_unedu[p,i_p]*self.Prob_ag[i_q]*self.marginal_u(self.val_ag[i_q]*pf_c[j+1,i,i_p,f,i_q])
                                if j == self.R:
                                    m_cons_sum = self.marginal_u(pf_c[j+1,i,p,f,q])
                                cons = self.inv_marginal_u(self.prob_surv[j]*self.beta*(1+self.r)*m_cons_sum) #compute consumption
                                l = np.minimum(np.maximum(1/self.lambda_pref* cons**(-self.sigma/self.chi)*(w*1/self.lambda_pref)**(1/self.chi),0.0),1.0)

                                a = 1/(1+self.r)*(self.grid[i]+cons-w*l)*self.val_ag[q] #compute endogenous grid values
                                a = np.maximum(self.grid_min,a)

                                end_grid[i] = a
                                end_grid_cons[i] = cons
                            pf_a[j,:,p,f,q] = np.maximum(np.interp(self.grid, end_grid, self.grid),self.grid_min) #interpolate on exogenous grid
                            pf_c[j,:,p,f,q] = np.maximum(np.interp(self.grid, end_grid, end_grid_cons),1e-9)
                            pf_l[j,:,p,f,q] =np.minimum(np.maximum(-1/w*((1+self.r)*self.grid/self.val_ag[q] - pf_a[j,:,p,f,q]- pf_c[j,:,p,f,q]),0.0),1.0)

          self.pf_a_unedu = pf_a
          self.pf_c_unedu = pf_c
          self.pf_l_unedu = pf_l
          return self.pf_a_edu, self.pf_c_edu, self.pf_l_edu, self.pf_a_unedu, self.pf_c_unedu, self.pf_l_unedu


############################################################################################################################################
############################################################################################################################################


    def simulate_life_cycle(self):
          '''
          Due to (possibly) aggregate shocks with no initial distribution,
          we simulate many possible shocks paths and saving and consumption paths
          instead of finding general distribution
          '''
          ##local definitions
          initial_prod = self.initial_prod
          initial_sav = self.initial_asset
          s_path = np.zeros((self.n_sim, self.n_sim_ag, self.T+1))
          c_path = np.zeros((self.n_sim, self.n_sim_ag, self.T+1))
          l_path = np.zeros((self.n_sim, self.n_sim_ag, self.T+1))
          income = np.zeros((self.n_sim, self.n_sim_ag, self.T+1,2))
          income_stable = np.zeros((self.n_sim, self.n_sim_ag, self.T+1,2))
          ## true (not divided by aggregate productivity) values of savings and consumption
          s_path_true = np.zeros((self.n_sim, self.n_sim_ag, self.T+1))
          c_path_true = np.zeros((self.n_sim, self.n_sim_ag, self.T+1))
          z_ag_hist =  np.zeros((self.n_sim_ag, self.T+1)) #history of aggregate shocks
          zsim1 = np.zeros((self.n_sim, self.n_sim_ag, self.T+1)) #table of shocks, used for
          fsim1 = np.zeros((self.n_sim, self.n_sim_ag))
          prod_history = np.ones((self.n_sim_ag,self.T+1)) #vaues of aggregate shocks
          bc = np.zeros((self.n_sim, self.n_sim_ag, self.T+1)) #budget constraint
          val_shock = np.zeros((self.n_fe,self.a_r,self.n_ag,2))
          n_shock = np.zeros((self.n_fe,self.a_r,self.n_ag,2))
## AR productivity shocks
          zsim = initial_prod
          zsim_old = initial_prod
## initiazlization of the shock values and savings
          zsim1[:,:,0] = zsim
          s_path[:,:,0] = initial_sav #initial productivity
           #initial conusumption


           #aq is indicator pro aggregate shock
          for aq in range(self.n_sim_ag):
              #simulate aggregate shocks history
              for j in range(self.R+1):
                  rand = np.random.uniform(low=0.0, high=1.0)
                  for q in range(self.n_ag):
                        if ( rand <np.sum(self.Prob_ag[0:q+1]) ):

                             z_ag_hist[aq,j] = q
                             prod_history[aq,j] = self.val_ag[q]

                             break
                #indicidual simulations for educated
              for s in range(self.edu_numb):
                zsim = initial_prod
                #simulate the individual fixed effect
                rand1 = np.random.uniform(low=0.0, high=1.0)
                f=1
                if ( rand1 <=self.P_ind_fe_edu[0] ):
                     f = 0
                fsim1[s,aq] = f
                #initialize consumption
                l_path[s,aq,0] = self.pf_c_edu[0,0,zsim,f,int(z_ag_hist[aq,0])]
                c_path[s,aq,0] = self.pf_c_edu[0,0,zsim,f,int(z_ag_hist[aq,0])]
                c_path_true[s,0] = c_path[s,0]* prod_history[aq,0]

                for j in range(1,self.T+1,1):
                     rand2 = np.random.uniform(low=0.0, high=1.0)
                     for p in range(self.a_r):

                          if ( rand2 <=np.sum(self.P_edu[zsim, 0:p+1]) ):
                               zsim_old = zsim
                               zsim =p
                               break
                     zsim1[s,aq,j] = zsim
                     s_path[s,aq,j] = np.interp(s_path[s,aq,j-1], self.grid, self.pf_a_edu[j-1,:,zsim_old,f,int(z_ag_hist[aq,j-1])])
                     c_path[s,aq,j] = np.interp(s_path[s,aq,j], self.grid, self.pf_c_edu[j,:,zsim,f,int(z_ag_hist[aq,j])])
                     l_path[s,aq,j] = np.interp(s_path[s,aq,j], self.grid, self.pf_l_edu[j,:,zsim,f,int(z_ag_hist[aq,j])])
                     s_path_true[s,aq,j] = s_path[s,aq,j]*np.prod(prod_history[aq,0:j])
                     c_path_true[s,aq,j] = c_path[s,aq,j]*np.prod(prod_history[aq,0:j+1])
                     #check budget constraint
                     w = (1-self.tau_l)*self.edu_premium[1]*np.prod(prod_history[aq,0:j])*self.w*m.exp(self.ind_fe_edu[f]+self.val_edu[zsim_old]+self.age_eff_coef_1*(j-1) +self.age_eff_coef_2*(j-1)**2+ self.age_eff_coef_3*(j-1)**3)
                     w_stable = (1-self.tau_l)*self.edu_premium[0]*self.w*m.exp(self.ind_fe_unedu[f]+self.val_unedu[zsim_old]+self.age_eff_coef_1*(j-1) +self.age_eff_coef_2*(j-1)**2+ self.age_eff_coef_3*(j-1)**3)

                     if j<=self.R+1:
                         bc[s,aq,j-1] = s_path_true[s,aq,j]+ c_path_true[s,aq,j-1] - (1+self.r)*s_path_true[s,aq,j-1] - w*l_path[s,aq,j-1]
                         income[s,aq,j-1,1] = w*l_path[s,aq,j-1]
                         income_stable[s,aq,j-1,1] = w_stable*l_path[s,aq,j-1]
                     else:
                         bc[s,aq,j-1] = s_path_true[s,aq,j]+ c_path_true[s,aq,j-1] - (1+self.r)*s_path_true[s,aq,j-1] - \
                         self.pension(f,zsim_old,int(z_ag_hist[aq,self.R]),1)*np.prod(prod_history[aq,0:j])

                         income[s,aq,j-1,1] = self.pension(f,zsim_old,int(z_ag_hist[aq,self.R]),1)*np.prod(prod_history[aq,0:j])
                         income_stable[s,aq,j-1,0] = self.pension(f,zsim_old,int(z_ag_hist[aq,self.R]),0)
            #write local variables to globa;
              self.sav_distr[0:self.edu_numb, aq, :] = s_path_true[0:self.edu_numb, aq, :]
              self.cons_distr[0:self.edu_numb, aq, :] = c_path_true[0:self.edu_numb, aq, :]
              self.l_distr[0:self.edu_numb, aq, :] = l_path[0:self.edu_numb, aq, :]
              self.bc[0:self.edu_numb, aq, :] = bc[0:self.edu_numb, aq, :]
              self.income_distr[0:self.edu_numb, aq, :] = income[0:self.edu_numb, aq, :]
              #do the same for uneducated
              for s in range(self.edu_numb, self.n_sim,1):
                 zsim = initial_prod
                 rand1 = np.random.uniform(low=0.0, high=1.0)
                 f=1
                 if ( rand1 <=self.P_ind_fe_unedu[0] ):
                      f = 0
                 fsim1[s,aq] = f
                 c_path[s,aq,0] = self.pf_c_unedu[0,0,zsim,f,int(z_ag_hist[aq,0])]
                 c_path_true[s,aq,0] = c_path[s,aq,0]*prod_history[aq,0]
                 l_path[s,aq,0] = self.pf_c_edu[0,0,zsim,f,int(z_ag_hist[aq,0])]
                 for j in range(1,self.T+1,1):

                       rand = np.random.uniform(low=0.0, high=1.0)
                       for p in range(self.a_r):
                            temp = np.sum(self.P_unedu[zsim, 0:p+1])
                            if ( rand <=temp ):
                                 zsim_old = zsim
                                 zsim =p
                                 break

                       zsim1[s,aq,j] = zsim
                       s_path[s,aq,j] = np.interp(s_path[s,aq,j-1], self.grid, self.pf_a_unedu[j-1,:,zsim_old,f,int(z_ag_hist[aq,j-1])])
                       c_path[s,aq,j] = np.interp(s_path[s,aq,j], self.grid, self.pf_c_unedu[j,:,zsim,f,int(z_ag_hist[aq,j])])
                       l_path[s,aq,j] = np.interp(s_path[s,aq,j], self.grid, self.pf_l_unedu[j,:,zsim,f,int(z_ag_hist[aq,j])])
                       s_path_true[s,aq,j] = s_path[s,aq,j]*np.prod(prod_history[aq,0:j])
                       c_path_true[s,aq,j] = c_path[s,aq,j]*np.prod(prod_history[aq,0:j+1])

                       w = (1-self.tau_l)*self.edu_premium[0]*np.prod(prod_history[aq,0:j])*self.w*m.exp(self.ind_fe_unedu[f]+self.val_unedu[zsim_old]+self.age_eff_coef_1*(j-1) +self.age_eff_coef_2*(j-1)**2+ self.age_eff_coef_3*(j-1)**3)
                       w_stable = (1-self.tau_l)*self.edu_premium[0]*self.w*m.exp(self.ind_fe_unedu[f]+self.val_unedu[zsim_old]+self.age_eff_coef_1*(j-1) +self.age_eff_coef_2*(j-1)**2+ self.age_eff_coef_3*(j-1)**3)
                       if j<=self.R+1:
                           bc[s,aq,j-1] = s_path_true[s,aq,j]+ c_path_true[s,aq,j-1] - (1+self.r)*s_path_true[s,aq,j-1] - w*l_path[s,aq,j-1]
                           income[s,aq,j-1,0] = w*l_path[s,aq,j-1]
                           income_stable[s,aq,j-1,0] = w_stable*l_path[s,aq,j-1]
                       else:
                           bc[s,aq,j-1] = s_path_true[s,aq,j]+ c_path_true[s,aq,j-1] - (1+self.r)*s_path_true[s,aq,j-1] - \
                           self.pension(f,zsim_old,int(z_ag_hist[aq,self.R]),0)*np.prod(prod_history[aq,0:j])

                           income[s,aq,j-1,0] = self.pension(f,zsim_old,int(z_ag_hist[aq,self.R]),0)*np.prod(prod_history[aq,0:j])
                           income_stable[s,aq,j-1,0] = self.pension(f,zsim_old,int(z_ag_hist[aq,self.R]),0)
              self.sav_distr[self.edu_numb: self.n_sim,aq,:] = s_path_true[self.edu_numb: self.n_sim,aq,:]
              self.cons_distr[self.edu_numb: self.n_sim,aq,:] =c_path_true[self.edu_numb: self.n_sim,aq,:]
              self.bc[self.edu_numb: self.n_sim, aq, :] = bc[self.edu_numb: self.n_sim, aq, :]
              self.l_distr[self.edu_numb: self.n_sim,aq,:] =l_path[self.edu_numb: self.n_sim,aq,:]
              self.income_distr[self.edu_numb: self.n_sim,aq,:] =income[self.edu_numb: self.n_sim,aq,:]

              self.median_income_old = self.median_income



          for aq in range(self.n_sim_ag):
              for s in range(self.n_sim):
                  if(s<self.edu_numb):
                      val_shock[int( fsim1[s,aq]),  int(zsim1[s,aq,self.R]),int( z_ag_hist[aq,self.R]),1] += income_stable[s,aq,self.R,1]
                      n_shock[ int(fsim1[s,aq]), int(zsim1[s,aq,self.R]),int( z_ag_hist[aq,self.R]),1] +=1
                  else:
                      val_shock[int( fsim1[s,aq]),  int(zsim1[s,aq,self.R]),int( z_ag_hist[aq,self.R]),0] += income_stable[s,aq,self.R,0]
                      n_shock[ int(fsim1[s,aq]), int(zsim1[s,aq,self.R]),int( z_ag_hist[aq,self.R]),0] +=1


          for edu in range(2):
              for f in range(self.n_fe):
                   for p in range(self.a_r):
                       for q in range(self.n_ag):
                           if n_shock[f,p,q,edu] ==0:
                               self.final_pernament[p,f,q,edu] = 0.5*self.final_pernament_old[p,f,q,edu] + 0.5*(1-self.tau_l)*self.edu_premium[0]*np.prod(prod_history[aq,0:j])*self.w*m.exp(self.ind_fe_unedu[f]+self.val_unedu[zsim_old]+self.age_eff_coef_1*(j-1) +self.age_eff_coef_2*(j-1)**2+ self.age_eff_coef_3*(j-1)**3)*0.5
                           else:
                               self.final_pernament[p,f,q,edu] = 0.5*self.final_pernament_old[p,f,q,edu]+ 0.5*val_shock[f,p,q,edu]/n_shock[f,p,q,edu]



          self.median_income = 0.5*self.median_income_old + 0.5*np.median(income_stable[:,:,0:self.R+1])
          print("median income")
          print(np.median(income[:,:,0:self.R+1]))
          print(np.median(l_path[:,:,0:self.R+1]))
          print("_____________________")
          #print(self.final_pernament)
          #print(val_shock)
          #print(n_shock)
          return self.sav_distr, self.cons_distr, self.zsim1, self.l_distr
##############################################################################################################
    def plot_life_cycle(self):
         """
         Some plots. Savongs and consumption distribution, policy function for the period 44 (just before retirment)
         """

         s_mean = np.zeros(self.T+1)
         s_max = np.zeros(self.T+1)
         s_min = np.zeros(self.T+1)
         s_median = np.zeros(self.T+1)
         c_mean = np.zeros(self.T+1)
         c_max = np.zeros(self.T+1)
         c_min = np.zeros(self.T+1)

         z_mean = np.zeros(self.T+1)
         z_max = np.zeros(self.T+1)
         z_min = np.zeros(self.T+1)

         bc_mean = np.zeros(self.T+1)
         bc_max = np.zeros(self.T+1)
         bc_min = np.zeros(self.T+1)

         l_mean = np.zeros(self.T+1)
         l_max = np.zeros(self.T+1)
         l_min = np.zeros(self.T+1)


         for j in range(1,self.T+1,1):
             s_mean[j] = np.mean(self.sav_distr[:,:,j])
             s_max[j] = np.percentile(self.sav_distr[:,:,j],90)
             s_min[j] = np.percentile(self.sav_distr[:,:,j],10)
             s_median[j] = np.median(self.sav_distr[:,:,j])
         plt.plot(range(self.T+1), s_mean, label = "mean savings")
         plt.plot(range(self.T+1), s_max, label = "90th percentile of savings")
         plt.plot(range(self.T+1), s_min, label = "10th percentile of savings")
         plt.plot(range(self.T+1), s_median, label = "median")
         plt.ylabel('Savings profile')
         plt.legend(loc='best')
         plt.show()
         for j in range(1,self.T+1,1):
             z_mean[j] = np.mean(self.zsim1[:,j])
             z_max[j] = np.max(self.zsim1[:,j])
             z_min[j] = np.min(self.zsim1[:,j])
         plt.plot(range(self.T+1), z_mean, label = "mean shocks")
         plt.plot(range(self.T+1), z_max, label = "max shocks")
         plt.plot(range(self.T+1), z_min, label = "min shocks")
         plt.ylabel('shocks')
         plt.legend(loc='best')
         plt.show()

         for j in range(1,self.T+1,1):
             c_mean[j] = np.mean(self.cons_distr[:,:,j])
             c_max[j] = np.percentile(self.cons_distr[:,:,j],90)
             c_min[j] = np.percentile(self.cons_distr[:,:,j],10)
         plt.plot(range(self.T+1), c_mean, label = "mean consumption")
         plt.plot(range(self.T+1), c_max, label = "90th percentile of consumption")
         plt.plot(range(self.T+1), c_min, label = "10th percentile consumption")
         plt.ylabel('Consumption path')
         plt.legend(loc='best')
         plt.show()

         for j in range(1,self.T+1,1):
             bc_mean[j] = np.mean(self.bc[:,:,j])
             bc_max[j] = np.percentile(self.bc[:,:,j],99)
             bc_min[j] = np.percentile(self.bc[:,:,j],1)
         plt.plot(range(self.T+1), bc_mean, label = "bc consumption")
         plt.plot(range(self.T+1), bc_max, label = "bc max")
         plt.plot(range(self.T+1), bc_min, label = "bc min")
         plt.ylabel('budget constraint')
         plt.legend(loc='best')
         plt.show()

         for j in range(1,self.T+1,1):
             l_mean[j] = np.mean(self.l_distr[:,:,j])
             l_max[j] = np.percentile(self.l_distr[:,:,j],99)
             l_min[j] = np.percentile(self.l_distr[:,:,j],1)
         plt.plot(range(self.T+1), l_mean, label = "l consumption")
         plt.plot(range(self.T+1), l_max, label = "l max")
         plt.plot(range(self.T+1), l_min, label = "l min")
         plt.ylabel('labor')
         plt.legend(loc='best')
         plt.show()

         plt.plot(self.grid[0:50], self.pf_a_edu[40,0:50,0,1,2], label = "savings for worst group")
         plt.plot(self.grid[0:50], self.pf_a_edu[40,0:50,1,1,2], label = "savings for second worst group")
         plt.plot(self.grid[0:50], self.pf_a_edu[40,0:50,2,1,2], label = "savings for mednian group")
         plt.plot(self.grid[0:50], self.pf_a_edu[40,0:50,3,1,2], label = "savings for second best group")
         plt.plot(self.grid[0:50], self.pf_a_edu[40,0:50,4,1,2], label = "savings for best group")
         # plt.plot(self.grid[0:80], self.pf_a_edu[44,0:80,0,1,2], label = "savings for worst2 group")
         # plt.plot(self.grid[0:80], self.pf_a_edu[44,0:80,1,1,2], label = "savings for second2 worst group")
         # plt.plot(self.grid[0:80], self.pf_a_edu[44,0:80,2,1,2], label = "savings for mednian2 group")
         # plt.plot(self.grid[0:80], self.pf_a_edu[44,0:80,3,1,2], label = "savings for second best2 group")
         # plt.plot(self.grid[0:80], self.pf_a_edu[44,0:80,4,1,2], label = "savings for best 2group")
         plt.ylabel('saving policy function for eductated workers, at medium aggregate shock, with high fixed effect')
         plt.legend(loc='best')
         plt.show()

    def labour_rate(self, x):
        self.lambda_pref = x
        self.policy_functions()
        self.simulate_life_cycle()
        #print(x)
        #print(np.mean(self.l_distr[:,:,0:self.R+1]))
        #print(np.mean(self.l_distr[:,:,0:self.R+1]) - self.l_chosen)
        return np.mean(self.l_distr[:,:,0:self.R+1]) - self.l_chosen

    def calibrate(self):
        for i in range(100):
            print i
            self.policy_functions()
            #print(self.final_pernament)
            self.final_pernament_old = self.final_pernament
            self.simulate_life_cycle()
            # print("after simulation")
            # print self.final_pernament, -1*self.final_pernament_old
            # print(np.amax(np.absolute(np.add(self.final_pernament,-1*self.final_pernament_old))))
            # print( self.final_pernament-1*self.final_pernament_old)
            # print(self.median_income)
            # print("end")
            if np.amax(np.absolute(np.add(self.final_pernament,-1*self.final_pernament_old)))<1e-4 and abs(self.median_income-self.median_income_old)<1e-4 :
                print(np.amax(np.absolute(np.add(self.final_pernament,-1*self.final_pernament_old))))
                print( self.final_pernament-self.final_pernament_old)
                print(self.median_income)
                break

        pension_income = np.sum(self.income_distr[:,:,self.R+1:self.T+1])/np.sum(self.income_distr[:,:,0:self.R+1])
        print(pension_income)
        self.plot_life_cycle()
############################################################################################################################
    def execute_life_cycle(self):
          self.policy_functions()
          self.simulate_life_cycle()
          self.plot_life_cycle() #some basic plots
############################################################################################################################
############################################################################################################################





