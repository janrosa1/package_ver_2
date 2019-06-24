# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 19:00:17 2019

@author: janro
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import math as m
import add_func as af

class life_cycle_1:
    """
    The life cycle model is here constructed.
    Consumers life T periods, retire after R.  Consumer may be educated or unecudated (i=0 or 1 for eductaed).
    Her productivity process is givan as:
    \ log p_{j,t,i} &= \kappa_i + \eta_{j,t,i} + \gamma_1 j -\gamma_2 j^{2} -\gamma_1 j^3 
    eta_{j+1,t+1} &=\rho_i eta_{j,t} + \epsilon_{j,t,i}

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
     r = 0.04, # interst rate- since it is partial equilibrium
     sigma  = 2.0, #CRRA function parameter
     rho_edu = 0.97, # autororelation parameter in AR(1) productivity process
     sigma_AR_edu = 0.0180, #conditional variance  in AR(1) productivity process
     rho_unedu = 0.98, # autororelation parameter in AR(1) productivity process
     sigma_AR_unedu = 0.0346, #conditional variance  in AR(1) productivity process
     a_n =100, #number of points
     a_r = 5, #number of grid points at Rouwenhorst discretization
     n_sim = 100, #number of simulations to get distribution of consumers
     n_sim_ag = 10, #number of simulations to get distribution of consumers
     grid_min =1e-4, #minimal assets d
     grid_max = 60.0, #maximal assets
     edu_premium = [1.0,1.34], #education premimum
     edu_prop = [0.7,0.3], #proportion of eductated
     ind_fe = [0.1517,  0.2061], #variation of the fixed effects for the different education groups
     n_fe =2, #number of fixed effect shocks
     age_eff_coef_1 = 0.048,
     age_eff_coef_2 = -0.0008,
     age_eff_coef_3 = -6.46e-7,
     n_ag = 5, #number of grid points of for aggregated shocks
     mu_ag = 1.0, #expected value of the aggregate shock
     sigma_ag = 0.001 #variation of the aggregate shock
     ):

        self.T, self.R, self.alpha, self.beta, self.r, self.sigma, self.rho_edu, self.sigma_AR_edu, self.rho_unedu, self.sigma_AR_unedu, self.a_n,\
        self.a_r,self.n_sim, self.grid_min, self.grid_max, self.edu_premium, self.edu_prop, self.ind_fe, self.n_fe, self.age_eff_coef_1, self.age_eff_coef_2, self.age_eff_coef_3, self.n_ag, \
        self.n_sim_ag\
        = T, R, alpha, beta, r, sigma, rho_edu, sigma_AR_edu, rho_unedu, sigma_AR_unedu,\
        a_n, a_r, n_sim, grid_min, grid_max, edu_premium, edu_prop, ind_fe, n_fe, age_eff_coef_1, age_eff_coef_2, age_eff_coef_3, n_ag, n_sim_ag
        self.inv_sigma = 1/sigma
#grid definition
        self.grid = np.geomspace(self.grid_min, self.grid_max, num=self.a_n+1) #grid definition
# wage
        self.w = (1-alpha)*((1+r)/alpha)**(alpha/(alpha-1)) #wage guess

# life tables
        #prob_dead = np.genfromtxt('life_table.csv', delimiter=',')
        prob_dead = np.zeros(self.T+1)
        self.prob_surv = 1.0 - prob_dead
        self.prob_absol = np.zeros(T+1)
        for j in range(self.T+1):
            self.prob_absol[j] =np.prod(self.prob_surv[0:j+1])

#Aggregate shock disretization
        [self.val_ag, self.Prob_ag] =  af.normal_discrete_1(mu_ag, sigma_ag,n_ag)
        self.val_ag = [1.0,1.0,1.0,1.0,1.0]
#education premum and poprotion of educated

        self.edu_premium = edu_premium
        self.edu_prop = edu_prop
### productivity shocks for educated
        [val,P] =  af.approx_markov(rho_edu, m.sqrt(sigma_AR_edu), m=2, n=a_r) #take values of shocks and transition matrix for AR productivity shocks
        self.val_edu = val #[1.0,1.0,1.0,1.0,1.0]# values of the shock
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
        self.pf_a_unedu = np.zeros((self.T+1, self.a_n+1, self.a_r,n_fe, n_ag)) #saving policy function
        self.pf_c_unedu = np.zeros((self.T+1,self.a_n+1,self.a_r,n_fe, n_ag)) #consumption policy unction


        self.pf_a_edu_hyp = np.zeros((self.T+1, self.a_n+1, self.a_r, n_fe, n_ag)) #saving policy function
        self.pf_c_edu_hyp = np.zeros((self.T+1,self.a_n+1,self.a_r,n_fe, n_ag)) #consumption policy unction
        self.pf_a_unedu_hyp = np.zeros((self.T+1, self.a_n+1, self.a_r,n_fe, n_ag)) #saving policy function
        self.pf_c_unedu_hyp = np.zeros((self.T+1,self.a_n+1,self.a_r,n_fe, n_ag)) #consumption policy unction


# distribution of savings and consumption
        self.sav_distr =  np.zeros((n_sim, n_sim_ag, T+1)) #distribution of savings from simulation
        self.cons_distr = np.zeros((n_sim, n_sim_ag, T+1)) #distribution of consumption from simulation

        self.sav_distr_opt =  np.zeros((n_sim, n_sim_ag, T+1)) #distribution of savings from simulation
        self.cons_distr_opt = np.zeros((n_sim, n_sim_ag, T+1))
        self.underself_cons = np.zeros((n_sim*n_sim_ag, T+1))
        self.underself_cons_opt = np.zeros((n_sim*n_sim_ag, T+1))

        self.undersave = 0
# aditional table to control shock values
        self.zsim1 = np.zeros((n_sim, T+1))
# aditional table to check a budget constraint
        self.bc = np.zeros((n_sim, n_sim_ag, T+1))

        self.median_income = self.w


        self.hyp = 1.0
######################################################################################################################
#####################################################################################################################
# some simple functions

    def utility(self,x): #calculate utility
         '''
         Define  utility
         '''
         if(x<1e-6):
              print("error in utility")
         return x**(1-self.sigma)/(1-self.sigma)

    def marginal_u(self,x): #marginal utility
         '''
         Define marginal utility
         '''
         if(x<1e-6):
              print("error")
         return x**(-self.sigma)

    def inv_marginal_u(self,x): #inverse of marginal utility
         '''
         Define inverse marginal utility
         '''
         if(x<1e-6):
              print("error",x)
              return 1e-6**(-self.inv_sigma)
         return x**(-self.inv_sigma)
    def life_time_utlility(self, util):
        '''
        Define liftetime utility
        '''
        sum_util = 0.0
        for i in range(self.T):
            sum_util+=self.prob_absol[i]*self.beta**i*self.utility(util[i])
        if sum_util ==0:
            print("error, sum equal 0")
        print(sum_util)
        return sum_util

    def set_pensions(self):
        '''
        Define pensions levels
        '''
        for f in range(self.n_fe):
             for p in range(self.a_r):
                 for q in range(self.n_ag):
                     self.final_pernament[p,f,q,1] = (1-self.tau_l)*self.edu_premium[1]*self.w*m.exp(self.ind_fe_edu[f]+self.val_edu[p]+self.age_eff_coef_1*self.R +self.age_eff_coef_2*(self.R)**2+ self.age_eff_coef_3*(self.R)**3)
        for f in range(self.n_fe):
             for p in range(self.a_r):
                 for q in range(self.n_ag):
                     self.final_pernament[p,f,q,0] = (1-self.tau_l)*self.edu_premium[0]*self.w*m.exp(self.ind_fe_unedu[f]+self.val_unedu[p]+self.age_eff_coef_1*self.R +self.age_eff_coef_2*(self.R)**2+ self.age_eff_coef_3*(self.R)**3)

    def pension(self, f,p,q,edu):
        '''
        Define pensions
        '''
        if(0.38*self.median_income>self.final_pernament[p,f,q,edu]):
            return 0.9*self.final_pernament[p,f,q,edu]*self.y_pen
        elif((0.38*self.median_income<=self.final_pernament[p,f,q,edu]) and (1.59*self.median_income>self.final_pernament[p,f,q,edu])):
            return 0.9*0.38*self.median_income*self.y_pen + 0.32*(self.final_pernament[p,f,q,edu] - 0.38*self.median_income)*self.y_pen
        else:
            return 0.9*0.38*self.median_income*self.y_pen + 0.32*((1.59 - 0.38)*self.median_income)*self.y_pen + 0.15*(self.final_pernament[p,f,q,edu] - 1.59*self.median_income)*self.y_pen

##########################################################################################################################
##########################################################################################################################
#Calculate the policy function for consumption and savings
    def policy_functions(self):
          """
          Find policy functions using endogenous grid method.
          """
          #grid definition
          end_grid = np.zeros(self.a_n+1) #endogenous grid definition
          pf_a = np.zeros((self.T+1, self.a_n+1, self.a_r,self.n_fe,self.n_ag)) #local table for the asset policy function
          pf_c = np.zeros((self.T+1,self.a_n+1,self.a_r,self.n_fe, self.n_ag)) #local table for the consumption policy function


########################################POLICY FUNCTION FOR EDUCTAED#####################################################
          #iteration for last year
          for f in range(self.n_fe):
               for p in range(self.a_r):
                   for q in range(self.n_ag):
                       for i in range(self.a_n+1):
                           pf_c[self.T,i,p,f,q] = (1+self.r)*self.grid[i] + self.pens
                           pf_a[self.T,i,p,f,q] = 0.0
         #iteration for the retirees
          for j in range(self.T-1,self.R,-1):
               for f in range(self.n_fe):
                    for p in range(self.a_r):
                        for q in range(self.n_ag):
                            for i in range(self.a_n+1):

                                m_cons_sum =  self.marginal_u(pf_c[j+1,i,p,f,q])

                                cons = self.inv_marginal_u(self.prob_surv[j]*self.beta*(1+self.r)*m_cons_sum)
                                a = 1/(1+self.r)*(self.grid[i]+cons-self.pens)
                                a = np.maximum(self.grid_min,a)


                                end_grid[i] = a
                            pf_a[j,:,p,f,q] = np.maximum(np.interp(self.grid, end_grid, self.grid),self.grid_min) #interpolate on exogenous grid
                            pf_c[j,:,p,f,q] =(1+self.r)*self.grid+ self.pens - pf_a[j,:,p,f,q]

        #iteration for the workes
          for j in range(self.R,-1,-1):
               for f in range(self.n_fe):
                    for p in range(self.a_r):
                        for q in range(self.n_ag):
                            w = self.edu_premium[1]*self.w*m.exp(self.ind_fe_edu[f]+self.val_edu[p]+self.age_eff_coef_1*j +self.age_eff_coef_2*(j)**2+ self.age_eff_coef_3*(j)**3)

                            for i in range(self.a_n+1):
                                m_cons_sum = 0
                                for i_p in range(self.a_r):
                                    for i_q in range(self.n_ag):

                                        m_cons_sum = m_cons_sum+ self.P_edu[p,i_p]*self.Prob_ag[i_q]*self.marginal_u(self.val_ag[i_q]*pf_c[j+1,i,i_p,f,i_q])
                                if j == self.R:
                                    m_cons_sum = self.marginal_u(pf_c[j+1,i,p,f,q])
                                cons = self.inv_marginal_u(self.prob_surv[j]*self.beta*(1+self.r)*m_cons_sum) #compute consumption


                                a = 1/(1+self.r)*(self.grid[i]+cons-w)*self.val_ag[q] #compute endogenous grid values
                                a = np.maximum(self.grid_min,a)

                                end_grid[i] = a
                            pf_a[j,:,p,f,q] = np.maximum(np.interp(self.grid, end_grid, self.grid),self.grid_min) #interpolate on exogenous grid

                            pf_c[j,:,p,f,q] = (1+self.r)*self.grid/self.val_ag[q]+ w - pf_a[j,:,p,f,q] #find consumption policy function


          self.pf_a_edu = pf_a
          self.pf_c_edu = pf_c
########################################POLICY FUNCTION FOR UNEDUCTAED#####################################################

          pf_a = np.zeros((self.T+1, self.a_n+1, self.a_r,self.n_fe, self.n_ag)) #asset policy function
          pf_c = np.zeros((self.T+1,self.a_n+1,self.a_r,self.n_fe, self.n_ag)) #cnsumption policy function
          #iteration for last year
          for f in range(self.n_fe):
               for p in range(self.a_r):
                   for q in range(self.n_ag):
                       for i in range(self.a_n+1):
                           pf_c[self.T,i,p,f,q] = (1+self.r)*self.grid[i] + self.pens
                           pf_a[self.T,i,p,f,q] = 0.0

          #iteration for the retirees
          for j in range(self.T-1,self.R,-1):
               for f in range(self.n_fe):
                    for p in range(self.a_r):
                        for q in range(self.n_ag):
                            for i in range(self.a_n+1):

                                m_cons_sum = self.marginal_u(pf_c[j+1,i,p,f,q])
                                cons = self.inv_marginal_u(self.prob_surv[j]*self.beta*(1+self.r)*m_cons_sum)
                                a = 1/(1+self.r)*(self.grid[i]+cons-self.pens)
                                a = np.maximum(self.grid_min,a)

                                end_grid[i] = a
                            pf_a[j,:,p,f,q] = np.maximum(np.interp(self.grid, end_grid, self.grid),self.grid_min) #interpolate on exogenous grid
                            pf_c[j,:,p,f,q] = (1+self.r)*self.grid + self.pens - pf_a[j,:,p,f,q]

        #iteration for the workes
          for j in range(self.R,-1,-1):
               for f in range(self.n_fe):
                    for p in range(self.a_r):
                        for q in range(self.n_ag):
                            w = self.edu_premium[0]*self.w*m.exp(self.ind_fe_unedu[f]+self.val_unedu[p]+self.age_eff_coef_1*j +self.age_eff_coef_2*(j)**2+ self.age_eff_coef_3*(j)**3)
                            for i in range(self.a_n+1):
                                m_cons_sum = 0
                                for i_p in range(self.a_r):
                                    for i_q in range(self.n_ag):
                                        m_cons_sum = m_cons_sum + self.P_unedu[p,i_p]*self.Prob_ag[i_q]*self.marginal_u(self.val_ag[i_q]*pf_c[j+1,i,i_p,f,i_q])
                                if j == self.R:
                                    m_cons_sum = self.marginal_u(pf_c[j+1,i,p,f,q])
                                cons = self.inv_marginal_u(self.prob_surv[j]*self.beta*(1+self.r)*m_cons_sum) #compute consumption

                                a = 1/(1+self.r)*(self.grid[i]+cons-w)*self.val_ag[q] #compute endogenous grid values
                                a = np.maximum(self.grid_min,a)

                                end_grid[i] = a
                            pf_a[j,:,p,f,q] = np.maximum(np.interp(self.grid, end_grid, self.grid),self.grid_min) #interpolate on exogenous grid

                            pf_c[j,:,p,f,q] = (1+self.r)*self.grid/self.val_ag[q]+ w - pf_a[j,:,p,f,q] #find consumption policy function

          self.pf_a_unedu = pf_a
          self.pf_c_unedu = pf_c
          return self.pf_a_edu, self.pf_c_edu, self.pf_a_unedu, self.pf_c_unedu


    #Calculate the policy function for consumption and savings
    def policy_functions_hyp(self):
              """
              Find policy functions using endogenous grid method.
              """
              #grid definition
              end_grid = np.zeros(self.a_n+1) #endogenous grid definition
              pf_a_hyp = np.zeros((self.T+1, self.a_n+1, self.a_r,self.n_fe,self.n_ag)) #local table for the asset policy function
              pf_c_hyp = np.zeros((self.T+1,self.a_n+1,self.a_r,self.n_fe, self.n_ag)) #local table for the consumption policy function
              pf_a = self.pf_a_edu
              pf_c = self.pf_c_edu

    ########################################POLICY FUNCTION FOR EDUCTAED#####################################################
              #iteration for last year
              for f in range(self.n_fe):
                   for p in range(self.a_r):
                       for q in range(self.n_ag):
                           for i in range(self.a_n+1):
                               pf_c_hyp[self.T,i,p,f,q] = (1+self.r)*self.grid[i] + self.pens
                               pf_a_hyp[self.T,i,p,f,q] = 0.0
             #iteration for the retirees
              for j in range(self.T-1,self.R,-1):
                   for f in range(self.n_fe):
                        for p in range(self.a_r):
                            for q in range(self.n_ag):
                                for i in range(self.a_n+1):

                                    m_cons_sum =  self.marginal_u(pf_c[j+1,i,p,f,q])

                                    cons = self.inv_marginal_u(self.prob_surv[j]*self.beta*self.hyp*(1+self.r)*m_cons_sum)
                                    a = 1/(1+self.r)*(self.grid[i]+cons-self.pens)
                                    a = np.maximum(self.grid_min,a)


                                    end_grid[i] = a
                                pf_a_hyp[j,:,p,f,q] = np.maximum(np.interp(self.grid, end_grid, self.grid),self.grid_min) #interpolate on exogenous grid
                                pf_c_hyp[j,:,p,f,q] =(1+self.r)*self.grid+ self.pens - pf_a_hyp[j,:,p,f,q]

            #iteration for the workes
              for j in range(self.R,-1,-1):
                   for f in range(self.n_fe):
                        for p in range(self.a_r):
                            for q in range(self.n_ag):
                                w = self.edu_premium[1]*self.w*m.exp(self.ind_fe_edu[f]+self.val_edu[p]+self.age_eff_coef_1*j +self.age_eff_coef_2*(j)**2+ self.age_eff_coef_3*(j)**3)

                                for i in range(self.a_n+1):
                                    m_cons_sum = 0
                                    for i_p in range(self.a_r):
                                        for i_q in range(self.n_ag):

                                            m_cons_sum = m_cons_sum+ self.P_edu[p,i_p]*self.Prob_ag[i_q]*self.marginal_u(self.val_ag[i_q]*pf_c[j+1,i,i_p,f,i_q])
                                    if j == self.R:
                                        m_cons_sum = self.marginal_u(pf_c[j+1,i,p,f,q])
                                    cons = self.inv_marginal_u(self.prob_surv[j]*self.beta*self.hyp*(1+self.r)*m_cons_sum) #compute consumption


                                    a = 1/(1+self.r)*(self.grid[i]+cons-w)*self.val_ag[q] #compute endogenous grid values
                                    a = np.maximum(self.grid_min,a)

                                    end_grid[i] = a
                                pf_a_hyp[j,:,p,f,q] = np.maximum(np.interp(self.grid, end_grid, self.grid),self.grid_min) #interpolate on exogenous grid

                                pf_c_hyp[j,:,p,f,q] = (1+self.r)*self.grid/self.val_ag[q]+ w - pf_a_hyp[j,:,p,f,q] #find consumption policy function


              self.pf_a_edu_hyp = pf_a_hyp
              self.pf_c_edu_hyp = pf_c_hyp
    ########################################POLICY FUNCTION FOR UNEDUCTAED#####################################################

              pf_a_hyp = np.zeros((self.T+1, self.a_n+1, self.a_r,self.n_fe,self.n_ag)) #local table for the asset policy function
              pf_c_hyp = np.zeros((self.T+1,self.a_n+1,self.a_r,self.n_fe, self.n_ag)) #local table for the consumption policy function
              pf_a = self.pf_a_unedu
              pf_c = self.pf_c_unedu #cnsumption policy function
              #iteration for last year
              for f in range(self.n_fe):
                   for p in range(self.a_r):
                       for q in range(self.n_ag):
                           for i in range(self.a_n+1):
                               pf_c_hyp[self.T,i,p,f,q] = (1+self.r)*self.grid[i] + self.pens
                               pf_a_hyp[self.T,i,p,f,q] = 0.0

              #iteration for the retirees
              for j in range(self.T-1,self.R,-1):
                   for f in range(self.n_fe):
                        for p in range(self.a_r):
                            for q in range(self.n_ag):
                                for i in range(self.a_n+1):

                                    m_cons_sum = self.marginal_u(pf_c[j+1,i,p,f,q])
                                    cons = self.inv_marginal_u(self.prob_surv[j]*self.beta*self.hyp*(1+self.r)*m_cons_sum)
                                    a = 1/(1+self.r)*(self.grid[i]+cons-self.pens)
                                    a = np.maximum(self.grid_min,a)

                                    end_grid[i] = a
                                pf_a_hyp[j,:,p,f,q] = np.maximum(np.interp(self.grid, end_grid, self.grid),self.grid_min) #interpolate on exogenous grid
                                pf_c_hyp[j,:,p,f,q] = (1+self.r)*self.grid + self.pens - pf_a_hyp[j,:,p,f,q]

            #iteration for the workes
              for j in range(self.R,-1,-1):
                   for f in range(self.n_fe):
                        for p in range(self.a_r):
                            for q in range(self.n_ag):
                                w = self.edu_premium[0]*self.w*m.exp(self.ind_fe_unedu[f]+self.val_unedu[p]+self.age_eff_coef_1*j +self.age_eff_coef_2*(j)**2+ self.age_eff_coef_3*(j)**3)
                                for i in range(self.a_n+1):
                                    m_cons_sum = 0
                                    for i_p in range(self.a_r):
                                        for i_q in range(self.n_ag):
                                            m_cons_sum = m_cons_sum + self.P_unedu[p,i_p]*self.Prob_ag[i_q]*self.marginal_u(self.val_ag[i_q]*pf_c[j+1,i,i_p,f,i_q])
                                    if j == self.R:
                                        m_cons_sum = self.marginal_u(pf_c[j+1,i,p,f,q])
                                    cons = self.inv_marginal_u(self.prob_surv[j]*self.beta*self.hyp*(1+self.r)*m_cons_sum) #compute consumption

                                    a = 1/(1+self.r)*(self.grid[i]+cons-w)*self.val_ag[q] #compute endogenous grid values
                                    a = np.maximum(self.grid_min,a)

                                    end_grid[i] = a
                                pf_a_hyp[j,:,p,f,q] = np.maximum(np.interp(self.grid, end_grid, self.grid),self.grid_min) #interpolate on exogenous grid

                                pf_c_hyp[j,:,p,f,q] = (1+self.r)*self.grid/self.val_ag[q]+ w - pf_a_hyp[j,:,p,f,q] #find consumption policy function

              self.pf_a_unedu_hyp = pf_a_hyp
              self.pf_c_unedu_hyp = pf_c_hyp
              return self.pf_a_edu_hyp, self.pf_c_edu_hyp, self.pf_a_unedu_hyp, self.pf_c_unedu_hyp


############################################################################################################################################
############################################################################################################################################
    def funcion_sim(self, income):
         #print("funkcja")
         '''
         Find an optimal asset and consumption policy functions, simulate the paths
         '''
         end_grid = np.zeros(self.a_n+1) #endogenous grid definition
         pf_a = np.zeros((self.T+1, self.a_n+1)) #asset policy function
         pf_c = np.zeros((self.T+1,self.a_n+1)) #cnsumption policy function
         pf_l = np.zeros((self.T+1,self.a_n+1))
         path_c = np.zeros(self.T+1)
         path_s = np.zeros(self.T+1)
         bc = np.zeros(self.T+1)

         for i in range(self.a_n+1):
             pf_c[self.T,i] = (1.0+self.r)*self.grid[i] + income[self.T]
             pf_a[self.T,i] = 0.0


        #iteration for the retirees
         for j in range(self.T-1,self.R,-1):
             for i in range(self.a_n+1):

                 m_cons_sum =  self.marginal_u(pf_c[j+1,i])

                 cons = self.inv_marginal_u(self.prob_surv[j]*self.beta*(1.0+self.r)*m_cons_sum)
                 a = 1/(1+self.r)*(self.grid[i]+cons-income[j])
                 a = np.maximum(self.grid_min,a)


                 end_grid[i] = a
             pf_a[j,:] = np.maximum(np.interp(self.grid, end_grid, self.grid),self.grid_min) #interpolate on exogenous grid
             pf_c[j,:] =(1+self.r)*self.grid+ income[j] - pf_a[j,:]


         for j in range(self.R,-1,-1):
                for i in range(self.a_n+1):
                    m_cons_sum =  self.marginal_u(pf_c[j+1,i])
                    cons = self.inv_marginal_u(self.prob_surv[j]*self.beta*(1.0+self.r)*m_cons_sum) #compute consumption

                    a = 1/(1+self.r)*(self.grid[i]+cons-income[j]) #compute endogenous grid values
                    a = np.maximum(self.grid_min,a)

                    end_grid[i] = a
                pf_a[j,:] = np.maximum(np.interp(self.grid, end_grid, self.grid),self.grid_min) #interpolate on exogenous grid
                pf_c[j,:] =(1+self.r)*self.grid+ income[j] - pf_a[j,:] #interpolate on exogenous grid

         path_c[0] = pf_c[0,0]


         for j in range(1,self.T+1,1):

             path_s[j] = np.interp(path_s[j-1], self.grid, pf_a[j-1,:])
             path_c[j] = np.interp(path_s[j], self.grid, pf_c[j,:])
             bc[j] = (1+self.r)*path_s[j-1]+income[j-1] -path_c[j-1]- path_s[j]
         if np.sum(bc)>1e-9:
            print("error in function")
         return  path_s, path_c

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
          ## true (not divided by aggregate productivity) values of savings and consumption
          s_path_true = np.zeros((self.n_sim, self.n_sim_ag, self.T+1))
          c_path_true = np.zeros((self.n_sim, self.n_sim_ag, self.T+1))
          s_path_optimal = np.zeros((self.n_sim, self.n_sim_ag, self.T+1))
          c_path_optimal = np.zeros((self.n_sim, self.n_sim_ag, self.T+1))
          z_ag_hist =  np.zeros((self.n_sim_ag, self.T+1)) #history of aggregate shocks
          zsim1 = np.zeros((self.n_sim, self.n_sim_ag, self.T+1)) #table of shocks, used for
          prod_history = np.ones((self.n_sim_ag,self.T+1)) #vaues of aggregate shocks
          bc = np.zeros((self.n_sim, self.n_sim_ag, self.T+1)) #budget constraint
          income = np.zeros((self.n_sim, self.n_sim_ag, self.T+1))
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
                        if ( rand <=np.sum(self.Prob_ag[0:q+1]) ):
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

                #initialize consumption
                c_path[s,aq,0] = self.pf_c_edu[0,0,zsim,f,int(z_ag_hist[aq,0])]
                c_path_true[s,aq,0] = c_path[s,aq,0]* prod_history[aq,0]
                #print(c_path_true[s,aq,0])
                for j in range(1,self.T+1,1):
                     rand2 = np.random.uniform(low=0.0, high=1.0)
                     for p in range(self.a_r):

                          if ( rand2 <=np.sum(self.P_edu[zsim, 0:p+1]) ):
                               zsim_old = zsim
                               zsim =p
                               break
                     zsim1[s,aq,j] = zsim
                     s_path[s,aq,j] = np.interp(s_path[s,aq,j-1], self.grid, self.pf_a_edu_hyp[j-1,:,zsim_old,f,int(z_ag_hist[aq,j-1])])
                     c_path[s,aq,j] = np.interp(s_path[s,aq,j], self.grid, self.pf_c_edu_hyp[j,:,zsim,f,int(z_ag_hist[aq,j])])
                     s_path_true[s,aq,j] = s_path[s,aq,j]*np.prod(prod_history[aq,0:j])
                     c_path_true[s,aq,j] = c_path[s,aq,j]*np.prod(prod_history[aq,0:j+1])
                     #check budget constraint
                     w = self.edu_premium[1]*np.prod(prod_history[aq,0:j])*self.w*m.exp(self.ind_fe_edu[f]+self.val_edu[zsim_old]+self.age_eff_coef_1*(j-1) +self.age_eff_coef_2*(j-1)**2+ self.age_eff_coef_3*(j-1)**3)
                     if j<=self.R+1:
                         bc[s,aq,j-1] = s_path_true[s,aq,j]+ c_path_true[s,aq,j-1] - (1+self.r)*s_path_true[s,aq,j-1] - w
                         income[s,aq,j-1] = w
                     else:
                         bc[s,aq,j-1] = s_path_true[s,aq,j]+ c_path_true[s,aq,j-1] - (1+self.r)*s_path_true[s,aq,j-1] - \
                         self.pens*np.prod(prod_history[aq,0:j+1])
                         income[s,aq,j-1] = self.pens*np.prod(prod_history[aq,0:j+1])
                income[s,aq,self.T] = self.pens*np.prod(prod_history[aq,0:j+1])
                [s_path_optimal[s,aq,:],c_path_optimal[s,aq,:]]  = self.funcion_sim(income[s,aq,:])
                if np.sum(s_path_optimal[s,aq,0:self.R+1])>np.sum(s_path[s,aq,0:self.R+1]+1e-9):
                    self.undersave+=1
                    self.underself_cons[self.undersave-1, :] = c_path_true[s,aq,:]
                    self.underself_cons_opt[self.undersave-1, :] = c_path_optimal[s,aq,:]
                    print( self.undersave)
            #write local variables to globa;
              self.sav_distr[0:self.edu_numb, aq, :] = s_path_true[0:self.edu_numb, aq, :]
              self.cons_distr[0:self.edu_numb, aq, :] = c_path_true[0:self.edu_numb, aq, :]
              self.sav_distr_opt[0:self.edu_numb, aq, :] = s_path_optimal[0:self.edu_numb, aq, :]
              self.cons_distr_opt[0:self.edu_numb, aq, :] =c_path_optimal[0:self.edu_numb, aq, :]
              self.bc[0:self.edu_numb, aq, :] = bc[0:self.edu_numb, aq, :]
              #do the same for uneducated
              for s in range(self.edu_numb, self.n_sim,1):
                 zsim = initial_prod
                 rand1 = np.random.uniform(low=0.0, high=1.0)
                 f=1
                 if ( rand1 <=self.P_ind_fe_unedu[0] ):
                      f = 0
                 c_path[s,aq,0] = self.pf_c_unedu[0,0,zsim,f,int(z_ag_hist[aq,0])]
                 c_path_true[s,aq,0] = c_path[s,aq,0]*prod_history[aq,0]
                 for j in range(1,self.T+1,1):

                       rand = np.random.uniform(low=0.0, high=1.0)
                       for p in range(self.a_r):
                            temp = np.sum(self.P_unedu[zsim, 0:p+1])
                            if ( rand <=temp ):
                                 zsim_old = zsim
                                 zsim =p
                                 break

                       zsim1[s,aq,j] = zsim
                       s_path[s,aq,j] = np.interp(s_path[s,aq,j-1], self.grid, self.pf_a_unedu_hyp[j-1,:,zsim_old,f,int(z_ag_hist[aq,j-1])])
                       c_path[s,aq,j] = np.interp(s_path[s,aq,j], self.grid, self.pf_c_unedu_hyp[j,:,zsim,f,int(z_ag_hist[aq,j])])
                       s_path_true[s,aq,j] = s_path[s,aq,j]*np.prod(prod_history[aq,0:j])
                       c_path_true[s,aq,j] = c_path[s,aq,j]*np.prod(prod_history[aq,0:j+1])
                       w = self.edu_premium[0]*np.prod(prod_history[aq,0:j])*self.w*m.exp(self.ind_fe_unedu[f]+self.val_unedu[zsim_old]+self.age_eff_coef_1*(j-1) +self.age_eff_coef_2*(j-1)**2+ self.age_eff_coef_3*(j-1)**3)
                       if j<=self.R+1:
                           bc[s,aq,j-1] = s_path_true[s,aq,j]+ c_path_true[s,aq,j-1] - (1+self.r)*s_path_true[s,aq,j-1] - w
                           income[s,aq,j-1] = w
                       else:
                           bc[s,aq,j-1] = s_path_true[s,aq,j]+ c_path_true[s,aq,j-1] - (1+self.r)*s_path_true[s,aq,j-1] - \
                           self.pens*np.prod(prod_history[aq,0:j+1])
                           income[s,aq,j-1] = self.pens*np.prod(prod_history[aq,0:j+1])
                 income[s,aq,self.T] = self.pens*np.prod(prod_history[aq,0:j+1])
                 [s_path_optimal[s,aq,:],c_path_optimal[s,aq,:]]  = self.funcion_sim(income[s,aq,:])
                 if np.sum(s_path_optimal[s,aq,0:self.R+1])>np.sum(s_path[s,aq,0:self.R+1]+1e-9):
                     self.undersave+=1
                     self.underself_cons[self.undersave-1, :] = c_path_true[s,aq,:]
                     self.underself_cons_opt[self.undersave-1, :] = c_path_optimal[s,aq,:]
                     print( self.undersave)
              self.sav_distr[self.edu_numb: self.n_sim,aq,:] = s_path_true[self.edu_numb: self.n_sim,aq,:]
              self.cons_distr[self.edu_numb: self.n_sim,aq,:] =c_path_true[self.edu_numb: self.n_sim,aq,:]
              self.sav_distr_opt[self.edu_numb: self.n_sim,aq,:] = s_path_optimal[self.edu_numb: self.n_sim,aq,:]
              self.cons_distr_opt[self.edu_numb: self.n_sim,aq,:] =c_path_optimal[self.edu_numb: self.n_sim,aq,:]
              self.bc[self.edu_numb: self.n_sim, aq, :] = bc[self.edu_numb: self.n_sim, aq, :]



          return self.sav_distr, self.cons_distr, self.zsim1
    def calculate_cons_equivalent(self):
        cons_lambda = np.zeros(self.undersave)
        for i in range(self.undersave):
            #print(self.underself_cons_opt[i, :])
            cons_lambda[i] = (self.life_time_utlility(self.underself_cons_opt[i, :])/ self.life_time_utlility(self.underself_cons[i, :]))**(1/(1-self.sigma))-1.0

        n, bins, patches = plt.hist(x=cons_lambda, bins='auto', color='#0504aa',
                      alpha=0.7, rwidth=0.85)
        plt.grid(axis='y', alpha=0.75)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title('Consumption equivalen differences')
        plt.show()
        maxfreq = n.max()


        plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

        print("end")
        return cons_lambda
##############################################################################################################
    def plot_life_cycle(self):
         """
         Some plots. Savongs and consumption distribution, policy function for the period 44 (just before retirment)
         """

         s_mean = np.zeros(self.T+1)
         s_max = np.zeros(self.T+1)
         s_min = np.zeros(self.T+1)

         c_mean = np.zeros(self.T+1)
         c_max = np.zeros(self.T+1)
         c_min = np.zeros(self.T+1)

         c_opt_mean = np.zeros(self.T+1)
         c_opt_max = np.zeros(self.T+1)
         c_opt_min = np.zeros(self.T+1)


         c_diff_mean = np.zeros(self.T+1)
         c_diff_max = np.zeros(self.T+1)
         c_diff_min = np.zeros(self.T+1)

         z_mean = np.zeros(self.T+1)
         z_max = np.zeros(self.T+1)
         z_min = np.zeros(self.T+1)

         bc_mean = np.zeros(self.T+1)
         bc_max = np.zeros(self.T+1)
         bc_min = np.zeros(self.T+1)


         for j in range(1,self.T+1,1):
             s_mean[j] = np.mean(self.sav_distr[:,:,j])
             s_max[j] = np.percentile(self.sav_distr[:,:,j],90)
             s_min[j] = np.percentile(self.sav_distr[:,:,j],10)
         plt.plot(range(self.T+1), s_mean, label = "mean savings")
         plt.plot(range(self.T+1), s_max, label = "90th percentile of savings")
         plt.plot(range(self.T+1), s_min, label = "90th percentile of savings")
         plt.ylabel('savings profile')
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
         plt.ylabel('savings')
         plt.legend(loc='best')
         plt.show()

         for j in range(1,self.T+1,1):
             c_opt_mean[j] = np.mean(self.cons_distr_opt[:,:,j])
             c_opt_max[j] = np.percentile(self.cons_distr_opt[:,:,j],90)
             c_opt_min[j] = np.percentile(self.cons_distr_opt[:,:,j],10)
         plt.plot(range(self.T+1), c_opt_mean, label = "mean optimal consumption")
         plt.plot(range(self.T+1), c_opt_max, label = "90th percentile of optimal consumption")
         plt.plot(range(self.T+1), c_opt_min, label = "10th percentile of optimal consumption")
         plt.ylabel('savings')
         plt.legend(loc='best')
         plt.show()

         c_diff = self.cons_distr_opt - self.cons_distr

         for j in range(1,self.T+1,1):
             c_diff_mean[j] = np.mean(c_diff[:,:,j])
             c_diff_max[j] = np.percentile(c_diff[:,:,j],99)
             c_diff_min[j] = np.percentile(c_diff[:,:,j],1)
         plt.plot(range(self.T+1), c_diff_mean, label = "mean diff consumption")
         plt.plot(range(self.T+1), c_diff_max, label = "90th percentile of diff consumption")
         plt.plot(range(self.T+1), c_diff_min, label = "10th percentile of diff consumption")
         plt.ylabel('savings')
         plt.legend(loc='best')
         plt.show()


         for j in range(1,self.T+1,1):
             bc_mean[j] = np.mean(self.bc[:,:,j])
             bc_max[j] = np.percentile(self.bc[:,:,j],99)
             bc_min[j] = np.percentile(self.bc[:,:,j],1)
         plt.plot(range(self.T+1), bc_mean, label = "bc consumption")
         plt.plot(range(self.T+1), bc_max, label = "bc max")
         plt.plot(range(self.T+1), bc_min, label = "bc min")
         plt.ylabel('bc')
         plt.legend(loc='best')
         plt.show()



############################################################################################################################
    def execute_life_cycle(self):
          self.policy_functions()
          self.policy_functions_hyp()
          self.simulate_life_cycle()
          self.plot_life_cycle() #some basic plots
          print(self.undersave)
          self.calculate_cons_equivalent()
