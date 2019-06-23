# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 18:43:08 2019

@author: janro
"""

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#FUNCTIONS

######################################################################################################################
#Tauchen method from QuantEcon, this section is from QUANtECON toolbocks

"""
Filename: tauchen.py

Authors: Thomas Sargent, John Stachurski

Discretizes Gaussian linear AR(1) processes via Tauchen's method

"""

from scipy.stats import norm
import math as m
import numpy as np


def approx_markov(rho, sigma_u, m=3, n=7):
    """
    Computes the Markov matrix associated with a discretized version of
    the linear Gaussian AR(1) process

        y_{t+1} = rho * y_t + u_{t+1}

    according to Tauchen's method.  Here {u_t} is an iid Gaussian
    process with zero mean.

    Parameters
    ----------
    rho : scalar(float)
        The autocorrelation coefficient
    sigma_u : scalar(float)
        The standard deviation of the random process
    m : scalar(int), optional(default=3)
        The number of standard deviations to approximate out to
    n : scalar(int), optional(default=7)
        The number of states to use in the approximation

    Returns
    -------

    x : array_like(float, ndim=1)
        The state space of the discretized process
    P : array_like(float, ndim=2)
        The Markov transition matrix where P[i, j] is the probability
        of transitioning from x[i] to x[j]

    """
    F = norm(loc=0, scale=sigma_u).cdf

    # standard deviation of y_t
    std_y = np.sqrt(sigma_u**2 / (1-rho**2))

    # top of discrete state space
    x_max = m * std_y

    # bottom of discrete state space
    x_min = - x_max

    # discretized state space
    x = np.linspace(x_min, x_max, n)

    step = (x_max - x_min) / (n - 1)
    half_step = 0.5 * step
    P = np.empty((n, n))

    for i in range(n):
        P[i, 0] = F(x[0]-rho * x[i] + half_step)
        P[i, n-1] = 1 - F(x[n-1] - rho * x[i] - half_step)
        for j in range(1, n-1):
            z = x[j] - rho * x[i]
            P[i, j] = F(z + half_step) - F(z - half_step)

    return x, P
######################################################################################################################
######################################################################################################################
#Discretization of the normal distribution from Kinderman's toolbocks
def normal_discrete_1(mu, sigma,n):


       ###### OTHER VARIABLES ####################################################

       x = np.zeros(n)
       prob = np.zeros(n)
       maxit = 200
       pi = m.pi
       z = 0.0
       mu_c = mu
       sigma_c = sigma

       ###### ROUTINE CODE #######################################################

       # initialize parameters

       # calculate 1/pi^0.25
       pim4 = 1.0/pi**0.25
       # get number of points
       m1 = (n+1)/2



       # start iteration
       for i in range(m1):

           # set reasonable starting values
           if(i == 0):
               z = m.sqrt(float(2*n+1))-1.85575*(float(2*n+1)**(-1/6))
           elif(i == 1):
               z = z - 1.14*(float(n)**0.426)/z
           elif(i == 2):
               z = 1.86*z+0.86*x[0]
           elif(i == 3):
               z = 1.91*z+0.91*x[1];
           else:
               temp = i-2
               z = 2.0*z+x[temp];

           #! root finding iterations
           its = 0
           while its < maxit:
               its = its+1
               p1 = pim4
               p2 = 0.0
               for j in range(n):
                   p3 = p2
                   p2 = p1
                   p1 = z*m.sqrt(2.0/float(j+1))*p2-m.sqrt(float(j)/float(j+1))*p3
               pp = m.sqrt(2.0*float(n))*p2
               z1 = z
               z  = z1-p1/pp
               if abs(z-z1) < 1e-14:
                   break
           if its >= maxit:
               print('normal_discrete','Could not discretize normal distribution')
           # endif
           temp = n-i-1
           x[temp] = z
           x[i] = -z
           prob[i] = 2.0/pp**2
           prob[temp] = prob[i]

       prob = prob/m.sqrt(pi)
       x = x*m.sqrt(2.0)*sigma_c + mu_c
       return x, prob
####################