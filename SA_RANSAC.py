"""
RANSAC: Random Sample Consensus
Strategy :
Find a model that accords with the maximum number of samples
Assumptions:
1- Majority of good samples agree with the underlying model
2- Bad samples does not consistently agree with a single model

RANSAC ALGORITHM
for number of iteration:
    select randomly a subset of data(S)
    find the model for the selected data = fit a model foe data
     test all data against the model and determine the inliers based on threshold
    if the new model is better (based on number of inliers)
    best model = new model
repeat

How many samples should we choose ?
N= number of samples
e= probability that a point is an outlier
s number of point in a sampleschmidt.guillaume@wanadoo.fr
p= desired probability that we get a good sample

N=log(1-p)/log(1-(1-e)s)

Step 1: Random sampling
Step 2: Model building: and fit with Least square Method
Step 3: Thresholding : measrue how well the points fitting the model
Step 4: Inlier counting
 repeat untill we have the maximum number of inliers

"""

""" ALL IMPORTS"""
#=======================================================================
import numpy as np
import matplotlib.pylab as plt
from netCDF4 import Dataset
from scipy.stats import norm, chi2
import random
from scipy.optimize import fsolve
#=======================================================================
"""Loading Data"""
def dataset():
    date = []
    data_mm=[]

    #line =0
    data_file = "mLCONQ_dec10.slv"
    with open(data_file) as f:
        for line in f :
            date.append(float(line[0:9]))
            data_mm.append(int(line[10:-1]))

    return (date,data_mm)

""" EXtract Jason-2 Satellite Altimetry data"""
date0,h_0= dataset()

print("data0=\n",date0)
print("h_0=\n",h_0)
#=======================================================================
""" Initial solution: Obtained from MeanSquare Calculation"""
parametre_init=[4.1497057,19.32060092,-6.10410342,6.25634226,3.01462022
 ,-3.67541036,1.97400679,-3.69483161,-1.55771433,0.85590144,-1.6703370]
# Number of parameters
n_pc=len(parametre_init)
print("n_pc=",n_pc)
#=======================================================================
"""Mymodel: FOR SOLVING THE NON-LINEAR PROBLEM at step 1"""
#=======================================================================
def Mymodel(pc):
    a= time0[0]
    f=pc[0]*(time0-a)+pc[1]+pc[2]*np.sin(pc[3]*(time0-a)+pc[4])\
		+pc[5]*np.sin(pc[6]*(time0-a)+pc[7])\
		+pc[8]*np.sin(pc[9]*(time0-a)+pc[10])-height0
    return f
#=======================================================================
"""Method Solve: Allows to solve the nonlinear problem using time series 
as entries """
#=======================================================================
def Solve(time, obs):

    pc=fsolve(Mymodel,parametre_init)
    print("fsolve=",pc)
    return pc
#=======================================================================
"""Mean Square! Method for solving the the redundant system of equations 
"""
#=======================================================================
def Meansquare(time,height,parametre_init):
    SIGMA0 = 1
    sigma_obs=SIGMA0
    i=0
    t=np.asarray(time)
    pc=np.asarray(parametre_init)
    A=np.zeros((len(t),len(parametre_init)))
    Kl=(sigma_obs**2)*np.diag(np.ones(len(t)))
    QL=(1/SIGMA0**2)*Kl
    P=np.linalg.inv(QL)
    i=0
    Residual_before=np.zeros(len(t))
    Residual_after=np.ones(len(t))
    epsilon=10e-7
    a=t[0]
    while (np.any(np.abs(Residual_after-Residual_before)>epsilon) and i<300):
        Residual_before=Residual_after
    #   Definition of the Jacobian Matrix related to the problem
        A0=t-a
        A1=np.ones(len(t))
        A2=np.sin(pc[3]*(t-a)+pc[4])
        A3=pc[2]*(t-a)*np.cos(pc[3]*(t-a)+pc[4])
        A4=pc[2]*np.cos(pc[3]*(t-a)+pc[4])
        A5=np.sin(pc[6]*(t-a)+pc[7])
        A6=pc[5]*(t-a)*np.cos(pc[6]*(t-a)+pc[7])
        A7=pc[5]*np.cos(pc[6]*(t-a)+pc[7])
        A8=np.sin(pc[9]*(t-a)+pc[10])
        A9=pc[8]*(t-a)*np.cos(pc[9]*(t-a)+pc[10])
        A10=pc[8]*np.cos(pc[9]*(t-a)+pc[10])
        A[:,0]=A0
        A[:,1]=A1
        A[:,2]=A2
        A[:,3]=A3
        A[:,4]=A4
        A[:,5]=A5
        A[:,6]=A6
        A[:,7]=A7
        A[:,8]=A8
        A[:,9]=A9
        A[:,10]=A10
    # Calculating the closing biasis w
        f_0=pc[0]*(t-t[0])+pc[1]+pc[2]*np.sin(pc[3]*(t-t[0])+pc[4])\
        +pc[5]*np.sin(pc[6]*(t-t[0])+pc[7])+pc[8]*np.sin(pc[9]*(t-t[0])+pc[10])
        W= np.asarray(height)-f_0
    # Calculating  Dx_chp, Qx_chp
        AtPA_1 = np.linalg.inv(np.asarray(np.dot(np.dot(A.T, P), A), dtype='double'))
        N = np.dot(np.dot(AtPA_1, A.T), P)
        dx_chap = np.dot(N, W)
        Qx_chap=AtPA_1
        # Residual Values
        Residu=W-A.dot(dx_chap)
        Residual_after=Residu
        pc=pc+dx_chap
        Q_l_chapeau=np.dot(A,np.dot(Qx_chap,A.T))
    # ======================================================================
    #   Evaluating  Q_residu=Q_l-Q_l_chapeau
    # ======================================================================
        Id=np.diag(np.ones(len(height)))
        Mat=Id-np.dot(A,N)
        Q_residu= QL-Q_l_chapeau
        K_residu=(SIGMA0**2)*Q_residu
        i=i+1
    h_cp=height-Residu
    Sigma_chp_square=Residu.T.dot(P.dot(Residu))/(len(t)-len(pc))
    print(np.sqrt(Sigma_chp_square))
    Res_norm=Residu/np.sqrt(Sigma_chp_square*Q_residu.diagonal())
    return h_cp,pc,Residu, Res_norm

# ======================================================================
"""ICDF: for Inverse Cumulative Distribution function: Used to evaluate 
the rate of acceptance of observations based on a probability of inliers 
among the observations: Ref. Zuliani RANSAC for Dummies 2009
: It assumes that the offset between observations and the model is a Gaussian 
Noise of parameters mu, std.
"""
# ======================================================================
def ICDF_threshold(df, pb_inliers):
    icdf=np.sqrt(chi2.ppf(pb_inliers, df))
    print("icdf=",icdf)
    return icdf
#=======================================================================
Jeux=[]
flag=True
K=0
#=======================================================================
"""Global variables to be used by the non linear solving function"""
#=======================================================================
global time
global height
global time0
global height0
#=======================================================================
"""The RANSAC METHOD"""
#=======================================================================
while (flag==True and K<28):
    date,h_=dataset()

    nc=len(date)
    # Get random observations among the suggested obs
    indices = random.sample(range(len(date)), n_pc)
    print("indices=",indices)
    time0=[date[i] for i in sorted(indices)]
    height0=[h_[i] for i in sorted(indices)]

    time0=np.asarray(time0)
    height0=np.asarray(height0)
    pc = Solve(Mymodel,parametre_init)
    # Search for data verifying the model to a certain threshold t
    #===================================================================
    for i in range(n_pc):
        date.remove(time0[i])
        h_.remove(height0[i])
    #===================================================================
    date=np.asarray(date)
    h_=np.asarray(h_)

    Model= pc[0]*(date-date[0])+pc[1]+pc[2]*np.sin(pc[3]*(date-date[0])+pc[4])\
          +pc[5]*np.sin(pc[6]*(date-date[0])+pc[7])\
          +pc[8]*np.sin(pc[9]*(date-date[0])+pc[10])
    print("step",K)
    D_pts_model=np.abs(h_-Model)/np.abs(Model) # distance des points par rapport au modèle
    print(D_pts_model)
    print("np.max(D_pts_model)=",np.max(D_pts_model))
    print(np.min(D_pts_model))
    # First Threshold
    # Suppose we have a probability of inliers
    P_inliers=0.85
    #t=ICDF_threshold(1,P_inliers)

    t=0.15
    #print(t)
    D_pts_model=D_pts_model.tolist()
    S=[D_pts_model.index(i) for i in D_pts_model if (i<t)]
    print(len(S))
    time=[date[i] for i in S]
    height=[h_[i] for i in S]
    # Second Threshold
    if (len(S)>P_inliers*nc):
        H_cp, pc,Residual,N_residual=Meansquare(time, height,parametre_init)
        flag=False
    else:
        Jeux.append((time, height))
    K=K+1
    date=date.tolist()
    h_=h_.tolist()
#=======================================================================
if (Jeux and flag==True):
	print(" We didn't reach a threshold of 85% \n")
	idx=0
	Max=len(Jeux[0][0])
	for j in Jeux[1:]:
		if len(j[0])>Max:
			Max=len(j[0])
			idx=Jeux.index(j)
	time=[j for j in Jeux[idx][0]]
	height=[j for j in Jeux[idx][1]]
	H_cp, pc,Residual, N_residual=Meansquare(time,height,parametre_init)

"""Display all estimated parameters """
print("Parameters")
print(pc)
# ======================================================================
# Displaying: Initial dataset, Compensated dataset, the Model Function,
# Residuals and THE normal pdf: prob density function, trend of the residuals
# ======================================================================
# Search for Indexes of data that were included in model calculation:
# ===> Distinguish the intervening data
Inc_indices=[date0.index(time[i]) for i in range(len(time))]
#print(Inc_indices)
# Retrieve the corresponding initial heights that have been conpensated
# in order to display the consensus data in green colors
Inc_heights=[h_0[i] for i in Inc_indices]
# ======================================================================
fig1=plt.figure()
axes=fig1.add_subplot(111)
axes.set_ylabel('Mean sea level data (mm)')
axes.set_xlabel('time Jason-2')
axes.set_title('Data available from NOOA ')
axes.scatter(np.asarray(date0),np.asarray(h_0),label='Initial Observations', s=30, color='blue')
axes.scatter(time,Inc_heights,label='Consenus data', s=30, color='green')
t=np.linspace(date0[0],date0[-1],200)
axes.scatter(time,H_cp,label='Compensated Measures',s=20,color='red')
f=[pc[0]*(i-t[0])+pc[1]+pc[2]*np.sin(pc[3]*(i-t[0])+pc[4])\
   +pc[5]*np.sin(pc[6]*(i-t[0])+pc[7])\
   +pc[8]*np.sin(pc[9]*(i-t[0])+pc[10]) for i in time]
axes.plot(time,f,label='Model Function',color='black')
axes.legend(loc=2)
axes.grid(True)
#=======================================================================
fig2=plt.figure()
axes=fig2.add_subplot(111)
axes.set_ylabel('Mean sea level Residuals')
axes.set_xlabel('time Jason2')
axes.set_title('Estimated Mean Square Residuals ')
axes.scatter(time,N_residual,label='Residuals', s=30, color='green')
#=======================================================================
"""Check if normalized residuals follow normal Gaussian distribtuion
of: mean=0 and Standard deviation=1"""
#=======================================================================
fig3=plt.figure()
axes=fig3.add_subplot(111)
mu,std=norm.fit(N_residual)
# Plotting data histogram
axes.hist(N_residual,bins=25,normed=True,alpha=0.6,color='g')
# Plotting the Normal distribution
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
plt.title(title)
#=======================================================================
plt.show()



