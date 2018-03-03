"""
The context:
Evaluating the GLobal Mean Sea Level trend of Le CONQUET tide-gauge
using RANSAC method
#=======================================================================
1. Evaluate an initial solution using only a set of obs which have the
same number of parameters ans randomly ectracted
#=======================================================================
2. Continue 1 until some threshold is attained regarding the number of
observations laying within a ceratin margin from the 1-step calculated model
#=======================================================================
2. If there is a model that satisfies T threshold condition,
perform a mean square regression and stop. Else, choose among all datasets Si
the one with the biggest size and perform a mean square analysis
#=======================================================================
"""
#=======================================================================
""" IMPORTS"""
#=======================================================================
import os
import numpy as np
import matplotlib.pylab as plt

#===========================Loading Data=================================
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
#==============================Step1=========================================
"""
Last squares method: in order to estimate an affine relation between the variables
( data_mm[]=sea level and date[]= dates),in other words look for a line that fits
as well as possible to this cloud of points.
INPUT:
     data_mm[]=sea level
     date[]= dates

The model:f(t) = a(t-t0)+ b

"""
#============================================================================
def lastsquare_1(date,data_mm):
    date_ = np.asarray(date)
    #stochastic model
    l = np.asarray(data_mm)                             # observation
    n = np.shape(l)[0]                                  # observation dimension
    Kl = np.eye(n)                                      # matrix var/covar
    sigma_0 = 1
    Ql = (1/(sigma_0)**2)*Kl
    P = np.linalg.inv(Ql)                               # weight matrix
    # X parameter =[a,b]
    #initial parameter
    X = [0.2948, 5103]
    # closing gap matrix : B
    B =0*l
    for i in range (n):
        B[i] = l[i]- X[0]*(date_[i]-date_[0])-X[1]

    #Jacobian matrix: A
    A = np.zeros((n,len(X)),dtype='double')
    for i in range(n):
        A[i,0] = date_[i]-date_[0]
        A[i,1] = 1

    dxx = calculation_matrix(A=A,P=P,B=B,X0=X,Obs=l,Ql=Ql)[0]
    K = 0
    while any(abs(dxx) > 10e-7) and (K <= 100):
        B = np.zeros((n),dtype='double')
        for i in range(n):
            B[i] = l[i] - X[0] * (date_[i] - date_[0]) - X[1]

        A = np.zeros((n, len(X)))
        for i in range(n):
            A[i, 0] = date_[i] - date_[0]
            A[i, 1] = 1


        dxx,X_chap,Qxx,V_chap,ll,Qvv,Qll = calculation_matrix(A=A,P=P,B=B,X0=X,Obs=l,Ql=Ql)
        X = X_chap
        K = K + 1

    print("X=",X)

    return X

def calculation_matrix(A,P,B,X0,Obs,Ql):
    #dx_chap_matrix
    S1=np.linalg.inv(np.dot(A.T,np.dot(P,A)))
    dx_chap = np.dot(S1,np.dot(A.T,np.dot(P,B)))
    #X_chap
    x_chap = X0 + dx_chap
    #Qxx
    Qxx = np.linalg.inv(np.dot(A.T,np.dot(P,A)))
    #V_chap
    V_chap = B - np.dot(A, dx_chap)
    #L_chap
    l_chapeau = Obs - V_chap
    #Qv_chap
    Qvv = Ql - np.dot(A, np.dot(Qxx, A.T))
    #Ql_chap
    Qll = Ql - Qvv

    return (dx_chap,x_chap,Qxx,V_chap,l_chapeau,Qvv,Qll)

# ======================================================================
# Displaying the initial obs, the compensated heights, the residuals,
# the Residuals' test
# ======================================================================
def display(date,h_mm,X):
    #La droite de 'estimation par MC
    #f_lastsquare_1 = X[0]*(np.asarray(date)-date[0])+X[1]
    #X = [17.68, 3960]

    f_lastsquare_1 = [(X[0]*(i-date[0])+X[1]) for i in np.asarray(date)]

    fig1 = plt.figure()
    axes = fig1.add_subplot(111)
    axes.set_ylabel('Sea level data (mm)')
    axes.set_xlabel('Date')
    axes.set_title('Le CONQUET-Monthly means ')
    #plt.plot(np.array(date), np.asarray(h_mm), label='Initial Observations', color='blue',linestyle='dotted')
    axes.scatter(np.asarray(date),np.asarray(h_mm) , label='Initial dataset', s=3, color='blue')
    plt.plot(np.asarray(date), f_lastsquare_1, label='La droite par MC', color='red', linestyle='solid')


    plt.legend(loc="best")
    axes.grid(True)

    fig2 = plt.figure()
    axes = fig2.add_subplot(111)
    axes.set_ylabel('Sea level data (mm)')
    axes.set_xlabel('Date')
    axes.set_title('Le CONQUET-Monthly means ')

    #plt.plot(np.asarray(date), f_lastsquare_1, label='La droite par MC', color='red',linestyle='solid' )


    plt.legend(loc="best")
    axes.grid(True)

    plt.show()
    return



if __name__ == '__main__':
    date,h_mm = dataset()
    print(date)
    print(h_mm)
    X = lastsquare_1(date=date,data_mm=h_mm)
    display(date= date,h_mm= h_mm,X= X)


