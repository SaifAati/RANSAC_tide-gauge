import os
import numpy as np
import matplotlib.pylab as plt
from scipy.stats import norm
import statistics as stat

#===========================Loading Data========================================
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
#==============================Step1=============================================
"""
Last squares method: in order to estimate an affine relation between the variables
( data_mm[]=sea level and date[]= dates),in other words look for a line that fits
as well as possible to this cloud of points.
INPUT:
     data_mm[]=sea level
     date[]= dates

The model:f(t) = a(t-t0)+ b

"""
#================================================================================
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

    dxx=np.ones(len(l))
    K = 0
    while any(abs(dxx) > 10e-7) and (K <= 100):
        # closing gap matrix : B
        B = np.zeros((n),dtype='double')
        for i in range(n):
            B[i] = l[i] - X[0] * (date_[i] - date_[0]) - X[1]
        # Jacobian matrix: A
        A = np.zeros((n, len(X)),dtype='double')
        for i in range(n):
            A[i, 0] = date_[i] - date_[0]
            A[i, 1] = 1


        dxx,X_chap,Qxx,V_chap,ll,Qvv,Qll = calculation_matrix(A=A,P=P,B=B,X0=X,Obs=l,Ql=Ql)
        X = X_chap
        K = K + 1

    print("X=",X)

    return X


def Meansquare(date,h_,X_droite):
    t = np.asarray(date)
    # stochastic model
    l = np.asarray(h_)  # observation
    n = np.shape(l)[0]  # observation dimension
    Kl = np.eye(n)      # matrix var/covar
    sigma_0 = 1
    Ql = (1 / (sigma_0) ** 2) * Kl
    P = np.linalg.inv(Ql)  # weight matrix
    # X parameter =[a,b]
    # initial parameter
    parametre_init = [X_droite[0], X_droite[1], -6.03148247, 6.25907236, 2.98826516,
                      -3.31037008, 2.00716027, -3.89813205,-3.89813205,-3.89813205,-3.89813205]

    pc = np.asarray(parametre_init)
    dxx = np.ones(len(h_))

    k = 0
    epsilon = 10e-7
    B = np.zeros((n), dtype='double')
    f_0 = np.zeros((n), dtype='double')
    A = np.zeros((n, len(pc)), dtype='double')

    # a=stat.mean(date) #période caractéristique
    a = t[0]
    while (np.any(np.abs(dxx) > epsilon) and k <= 100):
        print("MC1=", k)
        # closing gap matrix : B   # Evaluate closing Biasis
        # f_0=pc[0]*(t-t[0])+pc[1]+pc[2]*np.sin(pc[3]*(t-t[0])+pc[4])+pc[5]*np.sin(pc[6]*(t-t[0])+pc[7])-1.55771433*np.sin(0.85590144*(t-t[0])-1.67033704)-pc_res[0] * np.sin(pc_res[1] * (t[i] - date[0]) +pc_res[2])
        for i in range(n):
            f_0[i] = pc[0] * (t[i] - a) + pc[1] + \
                     pc[2] * np.sin(pc[3] * (t[i] - a) + pc[4]) + \
                     pc[5] * np.sin(pc[6] * (t[i] - a) + pc[7]) + \
                     pc[8] * np.sin(pc[9] * (t[i] - a) + pc[10])
            B[i] = np.asarray(h_)[i] - f_0[i]

        # Jacobian matrix: A
        for i in range(n):
            A[i, 0] = t[i] - a
            A[i, 1] = 1
            A[i, 2] = np.sin(pc[3] * (t[i] - a) + pc[4])
            A[i, 3] = pc[2] * (t[i] - a) * np.cos(pc[3] * (t[i] - a) + pc[4])
            A[i, 4] = pc[2] * np.cos(pc[3] * (t[i] - a) + pc[4])
            A[i, 5] = np.sin(pc[6] * (t[i] - a) + pc[7])
            A[i, 6] = pc[5] * (t[i] - a) * np.cos(pc[6] * (t[i] - a) + pc[7])
            A[i, 7] = pc[5] * np.cos(pc[6] * (t[i] - a) + pc[7])
            A[i, 8] = np.sin(pc[9] * (t[i] - a) + pc[10])
            A[i, 9] = pc[8] * (t[i] - a) * np.cos(pc[9] * (t[i] - a) + pc[10])
            A[i, 10] = pc[8] * np.cos(pc[9] * (t[i] - a) + pc[10])



        dxx, X_chap, Qxx, V_chap, ll, Qvv, Qll = calculation_matrix(A=A, P=P, B=B, X0=pc, Obs=l, Ql=Ql)
        pc = X_chap
        k+=1
    print("pc=",pc)
    Res_norm,Sigma_chp_square = normalized_residuals(Residu=V_chap, P=P, obs=l, parameters=pc, Qvv=Qvv)


    #print("norm.mean(Res_norm)=",norm.mean(date))
    return pc,Res_norm,V_chap,Qll,np.sqrt(Sigma_chp_square),Qvv

def Meansquare_Res(date,Res_norm,mat_QL):
    t = np.asarray(date)
    # stochastic model
    l = np.asarray(Res_norm)  # observation
    n = np.shape(l)[0]  # observation dimension
    Kl = np.eye(n)  # matrix var/covar
    sigma_0 = 1
    Ql = (1 / (sigma_0) ** 2) * Kl
    #Ql =mat_QL
    P = np.linalg.inv(Ql)  # weight matrix
    # X parameter =[a,b,c]
    # initial parameter
    parametre_init = [30.71,18.33,-1.982,39.91,27.07,0.1655]
    pc=np.asarray(parametre_init)

    dxx = np.ones(len(Res_norm))
    k = 0
    epsilon = 10e-7
    B = np.zeros((n), dtype='double')
    f_0 = np.zeros((n), dtype='double')
    A = np.zeros((n, len(pc)), dtype='double')

    a= t[0]
    while (np.any(np.abs(dxx)>epsilon) and k<=100):
        # closing gap matrix : B   # Evaluate closing Biasis

        for i in range (n):
                    f_0[i] = pc[0] * np.sin(pc[1] * (t[i]-a)+pc[2])+ pc[3] * np.sin(pc[4] * (t[i]-a)+pc[5])
                    B[i] = np.asarray(Res_norm)[i] - f_0[i]

        # Jacobian matrix: A
        for i in range (n):
            A[i, 0] = np.sin(pc[1]*(t[i]-a)+pc[2])
            A[i, 1] = pc[0]*(t[i]-a)*np.cos(pc[1]*(t[i]-a)+pc[2])
            A[i, 2] = pc[0]*np.cos(pc[1]*(t[i]-a)+pc[2])
            A[i, 3] = np.sin(pc[4] * (t[i]-a) + pc[5])
            A[i, 4] = pc[3] * (t[i]-a) * np.cos(pc[4] * (t[i]-a) + pc[5])
            A[i, 5] = pc[3] * np.cos(pc[4] * (t[i]-a) + pc[5])

        dxx, X_chap, Qxx, V_chap, ll, Qvv, Qll = calculation_matrix(A=A, P=P, B=B, X0=pc, Obs=l, Ql=Ql)
        pc = X_chap
        k+=1
    print("pc_res=",pc)
    Res_norm_res,Sigma_chp_square = normalized_residuals(Residu=V_chap, P=P, obs=l, parameters=pc, Qvv=Qvv)

    return pc

def Meansquare_norm(date,h_,X_droite,pc_res,mat_poids,sigma):
    t = np.asarray(date)
    # stochastic model
    l = np.asarray(h_)  # observation
    n = np.shape(l)[0]  # observation dimension
    Kl = np.eye(n)  # matrix var/covar
    #sigma_0 = 1
    sigma_0 = sigma
    Ql = (1 / (sigma_0) ** 2) * Kl
    #Ql =mat_poids
    P = np.linalg.inv(Ql)  # weight matrix
    # X parameter =[a,b]
    # initial parameter
    parametre_init = [X_droite[0], X_droite[1], -6.03148247, 6.25907236, 2.98826516,
                      -3.31037008, 2.00716027, -3.89813205,-3.89813205,-3.89813205,-3.89813205]
    pc=np.asarray(parametre_init)
    dxx = np.ones(len(h_))

    k = 0
    epsilon = 10e-7
    B = np.zeros((n), dtype='double')
    f_0 = np.zeros((n), dtype='double')
    A = np.zeros((n, len(pc)), dtype='double')

    #a=stat.mean(date) #période caractéristique
    a=2008
    while (np.any(np.abs(dxx)>epsilon) and k<=100):
        print("normilezied",k)
        # closing gap matrix : B   # Evaluate closing Biasis
        # f_0=pc[0]*(t-t[0])+pc[1]+pc[2]*np.sin(pc[3]*(t-t[0])+pc[4])+pc[5]*np.sin(pc[6]*(t-t[0])+pc[7])-1.55771433*np.sin(0.85590144*(t-t[0])-1.67033704)-pc_res[0] * np.sin(pc_res[1] * (t[i] - date[0]) +pc_res[2])
        for i in range (n):
            f_0[i] = pc[0] * (t[i] - a) + pc[1] +\
                     pc[2] * np.sin(pc[3] * (t[i] - a)+pc[4])+\
                     pc[5] * np.sin(pc[6] * (t[i] - a)+pc[7])+\
                     pc[8] * np.sin(pc[9] * (t[i] - a)+ pc[10])
            B[i] = np.asarray(h_)[i] - f_0[i]

        # Jacobian matrix: A
        for i in range (n):
            A[i, 0] = t[i] - a
            A[i, 1] = 1
            A[i, 2] = np.sin(pc[3] * (t[i] - a) + pc[4])
            A[i, 3] = pc[2] * (t[i] - a) * np.cos(pc[3] * (t[i] - a) + pc[4])
            A[i, 4] = pc[2] * np.cos(pc[3] * (t[i] - a) + pc[4])
            A[i, 5] = np.sin(pc[6] * (t[i] - a) + pc[7])
            A[i, 6] = pc[5] * (t[i] - a) * np.cos(pc[6] * (t[i] - a) + pc[7])
            A[i, 7] = pc[5] * np.cos(pc[6] * (t[i] - a) + pc[7])
            A[i, 8] = np.sin(pc[9] * (t[i] - a) + pc[10])
            A[i, 9] = pc[8] * (t[i] - a) * np.cos(pc[9] * (t[i] - a) + pc[10])
            A[i, 10] = pc[8] * np.cos(pc[9] * (t[i] - a) + pc[10])


        dxx, X_chap, Qxx, V_chap, ll, Qvv, Qll = calculation_matrix(A=A, P=P, B=B, X0=pc, Obs=l, Ql=Ql)
        pc = X_chap
        k+=1
    pc_final = pc
    print("pc_final=",pc_final)
    Res_norm_final,Sigma_chp_square = normalized_residuals(Residu=V_chap, P=P, obs=l, parameters=pc, Qvv=Qvv)

    #print("norm.mean(Res_norm)=",norm.mean(date))
    return pc_final,Res_norm_final,V_chap

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

def normalized_residuals(Residu,P,obs,parameters,Qvv):
    Sigma_chp_square = Residu.T.dot(P.dot(Residu)) / (len(obs) - len(parameters))
    Res_norm = Residu / np.sqrt(Sigma_chp_square * Qvv.diagonal())
    return Res_norm,Sigma_chp_square

# ======================================================================
# Displaying the initial obs, the compensated heights, the residuals,
# the Residuals' test
# ======================================================================
def display(date,h_mm,X,pc,Res_norm,Res,mean,sd,pc_final1, Res_norm_final1, mean_final1, sd_final1,pc_res1):
    #La droite de 'estimation par MC
    #f_lastsquare_1 = X[0]*(np.asarray(date)-date[0])+X[1]
    #X = [17.68, 3960]
    aa= date[0]

    f_lastsquare_1 = [(X[0]*(i-date[0])+X[1]) for i in np.asarray(date)]
    f_0 = [(pc[0] * (i - aa) + pc[1]+
                   pc[2] * np.sin(pc[3] * (i - aa)+pc[4]) +
                   pc[5] * np.sin(pc[6] * (i - aa) + pc[7])+
                   pc[8] * np.sin(pc[9] * (i - aa) + pc[10]))
                  for i in np.asarray(date)]

    fig1 = plt.figure()
    axes = fig1.add_subplot(111)
    axes.set_ylabel('Mean sea level (mm)')
    axes.set_xlabel('Date')
    axes.set_title('Le CONQUET-Monthly means ')
    #plt.plot(np.array(date), np.asarray(h_mm), label='Initial Observations', color='blue',linestyle='dotted')
    axes.scatter(np.asarray(date),np.asarray(h_mm) , label='Initial dataset', s=3, color='blue')
    plt.plot(np.asarray(date), np.asarray(h_mm), label='Interpolated Data', color='green', linestyle='solid',linewidth=0.3)
    plt.plot(np.asarray(date), f_lastsquare_1, label='Least square polynom1', color='red', linestyle='solid',linewidth=0.5)
    plt.plot(np.asarray(date), f_0, color= 'k',label='Least square model',linestyle = 'solid',linewidth=1)
    plt.legend(loc="best")
    axes.grid(True)

    fig2=plt.figure()
    axes=fig2.add_subplot(211)
    axes.set_ylabel('Mean sea level Residuals (mm)')
    axes.set_xlabel('Date')
    axes.set_title('Estimated Least Square Residuals(Normalized) ')
    axes.scatter(np.asarray(date),Res,label=' Residuals', s=10,color='blue')
    plt.plot(np.asarray(date), Res, label='Interpolated Residuals', color='red', linestyle='solid',linewidth=0.5)

    #model for redisu
    a_res=date[0]
    f_model_res = [(pc_res1[0] * np.sin(pc_res1[1] * (i - a_res) + pc_res1[2])+pc_res1[3] * np.sin(pc_res1[4] * (i - a_res) + pc_res1[5]))
                   for i in np.asarray(date)]
    #f_model_res = [(1.55771433 * np.sin(0.85590144 * (i - date[0]) - 1.67033704)) for i in np.asarray(date)]
    plt.plot(np.asarray(date), f_model_res, label=' Estimated Least Square model forNormalized Residuals1',
             color='green', linestyle='solid',linewidth=0.5)
    plt.legend(loc="best")
    axes.grid(True)
    # =======================================================================
    """FFT of normalized residuals"""
    # ===================================================================
    axes = fig2.add_subplot(212)
    tf_res_norm = np.fft.fft(Res_norm)
    freq = np.fft.fftfreq(np.size(Res_norm), d=0.4)
    f = np.fft.fftshift(freq)
    plt.plot(f, np.real(tf_res_norm),linewidth=0.5)
    plt.legend(loc="best")
    axes.grid(True)

    # =======================================================================
    """Check if normalized residuals follow normal Gaussian distribtuion
    of: mean=0 and Standard deviation=1"""
    # ===================================================================
    fig3 = plt.figure()
    axes = fig3.add_subplot(111)
    # Plotting data histogram
    axes.hist(Res_norm, bins=25, normed=True, alpha=0.5, color='b')
    # Plotting the Normal distribution
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, loc= mean, scale= sd)
    plt.plot(x, p, 'k', linewidth=2)
    title = "Fit results: mean = %.2f,  std = %.2f" % (mean, sd)
    axes.set_title(title)
    plt.legend(loc="best")
    axes.grid(True)


    # After first normalization
    #a = stat.mean(date)
    a = 2008
    f_0_final1 = [(pc_final1[0] * (i - a) + pc_final1[1]+
                   pc_final1[2] * np.sin(pc_final1[3] * (i - a)+pc_final1[4]) +
                   pc_final1[5] * np.sin(pc_final1[6] * (i - a) + pc_final1[7])+
                   pc_final1[8] * np.sin(pc_final1[9] * (i - a) + pc_final1[10]))
                  for i in np.asarray(date)]

    fig4 = plt.figure()
    axes = fig4.add_subplot(111)
    axes.set_ylabel('Mean sea level (mm)')
    axes.set_xlabel('Date')
    axes.set_title('Le CONQUET-Monthly means ')
    axes.scatter(np.asarray(date), np.asarray(h_mm), label='Initial dataset', s=3, color='blue')
    plt.plot(np.asarray(date), np.asarray(h_mm), label='Interpolated Data', color='green', linestyle='solid',
             linewidth=0.3)
    plt.plot(np.asarray(date), f_lastsquare_1, label='Least square polynom1', color='red', linestyle='solid',
             linewidth=0.5)
    plt.plot(np.asarray(date), f_0_final1, color='k', label='Least square final model', linestyle='solid', linewidth=1)
    plt.legend(loc="best")

    axes.grid(True)


    fig5 = plt.figure()
    axes = fig5.add_subplot(111)
    axes.set_ylabel('Mean sea level Final Residuals (mm)')
    axes.set_xlabel('Date')
    axes.set_title('Estimated Final Least Square Residuals(Normalized) ')
    axes.scatter(np.asarray(date), Res_norm_final1, label='Normalized Residuals', s=10, color='blue')
    plt.plot(np.asarray(date), Res_norm_final1, label='Interpolated Normalized Residuals', color='red', linestyle='solid',
             linewidth=0.5)
    plt.legend(loc="best")
    # model for redisu
    #f_model_res = [(pc_res1[0] * np.sin(pc_res1[1] * (i - date[0]) + pc_res1[2])) for i in np.asarray(date)]
    # f_model_res = [(1.55771433 * np.sin(0.85590144 * (i - date[0]) - 1.67033704)) for i in np.asarray(date)]
    #plt.plot(np.asarray(date), f_model_res, label=' Estimated Least Square model forNormalized Residuals1', color='green', linestyle='solid', linewidth=0.5)
    #plt.legend(loc="best")

    axes.grid(True)
    # =======================================================================
    """FFT of normalized residuals"""
    # ===================================================================
    """   
    axes = fig5.add_subplot(212)
    tf_res_norm = np.fft.fft(Res_norm_final1)
    freq = np.fft.fftfreq(np.size(Res_norm_final1), d=0.4)
    f = np.fft.fftshift(freq)
    plt.plot(f, np.real(tf_res_norm), linewidth=0.5)
    plt.legend(loc="best")
    axes.grid(True)
    """

    # =======================================================================
    """Check if normalized residuals follow normal Gaussian distribtuion
    of: mean=0 and Standard deviation=1"""
    # ===================================================================

    fig6 = plt.figure()
    axes = fig6.add_subplot(111)
    # Plotting data histogram
    axes.hist(Res_norm_final1, bins=25, normed=True, alpha=0.5, color='b')
    # Plotting the Normal distribution
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p_final1 = norm.pdf(x, loc=mean_final1, scale=sd_final1)
    plt.plot(x, p_final1, 'k', linewidth=2)
    title = "Final Fit results: mean = %.2f,  std = %.2f" % (mean_final1, sd_final1)
    axes.set_title(title)
    axes.grid(True)


    plt.show()
    return


if __name__ == '__main__':
    date,h_mm = dataset()
    print("date=",date)
    print("h_mm=",h_mm)
    # moindre carré de la droite
    X = lastsquare_1(date=date,data_mm=h_mm)
    # MC du modéle: 3 sin + Droite
    pc,Res_norm,Res,Qll,sigma0,Qvv = Meansquare(date=date, h_= h_mm, X_droite = X)
    print("RES=",Res)
    mean, sd = norm.fit(Res_norm)

    #model for residual
    pc_res1 = Meansquare_Res(date=date, Res_norm=Res,mat_QL=Qvv)

    # model droite+3sin-model- résidus

    pc_final1, Res_norm_final1,Res_final1= Meansquare_norm(date=date, h_=h_mm, X_droite=X, pc_res=pc_res1,mat_poids= Qll,sigma=sigma0)
    mean_final1, sd_final1 = norm.fit(Res_norm_final1)

    print("pc_final1=",pc_final1)
    t =date
    n = np.shape(h_mm)[0]  # observation dimension


    display(date= date,h_mm= h_mm,X= X,
            pc= pc,Res_norm= Res_norm,Res=Res,mean=mean,sd=sd,
            pc_final1=pc_final1, Res_norm_final1=Res_norm_final1, mean_final1=mean_final1, sd_final1=sd_final1,pc_res1=pc_res1)

