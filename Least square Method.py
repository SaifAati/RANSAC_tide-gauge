import os
import numpy as np
import matplotlib.pylab as plt
from scipy.stats import norm
from scipy.fftpack import fft, fftfreq
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

def LeastSquareLineFit(date, data_mm):
    # ==============================Step1=============================================
    """
    Using least squares approximation to fit a line to points
    Last squares method: in order to estimate an affine relation between the variables
    ( data_mm[]=sea level and date[]= dates),in other words look for a line that fits
    as well as possible to this cloud of points.
    INPUT:
         data_mm[]=sea level
         date[]= dates

    The model:f(t) = a(t-t0)+ b

    """
    # ================================================================================
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


        dxx,X_chap,Qxx,V_chap,ll,Qvv,Qll = CalculationMatrix(A=A, P=P, B=B, X0=X, Obs=l, Ql=Ql)
        X = X_chap
        K = K + 1

    print("X=",X)

    return X

def LeastSquareForModelDesign(date, h_, X_droite):
    # ==============================Step2=============================================
    """
    Using least squares approximation to design the final model
    INPUT:
         data_mm[]=sea level
         date[]= dates
         parameter of the line : a and b

    The model:f(t) = a(t-t0)+ b+Ai*sin(wi(t-t0)+phi)

    """
    # ================================================================================
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



        dxx, X_chap, Qxx, V_chap, ll, Qvv, Qll = CalculationMatrix(A=A, P=P, B=B, X0=pc, Obs=l, Ql=Ql)
        pc = X_chap
        k+=1
    print("pc=",pc)
    Res_norm,Sigma_chp_square = NormalizedResidues(Residu=V_chap, P=P, obs=l, parameters=pc, Qvv=Qvv)

    return pc,Res_norm,V_chap,Qll,np.sqrt(Sigma_chp_square),Qvv

def FinalLeastSquare(date, h_, X_droite, sigma):
    t = np.asarray(date)
    print(np.shape(date))
    # stochastic model
    l = np.asarray(h_)  # observation
    n = np.shape(l)[0]  # observation dimension
    print("n=",n)
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
    dvv = np.ones(n)
    dxx = np.ones(len(h_))

    k = 0
    epsilon = 10e-7
    B = np.zeros((n), dtype='double')
    f_0 = np.zeros((n), dtype='double')
    A = np.zeros((n, len(pc)), dtype='double')

    #a=stat.mean(date) #période caractéristique
    a=2008
    v_ini = np.zeros(np.shape(h_))
    while (np.any(np.abs(dvv)>epsilon) and k<=150):

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


        dxx, X_chap, Qxx, V_chap, ll, Qvv, Qll = CalculationMatrix(A=A, P=P, B=B, X0=pc, Obs=l, Ql=Ql)
        dvv = V_chap - v_ini
        v_ini =V_chap
        pc = X_chap
        k+=1
    pc_final = pc
    Res_norm_final,Sigma_chp_square = NormalizedResidues(Residu=V_chap, P=P, obs=l, parameters=pc, Qvv=Qvv)

    #print("norm.mean(Res_norm)=",norm.mean(date))

    return pc_final,Res_norm_final,V_chap,ll,np.sqrt(Sigma_chp_square)

def CalculationMatrix(A, P, B, X0, Obs, Ql):
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

def NormalizedResidues(Residu, P, obs, parameters, Qvv):
    Sigma_chp_square = Residu.T.dot(P.dot(Residu)) / (len(obs) - len(parameters))
    Res_norm = Residu / np.sqrt(Sigma_chp_square * Qvv.diagonal())
    return Res_norm,Sigma_chp_square


def FFTFunction(signal, Date, N):
    #====================================
    """
    Inputs:
        - Date
        - Signal
        - Number of points (sample)


    """

    t0 = Date[0]   # debut de l'acquisition du signal
    t1 = Date[-1]  # fin de l'acquisition

    # Definition des parametres d'echantillonnage (sampling)
    #========================================================
    """
    FreqEch = 1024              # frequence dechatillonage sampling
    PerEch = 1. / FreqEch       # periode of sampling
    N = FreqEch * (t1 - t0)     # Nunmber of samples (points)
    """
    #=========================================================
    FreqEch = N/(t1-t0)
    PerEch = 1. / FreqEch
    print("PerEch=",PerEch)


    # definition du temps
    #t = linspace(t0, t1, N)
    t = Date

    # definition des donnees de FFT
    FenAcq = signal.size      # taille de la fenetre temporelle =N


    # calcul de la TFD par l'algo de FFT
    signal_FFT = abs(fft(signal))   # on ne recupere que les composantes reelles

    # recuperation du domaine frequentiel
    signal_freq = fftfreq(FenAcq, PerEch)

    # extraction des valeurs reelles de la FFT et du domaine frequentiel
    signal_FFT = signal_FFT[0:len(signal_FFT)//2 ]
    signal_freq = signal_freq[0:len(signal_freq)//2]

    """
    # affichage du signal
    plt.subplot(211)
    plt.title('Signal et son spectre')
    #plt.ylim(-(A1 + 5), A1 + 5)
    plt.plot(t, signal)
    plt.xlabel('Temps (s)');
    plt.ylabel('Amplitude')

    # affichage du spectre du signal
    plt.subplot(212)
    #lt.xlim(N*signal_freq[0], N*signal_freq[-1])
    plt.plot(N*signal_freq, signal_FFT)
    plt.xlabel('Frequence (Hz)');
    plt.ylabel('Amplitude')
    # plt.title('Signal et son spectre')
    plt.show()
    """
    return signal_freq,signal_FFT

def Test3Sigma(date, h_mm, X_droite):
    N =np.shape((date)[0])
    # Deleting Observations corresponding to high residuals and doing 3sigma test
    sigma0 = 1
    diff=0
    initial_h = h_mm
    initial_date = date
    for j in range(5):

        print("\n\nStep", j)
        pc_final, Res_norm_final, Res_final, h_mm_compen, Sigma_chap_final = FinalLeastSquare(date=date, h_=h_mm,
                                                                                                 X_droite=X_droite, sigma=sigma0)
        print("Sigma_chap", Sigma_chap_final)
        print("Parameters", pc_final)
        print("Res_norm max", np.max(Res_norm_final))

        mu, std = norm.fit(Res_norm_final)

        # Condition The Residues by the Sigma Test  to get some outliers
        indices = []
        new_h_mm = []
        new_date =[]
        number=0
        for i in range(len(Res_final)):
            if (np.abs(Res_norm_final[i]) < 3 * std):
                number +=1
                new_date.append(date[i])
                new_h_mm.append(h_mm[i])
        diff = diff+ np.shape(date)[0]-number
        print("Number of Obs to be removed in Step ",j,"= ",diff )
        date = np.asarray(new_date)
        h_mm =np.asarray(new_h_mm)

    return

def Display(date, h_mm, X, pc, Res_norm, Res, mean, sd, pc_final1, Res_norm_final1, mean_final1, sd_final1, h_mm_compen):
    # ======================================================================
    # Displaying the initial obs, the compensated heights, the residuals,
    # the Residuals' test
    # ======================================================================

    #******************************  Display 1 ******************************************#
    # First model estimated by Least square Method
    aa= date[0]
    Line_model = [(X[0]*(i-date[0])+X[1]) for i in np.asarray(date)]
    Model_1 = [(pc[0] * (i - aa) + pc[1]+
                   pc[2] * np.sin(pc[3] * (i - aa)+pc[4]) +
                   pc[5] * np.sin(pc[6] * (i - aa) + pc[7])+
                   pc[8] * np.sin(pc[9] * (i - aa) + pc[10]))
                  for i in np.asarray(date)]

    fig1 = plt.figure(1)
    axes = fig1.add_subplot(111)
    axes.set_ylabel('Mean sea level (mm)')
    axes.set_xlabel('Date')
    axes.set_title('Le CONQUET-Monthly means ')
    axes.scatter(np.asarray(date),np.asarray(h_mm) , label='Initial dataset', s=3, color='blue')
    plt.plot(np.asarray(date), np.asarray(h_mm), label='Interpolated Data', color='green', linestyle='solid',linewidth=0.3)
    plt.plot(np.asarray(date), Line_model, label='Least square polynom1', color='red', linestyle='solid',linewidth=0.5)
    plt.plot(np.asarray(date), Model_1, color= 'k',label='Least square model 1',linestyle = 'solid',linewidth=1)
    plt.legend(loc="best")
    axes.grid(True)
    # ******************************  Display 2 ******************************************#
    # Display the residues
    # Display the Fourier transform of residues
    # sample spacing

    fig2 = plt.figure(2)
    axes = fig2.add_subplot(211)
    axes.set_ylabel('Mean sea level Residues (mm)')
    axes.set_xlabel('Date(decimal year)')
    axes.set_title('Least Square Residues')
    axes.scatter(np.asarray(date), Res, label=' Residues', s=10, color='blue')
    axes.grid(True)
    plt.plot(np.asarray(date), Res, label='Interpolated Residues', color='red', linestyle='solid', linewidth=0.5)

    N = np.shape(date)[0]
    print("N=",N)
    signal_freq, signal_FFT = FFTFunction(signal=Res, Date=np.asarray(date), N =N)

    axes = fig2.add_subplot(212)
    plt.plot(N*signal_freq, signal_FFT,linewidth=0.5)
    plt.xlabel('Frequence ');
    plt.ylabel('Amplitude')
    axes.grid(True)
    # plt.title('Signal et son spectre')
    plt.show()

    # ******************************  Display 3 ******************************************#
    # Check if normalized residuals follow normal Gaussian distribution
    # of: mean=0 and Standard deviation=1

    fig3 = plt.figure(3)
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
    plt.show()


    # ******************************  Display 4 ******************************************#

    # Final model

    a = 2008

    f_0_final1 = [(pc_final1[0] * (i - a) + pc_final1[1]+
                   pc_final1[2] * np.sin(pc_final1[3] * (i - a)+pc_final1[4]) +
                   pc_final1[5] * np.sin(pc_final1[6] * (i - a) + pc_final1[7])+
                   pc_final1[8] * np.sin(pc_final1[9] * (i - a) + pc_final1[10]))
                  for i in np.asarray(date)]

    fig4 = plt.figure(4)
    axes = fig4.add_subplot(111)
    axes.set_ylabel('Mean sea level (mm)')
    axes.set_xlabel('Date')
    axes.set_title('Le CONQUET-Monthly means ')
    axes.scatter(np.asarray(date), np.asarray(h_mm), label='Initial dataset', s=3, color='blue')
    plt.plot(np.asarray(date), np.asarray(h_mm), label='Interpolated Data', color='green', linestyle='solid',
             linewidth=0.3)
    plt.plot(np.asarray(date), Line_model, label='Least square polynom1', color='red', linestyle='solid',
             linewidth=0.5)
    plt.plot(np.asarray(date), f_0_final1, color='k', label='Least square final model', linestyle='solid', linewidth=1)
    axes.scatter(np.asarray(date), h_mm_compen, label='Re-estimated Measures', s=2, color='red')
    plt.legend(loc="best")

    axes.grid(True)

    # ******************************  Display 5 ******************************************#
    #  # Check if normalized residuals follow normal Gaussian distribution
    # of: mean=0 and Standard deviation=1
    fig5 = plt.figure()
    #fig6 = plt.figure()
    axes = fig5.add_subplot(111)
    # Plotting data histogram
    axes.hist(Res_norm_final1, bins=25, normed=True, alpha=0.5, color='b')
    # Plotting the Normal distribution
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p_final1 = norm.pdf(x, loc=mean_final1, scale=sd_final1)
    plt.plot(x, p_final1, 'k', linewidth=2)
    title = "Final Fit results: mean = %.2f,  std = %.2f" % (mean_final1, sd_final1)
    axes.set_title(title)
    plt.legend(loc="best")
    axes.grid(True)
    plt.show()
    return


if __name__ == '__main__':

    # Load data
    date,h_mm = dataset()
    print("date=",date)
    print("h_mm=",h_mm)

    # Least_square for line fit
    X = LeastSquareLineFit(date=date, data_mm=h_mm)

    # Least square first model = 3 sin + line
    pc,Res_norm,Res,Qll,sigma_chap1,Qvv = LeastSquareForModelDesign(date=date, h_= h_mm, X_droite = X)
    print("Sigma_chap1=",sigma_chap1)
    mean, sd = norm.fit(Res_norm)

    # Least squareFinal Model
    pc_final1, Res_norm_final1,Res_final1,h_mm_compen,Sigma_chap_final = FinalLeastSquare(date=date, h_=h_mm, X_droite=X, sigma=1)
    mean_final1, sd_final1 = norm.fit(Res_norm_final1)
    print("pc_final1=",pc_final1)
    print("Sigma_chap", Sigma_chap_final)

    # Plotting

    Display(date= date, h_mm= h_mm, X= X,
            pc= pc, Res_norm= Res_norm, Res=Res, mean=mean, sd=sd,
            pc_final1=pc_final1, Res_norm_final1=Res_norm_final1, mean_final1=mean_final1, sd_final1=sd_final1, h_mm_compen=h_mm_compen)


    Test3Sigma(date=date, h_mm=h_mm, X_droite=X)