import numpy as np
from scipy.optimize import *
import matplotlib.pyplot as plt
from scipy.stats import norm

"""Loading Data"""
def dataset():
    date = []
    data_mm=[]
    data_file = "mLCONQ_dec10.slv"
    with open(data_file) as f:
        for line in f :
            date.append(float(line[0:9]))
            data_mm.append(int(line[10:-1]))

    return (date,data_mm)

def function(parameters,time):
    # the aim of This function: for specific time and parameters returns sea level
    # INPUTS:
    #   parameters,
    #   time,
    #OUTPUt:
    #   h_ unit mm
    a= date[0]
    #a=2008
    h_=parameters[0]*(time-a)+parameters[1]+\
                   parameters[2]*np.sin(parameters[3]*(time-a)+parameters[4])+\
                   parameters[5]*np.sin(parameters[6]*(time-a)+parameters[7])+\
                   parameters[8]*np.sin(parameters[9]*(time-a)+parameters[10])
    return h_


def Model_function(parameters):
    # =======================================================================
    """The model: FOR SOLVING THE NON-LINEAR PROBLEM at step 1"""
    # =======================================================================
    date0 = maybe_points[:,0]
    h_mm0 = maybe_points[:,1]
    a = date[0]
    #a=2008
    f_model = np.empty(len(date0))
    for i in range(len(date0)):
        f_model[i]=parameters[0]*(date0[i]-a)+parameters[1]+\
                   parameters[2]*np.sin(parameters[3]*(date0[i]-a)+parameters[4])+\
                   parameters[5]*np.sin(parameters[6]*(date0[i]-a)+parameters[7])+\
                   parameters[8]*np.sin(parameters[9]*(date0[i]-a)+parameters[10])-\
                   h_mm0[i]
    return f_model

def Solve_Model_function(parameters):
    # =======================================================================
    """
    Aim of the function: solve the nonlinear problem using time series
    as entries
    Find a unique solution for the given points
    :pram points selected points for model fitting
    :return model
    """
    # =======================================================================

    # find a line model for these points
    pc=fsolve(Model_function,parameters)
    #print("fsolve=",pc)
    return pc


def Least_square_method(date_,h_,parameters):
    t = np.asarray(date_)
    # stochastic model
    l = np.asarray(h_)  # observation
    n = np.shape(l)[0]  # observation dimension
    Kl = np.eye(n)      # matrix var/covar
    sigma_0 = 1
    Ql = (1 / (sigma_0) ** 2) * Kl
    P = np.linalg.inv(Ql)  # weight matrix
    pc = np.asarray(parameters)
    dxx = np.ones(len(h_))

    k = 0
    epsilon = 10e-7
    B = np.zeros((n), dtype='double')
    f_0 = np.zeros((n), dtype='double')
    A = np.zeros((n, len(pc)), dtype='double')

    a = t[0]
    #a=2008
    while (np.any(np.abs(dxx) > epsilon) and k <= 100):
        # closing gap matrix : B   # Evaluate closing Biasis

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

    Res_norm,Sigma_chp_square = normalized_residuals(Residu=V_chap, P=P, obs=l, parameters=pc, Qvv=Qvv)
    #print("norm.mean(Res_norm)=",norm.mean(date))
    return pc,Res_norm,V_chap,np.sqrt(Sigma_chp_square)
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

def ransac_plot(it_, date_, h_mm_,parameters_, date_inliers_, h_inliers_, maybe_points_,status,Res_norm):
    #n, x, y, m, c, final=False, x_in=(), y_in=(), points=()):
    """ plot the current RANSAC step
    :param n      iteration
    :param points picked up points for modeling
    :param x      samples x
    :param y      samples y
    :param m      slope of the line model
    :param c      shift of the line model
    :param x_in   inliers x
    :param y_in   inliers y
    """

    fname = "output/figure_" + str(it_) + ".png"
    line_width = 1.
    line_color = '#0080ff'
    title = 'iteration ' + str(it_)

    if (status==0):
        # To present final results
        fname = "output/final.png"
        line_width = 3.
        line_color = '#ff0000'
        title = 'Final solution'



    plt.figure(11)
        # put grid on the plot
    plt.grid(b=True, which='major', color='0.75', linestyle='--')
        # plot input points
    plt.plot(np.asarray(date_), np.asarray(h_mm_), marker='o', label='Input points', color='#00cc00', linestyle='None', alpha=0.2)
    a=date[0]
    #a = 2008
    f_model =[(parameters_[0] * (i - a) + parameters_[1] + parameters_[2] * np.sin(parameters_[3] * (i - a)+ parameters_[4])
               +parameters_[5] * np.sin(parameters_[6] * (i - a) + parameters_[7])
               + parameters_[8] * np.sin(parameters_[9] * (i - a) + parameters_[10]))
              for i in np.asarray(date_)]
    # draw the current model
    plt.plot(np.asarray(date_), f_model, 'r', label='Line model', color=line_color, linewidth=line_width)
    plt.legend()


    if (status==1):
        plt.plot(date_inliers_, h_inliers_, marker='o', label='Inliers', linestyle='None', color='#ff0000', alpha=0.2)

    # draw points picked up for the modeling
        plt.plot(maybe_points_[:, 0], maybe_points_[:, 1], marker='x', label='Picked points', color='#0000cc', linestyle='None',
                 alpha=0.6)

    plt.title(title)
    plt.legend()
    plt.show()


    fig2 = plt.figure("histo")
    axes = fig2.add_subplot(111)
    mean_final, sd_final = norm.fit(Res_norm)
    # Plotting data histogram
    axes.hist(Res_norm, bins=25, normed=True, alpha=0.5, color='b')
    # Plotting the Normal distribution
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p_final1 = norm.pdf(x, loc=mean_final, scale=sd_final)
    plt.plot(x, p_final1, 'k', linewidth=2)
    title = "Final Fit results: mean = %.2f,  std = %.2f" % (mean_final, sd_final)
    axes.set_title(title)
    axes.grid(True)
    plt.legend()
    plt.show()
    # draw inliers
def random_points(n):

    all_indices = np.arange(date.shape[0])
    # print("all_indices=",all_indices)
    np.random.shuffle(all_indices)
    # print(" after shuffle\n",all_indices)
            # Take the first n indices of the all_indices
    indices_1 = all_indices[:n]
    # print("random_indice1: ",indices_1)

            # Take all the rest of the indices: Rest of points
    indices_2 = all_indices[n:]
    # print("random_indice2: ",indices_2)

            # Random points : n(=11) random values of the data
    global maybe_points
    maybe_points = data[indices_1, :]
    # print("maybe_points:\n",maybe_points)
    test_points = data[indices_2, :]  # all the rest points
    # print("test_points=\n",test_points)
    return maybe_points,test_points
if __name__ == '__main__':

    global date     # observation date
    global h_mm     # observation height (mm)

    date,h_mm =dataset()
    number_obs=len(date)
    date = np.asarray(date)
    h_mm = np.asarray(h_mm)
    data = np.hstack((date.reshape(number_obs,1), h_mm.reshape(number_obs,1)))

    print("number_obs=", number_obs)
    print("data=\n",data)


    # Ransac parameters
    ransac_iterations = 30  # number of iterations

    # Determine the set of points S  that are in agreement with the model at a threshold
    # t=ransac_threshold fixed.
    # S called the sample consensus game ("good" S points).
    ransac_threshold = 80   # fixed threshold (t mm)
    ransac_ratio = 0.85  # ratio of inliers required to assert that 85%
    ratio = 0.65   # verification threshold of consensus points (minimum 114 points)

    # Best Model
    model_para= np.zeros((1,11))
    #print("model_para=",model_para)
    it =0  #itérateur
    #meilleur ensembes le points
    best_points = []
    #Best error
    best_error = 1e10

    ini_parameters = [4.1497057, 19.32060092, -6.10410342, 6.25634226, 3.01462022
        , -3.67541036, 1.97400679, -3.69483161, -1.55771433, 0.85590144, -1.6703370]

    flag = False
    # Perform RANSAC iterations
    for it in range(ransac_iterations):
        # =======================================================
        # Minimum number of points needed to estimate the model
        #  pick up n random points
        # ========================================================
        n = 11
        maybe_points, test_points = random_points(n)

        #===============================================================
        # Find out possible parameters corresponding to the random points
        # by solving the non linear system
        #===============================================================
        possible_parameters = Solve_Model_function(parameters=ini_parameters)
        print("Possible_parameters=\n",possible_parameters)

        # Coordinates of consensus points (inlier points)
        date_list = []  # The date of the inliers points
        h_list = []     # The height of the inliers points
        num = 11   # number of the consensus points
        #=================================================================================
        # we test point by point of the remaining points
        # if the distance from the model is below the threshold(t=ransac_threshold)< ransac_threshold
        # For the purpose of determining the set of points of consensus
        #==================================================================================
        for ind in range(test_points.shape[0]):

            date0 = test_points[ind, 0]
            h_mm0 = test_points[ind, 1]
            h_with_estim_model =function(parameters=possible_parameters,time=date0)

            # Distance between the data point and the model
            dist = np.abs(h_mm0-h_with_estim_model)
            #print("distance=",dist)

            # Check whether it's an inlier or not
            if (dist <= ransac_threshold):
                #points of consensus set
                date_list.append(date0)
                h_list.append(h_mm0)
                num += 1

        # Add the n points chosen randomly
        for i in range(n):
            date_list.append(maybe_points[:, 0][i])
            h_list.append(maybe_points[:, 1][i])

        print("Number of consensus points=", num)

        # Coordinates of consensus points (inlier points) in array form
        date_inliers = np.asarray(date_list)
        h_inliers = np.asarray(h_list)

        # =================================================================================
        # We test if the number of consensus number > ratio (minimum number of points to estimate a model)
        # If the size of consensus points is greater than a given threshold (ratio),
        # We estimate the model again on the basis of consensus points
        # In case a new model is better - cache it
        # ==================================================================================

        if ((num / float(number_obs)) > ratio):
            flag= True
            #update the ratio = number of points/number_of_obs
            ratio = num / float(number_obs)
            print(" The new inlier ratio = ", ratio)
            # we adjust the model at all points of the consensus (inlier points)
            # we apply an estimation by the least square method
            updated_parameters, Res_norm, V_chap,sigma_chap=Least_square_method(
                date_=date_inliers,h_=h_inliers,parameters=possible_parameters)
            ini_parameters=updated_parameters
            max_points = num

            print("Updated_parameters=",updated_parameters)
            print("sigma_chap",sigma_chap)
            stat = 1
            # Plot the current step
            ransac_plot(it_=it, date_=date, h_mm_=h_mm, parameters_=updated_parameters, date_inliers_=date_inliers,
                        h_inliers_=h_inliers, maybe_points_=maybe_points, status=stat, Res_norm=Res_norm)

        if num > number_obs * ransac_ratio:
            # we are done in case we have enough inliers
            print("The model is found ! with a ratio of 90%")
            break

        print("\n****Next iteration************\n ")

    # Plot the final model
    if (flag== True):
        stat=0
        ransac_plot(it_=0, date_=date, h_mm_=h_mm,parameters_=updated_parameters, date_inliers_=date_inliers,
                    h_inliers_=h_inliers, maybe_points_=maybe_points,status=stat,Res_norm=Res_norm)

        print("\nFinal model:\n")
        print("ratio = ", ratio)
        print("the number of points=",max_points)
        print(" Final model = ",updated_parameters )
    else:
        print("did not meet fit acceptance criteria")


