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

#=======================================================================
"""Loading Data"""
#=======================================================================

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

# ======================================================================
# Displaying the initial obs, the compensated heights, the residuals,
# the Residuals' test
# ======================================================================
def display(date,h_mm):
    fig1 = plt.figure()
    axes = fig1.add_subplot(111)
    axes.set_ylabel('Sea level data (mm)')
    axes.set_xlabel('Date')
    axes.set_title('Le CONQUET-Monthly means ')
    plt.plot(np.asarray(date), np.asarray(h_mm), label='Initial Observations', color='blue', linestyle='-')
    axes.legend(loc=2)
    axes.grid(True)
    plt.show()
    return



if __name__ == '__main__':
    date,h_mm = dataset()
    print(date)
    print(h_mm)
    display(date,h_mm)


