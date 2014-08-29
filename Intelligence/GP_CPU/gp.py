import numpy as np
import os
from scipy import spatial
import math
from time import gmtime, strftime


class GP(object):
    def __init__(self):
        '''
        do nothing -- just init '''
        self.generate_data = False
        self.debug = False

    # Gaussian process returns an index of chosen element from datapoints_predict
    def GP(self, datapoints_shown, feedback, data, random_K, random_K_xx, time=1, sigma_n=0.5):
        datapoints_predict = np.setdiff1d(np.arange(len(data)), datapoints_shown)
        if self.generate_data:
            outfileprefix = 'output/' + str(len(feedback) - 12) + '_'
        if self.debug:
            print("Inside GP")
        kernel = data
        # beta = math.sqrt(math.log(time))
        #K = (kernel[datapoints_shown,:])[:,datapoints_shown]+np.diag((sigma_n**2)*np.random.normal(1,0.1,(len(datapoints_shown))))
        if self.debug:
            print(datapoints_shown.shape)
            print(random_K.shape)
        K = (kernel[datapoints_shown, :])[:, datapoints_shown] + np.diag(random_K)
        if self.debug:
            print("K computed")
        if self.generate_data:
            np.save(outfileprefix + "K.npy", K)
        K_x = (kernel[datapoints_predict, :])[:, datapoints_shown]
        if self.generate_data:
            np.save(outfileprefix + "K_x.npy", K_x)
        if self.debug:
            print("K_x computed")
        #K_xx = kernel[datapoints_predict,datapoints_predict]+np.diag((sigma_n**2)*np.random.normal(1,0.1,(len(datapoints_predict))))
        if self.debug:
            print("K_xx computed")
        if self.generate_data:
            np.save(outfileprefix + "K_inv.npy", np.linalg.inv(K))
        temp = np.dot(K_x, np.linalg.inv(K))
        if self.generate_data:
            np.save(outfileprefix + "temp.npy", temp)
        #print("Temp computed")
        mean = np.dot(temp, feedback)
        if self.debug:
            print("Mean computed")
        if self.generate_data:
            np.save(outfileprefix + "mean.npy", mean)
            np.save(outfileprefix + "diag.npy", np.dot(temp, K_x.T))
        if self.debug:
            print("K_xx shape")
            print(random_K_xx.shape)
            print(np.diag(random_K_xx))
        K_xKK_xT_diag = [np.sum(temp[idx, :] * K_x[idx, :]) for idx in range(len(random_K_xx))]
        var = random_K_xx - K_xKK_xT_diag
        if self.generate_data:
            np.save(outfileprefix + "var.npy", var)
        if self.debug:
            print("var computed")
        #ucb = mean + beta*np.sqrt(var)
        return mean, var
        # for demo only
        #print("For demo only")
        #return K, K_xx, mean, var



