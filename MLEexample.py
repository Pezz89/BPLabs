import numpy as np
from scipy.optimize import minimize

def logisticFunction(L, L_50, s_50):
    '''
    Calculate logistic function for SNRs L, 50% SRT point L_50, and slope
    s_50
    '''
    return 1./(1.+np.exp(4.*s_50*(L_50-L)))

def logisticFuncLiklihood(self, args):
    '''
    Calculate the log liklihood for given L_50 and s_50 parameters.
    This function is designed for use with the scipy minimize optimisation
    function to find the optimal L_50 and s_50 parameters.

    args: a tuple containing (L_50, s_50)
    self.wordsCorrect: an n dimensional binary array of shape (N, 5),
        containing the correctness of responses to each of the 5 words for N
        trials
    self.trackSNR: A sorted list of SNRs of shape N, for N trials
    '''
    L_50, s_50 = args
    ck = self.wordsCorrect[np.arange(self.trackSNR.shape[0])]
    p_lf = self.logisticFunction(self.trackSNR, L_50, s_50)
    # Reshape array for vectorized calculation of log liklihood
    p_lf = p_lf[:, np.newaxis].repeat(5, axis=1)
    # Calculate the liklihood
    res = (p_lf**ck)*(((1.-p_lf)**(1.-ck)))
    with np.errstate(divide='raise'):
        try:
            a = np.concatenate(res)
            a[a == 0] = a.max()
            out = -np.sum(np.log(a))
        except:
            set_trace()
    return out

# Called from within a class in my implementation, will not work without a
# class containing member variables for wordsCorrect and trackSNR
res = minimize(logisticFuncLiklihood, np.array([-5.0,1.0]), method='L-BFGS-B')
