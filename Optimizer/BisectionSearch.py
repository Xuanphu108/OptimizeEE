import numpy as np
from numpy import inf
import matplotlib.pyplot as plt
import pandas
import matplotlib as mpl
import time

class BisectionSearch(object):
    '''
    :param h: matrix channel (N x M)
    :param b: matrix decision (N x M)
    '''
    def __init__(self,
                 channelPsUsers,
                 channelApUsers, 
                 binaryMatrix,
                 noise=10**((-174/10)-3),
                 subBandwidth=78*10**3,
                 effPower=0.8,
                 powUserCir=10**(-3),
                 powPScir=10**((5/10)-3),
                 dinkel=1):
        self.channelPsUsers = channelPsUsers
        self.channelApUsers = channelApUsers
        self.binaryMatrix = binaryMatrix
        self.noise = noise
        self.powUserCir = powUserCir
        self.powPScir = powPScir
        self.effPower = effPower
        self.dinkel = dinkel
        self.T = 1
        self.subBandwidth = subBandwidth
        self.epsilon = self.subBandwidth/(np.log(2))
        self.apUsers = np.multiply(self.channelApUsers,self.binaryMatrix)

    def calcObject(self, power, per):
        psUsers = self.channelPsUsers
        channel = self.apUsers/(self.subBandwidth*self.noise)
        sumRate = (1-per)*self.epsilon*np.log(1 + np.multiply((self.effPower*power*per*psUsers/(1-per) - self.powUserCir),channel))
        sumRate = np.sum(sumRate)
        X1 = per*(power - self.effPower*power*np.sum(psUsers) + self.powPScir)
        X2 = (1-per)*np.sum(np.multiply((self.effPower*power*per*psUsers/(1-per)-self.powUserCir),self.binaryMatrix))
        obj = sumRate - self.dinkel*(X1 + X2)
        return obj 

    def calcPowDerivative(self, power, per):
        psUsers = self.channelPsUsers
        channel = self.apUsers/(self.subBandwidth*self.noise)
        tempX1 = 1 - self.powUserCir*channel
        tempX2 = self.effPower*np.multiply(psUsers,channel)*per/(1-per)
        tempY1 = per - self.effPower*per*np.sum(psUsers)
        tempY2 = np.sum(self.effPower*per*np.multiply(psUsers,self.binaryMatrix))
        powerDerivative = np.sum(self.epsilon*(1-per)*tempX2/(tempX1 + tempX2*power)) - \
                          self.dinkel*(tempY1 + tempY2)
        return powerDerivative

    def calcPerDerivative(self, power, per):
        PS_users = self.channelPsUsers
        channel = self.apUsers/(self.subBandwidth*self.noise)
        A_per = 1 - self.powUserCir*channel
        B_per = self.effPower*power*np.multiply(PS_users,channel)
        X1_PerDer = power - self.effPower*power*np.sum(PS_users) + self.powPScir
        X2_PerDer = np.sum(np.multiply((self.effPower*power*PS_users+self.powUserCir),self.binaryMatrix))
        ValPerDer = np.sum(self.epsilon*np.log((1-per)/((B_per-A_per)*per+A_per)) + self.epsilon*(B_per-A_per)*(1-per)/((B_per-A_per)*per+A_per) + self.epsilon) - \
                    self.dinkel*(X1_PerDer + X2_PerDer)
        return ValPerDer

    @staticmethod 
    def split_based(mat, val):
        mask = mat!=val
        temp = np.split(mat[mask],mask.sum(1)[:-1].cumsum())
        out = np.array(list(map(list,temp)))
        return out

    def initPower(self, Pmax, PerMax):
        PS_users = self.channelPsUsers
        channel = self.channelApUsers/(self.subBandwidth*self.noise)
        # Initialize power
        PowerList = np.arange(0, Pmax, 1/1000000)
        A_per = 1 - self.powUserCir*channel
        B_per = self.effPower*np.outer(PowerList[1:],np.multiply(PS_users,channel))
        B_per = B_per.reshape(len(PowerList[1:]), channel.shape[0], channel.shape[1])
        C_per = (B_per-A_per)
        A_per = np.repeat(A_per[np.newaxis,...], len(PowerList[1:]), axis=0)

        C_neg = C_per.clip(max=0)
        C_pos = C_per.clip(min=0)

        UpperBound = np.divide(-A_per, C_neg, out = np.zeros_like(-A_per), where=C_neg!=0)
        UpperBound = UpperBound.reshape(len(PowerList[1:]), channel.shape[0]*channel.shape[1])
        UpperBound[UpperBound==0] = +inf
        UpperBound = np.min(UpperBound, axis=1)
        UpperBound = np.minimum(UpperBound, PerMax-0.0000000000001)
        
        LowerBound = np.divide(-A_per, C_pos, out = np.zeros_like(-A_per), where=C_pos!=0)
        LowerBound = LowerBound.reshape(len(PowerList[1:]), channel.shape[0]*channel.shape[1])
        # LowerBound = self.split_based(LowerBound, 1) # Remove 1
        LowerBound = np.max(LowerBound, axis=1)
        LowerBound = np.maximum(LowerBound, 0.0000000000001)

        TrueList = np.where(LowerBound <= UpperBound)
        if len(TrueList[0]) == 0:
            raise ValueError('Parameters is not suitable. Please adjust the parameters!')
        PowerInit = PowerList[1:][TrueList[0][int(len(TrueList)/2)]]
        LowerBoundInit = LowerBound[TrueList[0][int(len(TrueList)/2)]]
        return PowerInit, LowerBoundInit

    def initPeriod(self, Pmax, PerMax):
        PS_users = self.channelPsUsers
        channel = self.channelApUsers/(self.subBandwidth*self.noise)
        # Initialize period
        PeriodList = np.arange(0, PerMax, 1/1000000)
        A_pow = 1 - self.powUserCir*channel
        B_pow = self.effPower*np.outer(PeriodList[1:]/(1-PeriodList[1:]),np.multiply(PS_users,channel))
        B_pow = B_pow.reshape(len(PeriodList[1:]), channel.shape[0], channel.shape[1])
        A_pow = np.repeat(A_pow[np.newaxis,...], len(PeriodList[1:]), axis=0)

        B_neg = B_pow.clip(max=0)
        B_pos = B_pow.clip(min=0)

        UpperBound = np.divide(-A_pow, B_neg, out = np.zeros_like(-A_pow), where=B_neg!=0)
        UpperBound = UpperBound.reshape(len(PeriodList[1:]), channel.shape[0]*channel.shape[1])
        UpperBound[UpperBound==0] = +inf
        UpperBound = np.min(UpperBound, axis=1)
        UpperBound = np.minimum(UpperBound, Pmax)
        
        LowerBound = np.divide(-A_pow, B_pos, out=np.zeros_like(-A_pow), where=B_pos!=0)
        LowerBound = LowerBound.reshape(len(PeriodList[1:]), channel.shape[0]*channel.shape[1])
        LowerBound = np.max(LowerBound, axis=1)
        LowerBound = np.maximum(LowerBound, 0.0000000000001)
        TrueList = np.where(LowerBound <= UpperBound)
        if len(TrueList[0]) == 0:
            raise ValueError('Parameters is not suitable. Please adjust the parameters!')
        # import ipdb; ipdb.set_trace()
        PeriodInit = PeriodList[1:][TrueList[0][int(len(TrueList)/2)]]
        LowerBoundInit = LowerBound[TrueList[0][int(len(TrueList)/2)]]
        return PeriodInit, LowerBoundInit
        
    def searchPower(self, Pmax, PerMax, per, OldPower, index):
        PS_users = self.channelPsUsers
        # channel = self.apUsers/(self.subBandwidth*self.noise)
        channel = self.channelApUsers/(self.subBandwidth*self.noise)
        A_pow = 1 - self.powUserCir*channel
        B_pow = self.effPower*np.multiply(PS_users,channel)*per/(1-per)
        B_neg = B_pow.clip(max=0)
        B_pos = B_pow.clip(min=0)
        delta = 0.00000000000001 # 0.000000000000001  # Standard: 5
        UpperBound = np.divide(-A_pow, B_neg, out = np.zeros_like(-A_pow), where=B_neg!=0)
        UpperBound[UpperBound==0] = +inf
        UpperBound = np.min(UpperBound)
        UpperBound = np.minimum(UpperBound, Pmax)

        LowerBound = np.divide(-A_pow, B_pos, out=np.zeros_like(-A_pow), where=B_pos!=0)
        LowerBound = np.max(LowerBound)
        # LowerBound = np.maximum(LowerBound, 0.0000000000001)
                
        if index == 1:
            if (UpperBound - LowerBound) <= delta:
                print('Starting point (period) is BAD!')
                per, LowerBound  = self.initPeriod(Pmax, PerMax)
            else:
                print('Starting point (period) is GOOD!')
                
        if (UpperBound - LowerBound) <= delta:
            power = OldPower 
        else: 
            while UpperBound - LowerBound > delta:
                power = (UpperBound + LowerBound)/2
                if self.calcPowDerivative(power, per) > 0:
                    LowerBound = power
                else:
                    UpperBound = power
        OptObjective = self.calcObject(power, per)
        return power, OptObjective

    def searchPeriod(self, Pmax, PerMax, power, OldPer, index):
        PS_users = self.channelPsUsers
        # channel = self.apUsers/(self.subBandwidth*self.noise)
        channel = self.channelApUsers/(self.subBandwidth*self.noise)
        A_per = 1 - self.powUserCir*channel
        B_per = self.effPower*power*np.multiply(PS_users, channel)
        C_per = (B_per-A_per)
        C_neg = C_per.clip(max=0)
        C_pos = C_per.clip(min=0)
        delta = 0.00000000000001 # 0.000000000000001 # Standard: 5

        UpperBound = np.divide(-A_per, C_neg, out = np.zeros_like(-A_per), where=C_neg!=0)
        UpperBound[UpperBound==0] = +inf
        UpperBound = np.min(UpperBound)
        UpperBound = np.minimum(UpperBound, PerMax-0.0000000000001)
        
        LowerBound = np.divide(-A_per, C_pos, out = np.zeros_like(-A_per), where=C_pos!=0)
        LowerBound = LowerBound.flatten()
        LowerBound = np.delete(LowerBound, np.where(LowerBound==1))
        LowerBound = np.max(LowerBound)
        LowerBound = np.maximum(LowerBound, 10**(-2))
        # LowerBound = np.maximum(LowerBound,0.0000000000001)

        # import ipdb; ipdb.set_trace()       
        if index == 1:
            if (UpperBound - LowerBound) <= delta:
                print('Starting point (power) is BAD!')
                power, LowerBound  = self.initPower(Pmax, PerMax)
            else:
                print('Starting point (power) is GOOD!')

        if (UpperBound - LowerBound) <= delta:
            per = OldPer
        else:
            while UpperBound - LowerBound > delta:
                per = (UpperBound + LowerBound)/2
                if self.calcPerDerivative(power, per) > 0:
                    LowerBound = per
                else:
                    UpperBound = per
        OptObjective = self.calcObject(power, per)
        return per, OptObjective


if __name__ == "__main__":
    noise=10**((-90/10)-3)
    subBandwidth=78*10**3
    effPower=0.8
    powUserCir=10**(-3)
    powPScir=10**((5/10)-3)
    dinkel= 10.804231578308528 # 0.5
    binaryMatrix = np.array([[0, 0, 0, 0, 0, 0], 
                             [1, 1, 1, 1, 1, 1], 
                             [0, 0, 0, 0, 0, 0], 
                             [0, 0, 0, 0, 0, 0]])
    print('The shape of binary: ', binaryMatrix.shape)
    channelPsUsers = np.array([[1,2,3,4]]).T
    channelApUsers = np.array([[1,2,3,4,5,6],
                                 [7,8,9,10,11,12],
                                 [13,14,15,16,17,18],
                                 [19,20,21,22,23,24]])
    start_time = time.time()
    power, OptObjective = BisectionSearch(channelPsUsers,
                                          channelApUsers,
                                          binaryMatrix,
                                          noise,
                                          subBandwidth,
                                          effPower,
                                          powUserCir,
                                          powPScir,
                                          dinkel).searchPower(0.3004101332156862)
                                
    total_time = time.time() - start_time
    print('Optimization time: %f' %total_time)
    print('Power: %f' %power)
    print('Objective: %f' %OptObjective)













































