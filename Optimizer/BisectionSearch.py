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
                 channel_PS_users,
                 channel_AP_users, 
                 BinaryMatrix,
                 noise=10**((-174/10)-3),
                 bandwidth_sub=78*10**3,
                 EffPower=0.8,
                 PowUserCir=10**(-3),
                 PowPScir=10**((5/10)-3),
                 dinkel=1):
        self.channel_PS_users = channel_PS_users
        self.channel_AP_users = channel_AP_users
        self.BinaryMatrix = BinaryMatrix
        self.noise = noise
        self.PowUserCir = PowUserCir
        self.PowPScir = PowPScir
        self.EffPower = EffPower
        self.dinkel = dinkel
        self.T = 1
        self.bandwidth_sub = bandwidth_sub
        self.epsilon = self.bandwidth_sub/(np.log(2))
        self.AP_users = np.multiply(self.channel_AP_users,self.BinaryMatrix)

    def Objective(self, power, per):
        PS_users = self.channel_PS_users
        channel = self.AP_users/(self.bandwidth_sub*self.noise)
        SumRate = (1-per)*self.epsilon*np.log(1 + np.multiply((self.EffPower*power*per*PS_users/(1-per) - self.PowUserCir),channel))
        SumRate = np.sum(SumRate)
        X1 = per*(power - self.EffPower*power*np.sum(PS_users) + self.PowPScir)
        X2 = (1-per)*np.sum(np.multiply((self.EffPower*power*per*PS_users/(1-per)-self.PowUserCir),self.BinaryMatrix))
        obj = SumRate - self.dinkel*(X1 + X2)
        return obj 

    def DerivativePower(self, power, per):
        PS_users = self.channel_PS_users
        channel = self.AP_users/(self.bandwidth_sub*self.noise)
        A_pow = 1 - self.PowUserCir*channel
        B_pow = self.EffPower*np.multiply(PS_users,channel)*per/(1-per)
        X1_PowDer = per - self.EffPower*per*np.sum(PS_users)
        X2_PowDer = np.sum(self.EffPower*per*np.multiply(PS_users,self.BinaryMatrix))
        ValPowDer = np.sum(self.epsilon*(1-per)*B_pow/(A_pow + B_pow*power)) - \
                    self.dinkel*(X1_PowDer + X2_PowDer)
        return ValPowDer

    def DerivativePeriod(self, power, per):
        PS_users = self.channel_PS_users
        channel = self.AP_users/(self.bandwidth_sub*self.noise)
        A_per = 1 - self.PowUserCir*channel
        B_per = self.EffPower*power*np.multiply(PS_users,channel)
        X1_PerDer = power - self.EffPower*power*np.sum(PS_users) + self.PowPScir
        X2_PerDer = np.sum(np.multiply((self.EffPower*power*PS_users+self.PowUserCir),self.BinaryMatrix))
        ValPerDer = np.sum(self.epsilon*np.log((1-per)/((B_per-A_per)*per+A_per)) + self.epsilon*(B_per-A_per)*(1-per)/((B_per-A_per)*per+A_per) + self.epsilon) - \
                    self.dinkel*(X1_PerDer + X2_PerDer)
        return ValPerDer

    @staticmethod 
    def split_based(mat, val):
        mask = mat!=val
        temp = np.split(mat[mask],mask.sum(1)[:-1].cumsum())
        out = np.array(list(map(list,temp)))
        return out

    @staticmethod 
    def CheckPos(mat):
        mask = mat!=val
        temp = np.split(mat[mask],mask.sum(1)[:-1].cumsum())
        out = np.array(list(map(list,temp)))
        return out

    def PowerInit(self, Pmax, PerMax):
        PS_users = self.channel_PS_users
        channel = self.channel_AP_users/(self.bandwidth_sub*self.noise)
        # Initialize power
        PowerList = np.arange(0, Pmax, 1/10000)
        A_per = 1 - self.PowUserCir*channel
        B_per = self.EffPower*np.outer(PowerList[1:],np.multiply(PS_users,channel))
        B_per = B_per.reshape(len(PowerList[1:]), channel.shape[0], channel.shape[1])
        C_per = (B_per-A_per)
        A_per = np.repeat(A_per[np.newaxis,...], len(PowerList[1:]), axis=0)

        C_neg = C_per.clip(max=0)
        C_pos = C_per.clip(min=0)

        UpperBound = np.divide(-A_per, C_neg, out = np.zeros_like(-A_per), where=C_neg!=0)
        UpperBound = UpperBound.reshape(len(PowerList[1:]), channel.shape[0]*channel.shape[1])
        UpperBound[UpperBound==0] = +inf
        UpperBound = np.min(UpperBound, axis=1)
        UpperBound = np.minimum(UpperBound, PerMax-0.000001)
        
        LowerBound = np.divide(-A_per, C_pos, out = np.zeros_like(-A_per), where=C_pos!=0)
        LowerBound = LowerBound.reshape(len(PowerList[1:]), channel.shape[0]*channel.shape[1])
        # LowerBound = self.split_based(LowerBound, 1) # Remove 1
        LowerBound = np.max(LowerBound, axis=1)
        LowerBound = np.maximum(LowerBound, 0.000001)

        TrueList = np.where(LowerBound <= UpperBound)
        if len(TrueList[0]) == 0:
            raise ValueError('Parameters is not suitable. Please adjust the parameters!')
        PowerInit = PowerList[1:][TrueList[0][int(len(TrueList)/2)]]
        LowerBoundInit = LowerBound[TrueList[0][int(len(TrueList)/2)]]
        return PowerInit, LowerBoundInit

    def PeriodInit(self, Pmax, PerMax):
        PS_users = self.channel_PS_users
        channel = self.channel_AP_users/(self.bandwidth_sub*self.noise)
        # Initialize period
        PeriodList = np.arange(0, PerMax, 1/100000)
        A_pow = 1 - self.PowUserCir*channel
        B_pow = self.EffPower*np.outer(PeriodList[1:]/(1-PeriodList[1:]),np.multiply(PS_users,channel))
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
        LowerBound = np.maximum(LowerBound, 0.000001)
        TrueList = np.where(LowerBound <= UpperBound)
        if len(TrueList[0]) == 0:
            raise ValueError('Parameters is not suitable. Please adjust the parameters!')
        # import ipdb; ipdb.set_trace()
        PeriodInit = PeriodList[1:][TrueList[0][int(len(TrueList)/2)]]
        LowerBoundInit = LowerBound[TrueList[0][int(len(TrueList)/2)]]
        return PeriodInit, LowerBoundInit
        
    def PowerSearch(self, Pmax, PerMax, per, OldPower, index):
        PS_users = self.channel_PS_users
        # channel = self.AP_users/(self.bandwidth_sub*self.noise)
        channel = self.channel_AP_users/(self.bandwidth_sub*self.noise)
        A_pow = 1 - self.PowUserCir*channel
        B_pow = self.EffPower*np.multiply(PS_users,channel)*per/(1-per)
        B_neg = B_pow.clip(max=0)
        B_pos = B_pow.clip(min=0)
        delta = 0.000000000000001 # 0.000001

        UpperBound = np.divide(-A_pow, B_neg, out = np.zeros_like(-A_pow), where=B_neg!=0)
        UpperBound[UpperBound==0] = +inf
        UpperBound = np.min(UpperBound)
        UpperBound = np.minimum(UpperBound, Pmax)

        LowerBound = np.divide(-A_pow, B_pos, out=np.zeros_like(-A_pow), where=B_pos!=0)
        LowerBound = np.max(LowerBound)
        LowerBound = np.maximum(LowerBound, 0.000001)

        if index == 1:
            if (UpperBound - LowerBound) <= delta:
                print('Starting point (period) is BAD!')
                per, LowerBound  = self.PeriodInit(Pmax, PerMax)
            else:
                print('Starting point (period) is GOOD!')
                
        if (UpperBound - LowerBound) <= delta:
            power = OldPower 
        else: 
            while UpperBound - LowerBound > delta:
                power = (UpperBound + LowerBound)/2
                if self.DerivativePower(power, per) > 0:
                    LowerBound = power
                else:
                    UpperBound = power

        OptObjective = self.Objective(power, per)
        return power, OptObjective

    def PeriodSearch(self, Pmax, PerMax, power, OldPer, index):
        # import ipdb; ipdb.set_trace()
             
        PS_users = self.channel_PS_users
        # channel = self.AP_users/(self.bandwidth_sub*self.noise)
        channel = self.channel_AP_users/(self.bandwidth_sub*self.noise)
        A_per = 1 - self.PowUserCir*channel
        B_per = self.EffPower*power*np.multiply(PS_users, channel)
        C_per = (B_per-A_per)
        C_neg = C_per.clip(max=0)
        C_pos = C_per.clip(min=0)
        delta = 0.00000000000001 # 0.000001

        UpperBound = np.divide(-A_per, C_neg, out = np.zeros_like(-A_per), where=C_neg!=0)
        UpperBound[UpperBound==0] = +inf
        UpperBound = np.min(UpperBound)
        UpperBound = np.minimum(UpperBound, PerMax-0.000001)
        
        LowerBound = np.divide(-A_per, C_pos, out = np.zeros_like(-A_per), where=C_pos!=0)
        LowerBound = LowerBound.flatten()
        LowerBound = np.delete(LowerBound, np.where(LowerBound==1))
        LowerBound = np.max(LowerBound)
        LowerBound = np.maximum(LowerBound,0.000001)

        if index == 1:
            if (UpperBound - LowerBound) <= delta:
                print('Starting point (power) is BAD!')
                power, LowerBound  = self.PowerInit(Pmax, PerMax)
            else:
                print('Starting point (power) is GOOD!')

        if (UpperBound - LowerBound) <= delta:
            per = OldPer
        else:
            while UpperBound - LowerBound > delta:
                per = (UpperBound + LowerBound)/2
                if self.DerivativePeriod(power, per) > 0:
                    LowerBound = per
                else:
                    UpperBound = per
        OptObjective = self.Objective(power, per)
        return per, OptObjective


if __name__ == "__main__":
    noise=10**((-90/10)-3)
    bandwidth_sub=78*10**3
    EffPower=0.8
    PowUserCir=10**(-3)
    PowPScir=10**((5/10)-3)
    dinkel= 10.804231578308528 # 0.5
    BinaryMatrix = np.array([[0, 0, 0, 0, 0, 0], 
                             [1, 1, 1, 1, 1, 1], 
                             [0, 0, 0, 0, 0, 0], 
                             [0, 0, 0, 0, 0, 0]])
    print('The shape of binary: ', BinaryMatrix.shape)
    channel_PS_users = np.array([[1,2,3,4]]).T
    channel_AP_users = np.array([[1,2,3,4,5,6],
                                 [7,8,9,10,11,12],
                                 [13,14,15,16,17,18],
                                 [19,20,21,22,23,24]])
    start_time = time.time()
    power, OptObjective = BisectionSearch(channel_PS_users,
                                          channel_AP_users,
                                          BinaryMatrix,
                                          noise,
                                          bandwidth_sub,
                                          EffPower,
                                          PowUserCir,
                                          PowPScir,
                                          dinkel).PowerSearch(0.3004101332156862)
                                
    total_time = time.time() - start_time
    print('Optimization time: %f' %total_time)
    print('Power: %f' %power)
    print('Objective: %f' %OptObjective)













































