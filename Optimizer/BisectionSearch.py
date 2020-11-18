import numpy as np
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

    def CheckFeasible(self, power, per):

    def InitPoint(self, per, power):


    def PowerSearch(self, per, OldPower, index):
        PS_users = self.channel_PS_users
        # channel = self.AP_users/(self.bandwidth_sub*self.noise)
        channel = self.channel_AP_users/(self.bandwidth_sub*self.noise)
        A_pow = 1 - self.PowUserCir*channel
        B_pow = self.EffPower*np.multiply(PS_users,channel)*per/(1-per)
        delta = 0.000000000000001 # 0.000001
        UpperBound = 3
        # LowerBound = 0
        LowerBound = np.divide(-A_pow, B_pow, out=np.zeros_like(-A_pow), where=B_pow!=0)
        LowerBound = np.max(LowerBound)
        if index == 1:

        else:
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

    def PeriodSearch(self, power, OldPer, index):
        """
        PL = []
        ss = 0
        for i in range(9999):
            PL.append(ss)
            ss = ss + 1/10000
        for p in PL[1:]:
            PS_users = self.channel_PS_users
            channel = self.channel_AP_users/(self.bandwidth_sub*self.noise)
            A_per = 1 - self.PowUserCir*channel
            B_per = self.EffPower*p*np.multiply(PS_users,channel)
            C_per = (B_per-A_per)
            UpperBound = 1 - 0.000001

            LowerBound = np.divide(-A_per, C_per, out = np.zeros_like(-A_per), where=C_per!=0)
            LowerBound = LowerBound.flatten()
            LowerBound = np.delete(LowerBound, np.where(LowerBound==1))
            LowerBound = np.max(LowerBound)
            LowerBound = np.maximum(LowerBound,0.000001)
            count = 0
            power_pos = []
            if (UpperBound-LowerBound) > 0:
                count = count + 1
                power_pos.append(p)
        """
        # import ipdb; ipdb.set_trace()
             
        PS_users = self.channel_PS_users
        # channel = self.AP_users/(self.bandwidth_sub*self.noise)
        channel = self.channel_AP_users/(self.bandwidth_sub*self.noise)
        A_per = 1 - self.PowUserCir*channel
        B_per = self.EffPower*power*np.multiply(PS_users,channel)
        C_per = (B_per-A_per)
        delta = 0.00000000000001 # 0.000001
        UpperBound = 1 - 0.000001
        
        # LowerBound = 0
        LowerBound = np.divide(-A_per, C_per, out = np.zeros_like(-A_per), where=C_per!=0)
        LowerBound = LowerBound.flatten()
        LowerBound = np.delete(LowerBound, np.where(LowerBound==1))
        LowerBound = np.max(LowerBound)
        LowerBound = np.maximum(LowerBound,0.000001)

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













































