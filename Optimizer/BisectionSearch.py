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
                 powPsCir=10**((5/10)-3),
                 dinkel=1):
        self.channelPsUsers = channelPsUsers
        self.channelApUsers = channelApUsers
        self.binaryMatrix = binaryMatrix
        self.noise = noise
        self.powUserCir = powUserCir
        self.powPsCir = powPsCir
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
        X1 = per*(power - self.effPower*power*np.sum(psUsers) + self.powPsCir)
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
        psUsers = self.channelPsUsers
        channel = self.apUsers/(self.subBandwidth*self.noise)
        tempX1 = 1 - self.powUserCir*channel
        tempX2 = self.effPower*power*np.multiply(psUsers,channel)
        tempY1 = power - self.effPower*power*np.sum(psUsers) + self.powPsCir
        tempY2 = np.sum(np.multiply((self.effPower*power*psUsers+self.powUserCir),self.binaryMatrix))
        periodDerivative = np.sum(self.epsilon*np.log((1-per)/((tempX2-tempX1)*per+tempX1)) + self.epsilon*(tempX2-tempX1)*(1-per)/((tempX2-tempX1)*per+tempX1) + self.epsilon) - \
                           self.dinkel*(tempY1 + tempY2)
        return periodDerivative

    @staticmethod 
    def split_based(mat, val):
        mask = mat!=val
        temp = np.split(mat[mask],mask.sum(1)[:-1].cumsum())
        out = np.array(list(map(list,temp)))
        return out

    def initPower(self, pMax, perMax):
        psUsers = self.channelPsUsers
        channel = self.channelApUsers/(self.subBandwidth*self.noise)
        # Initialize power
        powerList = np.arange(0, pMax, 1/1000000)
        tempA = 1 - self.powUserCir*channel
        tempB = self.effPower*np.outer(powerList[1:],np.multiply(psUsers,channel))
        tempB = tempB.reshape(len(powerList[1:]), channel.shape[0], channel.shape[1])
        tempC = (tempB-tempA)
        tempA = np.repeat(tempA[np.newaxis,...], len(powerList[1:]), axis=0)

        negC = tempC.clip(max=0)
        posC = tempC.clip(min=0)

        upperBound = np.divide(-tempA, negC, out = np.zeros_like(-tempA), where=negC!=0)
        upperBound = upperBound.reshape(len(powerList[1:]), channel.shape[0]*channel.shape[1])
        upperBound[upperBound==0] = +inf
        upperBound = np.min(upperBound, axis=1)
        upperBound = np.minimum(upperBound, perMax-0.0000000000001)
        
        lowerBound = np.divide(-tempA, posC, out = np.zeros_like(-tempA), where=posC!=0)
        lowerBound = lowerBound.reshape(len(powerList[1:]), channel.shape[0]*channel.shape[1])
        # lowerBound = self.split_based(lowerBound, 1) # Remove 1
        lowerBound = np.max(lowerBound, axis=1)
        lowerBound = np.maximum(lowerBound, 0.0000000000001)

        trueList = np.where(lowerBound <= upperBound)
        if len(trueList[0]) == 0:
            raise ValueError('Parameters is not suitable. Please adjust the parameters!')
        initPower = powerList[1:][trueList[0][int(len(trueList)/2)]]
        initLowerBound = lowerBound[trueList[0][int(len(trueList)/2)]]
        return initPower, initLowerBound

    def initPeriod(self, pMax, perMax):
        psUsers = self.channelPsUsers
        channel = self.channelApUsers/(self.subBandwidth*self.noise)
        # Initialize period
        periodList = np.arange(0, perMax, 1/1000000)
        tempA = 1 - self.powUserCir*channel
        tempB = self.effPower*np.outer(periodList[1:]/(1-periodList[1:]),np.multiply(psUsers,channel))
        tempB = tempB.reshape(len(periodList[1:]), channel.shape[0], channel.shape[1])
        tempA = np.repeat(tempA[np.newaxis,...], len(periodList[1:]), axis=0)

        negB = tempB.clip(max=0)
        posB = tempB.clip(min=0)

        upperBound = np.divide(-tempA, negB, out = np.zeros_like(-tempA), where=B_neg!=0)
        upperBound = upperBound.reshape(len(periodList[1:]), channel.shape[0]*channel.shape[1])
        upperBound[upperBound==0] = +inf
        upperBound = np.min(upperBound, axis=1)
        upperBound = np.minimum(upperBound, pMax)
        
        lowerBound = np.divide(-tempA, posB, out=np.zeros_like(-tempA), where=posB!=0)
        lowerBound = lowerBound.reshape(len(periodList[1:]), channel.shape[0]*channel.shape[1])
        lowerBound = np.max(lowerBound, axis=1)
        lowerBound = np.maximum(lowerBound, 0.0000000000001)
        trueList = np.where(lowerBound <= upperBound)
        if len(trueList[0]) == 0:
            raise ValueError('Parameters is not suitable. Please adjust the parameters!')
        # import ipdb; ipdb.set_trace()
        initPeriod = periodList[1:][trueList[0][int(len(trueList)/2)]]
        initLowerBound = lowerBound[trueList[0][int(len(trueList)/2)]]
        return initPeriod, initLowerBound
        
    def searchPower(self, pMax, perMax, per, oldPower, index):
        psUsers = self.channelPsUsers
        # channel = self.apUsers/(self.subBandwidth*self.noise)
        channel = self.channelApUsers/(self.subBandwidth*self.noise)
        tempA = 1 - self.powUserCir*channel
        tempB = self.effPower*np.multiply(psUsers,channel)*per/(1-per)
        negB = tempB.clip(max=0)
        posB = tempB.clip(min=0)
        delta = 0.00000000000001 # 0.000000000000001  # Standard: 5
        upperBound = np.divide(-tempA, negB, out = np.zeros_like(-tempA), where=negB!=0)
        upperBound[upperBound==0] = +inf
        upperBound = np.min(upperBound)
        upperBound = np.minimum(upperBound, pMax)

        lowerBound = np.divide(-tempA, posB, out=np.zeros_like(-tempA), where=posB!=0)
        lowerBound = np.max(lowerBound)
        # lowerBound = np.maximum(lowerBound, 0.0000000000001)
                
        if index == 1:
            if (upperBound - lowerBound) <= delta:
                print('Starting point (period) is BAD!')
                per, lowerBound  = self.initPeriod(pMax, perMax)
            else:
                print('Starting point (period) is GOOD!')
                
        if (upperBound - lowerBound) <= delta:
            power = oldPower 
        else: 
            while upperBound - lowerBound > delta:
                power = (upperBound + lowerBound)/2
                if self.calcPowDerivative(power, per) > 0:
                    lowerBound = power
                else:
                    upperBound = power
        optObjective = self.calcObject(power, per)
        return power, optObjective

    def searchPeriod(self, pMax, perMax, power, oldPer, index):
        psUsers = self.channelPsUsers
        # channel = self.apUsers/(self.subBandwidth*self.noise)
        channel = self.channelApUsers/(self.subBandwidth*self.noise)
        tempA = 1 - self.powUserCir*channel
        tempB = self.effPower*power*np.multiply(psUsers, channel)
        tempC = (tempB-tempA)
        negC = tempC.clip(max=0)
        posC = tempC.clip(min=0)
        delta = 0.00000000000001 # 0.000000000000001 # Standard: 5

        upperBound = np.divide(-tempA, negC, out = np.zeros_like(-tempA), where=negC!=0)
        upperBound[upperBound==0] = +inf
        upperBound = np.min(upperBound)
        upperBound = np.minimum(upperBound, perMax-0.0000000000001)
        
        lowerBound = np.divide(-tempA, posC, out = np.zeros_like(-tempA), where=posC!=0)
        lowerBound = lowerBound.flatten()
        lowerBound = np.delete(lowerBound, np.where(lowerBound==1))
        lowerBound = np.max(lowerBound)
        lowerBound = np.maximum(lowerBound, 10**(-2))
        # lowerBound = np.maximum(lowerBound,0.0000000000001)

        # import ipdb; ipdb.set_trace()       
        if index == 1:
            if (upperBound - lowerBound) <= delta:
                print('Starting point (power) is BAD!')
                power, lowerBound  = self.initPower(pMax, perMax)
            else:
                print('Starting point (power) is GOOD!')

        if (upperBound - lowerBound) <= delta:
            per = oldPer
        else:
            while upperBound - lowerBound > delta:
                per = (upperBound + lowerBound)/2
                if self.calcPerDerivative(power, per) > 0:
                    lowerBound = per
                else:
                    upperBound = per
        optObjective = self.calcObject(power, per)
        return per, optObjective


if __name__ == "__main__":
    noise=10**((-90/10)-3)
    subBandwidth=78*10**3
    effPower=0.8
    powUserCir=10**(-3)
    powPsCir=10**((5/10)-3)
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
    power, optObjective = BisectionSearch(channelPsUsers,
                                          channelApUsers,
                                          binaryMatrix,
                                          noise,
                                          subBandwidth,
                                          effPower,
                                          powUserCir,
                                          powPsCir,
                                          dinkel).searchPower(0.3004101332156862)
                                
    total_time = time.time() - start_time
    print('Optimization time: %f' %total_time)
    print('Power: %f' %power)
    print('Objective: %f' %optObjective)













































