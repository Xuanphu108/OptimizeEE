import numpy as np
import matplotlib.pyplot as plt
import pandas
import matplotlib as mpl
import time
from BisectionSearch import BisectionSearch
from Alternating import Alternating

def SumRate(power,
            per,
            BinaryMatrix,
            channel_PS_users,
            channel_AP_users,
            noise,
            bandwidth_sub,
            EffPower,
            PowUserCir):
    epsilon = bandwidth_sub/(np.log(2))
    PS_users = channel_PS_users
    AP_users = np.multiply(channel_AP_users,BinaryMatrix)
    channel = AP_users/(bandwidth_sub*noise)
    SumRate = (1-per)*epsilon*np.log(1+np.multiply((EffPower*power*per*PS_users/(1-per)-PowUserCir),channel))
    SumRate = np.sum(SumRate)
    return SumRate

def Energy(power,
           per,
           BinaryMatrix,
           channel_PS_users,
           channel_AP_users,
           noise,
           bandwidth_sub,
           EffPower,
           PowUserCir,
           PowPScir):
    PS_users = channel_PS_users
    X1 = per*(power-EffPower*power*np.sum(PS_users)+PowPScir)
    X2 = (1-per)*np.sum(np.multiply((EffPower*power*per*PS_users/(1-per)-PowUserCir),BinaryMatrix))
    energy = X1 + X2
    return energy

def Dinkelbach(DinkelInit,
               PowerInit,
               PeriodInit,
               BinaryMatrixInit,
               channel_PS_users,
               channel_AP_users,
               noise,
               bandwidth_sub,
               EffPower,
               PowUserCir,
               PowPScir):
    FlagDinkel = True
    PowerOpt = PowerInit
    PeriodOpt = PeriodInit
    BinaryMatrixOpt = BinaryMatrixInit
    DinkelOpt = DinkelInit
    DinkelList = [DinkelInit]
    while (FlagDinkel):
        print('DinkelOpt: %f' %DinkelOpt)
        print('PowerOpt: %f' %PowerOpt)
        print('PeriodOpt: %f' %PeriodOpt)
        # print('BinaryMatrixOpt:')
        # print(BinaryMatrixOpt)
        BinaryMatrixOpt, PowerOpt, PeriodOpt, ObjectiveList = Alternating(PowerOpt,
                                                                          PeriodOpt,
                                                                          BinaryMatrixOpt,
                                                                          channel_PS_users,
                                                                          channel_AP_users,
                                                                          noise,
                                                                          bandwidth_sub,
                                                                          EffPower,
                                                                          PowUserCir,
                                                                          PowPScir,
                                                                          DinkelOpt)
        # import ipdb; ipdb.set_trace()
        ObjectiveOpt = ObjectiveList[-1]
        print('ObjectiveOpt: %f' %ObjectiveOpt)
        if (ObjectiveOpt < 10**(-1)):
            FlagDinkel = False
        else:
            SR = SumRate(PowerOpt,
                         PeriodOpt,
                         BinaryMatrixOpt,
                         channel_PS_users,
                         channel_AP_users,
                         noise,
                         bandwidth_sub,
                         EffPower,
                         PowUserCir)
            energy = Energy(PowerOpt,
                            PeriodOpt,
                            BinaryMatrixOpt,
                            channel_PS_users,
                            channel_AP_users,
                            noise,
                            bandwidth_sub,
                            EffPower,
                            PowUserCir,
                            PowPScir)
            DinkelOpt = SR/energy
            # print('DinkelOpt: ', DinkelOpt)
            DinkelList.append(DinkelOpt)
    return DinkelList

if __name__ == "__main__":
    noise=10**((-174/10)-3)
    bandwidth_sub=78*10**3
    EffPower=0.7
    PowUserCir=10**-6
    PowPScir=10**(-3/10)*0.001 
    
    def GetBinaryMatrixInit(no_users,no_subcarriers):
        BinaryMatrix = np.zeros((no_users,no_subcarriers))
        for i in range(no_users):
            for j in range(no_subcarriers):
                if i==j:
                    BinaryMatrix[i,j] = 1
                else:
                    BinaryMatrix[i,j] =0
        return BinaryMatrix
    BinaryMatrix = GetBinaryMatrixInit(5,64)
    """
    BinaryMatrix = np.array([[1, 1, 0, 0, 0, 0], 
                             [0, 0, 1, 0, 0, 0], 
                             [0, 0, 0, 1, 0, 1], 
                             [0, 0, 0, 0, 1, 0]])
    channel_PS_users = np.array([[1,2,3,4]]).T
    channel_AP_users = np.array([[1,2,3,4,5,6],
                                 [7,8,9,10,11,12],
                                 [13,14,15,16,17,18],
                                 [19,20,21,22,23,24]])
    """
    
    PathPS = "../Channels/ChannelSet/OFDMA/PS_Users/frame_1.csv"
    channel_PS_users = np.array([np.genfromtxt(PathPS, delimiter=',')]).T    
    PathAP = "../Channels/ChannelSet/OFDMA/AP_Users/frame_1.csv"
    channel_AP_users = np.genfromtxt(PathAP, delimiter=',')

    power = 0.5 # 0.9999
    per = 0.5 # 0.9999
    start_time = time.time()
    DinkelInit = 10
    DinkelList = Dinkelbach(DinkelInit,
                            power,
                            per,
                            BinaryMatrix,
                            channel_PS_users,
                            channel_AP_users,
                            noise,
                            bandwidth_sub,
                            EffPower,
                            PowUserCir,
                            PowPScir)
    total_time = time.time() - start_time
    # print('DinkelList:')
    # print(DinkelList)
    print('Optimization time: %f' %total_time)        

