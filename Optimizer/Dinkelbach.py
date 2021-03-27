import numpy as np
import matplotlib.pyplot as plt
import pandas
import matplotlib as mpl
import time
import natsort
import os
from BisectionSearch import BisectionSearch
from Alternating import Alternating
from Alternating import AlternatingSumRate, AlternatingFixedPeriod


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
               PowPScir,
               Pmax,
               PerMax):
    FlagDinkel = True
    PowerOpt = PowerInit
    PeriodOpt = PeriodInit
    BinaryMatrixOpt = BinaryMatrixInit
    DinkelOpt = DinkelInit
    DinkelList = [DinkelInit]
    while (FlagDinkel):
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
                                                                          Pmax,
                                                                          PerMax,
                                                                          DinkelOpt)
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
        # import ipdb; ipdb.set_trace()
        ObjectiveOpt = ObjectiveList[-1]
        print('ObjectiveOpt: %f' %ObjectiveOpt)
        if (ObjectiveOpt < 10**(-1)):
            FlagDinkel = False
            DinkelOpt = SR/energy
            print('DinkelOpt: ', DinkelOpt)
            DinkelList.append(DinkelOpt)
        else:
            DinkelOpt = SR/energy
            print('DinkelOpt: ', DinkelOpt)
            DinkelList.append(DinkelOpt)
    return DinkelList, PowerOpt, PeriodOpt

def DinkelbachFixedPeriod(DinkelInit,
                          PowerInit,
                          PeriodInit,
                          BinaryMatrixInit,
                          channel_PS_users,
                          channel_AP_users,
                          noise,
                          bandwidth_sub,
                          EffPower,
                          PowUserCir,
                          PowPScir,
                          Pmax,
                          PerMax):
    print('P_MAX!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!: ', Pmax)
    FlagDinkel = True
    PowerOpt = PowerInit
    BinaryMatrixOpt = BinaryMatrixInit
    DinkelOpt = DinkelInit
    DinkelList = [DinkelInit]
    while (FlagDinkel):
        print('PowerOpt: %f' %PowerOpt)
        # print('BinaryMatrixOpt:')
        # print(BinaryMatrixOpt)
        result = AlternatingFixedPeriod(PowerOpt,
                                        PeriodInit,
                                        BinaryMatrixOpt,
                                        channel_PS_users,
                                        channel_AP_users,
                                        noise,
                                        bandwidth_sub,
                                        EffPower,
                                        PowUserCir,
                                        PowPScir,
                                        Pmax,
                                        PerMax,
                                        DinkelOpt)
        # BinaryMatrixOpt, PowerOpt, ObjectiveList
        if (result != None):
            BinaryMatrixOpt = result[0]
            PowerOpt = result[1] 
            ObjectiveList = result[2]
            SR = SumRate(PowerOpt,
                         PeriodInit,
                         BinaryMatrixOpt,
                         channel_PS_users,
                         channel_AP_users,
                         noise,
                         bandwidth_sub,
                         EffPower,
                         PowUserCir)
            energy = Energy(PowerOpt,
                            PeriodInit,
                            BinaryMatrixOpt,
                            channel_PS_users,
                            channel_AP_users,
                            noise,
                            bandwidth_sub,
                            EffPower,
                            PowUserCir,
                            PowPScir)
            ObjectiveOpt = ObjectiveList[-1]
            print('ObjectiveOpt: %f' %ObjectiveOpt)
            if (ObjectiveOpt < 10**(-1)):
                FlagDinkel = False
                DinkelOpt = SR/energy
                print('DinkelOpt: ', DinkelOpt)
                DinkelList.append(DinkelOpt)
            else:
                DinkelOpt = SR/energy
                print('DinkelOpt: ', DinkelOpt)
                DinkelList.append(DinkelOpt)
        else:
            return None
    return DinkelList, PowerOpt

def DinkelbachFixedPower(DinkelInit,
                         PowerInit,
                         PeriodInit,
                         BinaryMatrixInit,
                         channel_PS_users,
                         channel_AP_users,
                         noise,
                         bandwidth_sub,
                         EffPower,
                         PowUserCir,
                         PowPScir,
                         PerMax):
    FlagDinkel = True
    PeriodOpt = PeriodInit
    BinaryMatrixOpt = BinaryMatrixInit
    DinkelOpt = DinkelInit
    DinkelList = [DinkelInit]
    while (FlagDinkel):
        print('PeriodOpt: %f' %PeriodOpt)
        # print('BinaryMatrixOpt:')
        # print(BinaryMatrixOpt)
        BinaryMatrixOpt, PeriodOpt, ObjectiveList = AlternatingSumRate(PowerInit,
                                                                       PeriodOpt,
                                                                       BinaryMatrixOpt,
                                                                       channel_PS_users,
                                                                       channel_AP_users,
                                                                       noise,
                                                                       bandwidth_sub,
                                                                       EffPower,
                                                                       PowUserCir,
                                                                       PowPScir,
                                                                       Pmax,
                                                                       PerMax,
                                                                       DinkelOpt)
        SR = SumRate(PowerInit,
                     PeriodOpt,
                     BinaryMatrixOpt,
                     channel_PS_users,
                     channel_AP_users,
                     noise,
                     bandwidth_sub,
                     EffPower,
                     PowUserCir)
        energy = Energy(PowerInit,
                        PeriodOpt,
                        BinaryMatrixOpt,
                        channel_PS_users,
                        channel_AP_users,
                        noise,
                        bandwidth_sub,
                        EffPower,
                        PowUserCir,
                        PowPScir)
        # import ipdb; ipdb.set_trace()
        ObjectiveOpt = ObjectiveList[-1]
        print('ObjectiveOpt: %f' %ObjectiveOpt)
        if (ObjectiveOpt < 10**(-1)):
            FlagDinkel = False
            DinkelOpt = SR/energy
            print('DinkelOpt: ', DinkelOpt)
            DinkelList.append(DinkelOpt)
        else:
            DinkelOpt = SR/energy
            print('DinkelOpt: ', DinkelOpt)
            DinkelList.append(DinkelOpt)
    return DinkelList

if __name__ == "__main__":
    noise=10**((-174/10)-3)
    bandwidth_sub=78*10**3
    EffPower=0.7
    PowUserCir=10**-7 # 10**-8
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
    
    PathPS = "../Channels/ChannelSet/OFDMA/PS_Users/frame_3.csv"
    channel_PS_users = np.array([np.genfromtxt(PathPS, delimiter=',')]).T    
    PathAP = "../Channels/ChannelSet/OFDMA/AP_Users/frame_3.csv"
    channel_AP_users = np.genfromtxt(PathAP, delimiter=',')

    power = 0.99999 # 0.5
    per = 0.99999 # 0.5

    start_time = time.time()
    DinkelInit = 10
    Pmax = 1
    PerMax = 1

    PmaxList = [0.1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 40]
    EE_Din_List = []
    PowerOpt_List = []
    PeriodOpt_List = []

    EE_SR_List = []

    EE_FixedPeriod_List = []
    PoweOpt_FixedPeriod_List = []

    EE_FixedPower_List = []

    EE_Din_List_Average = np.zeros([1,len(PmaxList)])
    EE_SR_List_Average = np.zeros([1,len(PmaxList)])
    EE_FixedPeriod_List_Average = np.zeros([1,len(PmaxList)])
    EE_FixedPower_List_Average = np.zeros([1,len(PmaxList)])
    for i in range(0,1):
        PathPSs = "../Channels/ChannelSet/OFDMA/PS_Users/" 
        PathAPs = "../Channels/ChannelSet/OFDMA/AP_Users/"
        list_file_PSs = os.listdir(PathPSs)
        list_file_APs = os.listdir(PathAPs)
        list_file_order_PSs = natsort.natsorted(list_file_PSs, reverse=False)
        channel_PS_users = np.array([np.genfromtxt(PathPSs + list_file_order_PSs[11], delimiter=',')]).T
        list_file_order_APs = natsort.natsorted(list_file_APs, reverse=False)
        channel_AP_users = np.genfromtxt(PathAPs + list_file_order_APs[11], delimiter=',')
        for Pmax in PmaxList:
            """
            print('PROPOSED SCHEME')
            EE_Din, PowerOpt, PeriodOpt = Dinkelbach(DinkelInit,
                                                     power,
                                                     per,
                                                     BinaryMatrix,
                                                     channel_PS_users,
                                                     channel_AP_users,
                                                     noise,
                                                     bandwidth_sub,
                                                     EffPower,
                                                     PowUserCir,
                                                     PowPScir,
                                                     Pmax,
                                                     PerMax)
            EE_Din_List.append(EE_Din[-1])
            PowerOpt_List.append(PowerOpt)
            PeriodOpt_List.append(PeriodOpt)
            """

            """
            SR_opt = AlternatingSumRate(Pmax,
                                        per,
                                        BinaryMatrix,
                                        channel_PS_users,
                                        channel_AP_users,
                                        noise,
                                        bandwidth_sub,
                                        EffPower,
                                        PowUserCir,
                                        PowPScir,
                                        Pmax,
                                        PerMax,
                                        0)
            EE = Energy(Pmax,
                        SR_opt[1],
                        BinaryMatrix,
                        channel_PS_users,
                        channel_AP_users,
                        noise,
                        bandwidth_sub,
                        EffPower,
                        PowUserCir,
                        PowPScir)
            EE_SR = SR_opt[-1][-1]/EE
            EE_SR_List.append(EE_SR)
            """
            
            
            result_FixedPeriod = DinkelbachFixedPeriod(DinkelInit,
                                                       power,
                                                       0.8, # Fixed period
                                                       BinaryMatrix,
                                                       channel_PS_users,
                                                       channel_AP_users,
                                                       noise,
                                                       bandwidth_sub,
                                                       EffPower,
                                                       PowUserCir,
                                                       PowPScir,
                                                       Pmax,
                                                       PerMax)
            if (result_FixedPeriod != None):
                EE_FixedPeriod = result_FixedPeriod[0]
                PowerOpt_FixedPeriod = result_FixedPeriod[-1]
                EE_FixedPeriod_List.append(EE_FixedPeriod[-1])
                PoweOpt_FixedPeriod_List.append(PowerOpt_FixedPeriod)
            else:
                EE_FixedPeriod_List.append(0)
                PoweOpt_FixedPeriod_List.append(0)
            
            
            """
            print('FIXED POWER SCHEME')
            EE_FixedPower = DinkelbachFixedPower(DinkelInit,
                                                 Pmax,
                                                 per,
                                                 BinaryMatrix,
                                                 channel_PS_users,
                                                 channel_AP_users,
                                                 noise,
                                                 bandwidth_sub,
                                                 EffPower,
                                                 PowUserCir,
                                                 PowPScir,
                                                 PerMax)
            EE_FixedPower_List.append(EE_FixedPower[-1]) 
            """
        """          
        EE_Din_List = np.array(EE_Din_List)
        EE_Din_List_Average = EE_Din_List_Average + EE_Din_List
        """

        """
        EE_SR_List = np.array(EE_SR_List)
        EE_SR_List_Average = EE_SR_List_Average + EE_SR_List
        """

        
        EE_FixedPeriod_List = np.array(EE_FixedPeriod_List)
        EE_FixedPeriod_List_Average = EE_FixedPeriod_List_Average + EE_FixedPeriod_List 
        

        """
        EE_FixedPower_List = np.array(EE_FixedPower_List)
        EE_FixedPower_List_Average = EE_FixedPower_List_Average + EE_FixedPower_List
        """
        
        # EE_Din_List = []
        # EE_SR_List = []
        EE_FixedPeriod_List = []
        # EE_FixedPower_List = []
        
        
    # print('EE_Din_List: ', EE_Din_List_Average)
    # print('Popt List: ', PowerOpt_List)
    # print('Period List: ', PeriodOpt_List)

    # print('EE_SR_List: ', EE_SR_List_Average/100)
    
    print('EE_FixedPeriod_List: ', EE_FixedPeriod_List_Average/100)
    print('Popt List: ', PoweOpt_FixedPeriod_List)

    # print('EE_FixedPower_List: ', EE_FixedPower_List_Average/100)
    