import numpy as np
import matplotlib.pyplot as plt
import pandas
import matplotlib as mpl
import time
import natsort
import os
from BisectionSearch import BisectionSearch
from Alternating import alterAlgorithm, alterSumRate, alterFixedPeriod

def calcSumRate(power,
                per,
                binaryMatrix,
                channelPsUsers,
                channelApUsers,
                noise,
                subBandwidth,
                effPower,
                powUserCir):
    epsilon = subBandwidth/(np.log(2))
    psUsers = channelPsUsers
    apUsers = np.multiply(channelApUsers,binaryMatrix)
    channel = apUsers/(subBandwidth*noise)
    sumRate = (1-per)*epsilon*np.log(1+np.multiply((effPower*power*per*psUsers/(1-per)-powUserCir),channel))
    sumRate = np.sum(sumRate)
    return sumRate

def calcEnergy(power,
               per,
               binaryMatrix,
               channelPsUsers,
               channelApUsers,
               noise,
               subBandwidth,
               effPower,
               powUserCir,
               powPsCir):
    psUsers = channelPsUsers
    temp1 = per*(power-effPower*power*np.sum(psUsers)+powPsCir)
    temp2 = (1-per)*np.sum(np.multiply((effPower*power*per*psUsers/(1-per)-powUserCir),binaryMatrix))
    energy = temp1 + temp2
    return energy

def Dinkelbach(DinkelInit,
               initPower,
               PeriodInit,
               initBinaryMatrix,
               channelPsUsers,
               channelApUsers,
               noise,
               subBandwidth,
               effPower,
               powUserCir,
               powPsCir,
               pMax,
               perMax):
    flagDinkel = True
    optPower = initPower
    PeriodOpt = PeriodInit
    optBinaryMatrix = initBinaryMatrix
    DinkelOpt = DinkelInit
    DinkelList = [DinkelInit]
    while (flagDinkel):
        print('optPower: %f' %optPower)
        print('PeriodOpt: %f' %PeriodOpt)
        # print('optBinaryMatrix:')
        # print(optBinaryMatrix)
        optBinaryMatrix, optPower, PeriodOpt, ObjectiveList = alterAlgorithm(optPower,
                                                                             PeriodOpt,
                                                                             optBinaryMatrix,
                                                                             channelPsUsers,
                                                                             channelApUsers,
                                                                             noise,
                                                                             subBandwidth,
                                                                             effPower,
                                                                             powUserCir,
                                                                             powPsCir,
                                                                             pMax,
                                                                             perMax,
                                                                             DinkelOpt)
        SR = calcSumRate(optPower,
                         PeriodOpt,
                         optBinaryMatrix,
                         channelPsUsers,
                         channelApUsers,
                         noise,
                         subBandwidth,
                         effPower,
                         powUserCir)
        energy = calcEnergy(optPower,
                            PeriodOpt,
                            optBinaryMatrix,
                            channelPsUsers,
                            channelApUsers,
                            noise,
                            subBandwidth,
                            effPower,
                            powUserCir,
                            powPsCir)
        # import ipdb; ipdb.set_trace()
        ObjectiveOpt = ObjectiveList[-1]
        print('ObjectiveOpt: %f' %ObjectiveOpt)
        if (ObjectiveOpt < 10**(-1)):
            flagDinkel = False
            DinkelOpt = SR/energy
            print('DinkelOpt: ', DinkelOpt)
            DinkelList.append(DinkelOpt)
        else:
            DinkelOpt = SR/energy
            print('DinkelOpt: ', DinkelOpt)
            DinkelList.append(DinkelOpt)
    return DinkelList, optPower, PeriodOpt

def DinkelbachFixedPeriod(DinkelInit,
                          initPower,
                          PeriodInit,
                          initBinaryMatrix,
                          channelPsUsers,
                          channelApUsers,
                          noise,
                          subBandwidth,
                          effPower,
                          powUserCir,
                          powPsCir,
                          pMax,
                          perMax):
    print('P_MAX!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!: ', pMax)
    flagDinkel = True
    optPower = initPower
    optBinaryMatrix = initBinaryMatrix
    DinkelOpt = DinkelInit
    DinkelList = [DinkelInit]
    while (flagDinkel):
        print('optPower: %f' %optPower)
        # print('optBinaryMatrix:')
        # print(optBinaryMatrix)
        result = AlternatingFixedPeriod(optPower,
                                        PeriodInit,
                                        optBinaryMatrix,
                                        channelPsUsers,
                                        channelApUsers,
                                        noise,
                                        subBandwidth,
                                        effPower,
                                        powUserCir,
                                        powPsCir,
                                        pMax,
                                        perMax,
                                        DinkelOpt)
        # optBinaryMatrix, optPower, ObjectiveList
        if (result != None):
            optBinaryMatrix = result[0]
            optPower = result[1] 
            ObjectiveList = result[2]
            SR = calcSumRate(optPower,
                         PeriodInit,
                         optBinaryMatrix,
                         channelPsUsers,
                         channelApUsers,
                         noise,
                         subBandwidth,
                         effPower,
                         powUserCir)
            energy = calcEnergy(optPower,
                            PeriodInit,
                            optBinaryMatrix,
                            channelPsUsers,
                            channelApUsers,
                            noise,
                            subBandwidth,
                            effPower,
                            powUserCir,
                            powPsCir)
            ObjectiveOpt = ObjectiveList[-1]
            print('ObjectiveOpt: %f' %ObjectiveOpt)
            if (ObjectiveOpt < 10**(-1)):
                flagDinkel = False
                DinkelOpt = SR/energy
                print('DinkelOpt: ', DinkelOpt)
                DinkelList.append(DinkelOpt)
            else:
                DinkelOpt = SR/energy
                print('DinkelOpt: ', DinkelOpt)
                DinkelList.append(DinkelOpt)
        else:
            return None
    return DinkelList, optPower

def DinkelbachFixedPower(DinkelInit,
                         initPower,
                         PeriodInit,
                         initBinaryMatrix,
                         channelPsUsers,
                         channelApUsers,
                         noise,
                         subBandwidth,
                         effPower,
                         powUserCir,
                         powPsCir,
                         perMax):
    flagDinkel = True
    PeriodOpt = PeriodInit
    optBinaryMatrix = initBinaryMatrix
    DinkelOpt = DinkelInit
    DinkelList = [DinkelInit]
    while (flagDinkel):
        print('PeriodOpt: %f' %PeriodOpt)
        # print('optBinaryMatrix:')
        # print(optBinaryMatrix)
        optBinaryMatrix, PeriodOpt, ObjectiveList = AlternatingcalcSumRate(initPower,
                                                                       PeriodOpt,
                                                                       optBinaryMatrix,
                                                                       channelPsUsers,
                                                                       channelApUsers,
                                                                       noise,
                                                                       subBandwidth,
                                                                       effPower,
                                                                       powUserCir,
                                                                       powPsCir,
                                                                       pMax,
                                                                       perMax,
                                                                       DinkelOpt)
        SR = calcSumRate(initPower,
                     PeriodOpt,
                     optBinaryMatrix,
                     channelPsUsers,
                     channelApUsers,
                     noise,
                     subBandwidth,
                     effPower,
                     powUserCir)
        energy = calcEnergy(initPower,
                        PeriodOpt,
                        optBinaryMatrix,
                        channelPsUsers,
                        channelApUsers,
                        noise,
                        subBandwidth,
                        effPower,
                        powUserCir,
                        powPsCir)
        # import ipdb; ipdb.set_trace()
        ObjectiveOpt = ObjectiveList[-1]
        print('ObjectiveOpt: %f' %ObjectiveOpt)
        if (ObjectiveOpt < 10**(-1)):
            flagDinkel = False
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
    subBandwidth=78*10**3
    effPower=0.7
    powUserCir=10**-7 # 10**-8
    powPsCir=10**(-3/10)*0.001 
    
    def GetinitBinaryMatrix(no_users,no_subcarriers):
        binaryMatrix = np.zeros((no_users,no_subcarriers))
        for i in range(no_users):
            for j in range(no_subcarriers):
                if i==j:
                    binaryMatrix[i,j] = 1
                else:
                    binaryMatrix[i,j] =0
        return binaryMatrix
    binaryMatrix = GetinitBinaryMatrix(5,64)
    """
    binaryMatrix = np.array([[1, 1, 0, 0, 0, 0], 
                             [0, 0, 1, 0, 0, 0], 
                             [0, 0, 0, 1, 0, 1], 
                             [0, 0, 0, 0, 1, 0]])
    channelPsUsers = np.array([[1,2,3,4]]).T
    channelApUsers = np.array([[1,2,3,4,5,6],
                                 [7,8,9,10,11,12],
                                 [13,14,15,16,17,18],
                                 [19,20,21,22,23,24]])
    """
    
    PathPS = "../Channels/ChannelSet/OFDMA/PS_Users/frame_3.csv"
    channelPsUsers = np.array([np.genfromtxt(PathPS, delimiter=',')]).T    
    PathAP = "../Channels/ChannelSet/OFDMA/AP_Users/frame_3.csv"
    channelApUsers = np.genfromtxt(PathAP, delimiter=',')

    power = 0.99999 # 0.5
    per = 0.99999 # 0.5

    start_time = time.time()
    DinkelInit = 10
    pMax = 1
    perMax = 1

    pMaxList = [0.1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 40]
    EE_Din_List = []
    optPower_List = []
    PeriodOpt_List = []

    EE_SR_List = []

    EE_FixedPeriod_List = []
    PoweOpt_FixedPeriod_List = []

    EE_FixedPower_List = []

    EE_Din_List_Average = np.zeros([1,len(pMaxList)])
    EE_SR_List_Average = np.zeros([1,len(pMaxList)])
    EE_FixedPeriod_List_Average = np.zeros([1,len(pMaxList)])
    EE_FixedPower_List_Average = np.zeros([1,len(pMaxList)])
    for i in range(0,1):
        PathPSs = "../Channels/ChannelSet/OFDMA/PS_Users/" 
        PathAPs = "../Channels/ChannelSet/OFDMA/AP_Users/"
        list_file_PSs = os.listdir(PathPSs)
        list_file_APs = os.listdir(PathAPs)
        list_file_order_PSs = natsort.natsorted(list_file_PSs, reverse=False)
        channelPsUsers = np.array([np.genfromtxt(PathPSs + list_file_order_PSs[11], delimiter=',')]).T
        list_file_order_APs = natsort.natsorted(list_file_APs, reverse=False)
        channelApUsers = np.genfromtxt(PathAPs + list_file_order_APs[11], delimiter=',')
        for pMax in pMaxList:
            """
            print('PROPOSED SCHEME')
            EE_Din, optPower, PeriodOpt = Dinkelbach(DinkelInit,
                                                     power,
                                                     per,
                                                     binaryMatrix,
                                                     channelPsUsers,
                                                     channelApUsers,
                                                     noise,
                                                     subBandwidth,
                                                     effPower,
                                                     powUserCir,
                                                     powPsCir,
                                                     pMax,
                                                     perMax)
            EE_Din_List.append(EE_Din[-1])
            optPower_List.append(optPower)
            PeriodOpt_List.append(PeriodOpt)
            """

            """
            SR_opt = AlternatingcalcSumRate(pMax,
                                        per,
                                        binaryMatrix,
                                        channelPsUsers,
                                        channelApUsers,
                                        noise,
                                        subBandwidth,
                                        effPower,
                                        powUserCir,
                                        powPsCir,
                                        pMax,
                                        perMax,
                                        0)
            EE = calcEnergy(pMax,
                        SR_opt[1],
                        binaryMatrix,
                        channelPsUsers,
                        channelApUsers,
                        noise,
                        subBandwidth,
                        effPower,
                        powUserCir,
                        powPsCir)
            EE_SR = SR_opt[-1][-1]/EE
            EE_SR_List.append(EE_SR)
            """
            
            
            result_FixedPeriod = DinkelbachFixedPeriod(DinkelInit,
                                                       power,
                                                       0.8, # Fixed period
                                                       binaryMatrix,
                                                       channelPsUsers,
                                                       channelApUsers,
                                                       noise,
                                                       subBandwidth,
                                                       effPower,
                                                       powUserCir,
                                                       powPsCir,
                                                       pMax,
                                                       perMax)
            if (result_FixedPeriod != None):
                EE_FixedPeriod = result_FixedPeriod[0]
                optPower_FixedPeriod = result_FixedPeriod[-1]
                EE_FixedPeriod_List.append(EE_FixedPeriod[-1])
                PoweOpt_FixedPeriod_List.append(optPower_FixedPeriod)
            else:
                EE_FixedPeriod_List.append(0)
                PoweOpt_FixedPeriod_List.append(0)
            
            
            """
            print('FIXED POWER SCHEME')
            EE_FixedPower = DinkelbachFixedPower(DinkelInit,
                                                 pMax,
                                                 per,
                                                 binaryMatrix,
                                                 channelPsUsers,
                                                 channelApUsers,
                                                 noise,
                                                 subBandwidth,
                                                 effPower,
                                                 powUserCir,
                                                 powPsCir,
                                                 perMax)
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
    # print('Popt List: ', optPower_List)
    # print('Period List: ', PeriodOpt_List)

    # print('EE_SR_List: ', EE_SR_List_Average/100)
    
    print('EE_FixedPeriod_List: ', EE_FixedPeriod_List_Average/100)
    print('Popt List: ', PoweOpt_FixedPeriod_List)

    # print('EE_FixedPower_List: ', EE_FixedPower_List_Average/100)
    