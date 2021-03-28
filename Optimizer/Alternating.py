import numpy as np
from numpy import inf
import matplotlib.pyplot as plt
import pandas
import matplotlib as mpl
import time
from BisectionSearch import BisectionSearch

def checkInit(initbinaryMatrix, 
              initPower, 
              initPeriod,
              channelPsUsers,
              channelApUsers,
              noise,
              subBandwidth,
              effPower,
              powUserCir):
    ###### - Check binary matrix - #######
    for i in range(0, initbinaryMatrix.shape[1]):
        if ((initbinaryMatrix[:,i] == 1).sum() > 1):
            print('Binary matrix is not satisfy. Please try again!')
    ###### - Check power - #######
    if (initPower >= 1) or (initPower <= 0):
        print('Power is not statisfy. Please try again!')
    ###### - Check period - #######
    if (initPeriod >= 1) or (initPeriod <= 0):
        print('Period is not statisfy. Please try again!')

def allocateResource(power,  
                     per,
                     channelPsUsers,
                     channelApUsers,
                     noise,
                     subBandwidth,
                     effPower,
                     powUserCir,
                     powPsCir,
                     dinkel):
    epsilon = subBandwidth/(np.log(2))
    psUsers = channelPsUsers
    channel = channelApUsers/(subBandwidth*noise)
    if ((1+np.multiply((effPower*power*per*psUsers/(1-per)-powUserCir),channel)) < 0).any():
        print('FAIL CASE !!!!!!!!!!!!!!!')
        print(1+np.multiply((effPower*power*per*psUsers/(1-per)-powUserCir),channel))
        import ipdb; ipdb.set_trace()
    rateMatrix = (1-per)*epsilon*np.log(1+np.multiply((effPower*power*per*psUsers/(1-per)-powUserCir),channel))
    X1 = per*(power-effPower*power*np.sum(psUsers)+powPsCir)
    X2 = (1-per)*(np.multiply((effPower*power*per*psUsers/(1-per)-powUserCir),np.ones_like(rateMatrix)))
    matObj = rateMatrix - dinkel*(X1+X2)
    binaryMatrix = np.zeros_like(matObj)
    for i in range(0, matObj.shape[1]):
        binaryMatrix[np.argmax(matObj[:,i])][i] = 1

    apUsers = np.multiply(channelApUsers, binaryMatrix)
    channel = apUsers/(subBandwidth*noise)
    rateMatrix = (1-per)*epsilon*np.log(1+np.multiply((effPower*power*per*psUsers/(1-per)-powUserCir),channel))
    X2_sum = (1-per)*np.sum(np.multiply((effPower*power*per*psUsers/(1-per)-powUserCir),binaryMatrix))
    sumMatObj = np.sum(rateMatrix) - dinkel*(X1 + X2_sum)
    return binaryMatrix, sumMatObj

def Alternating(initPower,
                initPeriod,
                binaryMatrixInit,
                channelPsUsers,
                channelApUsers,
                noise,
                subBandwidth,
                effPower,
                powUserCir,
                powPsCir,
                pMax,
                perMax,
                dinkel):
    ObjInit = BisectionSearch(channelPsUsers,
                              channelApUsers, 
                              binaryMatrixInit,
                              noise,
                              subBandwidth,
                              effPower,
                              powUserCir,
                              powPsCir,
                              dinkel)
    psUsers = channelPsUsers
    channel = channelApUsers/(subBandwidth*noise)

    if (initPower <= pMax):
        if ((1+np.multiply((effPower*initPower*initPeriod*psUsers/(1-initPeriod)-powUserCir),channel)) > 0).all():
            print('Starting points is GOOD!')
        else:
            print('Starting points is BAD!')
            initPeriod, LowerBound = ObjInit.initPeriod(pMax, perMax)
            initPowerList = np.arange(LowerBound, pMax, 1/100000000)
            initPower = initPowerList[int(len(initPowerList)/2)]
    else:
        print('Starting points not correct!')
        # import ipdb; ipdb.set_trace()
        initPeriod, LowerBound = ObjInit.initPeriod(pMax, perMax)
        initPowerList = np.arange(LowerBound, pMax, 1/100000000)
        initPower = initPowerList[int(len(initPowerList)/2)]
        print('PowerOpt: ', initPower)
        print('PeriodOpt: ', initPeriod)
    
    ObjectiveInit = ObjInit.Objective(initPower, initPeriod)
    
    i = 1
    flag = True
    ObjectiveList = [ObjectiveInit]
    PowerOpt = initPower
    PeriodOpt = initPeriod
    binaryMatrixOpt = binaryMatrixInit 
    j = 1
    while (flag):
        print('Iteration: ', i)       
        obj = BisectionSearch(channelPsUsers,
                              channelApUsers, 
                              binaryMatrixOpt,
                              noise,
                              subBandwidth,
                              effPower,
                              powUserCir,
                              powPsCir,
                              dinkel)
        # import ipdb; ipdb.set_trace() 
        PeriodOpt, ObjectiveOpt_2 = obj.PeriodSearch(pMax, perMax, PowerOpt, PeriodOpt, j)  
        
        j += 1
              
        PowerOpt, ObjectiveOpt_3 = obj.PowerSearch(pMax, perMax, PeriodOpt, PowerOpt, j)

        binaryMatrixOpt, ObjectiveOpt_1 = allocateResource(PowerOpt, 
                                                           PeriodOpt,
                                                           channelPsUsers,
                                                           channelApUsers,
                                                           noise,
                                                           subBandwidth,
                                                           effPower,
                                                           powUserCir,
                                                           powPsCir,
                                                           dinkel)   
        obj = BisectionSearch(channelPsUsers,
                              channelApUsers, 
                              binaryMatrixOpt,
                              noise,
                              subBandwidth,
                              effPower,
                              powUserCir,
                              powPsCir,
                              dinkel) 
               
        ObjectiveOpt = obj.Objective(PowerOpt, PeriodOpt)
        ObjectiveList.append(ObjectiveOpt) 

        ######### - Check algorithm - ###########
        if (ObjectiveList[i] < ObjectiveList[i-1]):
            print('Algorithm is wrong!')
            break

        ######### - Convergence condition - #########
        if ((ObjectiveList[i]-ObjectiveList[i-1]) < 10**(-1)):
            flag = False
        else:
            flag = True
        i += 1
    return binaryMatrixOpt, PowerOpt, PeriodOpt, ObjectiveList

def AlternatingSumRate(initPower,
                       initPeriod,
                       binaryMatrixInit,
                       channelPsUsers,
                       channelApUsers,
                       noise,
                       subBandwidth,
                       effPower,
                       powUserCir,
                       powPsCir,
                       pMax,
                       perMax,
                       dinkel):
    ObjInit = BisectionSearch(channelPsUsers,
                              channelApUsers, 
                              binaryMatrixInit,
                              noise,
                              subBandwidth,
                              effPower,
                              powUserCir,
                              powPsCir,
                              dinkel)
    psUsers = channelPsUsers
    channel = channelApUsers/(subBandwidth*noise)
    if ((1+np.multiply((effPower*initPower*initPeriod*psUsers/(1-initPeriod)-powUserCir),channel)) > 0).all():
        print('Strating point (period) is GOOD!')
    else:
        print('Strating point (period) is BAD!')
        A_per = 1 - powUserCir*channel
        B_per = effPower*initPower*np.multiply(psUsers, channel)
        C_per = (B_per-A_per)
        C_neg = C_per.clip(max=0)
        C_pos = C_per.clip(min=0)
        delta = 0.000001 # 0.00000000000001

        UpperBound = np.divide(-A_per, C_neg, out = np.zeros_like(-A_per), where=C_neg!=0)
        UpperBound[UpperBound==0] = +inf
        UpperBound = np.min(UpperBound)
        UpperBound = np.minimum(UpperBound, perMax-0.000001)
        
        LowerBound = np.divide(-A_per, C_pos, out = np.zeros_like(-A_per), where=C_pos!=0)
        LowerBound = LowerBound.flatten()
        LowerBound = np.delete(LowerBound, np.where(LowerBound==1))
        LowerBound = np.max(LowerBound)
        LowerBound = np.maximum(LowerBound,0.000001)

        delta = 0.00000001
        if (UpperBound - LowerBound) <= delta:
            raise ValueError('Power is not suitable. Please try another!')
        else:
            initPeriodList = np.arange(LowerBound, UpperBound, 1/100000)
            initPeriod = initPeriodList[int(len(initPeriodList)/2)]
        
    ObjectiveInit = ObjInit.Objective(initPower, initPeriod)
    
    i = 1
    flag = True
    ObjectiveList = [ObjectiveInit]
    PeriodOpt = initPeriod
    binaryMatrixOpt = binaryMatrixInit 
    while (flag):
        print('Iteration: ', i)
        obj = BisectionSearch(channelPsUsers,
                              channelApUsers, 
                              binaryMatrixOpt,
                              noise,
                              subBandwidth,
                              effPower,
                              powUserCir,
                              powPsCir,
                              dinkel)
        PeriodOpt, ObjectiveOpt_2 = obj.PeriodSearch(pMax, perMax, initPower, PeriodOpt, i) 
        binaryMatrixOpt, ObjectiveOpt_1 = allocateResource(initPower, 
                                                           PeriodOpt,
                                                           channelPsUsers,
                                                           channelApUsers,
                                                           noise,
                                                           subBandwidth,
                                                           effPower,
                                                           powUserCir,
                                                           powPsCir,
                                                           dinkel)
        ObjectiveOpt = obj.Objective(initPower, PeriodOpt)
        ObjectiveList.append(ObjectiveOpt) 

        ######### - Check algorithm - ###########
        if (ObjectiveList[i] < ObjectiveList[i-1]):
            print('Algorithm is wrong!')
            break

        ######### - Convergence condition - #########
        if ((ObjectiveList[i]-ObjectiveList[i-1]) < 10**(-1)):
            flag = False
        else:
            flag = True
        i += 1
    return binaryMatrixOpt, PeriodOpt, ObjectiveList

def AlternatingFixedPeriod(initPower,
                           initPeriod,
                           binaryMatrixInit,
                           channelPsUsers,
                           channelApUsers,
                           noise,
                           subBandwidth,
                           effPower,
                           powUserCir,
                           powPsCir,
                           pMax,
                           perMax,
                           dinkel):
    ObjInit = BisectionSearch(channelPsUsers,
                              channelApUsers, 
                              binaryMatrixInit,
                              noise,
                              subBandwidth,
                              effPower,
                              powUserCir,
                              powPsCir,
                              dinkel)
    psUsers = channelPsUsers
    channel = channelApUsers/(subBandwidth*noise)
    if (initPower <= pMax):
        if ((1+np.multiply((effPower*initPower*initPeriod*psUsers/(1-initPeriod)-powUserCir),channel)) > 0).all():
            print('Strating point (power) is GOOD!')
        else:
            print('Strating point (power) is BAD!')
            A_pow = 1 - powUserCir*channel
            B_pow = effPower*np.multiply(psUsers,channel)*initPeriod/(1-initPeriod)
            B_neg = B_pow.clip(max=0)
            B_pos = B_pow.clip(min=0)
            UpperBound = np.divide(-A_pow, B_neg, out = np.zeros_like(-A_pow), where=B_neg!=0)
            UpperBound[UpperBound==0] = +inf
            UpperBound = np.min(UpperBound)
            UpperBound = np.minimum(UpperBound, pMax)

            LowerBound = np.divide(-A_pow, B_pos, out=np.zeros_like(-A_pow), where=B_pos!=0)
            LowerBound = np.max(LowerBound)
            LowerBound = np.maximum(LowerBound, 0.000000001)
            delta = 0.000000001 # 0.00000001
            if (UpperBound - LowerBound) <= delta:
                print('Channel fail!!!!!!!!!!!!!!!!!')
                return None
            else:
                initPowerList = np.arange(LowerBound, UpperBound, 1/100000)
                initPower = initPowerList[int(len(initPowerList)/2)]
    else:
        print('Starting points not correct!')
        initPeriod, LowerBound = ObjInit.initPeriod(pMax, perMax)
        initPowerList = np.arange(LowerBound, pMax, 1/100000000)
        initPower = initPowerList[int(len(initPowerList)/2)]
        print('PowerOpt: ', initPower)
        print('PeriodOpt: ', initPeriod)

    """
    if ((1+np.multiply((effPower*initPower*initPeriod*psUsers/(1-initPeriod)-powUserCir),channel)) > 0).all():
        print('Strating point (power) is GOOD!')
    else:
        print('Strating point (power) is BAD!')
        A_pow = 1 - powUserCir*channel
        B_pow = effPower*np.multiply(psUsers,channel)*initPeriod/(1-initPeriod)
        B_neg = B_pow.clip(max=0)
        B_pos = B_pow.clip(min=0)
        UpperBound = np.divide(-A_pow, B_neg, out = np.zeros_like(-A_pow), where=B_neg!=0)
        UpperBound[UpperBound==0] = +inf
        UpperBound = np.min(UpperBound)
        UpperBound = np.minimum(UpperBound, pMax)

        LowerBound = np.divide(-A_pow, B_pos, out=np.zeros_like(-A_pow), where=B_pos!=0)
        LowerBound = np.max(LowerBound)
        LowerBound = np.maximum(LowerBound, 0.000000001)
        delta = 0.000000001 # 0.00000001
        if (UpperBound - LowerBound) <= delta:
            print('Channel fail!!!!!!!!!!!!!!!!!')
            return None
        else:
            initPowerList = np.arange(LowerBound, UpperBound, 1/100000)
            initPower = initPowerList[int(len(initPowerList)/2)]
    """

    # import ipdb; ipdb.set_trace()
    ObjectiveInit = ObjInit.Objective(initPower, initPeriod)
    
    i = 1
    j = 2
    flag = True
    ObjectiveList = [ObjectiveInit]
    PowerOpt = initPower
    binaryMatrixOpt = binaryMatrixInit 
    while (flag):
        print('Iteration: ', i)
        
        obj = BisectionSearch(channelPsUsers,
                              channelApUsers, 
                              binaryMatrixOpt,
                              noise,
                              subBandwidth,
                              effPower,
                              powUserCir,
                              powPsCir,
                              dinkel)
        # import ipdb; ipdb.set_trace()
        PowerOpt, ObjectiveOpt_2 = obj.PowerSearch(pMax, perMax, initPeriod, PowerOpt, j)
        binaryMatrixOpt, ObjectiveOpt_1 = allocateResource(PowerOpt, 
                                                           initPeriod,
                                                           channelPsUsers,
                                                           channelApUsers,
                                                           noise,
                                                           subBandwidth,
                                                           effPower,
                                                           powUserCir,
                                                           powPsCir,
                                                           dinkel)
        obj = BisectionSearch(channelPsUsers,
                              channelApUsers, 
                              binaryMatrixOpt,
                              noise,
                              subBandwidth,
                              effPower,
                              powUserCir,
                              powPsCir,
                              dinkel)
        ObjectiveOpt = obj.Objective(PowerOpt, initPeriod)
        ObjectiveList.append(ObjectiveOpt) 

        ######### - Check algorithm - ###########
        if (ObjectiveList[i] < ObjectiveList[i-1]):
            print('Algorithm is wrong!')
            break

        ######### - Convergence condition - #########
        if ((ObjectiveList[i]-ObjectiveList[i-1]) < 10**(-1)):
            flag = False
        else:
            flag = True
        i += 1
    return binaryMatrixOpt, PowerOpt, ObjectiveList



if __name__ == "__main__":
    noise=10**((-174/10)-3)
    subBandwidth=78*10**3
    effPower=0.7
    powUserCir=10**-7 
    powPsCir=10**(-3/10)*0.001 
    
    def GetbinaryMatrixInit(no_users,no_subcarriers):
        binaryMatrix = np.zeros((no_users,no_subcarriers))
        for i in range(no_users):
            for j in range(no_subcarriers):
                if i==j:
                    binaryMatrix[i,j] = 1
                else:
                    binaryMatrix[i,j] =0
        return binaryMatrix
    binaryMatrix = GetbinaryMatrixInit(5,64)
    
    PathPS = "../Channels/ChannelSet/OFDMA/PS_Users/frame_11.csv"
    channelPsUsers = np.array([np.genfromtxt(PathPS, delimiter=',')]).T    
    PathAP = "../Channels/ChannelSet/OFDMA/AP_Users/frame_11.csv"
    channelApUsers = np.genfromtxt(PathAP, delimiter=',')

    power = 0.99999 
    per = 0.99999 
    start_time = time.time()
    DinkelInit = 10
    pMax = 100 # 3
    perMax = 1
    _, _, ObjectiveList = AlternatingSumRate(per,
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
    total_time = time.time() - start_time
    print('ObjectiveList:')
    print(ObjectiveList)
    print('Optimization time: %f' %total_time)

