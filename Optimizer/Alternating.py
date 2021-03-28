import numpy as np
from numpy import inf
import matplotlib.pyplot as plt
import pandas
import matplotlib as mpl
import time
from BisectionSearch import BisectionSearch

def checkInit(initBinaryMatrix, 
              initPower, 
              initPeriod,
              channelPsUsers,
              channelApUsers,
              noise,
              subBandwidth,
              effPower,
              powUserCir):
    ###### - Check binary matrix - #######
    for i in range(0, initBinaryMatrix.shape[1]):
        if ((initBinaryMatrix[:,i] == 1).sum() > 1):
            print('Binary matrix is not satisfy. Please try again!')
    ###### - Check power - #######
    if (initPower >= 1) or (initPower <= 0):
        print('Power is not satisfy. Please try again!')
    ###### - Check period - #######
    if (initPeriod >= 1) or (initPeriod <= 0):
        print('Period is not satisfy. Please try again!')

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
        print('FAILED CASE !!!!!!!!!!!!!!!')
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

def alterAlgorithm(initPower,
                   initPeriod,
                   initBinaryMatrix,
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
    initObj = BisectionSearch(channelPsUsers,
                              channelApUsers, 
                              initBinaryMatrix,
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
            initPeriod, lowerBound = initObj.initPeriod(pMax, perMax)
            initPowerList = np.arange(lowerBound, pMax, 1/100000000)
            initPower = initPowerList[int(len(initPowerList)/2)]
    else:
        print('Starting points are not correct!')
        # import ipdb; ipdb.set_trace()
        initPeriod, lowerBound = initObj.initPeriod(pMax, perMax)
        initPowerList = np.arange(lowerBound, pMax, 1/100000000)
        initPower = initPowerList[int(len(initPowerList)/2)]
        print('optPower: ', initPower)
        print('optPeriod: ', initPeriod)
    
    initObjective = initObj.calcObject(initPower, initPeriod)
    
    i = 1
    flag = True
    objectiveList = [initObjective]
    optPower = initPower
    optPeriod = initPeriod
    optBinaryMatrix = initBinaryMatrix 
    j = 1
    while (flag):
        print('Iteration: ', i)       
        obj = BisectionSearch(channelPsUsers,
                              channelApUsers, 
                              optBinaryMatrix,
                              noise,
                              subBandwidth,
                              effPower,
                              powUserCir,
                              powPsCir,
                              dinkel)
        # import ipdb; ipdb.set_trace() 
        optPeriod, optObjective_2 = obj.searchPeriod(pMax, perMax, optPower, optPeriod, j)  
        
        j += 1
              
        optPower, optObjective_3 = obj.searchPower(pMax, perMax, optPeriod, optPower, j)

        optBinaryMatrix, optObjective_1 = allocateResource(optPower, 
                                                           optPeriod,
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
                              optBinaryMatrix,
                              noise,
                              subBandwidth,
                              effPower,
                              powUserCir,
                              powPsCir,
                              dinkel) 
               
        optObjective = obj.calcObject(optPower, optPeriod)
        objectiveList.append(optObjective) 

        ######### - Check algorithm - ###########
        if (objectiveList[i] < objectiveList[i-1]):
            print('Algorithm is wrong!')
            break

        ######### - Convergence condition - #########
        if ((objectiveList[i]-objectiveList[i-1]) < 10**(-1)):
            flag = False
        else:
            flag = True
        i += 1
    return optBinaryMatrix, optPower, optPeriod, objectiveList

def alterSumRate(initPower,
                 initPeriod,
                 initBinaryMatrix,
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
    initObj = BisectionSearch(channelPsUsers,
                              channelApUsers, 
                              initBinaryMatrix,
                              noise,
                              subBandwidth,
                              effPower,
                              powUserCir,
                              powPsCir,
                              dinkel)
    psUsers = channelPsUsers
    channel = channelApUsers/(subBandwidth*noise)
    if ((1+np.multiply((effPower*initPower*initPeriod*psUsers/(1-initPeriod)-powUserCir),channel)) > 0).all():
        print('Starting point (period) is GOOD!')
    else:
        print('Starting point (period) is BAD!')
        tempA = 1 - powUserCir*channel
        tempB = effPower*initPower*np.multiply(psUsers, channel)
        tempC = (tempB-tempA)
        negC = tempC.clip(max=0)
        posC = tempC.clip(min=0)
        delta = 0.000001 # 0.00000000000001

        upperBound = np.divide(-tempA, negC, out = np.zeros_like(-tempA), where=negC!=0)
        upperBound[upperBound==0] = +inf
        upperBound = np.min(upperBound)
        upperBound = np.minimum(upperBound, perMax-0.000001)
        
        lowerBound = np.divide(-tempA, posC, out = np.zeros_like(-tempA), where=posC!=0)
        lowerBound = lowerBound.flatten()
        lowerBound = np.delete(lowerBound, np.where(lowerBound==1))
        lowerBound = np.max(lowerBound)
        lowerBound = np.maximum(lowerBound,0.000001)

        delta = 0.00000001
        if (upperBound - lowerBound) <= delta:
            raise ValueError('Power is not suitable. Please try another!')
        else:
            initPeriodList = np.arange(lowerBound, upperBound, 1/100000)
            initPeriod = initPeriodList[int(len(initPeriodList)/2)]
        
    initObjective = initObj.calcObject(initPower, initPeriod)
    
    i = 1
    flag = True
    objectiveList = [initObjective]
    optPeriod = initPeriod
    optBinaryMatrix = initBinaryMatrix 
    while (flag):
        print('Iteration: ', i)
        obj = BisectionSearch(channelPsUsers,
                              channelApUsers, 
                              optBinaryMatrix,
                              noise,
                              subBandwidth,
                              effPower,
                              powUserCir,
                              powPsCir,
                              dinkel)
        optPeriod, optObjective_2 = obj.searchPeriod(pMax, perMax, initPower, optPeriod, i) 
        optBinaryMatrix, optObjective_1 = allocateResource(initPower, 
                                                           optPeriod,
                                                           channelPsUsers,
                                                           channelApUsers,
                                                           noise,
                                                           subBandwidth,
                                                           effPower,
                                                           powUserCir,
                                                           powPsCir,
                                                           dinkel)
        optObjective = obj.calcObject(initPower, optPeriod)
        objectiveList.append(optObjective) 

        ######### - Check algorithm - ###########
        if (objectiveList[i] < objectiveList[i-1]):
            print('Algorithm is wrong!')
            break

        ######### - Convergence condition - #########
        if ((objectiveList[i]-objectiveList[i-1]) < 10**(-1)):
            flag = False
        else:
            flag = True
        i += 1
    return optBinaryMatrix, optPeriod, objectiveList

def alterFixedPeriod(initPower,
                     initPeriod,
                     initBinaryMatrix,
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
    initObj = BisectionSearch(channelPsUsers,
                              channelApUsers, 
                              initBinaryMatrix,
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
            upperBound = np.divide(-A_pow, B_neg, out = np.zeros_like(-A_pow), where=B_neg!=0)
            upperBound[upperBound==0] = +inf
            upperBound = np.min(upperBound)
            upperBound = np.minimum(upperBound, pMax)

            lowerBound = np.divide(-A_pow, B_pos, out=np.zeros_like(-A_pow), where=B_pos!=0)
            lowerBound = np.max(lowerBound)
            lowerBound = np.maximum(lowerBound, 0.000000001)
            delta = 0.000000001 # 0.00000001
            if (upperBound - lowerBound) <= delta:
                print('Channel fail!!!!!!!!!!!!!!!!!')
                return None
            else:
                initPowerList = np.arange(lowerBound, upperBound, 1/100000)
                initPower = initPowerList[int(len(initPowerList)/2)]
    else:
        print('Starting points not correct!')
        initPeriod, lowerBound = initObj.initPeriod(pMax, perMax)
        initPowerList = np.arange(lowerBound, pMax, 1/100000000)
        initPower = initPowerList[int(len(initPowerList)/2)]
        print('optPower: ', initPower)
        print('optPeriod: ', initPeriod)

    """
    if ((1+np.multiply((effPower*initPower*initPeriod*psUsers/(1-initPeriod)-powUserCir),channel)) > 0).all():
        print('Strating point (power) is GOOD!')
    else:
        print('Strating point (power) is BAD!')
        A_pow = 1 - powUserCir*channel
        B_pow = effPower*np.multiply(psUsers,channel)*initPeriod/(1-initPeriod)
        B_neg = B_pow.clip(max=0)
        B_pos = B_pow.clip(min=0)
        upperBound = np.divide(-A_pow, B_neg, out = np.zeros_like(-A_pow), where=B_neg!=0)
        upperBound[upperBound==0] = +inf
        upperBound = np.min(upperBound)
        upperBound = np.minimum(upperBound, pMax)

        lowerBound = np.divide(-A_pow, B_pos, out=np.zeros_like(-A_pow), where=B_pos!=0)
        lowerBound = np.max(lowerBound)
        lowerBound = np.maximum(lowerBound, 0.000000001)
        delta = 0.000000001 # 0.00000001
        if (upperBound - lowerBound) <= delta:
            print('Channel fail!!!!!!!!!!!!!!!!!')
            return None
        else:
            initPowerList = np.arange(lowerBound, upperBound, 1/100000)
            initPower = initPowerList[int(len(initPowerList)/2)]
    """

    # import ipdb; ipdb.set_trace()
    initObjective = initObj.calcObject(initPower, initPeriod)
    
    i = 1
    j = 2
    flag = True
    objectiveList = [initObjective]
    optPower = initPower
    optBinaryMatrix = initBinaryMatrix 
    while (flag):
        print('Iteration: ', i)
        
        obj = BisectionSearch(channelPsUsers,
                              channelApUsers, 
                              optBinaryMatrix,
                              noise,
                              subBandwidth,
                              effPower,
                              powUserCir,
                              powPsCir,
                              dinkel)
        # import ipdb; ipdb.set_trace()
        optPower, optObjective_2 = obj.searchPower(pMax, perMax, initPeriod, optPower, j)
        optBinaryMatrix, optObjective_1 = allocateResource(optPower, 
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
                              optBinaryMatrix,
                              noise,
                              subBandwidth,
                              effPower,
                              powUserCir,
                              powPsCir,
                              dinkel)
        optObjective = obj.calcObject(optPower, initPeriod)
        objectiveList.append(optObjective) 

        ######### - Check algorithm - ###########
        if (objectiveList[i] < objectiveList[i-1]):
            print('Algorithm is wrong!')
            break

        ######### - Convergence condition - #########
        if ((objectiveList[i]-objectiveList[i-1]) < 10**(-1)):
            flag = False
        else:
            flag = True
        i += 1
    return optBinaryMatrix, optPower, objectiveList



if __name__ == "__main__":
    noise=10**((-174/10)-3)
    subBandwidth=78*10**3
    effPower=0.7
    powUserCir=10**-7 
    powPsCir=10**(-3/10)*0.001 
    
    def getInitBinaryMatrix(noUsers,noSubcarriers):
        binaryMatrix = np.zeros((noUsers,noSubcarriers))
        for i in range(noUsers):
            for j in range(noSubcarriers):
                if i==j:
                    binaryMatrix[i,j] = 1
                else:
                    binaryMatrix[i,j] =0
        return binaryMatrix
    binaryMatrix = getInitBinaryMatrix(5,64)
    
    PathPS = "../Channels/ChannelSet/OFDMA/PS_Users/frame_11.csv"
    channelPsUsers = np.array([np.genfromtxt(PathPS, delimiter=',')]).T    
    PathAP = "../Channels/ChannelSet/OFDMA/AP_Users/frame_11.csv"
    channelApUsers = np.genfromtxt(PathAP, delimiter=',')

    power = 0.99999 
    per = 0.99999 
    start_time = time.time()
    initDinkel = 10
    pMax = 100 # 3
    perMax = 1
    _, _, objectiveList = alterSumRate(power,
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
    total_time = time.time() - start_time
    print('objectiveList:')
    print(objectiveList)
    print('Optimization time: %f' %total_time)

