import numpy as np
from numpy import inf
import matplotlib.pyplot as plt
import pandas
import matplotlib as mpl
import time
from BisectionSearch import BisectionSearch

def CheckInit(BinaryMatrixInit, 
              PowerInit, 
              PeriodInit,
              channel_PS_users,
              channel_AP_users,
              noise,
              bandwidth_sub,
              EffPower,
              PowUserCir):
    ###### - Check binary matrix - #######
    for i in range(0, BinaryMatrixInit.shape[1]):
        if ((BinaryMatrixInit[:,i] == 1).sum() > 1):
            print('Binary matrix is not satisfy. Please try again!')
    ###### - Check power - #######
    if (PowerInit >= 1) or (PowerInit <= 0):
        print('Power is not statisfy. Please try again!')
    ###### - Check period - #######
    if (PeriodInit >= 1) or (PeriodInit <= 0):
        print('Period is not statisfy. Please try again!')

def ResourceAllocation(power, 
                       per,
                       channel_PS_users,
                       channel_AP_users,
                       noise,
                       bandwidth_sub,
                       EffPower,
                       PowUserCir,
                       PowPScir,
                       dinkel):
    epsilon = bandwidth_sub/(np.log(2))
    PS_users = channel_PS_users
    channel = channel_AP_users/(bandwidth_sub*noise)
    if ((1+np.multiply((EffPower*power*per*PS_users/(1-per)-PowUserCir),channel)) < 0).any():
        print('FAIL CASE !!!!!!!!!!!!!!!')
        print(1+np.multiply((EffPower*power*per*PS_users/(1-per)-PowUserCir),channel))
        import ipdb; ipdb.set_trace()
    RateMatrix = (1-per)*epsilon*np.log(1+np.multiply((EffPower*power*per*PS_users/(1-per)-PowUserCir),channel))
    X1 = per*(power-EffPower*power*np.sum(PS_users)+PowPScir)
    X2 = (1-per)*(np.multiply((EffPower*power*per*PS_users/(1-per)-PowUserCir),np.ones_like(RateMatrix)))
    MatObj = RateMatrix - dinkel*(X1+X2)
    BinaryMatrix = np.zeros_like(MatObj)
    for i in range(0, MatObj.shape[1]):
        BinaryMatrix[np.argmax(MatObj[:,i])][i] = 1

    AP_users = np.multiply(channel_AP_users,BinaryMatrix)
    channel = AP_users/(bandwidth_sub*noise)
    RateMatrix = (1-per)*epsilon*np.log(1+np.multiply((EffPower*power*per*PS_users/(1-per)-PowUserCir),channel))
    X2_sum = (1-per)*np.sum(np.multiply((EffPower*power*per*PS_users/(1-per)-PowUserCir),BinaryMatrix))
    SumMatObj = np.sum(RateMatrix) - dinkel*(X1 + X2_sum)
    return BinaryMatrix, SumMatObj

def Alternating(PowerInit,
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
                PerMax,
                dinkel):
    ObjInit = BisectionSearch(channel_PS_users,
                              channel_AP_users, 
                              BinaryMatrixInit,
                              noise,
                              bandwidth_sub,
                              EffPower,
                              PowUserCir,
                              PowPScir,
                              dinkel)
    PS_users = channel_PS_users
    channel = channel_AP_users/(bandwidth_sub*noise)

    if (PowerInit <= Pmax):
        if ((1+np.multiply((EffPower*PowerInit*PeriodInit*PS_users/(1-PeriodInit)-PowUserCir),channel)) > 0).all():
            print('Strating points is GOOD!')
        else:
            print('Strating points is BAD!')
            PeriodInit, LowerBound = ObjInit.PeriodInit(Pmax, PerMax)
            PowerInitList = np.arange(LowerBound, Pmax, 1/100000000)
            PowerInit = PowerInitList[int(len(PowerInitList)/2)]
    else:
        print('Starting points not correct!')
        # import ipdb; ipdb.set_trace()
        PeriodInit, LowerBound = ObjInit.PeriodInit(Pmax, PerMax)
        PowerInitList = np.arange(LowerBound, Pmax, 1/100000000)
        PowerInit = PowerInitList[int(len(PowerInitList)/2)]
        print('PowerOpt: ', PowerInit)
        print('PeriodOpt: ', PeriodInit)
    
    ObjectiveInit = ObjInit.Objective(PowerInit, PeriodInit)
    
    i = 1
    flag = True
    ObjectiveList = [ObjectiveInit]
    PowerOpt = PowerInit
    PeriodOpt = PeriodInit
    BinaryMatrixOpt = BinaryMatrixInit 
    j = 1
    while (flag):
        print('Iteration: ', i)       
        obj = BisectionSearch(channel_PS_users,
                              channel_AP_users, 
                              BinaryMatrixOpt,
                              noise,
                              bandwidth_sub,
                              EffPower,
                              PowUserCir,
                              PowPScir,
                              dinkel)
        # import ipdb; ipdb.set_trace() 
        PeriodOpt, ObjectiveOpt_2 = obj.PeriodSearch(Pmax, PerMax, PowerOpt, PeriodOpt, j)  
        
        j += 1
              
        PowerOpt, ObjectiveOpt_3 = obj.PowerSearch(Pmax, PerMax, PeriodOpt, PowerOpt, j)

        BinaryMatrixOpt, ObjectiveOpt_1 = ResourceAllocation(PowerOpt, 
                                                             PeriodOpt,
                                                             channel_PS_users,
                                                             channel_AP_users,
                                                             noise,
                                                             bandwidth_sub,
                                                             EffPower,
                                                             PowUserCir,
                                                             PowPScir,
                                                             dinkel)   
        obj = BisectionSearch(channel_PS_users,
                              channel_AP_users, 
                              BinaryMatrixOpt,
                              noise,
                              bandwidth_sub,
                              EffPower,
                              PowUserCir,
                              PowPScir,
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
    return BinaryMatrixOpt, PowerOpt, PeriodOpt, ObjectiveList

def AlternatingSumRate(PowerInit,
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
                       PerMax,
                       dinkel):
    ObjInit = BisectionSearch(channel_PS_users,
                              channel_AP_users, 
                              BinaryMatrixInit,
                              noise,
                              bandwidth_sub,
                              EffPower,
                              PowUserCir,
                              PowPScir,
                              dinkel)
    PS_users = channel_PS_users
    channel = channel_AP_users/(bandwidth_sub*noise)
    if ((1+np.multiply((EffPower*PowerInit*PeriodInit*PS_users/(1-PeriodInit)-PowUserCir),channel)) > 0).all():
        print('Strating point (period) is GOOD!')
    else:
        print('Strating point (period) is BAD!')
        A_per = 1 - PowUserCir*channel
        B_per = EffPower*PowerInit*np.multiply(PS_users, channel)
        C_per = (B_per-A_per)
        C_neg = C_per.clip(max=0)
        C_pos = C_per.clip(min=0)
        delta = 0.000001 # 0.00000000000001

        UpperBound = np.divide(-A_per, C_neg, out = np.zeros_like(-A_per), where=C_neg!=0)
        UpperBound[UpperBound==0] = +inf
        UpperBound = np.min(UpperBound)
        UpperBound = np.minimum(UpperBound, PerMax-0.000001)
        
        LowerBound = np.divide(-A_per, C_pos, out = np.zeros_like(-A_per), where=C_pos!=0)
        LowerBound = LowerBound.flatten()
        LowerBound = np.delete(LowerBound, np.where(LowerBound==1))
        LowerBound = np.max(LowerBound)
        LowerBound = np.maximum(LowerBound,0.000001)

        delta = 0.00000001
        if (UpperBound - LowerBound) <= delta:
            raise ValueError('Power is not suitable. Please try another!')
        else:
            PeriodInitList = np.arange(LowerBound, UpperBound, 1/100000)
            PeriodInit = PeriodInitList[int(len(PeriodInitList)/2)]
        
    ObjectiveInit = ObjInit.Objective(PowerInit, PeriodInit)
    
    i = 1
    flag = True
    ObjectiveList = [ObjectiveInit]
    PeriodOpt = PeriodInit
    BinaryMatrixOpt = BinaryMatrixInit 
    while (flag):
        print('Iteration: ', i)
        obj = BisectionSearch(channel_PS_users,
                              channel_AP_users, 
                              BinaryMatrixOpt,
                              noise,
                              bandwidth_sub,
                              EffPower,
                              PowUserCir,
                              PowPScir,
                              dinkel)
        PeriodOpt, ObjectiveOpt_2 = obj.PeriodSearch(Pmax, PerMax, PowerInit, PeriodOpt, i) 
        BinaryMatrixOpt, ObjectiveOpt_1 = ResourceAllocation(PowerInit, 
                                                             PeriodOpt,
                                                             channel_PS_users,
                                                             channel_AP_users,
                                                             noise,
                                                             bandwidth_sub,
                                                             EffPower,
                                                             PowUserCir,
                                                             PowPScir,
                                                             dinkel)
        ObjectiveOpt = obj.Objective(PowerInit, PeriodOpt)
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
    return BinaryMatrixOpt, PeriodOpt, ObjectiveList

def AlternatingFixedPeriod(PowerInit,
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
                           PerMax,
                           dinkel):
    ObjInit = BisectionSearch(channel_PS_users,
                              channel_AP_users, 
                              BinaryMatrixInit,
                              noise,
                              bandwidth_sub,
                              EffPower,
                              PowUserCir,
                              PowPScir,
                              dinkel)
    PS_users = channel_PS_users
    channel = channel_AP_users/(bandwidth_sub*noise)
    if (PowerInit <= Pmax):
        if ((1+np.multiply((EffPower*PowerInit*PeriodInit*PS_users/(1-PeriodInit)-PowUserCir),channel)) > 0).all():
            print('Strating point (power) is GOOD!')
        else:
            print('Strating point (power) is BAD!')
            A_pow = 1 - PowUserCir*channel
            B_pow = EffPower*np.multiply(PS_users,channel)*PeriodInit/(1-PeriodInit)
            B_neg = B_pow.clip(max=0)
            B_pos = B_pow.clip(min=0)
            UpperBound = np.divide(-A_pow, B_neg, out = np.zeros_like(-A_pow), where=B_neg!=0)
            UpperBound[UpperBound==0] = +inf
            UpperBound = np.min(UpperBound)
            UpperBound = np.minimum(UpperBound, Pmax)

            LowerBound = np.divide(-A_pow, B_pos, out=np.zeros_like(-A_pow), where=B_pos!=0)
            LowerBound = np.max(LowerBound)
            LowerBound = np.maximum(LowerBound, 0.000000001)
            delta = 0.000000001 # 0.00000001
            if (UpperBound - LowerBound) <= delta:
                print('Channel fail!!!!!!!!!!!!!!!!!')
                return None
            else:
                PowerInitList = np.arange(LowerBound, UpperBound, 1/100000)
                PowerInit = PowerInitList[int(len(PowerInitList)/2)]
    else:
        print('Starting points not correct!')
        PeriodInit, LowerBound = ObjInit.PeriodInit(Pmax, PerMax)
        PowerInitList = np.arange(LowerBound, Pmax, 1/100000000)
        PowerInit = PowerInitList[int(len(PowerInitList)/2)]
        print('PowerOpt: ', PowerInit)
        print('PeriodOpt: ', PeriodInit)

    """
    if ((1+np.multiply((EffPower*PowerInit*PeriodInit*PS_users/(1-PeriodInit)-PowUserCir),channel)) > 0).all():
        print('Strating point (power) is GOOD!')
    else:
        print('Strating point (power) is BAD!')
        A_pow = 1 - PowUserCir*channel
        B_pow = EffPower*np.multiply(PS_users,channel)*PeriodInit/(1-PeriodInit)
        B_neg = B_pow.clip(max=0)
        B_pos = B_pow.clip(min=0)
        UpperBound = np.divide(-A_pow, B_neg, out = np.zeros_like(-A_pow), where=B_neg!=0)
        UpperBound[UpperBound==0] = +inf
        UpperBound = np.min(UpperBound)
        UpperBound = np.minimum(UpperBound, Pmax)

        LowerBound = np.divide(-A_pow, B_pos, out=np.zeros_like(-A_pow), where=B_pos!=0)
        LowerBound = np.max(LowerBound)
        LowerBound = np.maximum(LowerBound, 0.000000001)
        delta = 0.000000001 # 0.00000001
        if (UpperBound - LowerBound) <= delta:
            print('Channel fail!!!!!!!!!!!!!!!!!')
            return None
        else:
            PowerInitList = np.arange(LowerBound, UpperBound, 1/100000)
            PowerInit = PowerInitList[int(len(PowerInitList)/2)]
    """

    # import ipdb; ipdb.set_trace()
    ObjectiveInit = ObjInit.Objective(PowerInit, PeriodInit)
    
    i = 1
    j = 2
    flag = True
    ObjectiveList = [ObjectiveInit]
    PowerOpt = PowerInit
    BinaryMatrixOpt = BinaryMatrixInit 
    while (flag):
        print('Iteration: ', i)
        
        obj = BisectionSearch(channel_PS_users,
                              channel_AP_users, 
                              BinaryMatrixOpt,
                              noise,
                              bandwidth_sub,
                              EffPower,
                              PowUserCir,
                              PowPScir,
                              dinkel)
        # import ipdb; ipdb.set_trace()
        PowerOpt, ObjectiveOpt_2 = obj.PowerSearch(Pmax, PerMax, PeriodInit, PowerOpt, j)
        BinaryMatrixOpt, ObjectiveOpt_1 = ResourceAllocation(PowerOpt, 
                                                             PeriodInit,
                                                             channel_PS_users,
                                                             channel_AP_users,
                                                             noise,
                                                             bandwidth_sub,
                                                             EffPower,
                                                             PowUserCir,
                                                             PowPScir,
                                                             dinkel)
        obj = BisectionSearch(channel_PS_users,
                              channel_AP_users, 
                              BinaryMatrixOpt,
                              noise,
                              bandwidth_sub,
                              EffPower,
                              PowUserCir,
                              PowPScir,
                              dinkel)
        ObjectiveOpt = obj.Objective(PowerOpt, PeriodInit)
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
    return BinaryMatrixOpt, PowerOpt, ObjectiveList



if __name__ == "__main__":
    noise=10**((-174/10)-3)
    bandwidth_sub=78*10**3
    EffPower=0.7
    PowUserCir=10**-7 
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
    
    PathPS = "../Channels/ChannelSet/OFDMA/PS_Users/frame_11.csv"
    channel_PS_users = np.array([np.genfromtxt(PathPS, delimiter=',')]).T    
    PathAP = "../Channels/ChannelSet/OFDMA/AP_Users/frame_11.csv"
    channel_AP_users = np.genfromtxt(PathAP, delimiter=',')

    power = 0.99999 
    per = 0.99999 
    start_time = time.time()
    DinkelInit = 10
    Pmax = 100 # 3
    PerMax = 1
    _, _, ObjectiveList = AlternatingSumRate(per,
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
    total_time = time.time() - start_time
    print('ObjectiveList:')
    print(ObjectiveList)
    print('Optimization time: %f' %total_time)

