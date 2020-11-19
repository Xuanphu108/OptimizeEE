import numpy as np
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
    if ((1+np.multiply((EffPower*power*per*PS_users/(1-per)-PowUserCir),channel))[0][0] < 0):
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
    if ((1+np.multiply((EffPower*PowerInit*PeriodInit*PS_users/(1-PeriodInit)-PowUserCir),channel)) > 0).all():
        print('Strating points is GOOD!')
    else:
        print('Strating points is BAD!')
        PeriodInit, LowerBound = ObjInit.PeriodInit(Pmax, PerMax)
        PowerInitList = np.arange(LowerBound, Pmax, 1/100000)
        # import ipdb; ipdb.set_trace()
        PowerInit = PowerInitList[int(len(PowerInitList)/2)]

    
    ObjectiveInit = ObjInit.Objective(PowerInit, PeriodInit)
    
    i = 1
    flag = True
    ObjectiveList = [ObjectiveInit]
    PowerOpt = PowerInit
    PeriodOpt = PeriodInit
    BinaryMatrixOpt = BinaryMatrixInit 
    while (flag):
        print('Iteration: ', i)
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
        PowerOpt, ObjectiveOpt_2 = obj.PowerSearch(Pmax, PerMax, PeriodOpt, PowerOpt, i)
        #if (PowerOpt < 2.9999):
        #    import ipdb; ipdb.set_trace()
        PeriodOpt, ObjectiveOpt_3 = obj.PeriodSearch(Pmax, PerMax, PowerOpt, PeriodOpt, i) 
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

if __name__ == "__main__":
    noise=10**((-90/10)-3)
    bandwidth_sub=78*10**3
    EffPower=0.8
    PowUserCir=10**(-3)
    PowPScir=10**((5/10)-3)
    dinkel=10.804232
    BinaryMatrix = np.array([[0, 0, 0, 0, 0, 0], 
                             [1, 1, 1, 1, 1, 1], 
                             [0, 0, 0, 0, 0, 0], 
                             [0, 0, 0, 0, 0, 0]])
    print('The shape of binary matrix: ', BinaryMatrix.shape)
    channel_PS_users = np.array([[1,2,3,4]]).T
    channel_AP_users = np.array([[1,2,3,4,5,6],
                                 [7,8,9,10,11,12],
                                 [13,14,15,16,17,18],
                                 [19,20,21,22,23,24]])
    power = 0.002911
    per = 0.300410

    start_time = time.time()
    _, _, _, ObjectiveList = Alternating(power,
                                         per,
                                         BinaryMatrix,
                                         channel_PS_users,
                                         channel_AP_users,
                                         noise,
                                         bandwidth_sub,
                                         EffPower,
                                         PowUserCir,
                                         PowPScir,
                                         dinkel)
    total_time = time.time() - start_time
    print('ObjectiveList:')
    print(ObjectiveList)
    print('Optimization time: %f' %total_time)

