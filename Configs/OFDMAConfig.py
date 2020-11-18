import numpy as np

path_PS_users = "../Channels/ChannelSet/OFDMA/PS_Users/"
gen_path_PS_users = "../Channels/ChannelSet/OFDMA/PS_Users/"
path_AP_users = "../Channels/ChannelSet/OFDMA/AP_Users/"
gen_path_AP_users = "../Channels/ChannelSet/OFDMA/AP_Users/"

channel_PS_users = dict(
    dataset='no', # multiple files
    no_samples=100,
    no_subcarriers=1,
    no_TransmitterAntens=1,
    no_ReceiverAntens=1,
    no_transmitters=1,
    pos_transmitters=np.array([0+1j*0]),
    no_receivers=5,
    pos_receivers=None,
    pos_receivers_area=np.array([-50, 50, -50 , 50]),
    pos_receiver_mode='random',
    gen_data_root=gen_path_PS_users,
    data_root=path_PS_users)

channel_AP_users = dict(
    dataset='no',
    no_samples=100,
    no_subcarriers=64,
    no_TransmitterAntens=1,
    no_ReceiverAntens=1,
    no_transmitters=1,
    pos_transmitters=np.array([0+1j*50]),
    no_receivers=5,
    pos_receivers=None,
    pos_receivers_area=np.array([-50, 50, -50 , 50]),
    pos_receiver_mode=None,
    gen_data_root=gen_path_AP_users,
    data_root=path_AP_users)

phy_params = dict(
    EH_model='linear',
    Tmax=1,
    Pmax=10**(np.arange(10,25,5)/10)*0.001,
    P_PScir=10**((5/10)-3), # Circuit power of the PS = 5 dBm
    P_APcir=10**((5/10)-3), # Circuit power of the AP = 5 dBm 
    P_UserCir=10**(-3), # Circuit power of each user 
    bandwidth=5*10**(6), # system bandwidth Hz
    bandwidth_sub=78*10**3, # bandwidth for each subcarriers Hz
    noise=10**((-174/10)-3), # W/Hz = -174 dBm/Hz 
    eff_path=2.5,
    eff_power=0.8,
    threshold=2*10**6)