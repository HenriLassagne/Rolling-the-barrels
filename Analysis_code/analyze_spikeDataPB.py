
import numpy as np
import matplotlib.pyplot as plt
#from klusta.kwik.model import KwikModel
#from attrdict import AttrDict 
from brpylib             import NsxFile, NevFile, brpylib_ver
from scipy.signal import butter, lfilter

# Functions
def getAll_events(DATA):
    Frames = []
    Reward = []
    StartTime=[]
    TrialStart=[]
    PosUnit=[]
    NegUnit=[]
    PosUnits=[]
    NegUnits=[]
    EnteringEvent=[]
    FreeWater=[]



    for i in range(len(DATA['comments']['Comment'])):        
        f = DATA['comments']['Comment'][i]
        t = DATA['comments']['TimeStamps'][i]
        if f[0:2] == "F:":        
            Frames.append([int(f[2:]), int(t)])
        if f[0:14] == "OneWaterReward":
            Reward.append(int(t))
        if f[0:12]=="Init session":
            StartTime.append(int(t))
            TrialStart.append(int(t))
        if f[0:5] == "Trial":
            TrialStart.append(int(t))
        if f[0:2] == "Po": 
            if len(f)==7:
                PosUnit.append([int(f[4:6])-1,int(f[6:7])-1])
            if len(f)==6:
                PosUnit.append([int(f[4:5])-1,int(f[5:6])-1])
        if f[0:2] == "Ne":
            if len(f)==7:
                NegUnit.append([int(f[4:6])-1,int(f[6:7])-1])
            if len(f)==6:
                NegUnit.append([int(f[4:5])-1,int(f[5:6])-1])        
        if f[0:5] == "FmaxP":
            fmaxpos=float(f[6:])
        if f[0:5] == "FmaxN":
            fmaxneg=float(f[6:])
        if f[0:4] == "Gain":
            Gain=float(f[5:]) 
        if f[0:7] == "Latency":
            latency = int(f[8:])
        if f[0:11] == "RewardUpper":
            RewardUp = int(f[12:])
        if f[0:11] == "RewardLower":
            RewardLow = int(f[12:])
        if f[0:4]=="Ente":
            EnteringEvent.append(int(t))
        if f[0:4]=='OneF':
            FreeWater.append(int(t))
            
        
             
    for k in range(0,len(PosUnit),4):
        PosUnits.append(PosUnit[k])
    for k in range(0,len(NegUnit),4):
        NegUnits.append(NegUnit[k])


    
    Frames = np.array(Frames)     
    
    Licks = np.array([])
    if DATA.get('dig_events')!=None:
        Licks = np.squeeze(np.array(DATA['dig_events']['TimeStamps']))
    Reward = np.array(Reward)
    
        
    return Frames, Reward, Licks, TrialStart, EnteringEvent, FreeWater

def get_spikes(DATA):
    
    if DATA.get('spike_events')!=None:
        
    
        Separate=[]
        for channel in range(32):
            Separate.append([])
            for unit in range(1,6,1):
                Separate[channel].append([])

        for index in range(len(DATA['spike_events']['ChannelID'])):
            for unit in range(1,6,1):
                for time in range(len(DATA['spike_events']['Classification'][index])):
                    if DATA['spike_events']['Classification'][index][time]==unit:
                        Separate[DATA['spike_events']['ChannelID'][index]-1][unit-1].append(DATA['spike_events']['TimeStamps'][index][time])                           
        return Separate
    else:
        return 0

def Frames_Trials(TrialStart, Frames):
    time=0
    Frames_individual_Trial=[]
    for k in range(len(TrialStart)-1):
        Frames_individual_Trial.append([])
        while Frames[:][:,1][time] < TrialStart[k+1]:
            Frames_individual_Trial[k].append(Frames[time])
            time+=1
        Frames_individual_Trial[k]=np.array(Frames_individual_Trial[k])
    
    Frames_individual_Trial.append([])
    while time<len(Frames):
        Frames_individual_Trial[-1].append(Frames[time])
        time+=1
    Frames_individual_Trial[-1]=np.array(Frames_individual_Trial[-1])
    return Frames_individual_Trial
    
    

def reconstruct_trajectory(Frames, TrialStart, Separate, latency, PosUnits, NegUnits, fmaxpos, fmaxneg, gain):
    #Reconstruction of the position
    PreviousSpeed=0
    PreviousPosition=Frames[:][:,0][int(1+latency/10.)]+0.5


    ListPosition=[]
    ListTime=[]

    for time in range(Frames[:][:,1][0],(TrialStart[1]-30000*5),30):  
        CurrentAcceleration=0
        for j in range(len(PosUnits)):
            timepos=0
            if Separate[PosUnits[j][0]][PosUnits[j][1]] != []: 
                while (Separate[PosUnits[j][0]][PosUnits[j][1]][timepos]) < (TrialStart[1]-30000*5):
                    if time-30<Separate[PosUnits[j][0]][PosUnits[j][1]][timepos]<=time:
                        CurrentAcceleration+=float(gain*4)*(float(fmaxneg)/(fmaxpos+fmaxneg))
                    timepos+=1         
        for k in range(len(NegUnits)):
            timeneg=0
            if Separate[NegUnits[k][0]][NegUnits[k][1]] != []:
                while (Separate[NegUnits[k][0]][NegUnits[k][1]][timeneg]) < (TrialStart[1]-30000*5):
                    if time-30<Separate[NegUnits[k][0]][NegUnits[k][1]][timeneg]<=time:
                        CurrentAcceleration-=float(gain*4)*(float(fmaxpos)/(fmaxpos+fmaxneg))
                    timeneg+=1
                   
        CurrentSpeed = (PreviousSpeed/1.05) + 1/1000.*CurrentAcceleration
        CurrentPosition = (PreviousPosition + 1/1000.*CurrentSpeed) 
    
        if (CurrentPosition>=360):
            CurrentPosition = CurrentPosition-360
        if (CurrentPosition<=0):
            CurrentPosition = CurrentPosition + 360
    
          
        ListPosition.append(int(CurrentPosition))
        ListTime.append(float(time)/30000.)
    
        PreviousSpeed = CurrentSpeed
        PreviousPosition = CurrentPosition
    
    return ListPosition, ListTime
    



def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def read_cont_data(datafile):
    # Version control
    brpylib_ver_req = "1.3.1"
    
    if brpylib_ver.split('.') < brpylib_ver_req.split('.'):
        raise Exception("requires brpylib " + brpylib_ver_req + " or higher, please use latest version")
   
    elec_ids     = 'all'  # 'all' is default for all (1-indexed)
    start_time_s = 0                       # 0 is default for all
    data_time_s  = 'all'                      # 'all' is default for all
    downsample   = 1                       # 1 is default
     
    # Open file and extract headers
    nsx_file = NsxFile(datafile)
    
    # Extract data - note: data will be returned based on *SORTED* elec_ids, see cont_data['elec_ids']
    cont_data = nsx_file.getdata(elec_ids, start_time_s, data_time_s, downsample)
    
    # Close the nsx file now that all data is out
    nsx_file.close()
    
    return cont_data    
    
def get_continous_signals(cont_data):
    
    channels = cont_data['elec_ids']
    
    cont_data_array = []
    
    for i in range(0,len(channels)):
        
        ch_idx  = cont_data['elec_ids'].index(i+1)
        cont_data_array.append(cont_data['data'][ch_idx])
    
    cont_data_array = np.array(cont_data_array,dtype=np.int16, order='C')
    cont_data_array = np.transpose(cont_data_array)
    
    return cont_data_array
    
def get_Multiunit_nevspikes(DATA):
    Multiu = []
    #Separate = []
    for c in range(len(DATA['spike_events']['TimeStamps'])): # channels
        Multiu.extend(list(np.array(DATA['spike_events']['TimeStamps'][c])[np.array(DATA['spike_events']['Classification'][c]) == "1"])) 
        Multiu.extend(list(np.array(DATA['spike_events']['TimeStamps'][c])[np.array(DATA['spike_events']['Classification'][c]) == "2"]))         
        Multiu.extend(list(np.array(DATA['spike_events']['TimeStamps'][c])[np.array(DATA['spike_events']['Classification'][c]) == "2"])) 
    return np.array(Multiu)

def get_Singleunit_nevspikes(DATA):
    SingleU = []
    #Separate = []
    for c in range(len(DATA['spike_events']['TimeStamps'])): # channels
        SingleU.append(list(np.array(DATA['spike_events']['TimeStamps'][c])[np.array(DATA['spike_events']['Classification'][c]) == "1"])) 
        SingleU.append(list(np.array(DATA['spike_events']['TimeStamps'][c])[np.array(DATA['spike_events']['Classification'][c]) == "2"]))         
        SingleU.append(list(np.array(DATA['spike_events']['TimeStamps'][c])[np.array(DATA['spike_events']['Classification'][c]) == "2"])) 
    return np.array(SingleU)

def klutadapt_cont_data(datafile, adaptedfile):
    #datafile = 'G:/DATA/Closed_Loop_BMI_Aamir/mouse61/session7/data.ns5' 
    #savePath = 'G:/DATA/Closed_Loop_BMI_Aamir/mouse61/session7'

    # Read data
    cont_data_struct = read_cont_data(datafile)
    channels = cont_data_struct['elec_ids']
    
    cont_data_array = []
    
    for i in range(0,len(channels)):
        
        ch_idx  = cont_data_struct['elec_ids'].index(i+1)
        cont_data_array.append(cont_data_struct['data'][ch_idx])
    
    cont_data_array = np.array(cont_data_array,dtype=np.int16, order='C')
    cont_data_array = np.transpose(cont_data_array)

    filename = adaptedfile + "/data.dat"
    fileobj = open(filename, mode='wb')
    cont_data_array.tofile(fileobj)
    fileobj.close()

def frame_event_data(DATA):
    Frames = []
    for i in range(len(DATA['comments']['Comment'])):        
        f = DATA['comments']['Comment'][i]
        t = DATA['comments']['TimeStamps'][i]
        if f[0:2] == "F:":        
            Frames.append([int(f[2:]), int(t)])
    Frames = np.array(Frames)            
    return Frames

def readkwikinfo(kwik, grupete,filetimes):
    model = KwikModel(kwik) # load kwik model from file
    spiketimes = model.spike_times # extract the absolute spike times
    clusters = model.cluster_groups # extract the cluster names
    sample_rate = model.sample_rate # extract sampling freq

    spikedata = {} # initialise dictionary
    for cluster in clusters.keys():
        clustergroup = clusters[cluster]
        if clustergroup==grupete: # only look at specified type of cluster, 0 = noise, 1 = MUA, 2 = GOOD, 3 = unsorted
            spiketimematrix = AttrDict({'spike_times': np.zeros(len(spiketimes[np.where(model.spike_clusters==cluster)]))})
            spiketimematrix.spike_times = spiketimes[np.where(model.spike_clusters==cluster)]
            spikedata[cluster] = spiketimematrix # data structure is a dictionary with attribute accessible spiketimes
            # attribute accessible means that spikedata.spike_times works, normal dictionaries would be spikedata[spike_times]

    nchan = model.metadata['n_channels']

    #model.close()

    return model, spikedata, sample_rate, nchan

def get_triggers(frame_list, frame_id, frame_list_type="continous"):
    f = frame_list[frame_list[:,0]==frame_id]
    if frame_list_type == "discrete": 
        f_triggers = []
        for i in range(0,len(f)):
            f_triggers.append(f[i,:][1])
    if frame_list_type == "continous":
        f_triggers = []
        for i in range(0,int(len(f)/150)):
            f_triggers.append(f[i*150,:][1])
    return f_triggers

def get_cluster_correlogram(clusters, cluster_id):
    clusnum = cluster_id
    spikes = clusters[clusnum].spike_times# getting the spike_times
    times1 = spikes
    acg1 = np.zeros(25)
    inds = np.arange(times1.shape[0])
    for s in np.arange(times1.shape[0]) :
        st = times1[s]
        times1nos = times1[np.where(s!=inds)[0]]
        bins = np.linspace(st, st+.025, 26)
        c, b = np.histogram(times1nos, bins=bins)
        acg1 += c
    acg1final = np.concatenate([acg1[::-1],[0], acg1])
    x=np.concatenate([(np.linspace(-0.025, 0, 26)+0.000)[:-1],[0],(np.linspace(0, 0.025, 26)-0.000)[1:]])
    x=x-0.0005

    correlogram = x
    return correlogram, acg1final 

def get_cluster_spikes(clusters, cluster_id):
    clusnum = cluster_id
    spikes = clusters[clusnum].spike_times# getting the spike_times

    return spikes

def get_cluster_waveforms (kwik_model,cluster_id):
    try:
        if (not(type(kwik_model) is KwikModel)):
            raise ValueError       
    except ValueError:
            print ("Exception: the first argument should be a KwikModel object")
            return
        
    clusters = kwik_model.spike_clusters
    try:
        if ((not(cluster_id in clusters))):
            raise ValueError       
    except ValueError:
            print ("Exception: cluster_id (%d) not found !! " % cluster_id)
            return
    
    idx=np.argwhere (clusters==cluster_id)
    #w=kwik_model.all_waveforms[idx]
    waveforms=w
    return waveforms

def plot_data(fig, peth, raster, before, after, nBins, correlogram, acg1final, waveforms, cluster_id, channel_id, experiment_id, frame_id):
    
    fig.suptitle(experiment_id+' Cluster '+str(cluster_id))
    #plot(x,acg1final,'or-',lw=2)
    
    plt.subplot(2,6,1)
    ax = plt.gca()
    for index, label in enumerate(ax.xaxis.get_ticklabels()): 
        if index % 2 == 0:
            label.set_visible(False)
#    spike_id = np.arange(waveforms.shape[0])
    waveform_size = waveforms.shape[1]
    ch = channel_id 
    m_spikes = np.mean(waveforms[:,:,ch],0)
    std_spikes = np.std(waveforms[:,:,ch],0)
#    for i in spike_id:
#        spike = waveforms[i,:,ch]
#        x= np.arange(0,waveform_size)/30000
#        plt.plot (x,spike,color="red",alpha=0.5)
    x= np.arange(0,waveform_size)/30000
    plt.plot(x,m_spikes+std_spikes,"--",color="red",linewidth=2,alpha=0.3)
    plt.plot(x,m_spikes-std_spikes,"--",color="red",linewidth=2,alpha=0.3)
    plt.plot(x,m_spikes,"-",color="black",linewidth=4,alpha=0.3)    
    
    plt.subplot(2,6,7)
    ax = plt.gca()
    for index, label in enumerate(ax.xaxis.get_ticklabels()): 
        if index % 2 == 0:
            label.set_visible(False)
    plt.bar(correlogram,acg1final,width=0.001,color='r',linewidth=2)
    plt.xlim([-0.0255,0.0255])
    #plt.axvline(0,color='b',linewidth=0.5)

    if frame_id==1:
        plt.subplot(2,6,2)
        ax = plt.gca()
        ax.axes.xaxis.set_ticklabels([])
    elif frame_id==2:
        plt.subplot(2,6,3)
        ax = plt.gca()
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
    elif frame_id==3:
        plt.subplot(2,6,4)
        ax = plt.gca()
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
    elif frame_id==4:
        plt.subplot(2,6,5)
        ax = plt.gca()
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
    else:
        plt.subplot(2,6,6)
        ax = plt.gca()
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
    
    plt.xlim(0,(before+after) * nBins)
    plt.ylim(0,150)
    plt.bar(np.arange(len(peth)),peth*nBins,linewidth=2,color='k')
    plt.axvline(x= before * nBins, linewidth=0.5, color='b') 
    plt.axvline(x= after * nBins, linewidth=0.5, color='b') 

    if frame_id==1:
        plt.subplot(2,6,8)
        ax = plt.gca()
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
    elif frame_id==2:
        plt.subplot(2,6,9)
        ax = plt.gca()
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
    elif frame_id==3:
        plt.subplot(2,6,10)
        ax = plt.gca()
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
    elif frame_id==4:
        plt.subplot(2,6,11)
        ax = plt.gca()
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
    else:
        plt.subplot(2,6,12)
        ax = plt.gca()
        ax.axes.yaxis.set_ticklabels([])
        plt.xlabel("                                  Time(ms)")
        for index, label in enumerate(ax.xaxis.get_ticklabels()):
            if index % 2 == 0:
                label.set_visible(False)
        
    plt.xlim(0,(before+after) * nBins)
    plt.axvline(x= before * nBins, linewidth=0.5, color='b') 
    plt.axvline(x= after * nBins, linewidth=0.5, color='b') 
    for i in range(0,len(raster)):
        plt.scatter(np.array(raster[i])*nBins, np.ones(len(raster[i]))*-500*i, marker = '.', color='k',s=20)
        
    
    