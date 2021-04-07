'''
Spike statistics
----------------
In all functions below, spikes is a sorted list of spike times
'''
from numpy import *
from scipy import interpolate

# First-order statistics
def firing_rate(spikes):
    '''rr
    Rate of the spike train.
    '''
    return (len(spikes)-1)/(spikes[-1]-spikes[0])

def CV(spikes):
    '''
    Coefficient of variation.
    '''
    ISI=diff(spikes) # interspike intervals
    return std(ISI)/mean(ISI)

# Second-order statistics
def correlogram(T1,T2,width=0.02,bin=0.001,T=None, N1=-1, N2=-1):
    '''
    Returns a cross-correlogram with lag in [-width,width] and given bin size.
    T is the total duration (optional) and should be greater than the duration of T1 and T2.
    The result is in Hz (rate of coincidences in each bin).

    N.B.: units are discarded.
    TODO: optimise?
    '''
    # Remove units
    width=float(width)
    T1=array(T1)[:N1]
    T2=array(T2)[:N2]
    i=0
    j=0
    n=int(ceil(width/bin)) # Histogram length
    l=[]
    for t in T1:
        while i<len(T2) and T2[i]<t-width: # other possibility use searchsorted
            i+=1
        while j<len(T2) and T2[j]<t+width:
            j+=1
        l.extend(T2[i:j]-t)
    H,_=histogram(l,bins=arange(2*n+1)*bin-n*bin)
    
    # Divide by time to get rate
    if T is None:
        T=max(T1[-1],T2[-1])-min(T1[0],T2[0])
    # Windowing function (triangle)
    W=zeros(2*n)
    W[:n]=T-bin*arange(n-1,-1,-1)
    W[n:]=T-bin*arange(n)
    
    return H/W

def correlogram2(T1,T2,width=0.02,bin=0.001,T=None, N1=-1, N2=-1):
    '''
    Returns a cross-correlogram with lag in [-width,width] and given bin size.
    T is the total duration (optional) and should be greater than the duration of T1 and T2.
    The result is in Hz (rate of coincidences in each bin).

    N.B.: units are discarded.
    TODO: optimise?
    '''
    # Remove units
    width=float(width)
    T1=array(T1)[:N1]
    T2=array(T2)[:N2]
    i=0
    j=0
    n=int(ceil(width/bin)) # Histogram length
    l=[]
    for t in T1:
        while i<len(T2) and T2[i]<t-width: # other possibility use searchsorted
            i+=1
        while j<len(T2) and T2[j]<t+width:
            j+=1
        l.extend(T2[i:j]-t)
    H,_=histogram(l,bins=arange(2*n+1)*bin-n*bin)
    
    # Divide by time to get rate
    if T is None:
        T=max(T1[-1],T2[-1])-min(T1[0],T2[0])
    # Windowing function (triangle)
    W=zeros(2*n)
    W[:n]=T-bin*arange(n-1,-1,-1)
    W[n:]=T-bin*arange(n)
    
    return H/W, H

def autocorrelogram(T0,width=0.02,bin=0.001,T=None, N0=-1):
    '''
    Returns an autocorrelogram with lag in [-width,width] and given bin size.
    T is the total duration (optional) and should be greater than the duration of T1 and T2.
    The result is in Hz (rate of coincidences in each bin).

    N.B.: units are discarded.
    '''
    C = correlogram(T0,T0,width,bin,T, N1=N0, N2=N0)
    C[len(C)/2] = 0
    return C

def CCF(T1,T2,width=0.02,bin=0.001,T=None):
    '''
    Returns the cross-correlation function with lag in [-width,width] and given bin size.
    T is the total duration (optional).
    The result is in Hz**2:
    CCF(T1,T2)=<T1(t)T2(t+s)>

    N.B.: units are discarded.
    '''
    return correlogram(T1,T2,width,bin,T)/bin

def ACF(T0,width=0.02,bin=0.001,T=None):
    '''
    Returns the autocorrelation function with lag in [-width,width] and given bin size.
    T is the total duration (optional).
    The result is in Hz**2:
    ACF(T0)=<T0(t)T0(t+s)>

    N.B.: units are discarded.
    '''
    return CCF(T0,T0,width,bin,T)

def CCVF(T1,T2,width=0.02,bin=0.001,T=None):
    '''
    Returns the cross-covariance function with lag in [-width,width] and given bin size.
    T is the total duration (optional).
    The result is in Hz**2:
    CCVF(T1,T2)=<T1(t)T2(t+s)>-<T1><T2>

    N.B.: units are discarded.
    '''
    return CCF(T1,T2,width,bin,T)-firing_rate(T1)*firing_rate(T2)

def ACVF(T0,width=0.02,bin=0.001,T=None):
    '''
    Returns the autocovariance function with lag in [-width,width] and given bin size.
    T is the total duration (optional).
    The result is in Hz**2:
    ACVF(T0)=<T0(t)T0(t+s)>-<T0>**2

    N.B.: units are discarded.
    '''
    return CCVF(T0,T0,width,bin,T)

def total_correlation(T1,T2,width=0.02,T=None):
    '''
    Returns the total correlation coefficient with lag in [-width,width].
    T is the total duration (optional).
    The result is a real (typically in [0,1]):
    total_correlation(T1,T2)=int(CCVF(T1,T2))/rate(T1)
    '''
    # Remove units
    width=float(width)
    T1=array(T1)
    T2=array(T2)
    # Divide by time to get rate
    if T is None:
        T=max(T1[-1],T2[-1])-min(T1[0],T2[0])
    i=0
    j=0
    x=0
    for t in T1:
        while i<len(T2) and T2[i]<t-width: # other possibility use searchsorted
            i+=1
        while j<len(T2) and T2[j]<t+width:
            j+=1
        x+=sum(1./(T-abs(T2[i:j]-t))) # counts coincidences with windowing (probabilities)
    return float(x/firing_rate(T1))-float(firing_rate(T2)*2*width)
    
def interspikeinterval(T1):
	bfr = int32(diff(T1)*1000)
	bfr = bfr[(bfr>-50) * (bfr<50)]+50
	bfr = append(bfr, 100-bfr)
	I = bincount(bfr, minlength=100)
	return I

def peth(event_times, trigger_times, before, after, bins_per_second):    
    ## Peri-event time histogram. All times in integer ms
    # go through triggers
    event_times = array(event_times)
    trigger_times = array(trigger_times)
    event_triggered_ensemble = []
    for i in range(len(trigger_times)):
        #  identify the events that are in the window around the stim 
        sel = array(((trigger_times[i] - before) < event_times) * (event_times < (trigger_times[i] + after)), dtype=bool)
        event_triggered_ensemble.extend(event_times[sel]-trigger_times[i]+before)
    if len(event_triggered_ensemble)>0:
        bfr = bincount(list(array(array(event_triggered_ensemble, dtype=float)*bins_per_second, dtype=int32)), minlength=int32((before+after)*bins_per_second))
    else:
        bfr = zeros(int32((before+after)*bins_per_second))
    bfr = bfr/float(len(trigger_times))
    return bfr

def raster(event_times, trigger_times, before, after):
	## Peri-event time histogram.
	# go through triggers
	event_triggered_list = []
	for i in range(len(trigger_times)):
		#  identify the events that are in the window around the stim 
		sel = array(((trigger_times[i] - before) < event_times) * (event_times < (trigger_times[i] + after)), dtype=bool)
		event_triggered_list.append(event_times[sel]-trigger_times[i]+before)
	return event_triggered_list
    
def RSU_or_FSU(spike_shape):
    # spike shape analysis to identify RSU versus FSU spike shapes .
    # select the largest waveform of the 8
#    a = -0.6
#    b = 110
    a = -1
    b = 0.51*320#110/320.    
    
    RSU_FSU = []
    M = []
    if shape(spike_shape)[0] == 64:
        for j in range(8):
            peak = max(spike_shape[10:41,j])
            M.append(peak)
        sel = spike_shape[10:41,argmax(M)]
    else:
        for j in range(8):
            peak = max(spike_shape[:,j])
            M.append(peak)
        sel = spike_shape[:,argmax(M)]
        
    # smooth the spike shape to best estimate the spike half-width 
    X = interpolate.UnivariateSpline(linspace(0,1,len(sel)), sel)
    sel_interp = X(linspace(0,1,320))
    sel_interp = (sel_interp-min(sel_interp))/(max(sel_interp)-min(sel_interp))
    # compute the 2 dimensions of the space to split RSU/FSU
    M = argmax(sel_interp)
    if M<len(sel_interp) and sum(sel_interp[M:]<0.15):
        downward_duration = min(where(sel_interp[M:]<0.15)[0])
    else:
        downward_duration = NaN

    if M>0 and sum(sel_interp[:M]<0.5):
        upward_duration = M-max(where(sel_interp[:M]<0.5)[0])
    else:
        upward_duration = NaN
            
    RSU_FSU = 0 # type cannot be identified    
    if not(isnan(upward_duration)) and not(isnan(downward_duration)):
        if (a*downward_duration + b) < upward_duration :
            RSU_FSU = 1 # regular spiker
        else:
            RSU_FSU = 2 # fast spiker
    return RSU_FSU