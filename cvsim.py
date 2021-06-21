# randles sevcik sim

import numpy as np
import matplotlib.pyplot as plt

# constants
F    = 96485        
R    = 8.314
T    = 298.15
Dox  = 5e-2
Dred = Dox
v    = 1
n    = 1

# formal potential
E0   = 0

# turning potentials
E1   = -0.3
E2   =  0.3

size = 500      # size of the simulated diffusion layer
step = 10       # every x-th tick, the result is plotted to the pyplots
j    = 0        # counter to index the current potential. since it's determined
dt0  = 2e-4     # dt, "infinitesimal" time step of a tick
trace = 2000    # this many values for i and u are stored, before they are deleted from the plots
U    = np.full((trace,),E1)     # array for potential values
I    = np.zeros((trace,))       # array for current values

c0   = 1        # starting concentration of prominent species
# index of these concentration profiles corresponds to distance from electrode
cOx  = np.full((size,),c0,dtype='float64')  # concentration profile of Ox
cRed = np.zeros((size,))                    # --"-- of Red

# triangular signal
E = np.linspace(E1,E2,round((abs(E2-E1)/v)/dt0))
E = np.concatenate((E,E[-1:0:-1]))

# plotting
plt.ion()
x = np.arange(size)

fig = plt.figure()
conc = plt.subplot(211)
cycv = plt.subplot(223)
poti = plt.subplot(224)
coxplt = conc.plot(x,cOx,linestyle='--')[0]
crdplt = conc.plot(x,cRed,c='#c40',linestyle=':')[0]
conc.spines['top'].set_visible(False)
conc.spines['right'].set_visible(False)
conc.set_xlabel('Distance from electrode')
conc.set_ylabel('Concentration')
conc.set_xticks([0])
conc.set_yticks([0,c0/2,c0])

i_cv = cycv.plot([],[])[0]
curr = cycv.plot([E1],[0],'o',c='#c40')[0]
cycv.spines['top'].set_visible(False)
cycv.spines['right'].set_visible(False)
cycv.set_xlabel('Electrode potential')
cycv.set_ylabel('Current')
cycv.set_yticks([0])
cycv.set_xticks([0])
cycv.set_xlim(E1-abs(E2-E1)*0.1,E2+abs(E2-E1)*0.1)
cycv.set_xticks([E1,E0,E2])
cycv.set_ylim((-0.011449162407792975, 0.015929193725920162))

VOLT = poti.plot(np.arange(-trace+1,1),[E1]*trace)[0]
volt = poti.plot([],[],'o',c='#c40')[0]
poti.set_xlim(-trace+1,trace//100+1)
poti.set_xticks([0])
poti.spines['top'].set_visible(False)
poti.spines['right'].set_visible(False)
poti.set_xlabel('Time')
poti.set_ylabel('Potential')
poti.set_xticks([-(trace+1)*0.95,0])
poti.set_xticklabels([-1,0])
poti.set_yticks([E1,(E1+E2)/2,E2])

fig.tight_layout()
fig.canvas.get_tk_widget().master.geometry("1400x1020+0+0")

# simulate diffusion
def flow(arr,D=1):
    lflow = arr*D*0.5
    rflow = arr*D*0.5
    
    ar  = np.zeros((arr.shape))
    ar[:-1]  -= rflow[:-1]
    ar[1:-1] -= lflow[1:-1]
    ar[:-1]  += lflow[1:]
    ar[1:-1] += rflow[:-2]
    
    return ar

# time steps
def tick():
    global cOx,cRed,Dox,Dred,n,t,U,I,dt0,j
    
    e = E[j%len(E)]
    Eq = np.exp(n*F/R/T * (e-E0))
    Q  = 1 / (Eq+1)
    O0 = cOx[0]
    R0 = cRed[0]
    
    cOx[0],cRed[0]  = (cOx[0]+cRed[0]) * Q, (cOx[0]+cRed[0]) * (1-Q)
    
    dO = cOx[0]  - O0
    dR = cRed[0] - R0
    
    fOx  = flow(cOx,Dox)
    dR  += fOx[0]
    fRed = flow(cRed,Dred)

    cOx  += fOx
    cRed += fRed
    
    ic  = -dO*n
    ia  = dR*n
    j  += 1
    return e,ic+ia

# plot stuff
def plot():
    coxplt.set_data(x,cOx)
    crdplt.set_data(x,cRed)
    curr.set_data([U[-1]],[I[-1]])
    
    cycv.set_ylim(I.min()-(I.max()-I.min())*0.1,I.max()+(I.max()-I.min())*0.1)
    
    i_cv.set_data(U,I)
    volt.set_data([0],[U[-1]])
    VOLT.set_ydata(U)
    plt.pause(0.001)

# infinite loop, call this method to start the simulation
# cancel with ctrl+c after selecting the python shell window with alt+tab
# because the pyplot windows are constantly updated, they are always on top
def draw(stopat=None,runs=None):
    global I,U

    def infty():
        i = 0
        while True:
            yield i
            i += 1

    s,z = 0,1
    for k in infty():
        E,i = tick()
        if not k%step:
            if E>U[-1] and z == -1:
                s += 1
                z  = 1
            if E<U[-1] and z == 1:
                s += 1
                z  = -1
            if stopat!=None and (E>stopat>U[-1] or U[-1]>stopat>E):
                break
            U = np.concatenate((U[1:],[E]))
            I = np.concatenate((I[1:],[i]))
            plot()
            if s//2 == runs: break
        
    
