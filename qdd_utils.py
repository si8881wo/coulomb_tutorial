import numpy as np
import qmeq
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colorbar import Colorbar

# for parallelelization
from concurrent.futures import ProcessPoolExecutor

# for progress bar
from tqdm import tqdm as tqdm

# define a class whose properties are the parameters of the system
class Parameters:
    """
    Class to store the parameters of the system.
    """
    def __init__(self, u=100., Tl=10., Tr=10., gamma=1., kerntype='Pauli'):
        """
        Initialize the Parameters object.

        Parameters:
        - vg (float): Gate voltage.
        - vb (float): Bias voltage.
        - B (float): Magnetic field.
        - u (float): Coulomb interaction strength.
        - Tl (float): Temperature of the left lead.
        - Tr (float): Temperature of the right lead.
        - gamma (float): Coupling strength between leads and dots.
        - kerntype (str): Type of kernel used for calculations.
        - dband (float): Energy band width.
        - countingleads (list): Leads for counting statistics.
        """
        self.u = u
        self.Tl = Tl
        self.Tr = Tr
        self.gamma = gamma
        self.kerntype = kerntype

class Stability_Diagram:
    def __init__(self, parameters, gate_range, bias_range):
        self.current = None
        self.conductance = None
        self.parameters = parameters
        self.gate = gate_range
        self.bias = bias_range
        self.temperatures = [parameters.Tl, parameters.Tr]
        self.gamma = parameters.gamma
        self.coulomb = parameters.u

    
    def anderson(self,vg=0.,vb=0.,parameters=Parameters()):
        """
        Build Anderson model with 4 leads and 2 dots, i.e. a single spinfull level.
        
        Parameters:
        - vg (float): Gate voltage.
        - vb (float): Bias voltage.
        - B (float): Magnetic field.
        - u (float): Coulomb interaction strength.
        - Tl (float): Temperature of the left lead.
        - Tr (float): Temperature of the right lead.
        - gamma (float): Coupling strength between leads and dots.
        - kerntype (str): Type of kernel used for calculations.
        - dband (float): Energy band width.
        - countingleads (list): Leads for counting statistics.
        
        Returns:
        - system (qmeq.Builder): Anderson model system.
        """
        # extract parameters from parameters object
        u = parameters.u
        Tl = parameters.Tl
        Tr = parameters.Tr
        gamma = parameters.gamma
        
        n = 2
        h = {(0,0):-vg, (1,1):-vg}
        U = {(0,1,1,0):u}

        nleads = 4
        mulst = {0:vb/2, 1:-vb/2, 2:vb/2, 3:-vb/2}
        tlst = {0:Tl, 1:Tr, 2:Tl, 3:Tr}

        t = np.sqrt(gamma/np.pi/2)
        tleads = {(0, 0):t, (1, 0):t, (2, 1):t, (3, 1):t}

        system = qmeq.Builder(nsingle=n, hsingle=h, coulomb=U, nleads=nleads, 
                            mulst=mulst, tlst=tlst, tleads=tleads, dband=1e4)
        
        return system
    
    def solve_bias_gate(self,vg,vb,params,dv=0.01):
        """
        Set the bias and gate voltages for a given system.

        Parameters:
        system (object): The system object representing the physical system.
        vg (float): Gate voltage.
        vb (float): Bias voltage.
        """
        system=self.anderson(vg,vb,params)
        system.solve()
        system_p=self.anderson(vg,vb+dv,params)
        system_p.solve()
        system_m=self.anderson(vg,vb-dv,params)
        system_m.solve()

        return system.current, (system_p.current - system_m.current) / (2*dv)
    
    def calculate_stability_diagram_parallel(self):
        """
        Calculate the stability diagram for the Anderson model.
        
        Parameters:
        - gate_range (list): Range of gate voltages.
        - bias_range (list): Range of bias voltages.
        
        Returns:
        - stability_diagram (np.ndarray): Stability diagram.
        """

        Vg = self.gate
        Vb = self.bias

        params = self.parameters

        with ProcessPoolExecutor() as executor:
            results = np.array(list(tqdm(executor.map(self.solve_bias_gate, np.tile(Vg,len(Vb)), np.repeat(Vb,len(Vg)), [params]*len(Vg)*len(Vb)),total=len(Vg)*len(Vb))),dtype=object)
        
        self.current = np.array([result[0][0]+result[0][2] for result in results]).reshape(len(Vg),len(Vb))
        self.conductance = np.array([result[1][0]+result[1][2] for result in results]).reshape(len(Vg),len(Vb))
           
    def calculate_stability_diagram(self):
        """
        Calculate the stability diagram for the Anderson model.
        
        Parameters:
        - gate_range (list): Range of gate voltages.
        - bias_range (list): Range of bias voltages.
        
        Returns:
        - stability_diagram (np.ndarray): Stability diagram.
        """

        Vg = self.gate
        Vb = self.bias

        params = self.parameters

        results = np.array(list(tqdm(map(self.solve_bias_gate, np.tile(Vg,len(Vb)), np.repeat(Vb,len(Vg)), [params]*len(Vg)*len(Vb)),total=len(Vg)*len(Vb))),dtype=object)
        
        self.current = np.array([result[0][0]+result[0][2] for result in results]).reshape(len(Vg),len(Vb))
        self.conductance = np.array([result[1][0]+result[1][2] for result in results]).reshape(len(Vg),len(Vb))
        
    def plot_stability_diagram(self,gate_cut=0.,bias_cut=0.):
        # define figure and grid
        fig = plt.figure(figsize=(10,5))
        gs = gridspec.GridSpec(3, 5, width_ratios=[1,4,0.33,4,1], height_ratios=[1,8,2], wspace=0.05, hspace=0.05)
        # current
        ax_curr = plt.subplot(gs[1,1])
        curr = ax_curr.imshow(self.current, extent=[self.gate[0], self.gate[-1], self.bias[0], self.bias[-1]], aspect='auto', origin='lower',cmap='bwr')
        ax_curr.set_xticks(np.linspace(self.gate[0], self.gate[-1], 5)) 
        ax_curr.set_xticklabels([])
        ax_curr.set_yticks(np.linspace(self.bias[0], self.bias[-1], 5))
        ax_curr.set_yticklabels([])
        ax_curr.tick_params(axis='both', direction='in', bottom=True, top=True, left=True, right=True)
        # colorbar
        cbax_curr = plt.subplot(gs[0,1])
        cb_curr = Colorbar(ax=cbax_curr, mappable=curr, orientation='horizontal',ticklocation='top')
        cb_curr.set_label('Current',labelpad=10, fontsize = 'large', fontweight = 'bold')
        # conductance
        ax_cond = plt.subplot(gs[1,3])
        cond = ax_cond.imshow(self.conductance, extent=[self.gate[0], self.gate[-1], self.bias[0], self.bias[-1]], aspect='auto', origin='lower',cmap='YlGnBu')
        # ticks on all sides pointing inwards without labels
        ax_cond.set_xticks(np.linspace(self.gate[0], self.gate[-1], 5)) 
        ax_cond.set_xticklabels([])
        ax_cond.set_yticks(np.linspace(self.bias[0], self.bias[-1], 5))
        ax_cond.set_yticklabels([])
        ax_cond.tick_params(axis='both', direction='in', bottom=True, top=True, left=True, right=True)
        # colorbar
        cbax_cond = plt.subplot(gs[0,3])
        cb_cond = Colorbar(ax=cbax_cond, mappable=cond, orientation='horizontal',ticklocation='top')
        cb_cond.set_label('Conductance',labelpad=10, fontsize = 'large', fontweight = 'bold')
        
        # gate and bias cuts
        # fixed bias cut
        # find the index of the biasvoltage closest to desired
        bias_cut_idx = np.argmin(np.abs(self.bias - bias_cut))
        bias_cut_v = self.bias[bias_cut_idx]
        # plot the current at given bias
        ax_curr_cut = plt.subplot(gs[2,1])
        curr_cut = ax_curr_cut.plot(self.gate,self.current[bias_cut_idx,:])
        ax_curr_cut.set_xmargin(0)
        ax_curr_cut.set_xticks(np.linspace(self.gate[0], self.gate[-1], 5))
        ax_curr_cut.set_xlabel('Gate voltage')
        ax_curr_cut.set_ylim(-1,1)
        ax_curr_cut.tick_params(axis='y',labelleft=False,left=False)
        # add a horizontal line at the bias cut
        ax_curr.axhline(y=bias_cut_v, color='k', linestyle='--')
        # plot the conductance at given bias
        ax_cond_cut = plt.subplot(gs[2,3])
        cond_cut = ax_cond_cut.plot(self.gate,self.conductance[bias_cut_idx,:])
        ax_cond_cut.set_xmargin(0)
        ax_cond_cut.set_xticks(np.linspace(self.gate[0], self.gate[-1], 5))
        ax_cond_cut.set_xlabel('Gate voltage')
        ax_cond_cut.tick_params(axis='y',labelleft=False,left=False)
        # add a horizontal line at the bias cut
        ax_cond.axhline(y=bias_cut_v, color='k', linestyle='--')

        # fixed gate cut
        # find the index of the gatevoltage closest to desired
        gate_cut_idx = np.argmin(np.abs(self.gate - gate_cut))
        gate_cut_v = self.gate[gate_cut_idx]
        # plot the current at given gate
        ax_curr_cut_g = plt.subplot(gs[1,0])
        curr_cut_g = ax_curr_cut_g.plot(self.current[:,gate_cut_idx],self.bias)
        ax_curr_cut_g.set_xlim(-1,1)
        ax_curr_cut_g.set_ymargin(0)
        ax_curr_cut_g.set_yticks(np.linspace(self.bias[0], self.bias[-1], 5))
        ax_curr_cut_g.set_ylabel('Bias voltage')
        #ax_curr_cut_g.set_xlabel('Current')
        ax_curr_cut_g.tick_params(axis='x',labelbottom=False,bottom=False)
        # add a vertical line at the gate cut
        ax_curr.axvline(x=gate_cut_v, color='k', linestyle='--')
        # plot the conductance at given gate
        ax_cond_cut_g = plt.subplot(gs[1,4])
        cond_cut_g = ax_cond_cut_g.plot(self.conductance[:,gate_cut_idx],self.bias)
        ax_cond_cut_g.set_yticks([])
        ax_cond_cut_g.set_ymargin(0)
        ax_cond_cut_g_r = ax_cond_cut_g.twinx()
        ax_cond_cut_g_r.set_yticks(np.linspace(self.bias[0], self.bias[-1], 5))
        ax_cond_cut_g_r.set_ylabel('Bias voltage',rotation=-90)
        #ax_cond_cut_g.set_xlabel('Conductance')
        ax_cond_cut_g.tick_params(axis='x',labelbottom=False,bottom=False)
        # add a vertical line at the gate cut
        ax_cond.axvline(x=gate_cut_v, color='k', linestyle='--')
        
        # add text to the gate and bias cut plots to show the cut values
        ax_curr_cut.text(-0.05, 0.5, f'$V_b = ${bias_cut_v:.2f}', horizontalalignment='center', verticalalignment='center', transform=ax_curr_cut.transAxes, rotation=90, fontsize='x-small')
        ax_cond_cut.text(1.05, 0.5, f'$V_b = ${bias_cut_v:.2f}', horizontalalignment='center', verticalalignment='center', transform=ax_cond_cut.transAxes, rotation=-90, fontsize='x-small')
        ax_curr_cut_g.text(0.5, 1.05, f'$V_g = ${gate_cut_v:.2f}', horizontalalignment='center', verticalalignment='center', transform=ax_curr_cut_g.transAxes, fontsize='x-small')
        ax_cond_cut_g.text(0.5, 1.05, f'$V_g = ${gate_cut_v:.2f}', horizontalalignment='center', verticalalignment='center', transform=ax_cond_cut_g.transAxes, fontsize='x-small')

        plt.show()

def quick_plot(u=100., Tl=10., Tr=10., gamma=1., gate_cut=0., bias_cut=0., npoints=31):
    params = Parameters(u, Tl, Tr, gamma, kerntype='Pauli')
    gate_range = np.linspace(-50,u+50,npoints)
    bias_range = np.linspace(-(u+80),u+80,npoints)
    stability = Stability_Diagram(params,gate_range,bias_range)
    stability.calculate_stability_diagram()
    stability.plot_stability_diagram(gate_cut,bias_cut)

def plot_temp(T=10., gate_cut=0., bias_cut=0., npoints=50):
    params = Parameters(0, T, T, 1., kerntype='Pauli')
    gate_range = np.linspace(-500,500,npoints)
    bias_range = np.linspace(-500,500,npoints)
    stability = Stability_Diagram(params,gate_range,bias_range)
    stability.calculate_stability_diagram()
    stability.plot_stability_diagram(gate_cut,bias_cut)

def plot_coulomb(u=0., gate_cut=0., bias_cut=0., npoints=50):
    params = Parameters(u, 10, 10, 1, kerntype='Pauli')
    gate_range = np.linspace(-100,400,npoints)
    bias_range = np.linspace(-550,550,npoints)
    stability = Stability_Diagram(params,gate_range,bias_range)
    stability.calculate_stability_diagram()
    stability.plot_stability_diagram(gate_cut,bias_cut)