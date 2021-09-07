from __future__ import division, unicode_literals, absolute_import
import numpy as np

from mlgw_bns import Model

class mlgw_bns_wrapper():
    
    def __init__(self, initial_frequency, df):

        self.model = Model(f0_hz=initial_frequency, df_hz=df, srate_interp_hz=srate)
        self.model.generate_dataset()
        self.model.train_nn()
        
        # self.srate      = srate
        # self.seglen     = seglen
        # self.dt         = 1./self.srate
        # self.times      = np.arange(int(self.srate*self.seglen))*self.dt - self.seglen/2.

    def __call__(self, f, params):
        """Evaluate

        Args:
            params (dict): Parameters to pass to mlgw_bns.
                mtot: total mass in solar masses
                q: ratio of the masses, larger/smaller
                s1z: aligned spin component of the larger body
                s2z: aligned spin component of the smaller body
                lambda1: tidal deformability of the larger body
                lambda2: tidal deformabiltiy of the smaller body
                distance: luminosity distance in Mpc
                iota: inclination angle 
                phi_ref: reference phase
        """
                
        hp, hc = self.model.predict_with_extrinsic(f, params)