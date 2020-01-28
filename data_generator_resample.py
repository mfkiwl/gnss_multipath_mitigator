import numpy as np
from data_generator import CorrDatasetV2

class CorrDatasetResample(CorrDatasetV2):
    def __init__(self, discr_size_fd, scale_code, 
               Tint=10**-3, 
               multipath_option=False,
               delta_tau_interv=None,
               delta_dopp_interv=None,
               alpha_att_interv=None,
               delta_phase_set=None,
               cn0_log_set=None,
               tau=[0,2], dopp=[-1000,1000]):
        
        # initialize parent class constructor
        CorrDatasetV2.__init__(self,
                               discr_size_fd=discr_size_fd, 
                               scale_code=scale_code, 
                               Tint=Tint,
                               multipath_option=multipath_option,
                               tau=tau, 
                               dopp=dopp)
        
        #self.discr_size_fd = discr_size_fd
        #self.scale_code = scale_code
        
        #self.Tint = Tint
        #self.w = w # correlator bandwidth
        #self.b_rf = 10**6 # RF frontend bandwidth
        
        #self.multipath_option = multipath_option
        self.delta_tau_interv = delta_tau_interv
        self.delta_dopp_interv = delta_dopp_interv
        self.alpha_att_interv = alpha_att_interv      
        self.delta_phase_set = delta_phase_set
        self.cn0_log_set = cn0_log_set
        
        #self.sign_amp = 1
        #self.sign_power = 8 * self.sign_amp / self.Tint**2
        #self.noise_psd = self.sign_power / 10**(0.1*self.cn0_log)
    
    def build(self, nb_samples=10, ref_features=False):
        data_samples = []
        for i in range(nb_samples):
            data = {}
            
            # Generate matrices: main, multipath
            if self.multipath_option:
                # Set random delta_tau/ delta_dopp/ alpha/ phase/ cn0_log
                delta_taui = np.random.uniform(low=self.delta_tau_interv[0], high=self.delta_tau_interv[1])
                delta_doppi = np.random.uniform(low=self.delta_dopp_interv[0], high=self.delta_dopp_interv[1])
                alpha_atti = np.random.uniform(low=self.alpha_att_interv[0], high=self.alpha_att_interv[1])
                
                delta_phasei = np.random.choice(self.delta_phase_set)
                self.cn0_log = np.random.choice(self.cn0_log_set)
                #print('phase check: ', self.delta_phase_set, delta_phasei)
                
                # generate main peak
                matrix, module, x, y = self.generate_peak()
                # generate multipath peak
                matrix_mp, module_mp, x, y = self.generate_peak(multipath=self.multipath_option,
                                                         delta_dopp=delta_doppi, 
                                                         delta_tau=delta_taui,
                                                         delta_phase=delta_phasei,
                                                         alpha_att=alpha_atti,
                                                         ref_features=ref_features)
                matrix[x:, y:] = matrix[x:, y:] + matrix_mp
                module[x:, y:] = module[x:, y:] + module_mp
                
                 # Log delta_tau/ delta_doppler 
                data['delta_tau'] = delta_taui
                data['delta_dopp'] = delta_doppi
                data['alpha'] = alpha_atti
                data['delta_phase'] = delta_phasei
            
            else:
                matrix, module, x, y = self.generate_peak(ref_features=ref_features)
                
                # Log delta_tau/ delta_doppler
                data['delta_tau'] = 0
                data['delta_dopp'] = 0
                data['alpha'] = 1
                data['delta_phase'] = 0
            
            data['table'] = matrix
            
            # Generate label
            #print('-------------------------------------')
            #print('multipath_option: ', self.multipath_option)
            
            
            if self.multipath_option:
                data['label'] = 1
            else:
                data['label'] = 0
            
            data_samples.append(data)
            
            # Compute reference features for given matrix
            # Not used here
            
        self.data_samples = np.array(data_samples)
        return self.data_samples