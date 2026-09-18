from scipy.special import softmax
import numpy as np
import pandas as pd

# landscape_args = {'start_params': {'f': 0, 'k': 0},
#                   'attractor_coords': [[0, triple_cusp_coords['y0']], [triple_cusp_coords['x1'], triple_cusp_coords['y1']]],
#                   'gradients': triple_cusp_grad,
#                  }

# dynamics_args = {'stored_states': attractor_states.T,
#                  'rhos': [0, 10]
#                 }

# We have some potential of the form V(x, y)
# To calculate V(m), we project between m-space to xy-space
# To find the matrix that transforms between x and m, we need
# translations, rotations, and scaling in 4 dimensions

# In other words, this takes (0, y0) to (0, 0, 1)
# (x1, y1) -> (0, 1, 0), and (-x1, y1) -> (1, 0, 0)

# First we define matrix M, which moves the 2-dimensional triangle vertices
# to the vertices of a simplex on three axes
T = np.array([[1, 0, 0, -1],
              [0, 1, 0, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])
Tinv = np.array([[1, 0, 0, 1],
                 [0, 1, 0, 0],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]])
X = np.array([[1, 0, 0, 0],
              [0, 0, -1, 0],
              [0, 1, 0, 0],
              [0, 0, 0, 1]])
Xinv = np.array([[1, 0, 0, 0],
                 [0, 0, 1, 0],
                 [0, -1, 0, 0],
                 [0, 0, 0, 1]])
sq2 = np.sqrt(2) / 2
psi = np.arcsin(np.sqrt(2/3))
Y = np.array([[sq2, 0, sq2, 0],
              [0, 1, 0, 0],
              [-sq2, 0, sq2, 0],
              [0, 0, 0, 1]])
Yinv = np.array([[sq2, 0, -sq2, 0],
                 [0, 1, 0, 0],
                 [sq2, 0, sq2, 0],
                 [0, 0, 0, 1]])
Z = np.array([[np.cos(psi), -np.sin(psi), 0, 0],
              [np.sin(psi), np.cos(psi), 0, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])
M = np.dot(Tinv, np.dot(Xinv, np.dot(Yinv, np.dot(Z, np.dot(Y, np.dot(X, T))))))

# Define potentials by their gradients and the approximate positions of their attractors
def triple_cusp_grad(x, y, params, rng):
    gradX = 6*x**5 - 4*x**3 + 2*x + 4*x*y + params['f']
    gradYNoK = 16*y**3 - 2*y + 2*x**2
    gradYWithK = gradYNoK + params['k']
    return gradX, gradYWithK, gradYNoK


def het_flip_grad(x, y, params, rng):
    gradX = 4*x**3 + 4*x*y + params['f'] 
    gradYNoK = 4*y**3 - 3*y**2 + 2*x**2 - 2*y
    gradYWithK = gradYNoK + params['k']
    return gradX, gradYWithK, gradYNoK


def double_cusp_grad(x, y, params, rng):
    gradX = 4*x**3 + 8*x*y + params['f']
    gradYNoK = 4*y**3 - 3*y**2 + 4*x**2 + 2*y
    gradYWithK = gradYNoK - params['k']
    return gradX, gradYWithK, gradYNoK


# Make adjustments for the positions of the attractors in xy-space
def generate_x_to_m_matrix(y0, x1, y1):
    S = np.array([
        [np.sqrt(2)/(2*x1), 0, 0, 0],
        [0, np.sqrt(3/2)/(y0 - y1), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    T1 = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, -np.sqrt(2)/2 - (y1 * np.sqrt(3/2))/(y0 - y1)],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    theta = (3 * np.pi)/4
    R1 = np.array([
        [np.cos(theta), -np.sin(theta), 0, 0],
        [np.sin(theta), np.cos(theta), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    
    return np.dot(M, np.dot(R1, np.dot(T1, S)))

bifurcationValuesMap = {}
bifurcationValuesMap["Triple Cusp"] = {'y0': 0.345, 'x1': 0.838, 'y1': -0.537, 'gradient': triple_cusp_grad}
bifurcationValuesMap["Heteroclinic Flip"] = {'y0': 0.82, 'x1': 0.964, 'y1': -0.877, 'gradient': het_flip_grad}
bifurcationValuesMap["Double Cusp"] = {'y0': 0.085, 'x1': 1.349, 'y1': -0.911, 'gradient': double_cusp_grad}


class AttractorNetwork:
    def __init__(self, attractors, initialState, f0, k0, bifurcationType, rhos=[0, 10]):

        # Set params
        self.attractors = pd.concat(attractors, axis=1, join="inner")
        genesInCommon = np.intersect1d(initialState.index, self.attractors.index)
        self.attractors = self.attractors.loc[genesInCommon].T
        self.initialState = initialState.loc[genesInCommon].copy()
        self.state = self.initialState.copy()
        self.start_params = {'f': f0, 'k': k0}
        self.params = self.start_params.copy()
        self.N = len(self.state)  # Number of elements in the state vector.
        self.temperature = 1 / 100 # low temperature, so that there's not too much random jumping
        self.rng = np.random.default_rng()
        self.noiseScale = np.std(self.state)

        # Calculate the correlation matrix
        self.corr = (1 / self.N) * self.attractors.dot(self.attractors.T)
        self.inv_corr = np.linalg.inv(self.corr)

        # Unpack the attractor coordinates
        bifurcation = bifurcationValuesMap[bifurcationType]
        self.gradients = bifurcation['gradient']
        self.x0, self.y0, self.x1, self.y1 = (0, bifurcation["y0"], bifurcation["x1"], bifurcation["y1"])
        self.x_hat = np.array([1, 0, 0, 0])
        self.y_hat = np.array([0, 1, 0, 0])
        self.rho_delta, self.rho_p = rhos

        # Set matrix operation values
        x_to_m = generate_x_to_m_matrix(self.y0, self.x1, self.y1)
        self.m_to_x = np.linalg.inv(x_to_m)
        self.hx = np.dot(self.x_hat, self.m_to_x)
        self.hy = np.dot(self.y_hat, self.m_to_x)
        print("Done!")
        
    def dynamics(self, zeroKRate=0.8):
        # Calculate the dynamics of the system.
        m_up = (1/self.N)*self.attractors.dot(self.state) # upper index/contravariant
        m_low = self.inv_corr.dot(m_up) # lower index/covariant
        m_low_4d = np.array([*m_low, 1])
        x = np.dot(self.hx, m_low_4d)
        y = np.dot(self.hy, m_low_4d)
        
        inv_corr_4d = np.vstack((self.inv_corr, np.array([0, 0, 0])))
        inv_corr_4d = np.hstack((inv_corr_4d, np.array([0, 0, 0, 1]).reshape(-1, 1)))
        
        x_chain = np.dot(self.hx, inv_corr_4d)
        y_chain = np.dot(self.hy, inv_corr_4d)
        
        gradX, gradYWithK, gradYNoK = self.gradients(x, y, self.params, self.rng)
        # gradYWithK += rng.binomial(1, zeroKRate, size=x.shape[0])
        potentialGradWithK = (gradX*x_chain + gradYWithK*y_chain)[:3]
        potentialGradNoK = (gradX*x_chain + gradYNoK*y_chain)[:3]

#         delta = 1 - self.rho_delta*potential_grad
#         delta = 1 + self.rho_delta*np.abs(potential_grad)
        zeroK = self.rng.binomial(1, zeroKRate, size=self.N)
        zeroKIndices = np.where(zeroK == 0)[0]
        attraction = self.attractors.T.dot(softmax( (1/self.temperature) * (m_low - self.rho_p*potentialGradWithK) ))
        # print(attraction)
        attractionNoK = self.attractors.T.dot(softmax( (1/self.temperature) * (m_low - self.rho_p*potentialGradNoK) ))
        attraction.iloc[zeroKIndices] = attractionNoK.iloc[zeroKIndices]
        return attraction - self.state

        
    def update(self, zeroKRate=0.2):
        # Update the state of the network according to the dynamics.
        
        # "asynchronous updates": randomly choose which genes to update each time
#         asynch = self.rng.binomial(1, 1/timing_scale, size=N) # allow 1 in __ genes to update at each step
#         self.state += asynch*self.dynamics()/20
        
        # synchronous updates: all genes update simultaneously
        self.state += self.dynamics(zeroKRate=zeroKRate) / 20

    def increment_param(self, t, param, signal_bounds, finish_params):
        if (t > signal_bounds['start'][param]) and (t < signal_bounds['end'][param]):
            self.params[param] += (finish_params[param] - self.start_params[param])/(signal_bounds['end'][param] - signal_bounds['start'][param])

    def calculate_potentials(self):
        m_up = (1/self.N)*self.attractors.dot(self.state) # upper index/contravariant
        m_low = np.linalg.solve(self.corr, m_up) # lower index/covariant
        
        m_low_4d = np.array([*m_low, 1])
        
        x = np.dot(self.hx, m_low_4d)
        y = np.dot(self.hy, m_low_4d)
        
        # triple cusp
        external_potential = x**6 - x**4 + x**2 + 4*y**4 - y**2 + 2*(x**2)*y + self.params['f']*x + self.params['k']*y
        kinetic = (1/2)*(self.state.T).dot(self.state)
        internal_potential = self.temperature*np.log(np.sum(np.exp(m_low/self.temperature)))
        
        return (-internal_potential, external_potential, kinetic)
    
    def simulate(self, total_time, signal_bounds, finish_params, recorded_genes=None, add_noise=False, reset=True, time_interval=1, zeroKRate=0.2):
        # Simulate the network for a set amount of update steps and return a dataframe of output.
        magnetizations = []
        # internal_potentials = []
        # external_potentials = []
        # kinetics = []

        # gene_dict = {}
        # if recorded_genes is not None:
        #     for gene in recorded_genes:
        #         gene_dict[gene] = []
        gene_dict = {gene: [] for gene in recorded_genes} if recorded_genes is not None else {}
        if reset:
            self.params = self.start_params.copy()
            self.state = self.initialState.copy()
        for t in range(0, total_time, time_interval):
            if t % 100 == 0:
                print("Time: " + str(t))
            # Choose which parameters to increment
            self.increment_param(t, 'f', signal_bounds, finish_params) 
            self.increment_param(t, 'k', signal_bounds, finish_params) # Make random on per cell basis what value of k is based on probability
            
            if add_noise: # this just adds noise to the observations, not to the system
                detection_rate = 0.4
                dropout = self.rng.binomial(1, detection_rate, size=self.N)
                zero_indices = np.where(dropout == 0)[0]
                
                dropout_state = self.state.T.copy()
                # dropout_state.iloc[zero_indices] = -1
                
                noise = self.rng.standard_normal(size=self.N) * self.noiseScale
                magnetizations += [(1/self.N)*self.attractors.dot(dropout_state + noise)]
            else:
                magnetizations += [(1/self.N)*self.attractors.dot(self.state.T)]
                
            if recorded_genes is not None:
                for gene in recorded_genes:
                    gene_dict[gene] += [self.state.loc[gene]]
                    
            # internal, external, kinetic = self.calculate_potentials()
            # internal_potentials += [internal]
            # external_potentials += [external]
            # kinetics += [kinetic]
            
            self.update(zeroKRate=zeroKRate)
            t += time_interval
                    
        magnetizations = pd.DataFrame(magnetizations)
        if len(magnetizations.columns[0]) < 2:
            magnetizations = pd.DataFrame(magnetizations, columns = ['m{}'.format(index) for index in range(self.N)])
        order_params = self.inv_corr.dot(magnetizations.T)
        # order_params = np.linalg.inv(self.corr).dot(magnetizations.T)
        # order_params = np.linalg.solve(magnetizations, self.corr)
        
        output_df = pd.concat([pd.DataFrame(order_params, index = magnetizations.columns).T, magnetizations, pd.DataFrame(gene_dict)], axis=1)

        if len(magnetizations.columns[0]) < 2:
            output_df = pd.concat([pd.DataFrame(order_params, index = ['a{}'.format(index) for index in range(self.N)]).T,
                                magnetizations, pd.DataFrame(gene_dict)], axis=1)
        # return (output_df, internal_potentials, external_potentials, kinetics)
        return output_df


