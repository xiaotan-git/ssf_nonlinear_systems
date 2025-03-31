import numpy as np
from typing import List
import itertools
from lib.estimate import get_poly_estimates,get_gauss_estimates
import logging

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def nchoosek(v: List[int], k: int) -> List[List[int]]:
    """
    Returns a list of lists containing all possible combinations of the elements of vector v taken k at a time.
    
    Args:
        v (List[int]): A list of elements to take combinations from.
        k (int): The number of elements in each combination.
    
    Returns:
        List[List[int]]: A list of lists where each sublist is a combination of k elements from v.
    """
    return [list(comb) for comb in itertools.combinations(v, k)]


class UnicycleModel:
    '''
    Unicycle ODE of the form:
    
    model inputs - u, w
    states - x, y, theta
    measured outputs - x, y

    \dot{x} = u*cos(theta)
    \dot{y} = u*sin(theta)
    \dot{theta} = w
    '''

    def __init__(self,parameters):
        self.parameters = parameters
        

    # Runge-Kutta (4th-order)
    def rk4(self, rhs, t_span, s0):
        t0, tf = t_span
        dt = tf - t0
        s = s0
        k1 = rhs(t0, s)
        k2 = rhs(t0 + dt / 2, s + dt * k1 / 2)
        k3 = rhs(t0 + dt / 2, s + dt * k2 / 2)
        k4 = rhs(t0 + dt, s + dt * k3)
        s_next = s + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        return [s, s_next]
    
    def forward_euler(self,rhs,t_span,s0):
        t0,tf = t_span
        dt = tf - t0
        s_next = s0 + rhs(t0,s0)*dt
        return [s0, s_next]

    def atan(self, y, x, y_b, x_b):
        theta_hat = np.arctan2(y, x)
        theta_low = np.zeros_like(theta_hat)
        theta_high = np.zeros_like(theta_hat)

        x_l = x - x_b
        x_h = x + x_b
        y_l = y - y_b
        y_h = y + y_b

        theta_low[(x_h>0) & (y_l>0)] = np.arctan2(y_l, x_h)[(x_h>0) & (y_l>0)]
        theta_low[(x_h<0) & (y_h>0)] = np.arctan2(y_h, x_h)[(x_h<0) & (y_h>0)]
        theta_low[(x_l<0) & (y_h<0)] = np.arctan2(y_h, x_l)[(x_l<0) & (y_h<0)]
        theta_low[(x_l>0) & (y_l<0)] = np.arctan2(y_l, x_l)[(x_l>0) & (y_l<0)]

        theta_high[(x_l>0) & (y_h>0)] = np.arctan2(y_h, x_l)[(x_l>0) & (y_h>0)]
        theta_high[(x_l<0) & (y_l>0)] = np.arctan2(y_l, x_l)[(x_l<0) & (y_l>0)]
        theta_high[(x_h<0) & (y_l<0)] = np.arctan2(y_l, x_h)[(x_h<0) & (y_l<0)]
        theta_high[(x_h>0) & (y_h<0)] = np.arctan2(y_h, x_h)[(x_h>0) & (y_h<0)]
        
        theta_low[(x_l<0) & (y_l<0) & (x_h>0) & (y_h>0)] = -10*np.pi
        theta_high[(x_l<0) & (y_l<0) & (x_h>0) & (y_h>0)] = 10*np.pi
        theta_high[theta_high<theta_hat] += 2*np.pi
        theta_low[theta_low>theta_hat] -= 2*np.pi

        return theta_hat, theta_low, theta_high


    def ode(self, x, u):
        '''
        RHS of ode
            x - state
            v - forward velocity
            w - angular velocity
        '''
        v, w = u[0], u[1]
        dxdt = np.zeros(3)
        dxdt[0] = v*np.cos(x[2])
        dxdt[1] = v*np.sin(x[2])
        dxdt[2] = w
        return dxdt
    
    def dt_dynamics(self,t,s,u):
        rhs = lambda t,x: self.ode(x,u)
        t_span = [t, t+self.parameters.sample_time]
        [s, s_next] = self.rk4(rhs,t_span,s)
        # [s, s_next] = self.forward_euler(rhs,t_span,s)
        s_next[2] = self.wrap_to_pi(s_next[2])
        return s_next

    # wrap to (-pi, pi]
    def wrap_to_pi(self,angle):
        wrapped = (angle + np.pi) % (2 * np.pi) - np.pi
        if wrapped == -np.pi: wrapped  = np.pi 
        return wrapped
    
    def invert_outputs(self, x_hat, y_hat, x_bound, y_bound):
        '''
        Return estimated states and state bounds from estimated outputs and output bounds
        for N time steps
        
        Inputs:
            x_hat - (2,N) - x output and it's first derivative
            y_hat - (2,N) - y output and it's first derivative
            x_bound - (2,N) - bound for x output and it's first derivative
            y_bound - (2,N) - bound for y output and it's first derivative

        Outputs:
            state_hat - (3,N) - estimated states
            state_low - (3,N) - lower bound for state estimate
            state_low - (3,N) - upper bound for state estimate
        '''

        theta_hat, theta_low, theta_high = self.atan(y_hat[1], x_hat[1], y_bound[1], x_bound[1])
        state_hat = np.array([x_hat[0], y_hat[0], theta_hat])
        state_low = np.array([x_hat[0] - x_bound[0], y_hat[0] - y_bound[0], theta_low])
        state_high = np.array([x_hat[0] + x_bound[0], y_hat[0] + y_bound[0], theta_high])

        return state_hat, state_low, state_high

    def measure(self,state,fake_state=None,attacked_sensors: List[int]=None):
        if fake_state is None:
            return self.c(state)
        else:
            measurement = [self.c(state,i) if i not in attacked_sensors else self.c(fake_state,i) 
                           for i in range(5)]
            return np.array(measurement)


    def c(self,x,ind= None):
        if ind is None:
            return np.array([
                x[0],
                x[1],
                np.sqrt(x[0]**2 + x[1]**2),
                np.arctan2(x[1],x[0]),
                x[2]])
        else:
            return self.c(x)[ind] # Return specific index from the computed array
        
    def get_plausible_state(self,timestep,controls,measurements:np.ndarray):
        s = self.parameters.attack_count
        p = len(self.parameters.sensor_dict.keys())
        l = self.parameters.data_length - 1
        if timestep<2*l+1:
            current_measurement = measurements[-1]
            return current_measurement[[0,1,4]]
        else:
            x_set = []
            combinations = nchoosek(range(p),p-s)
            err_list = []
            hat_x_curr_list = []
            for comb in combinations:
                control_data = controls[-2*l:] # shape (2l,m)
                measure_data_comb = measurements[-2*l-1:][:,comb]  # shape (2l+1,p-s)
                
                # hat_x_init is the estimated state L steps before the current time
                hat_x_init, hat_x_current = self.observ_map(control_data,measure_data_comb,comb)
                consistency,err = self.consistency_check(hat_x_init,control_data[-l:],measure_data_comb[-l-1:],comb)
                

                err_list.append(err)
                hat_x_curr_list.append(hat_x_current)
                if consistency:
                    x_set.append(hat_x_current)
                    # print(f'\n sensor comb:{comb}, consistency err:{err}')
                    # print(f'hat_x_current:{hat_x_current}')

            # if no combination passes the consistency check, give the combination with the least error bound
            if not x_set:
                print('no feasible state is found! Choose the one with least consistency error')
                # Find the index of the least number
                min_index = err_list.index(min(err_list))

                # Get the corresponding element from the second list
                hat_x_curr_min = hat_x_curr_list[min_index]
                x_set.append(hat_x_curr_min)
            
            x_set = np.array(x_set)
            if x_set.shape[0]>1:
                x_set = UnicycleModel.remove_duplicate_states(x_set)
            return x_set

    def get_average_state(self,x_set:np.ndarray):
        if x_set.ndim == 1:
            return x_set
        else:
            return np.mean(x_set, axis=0)
        
    def consistency_check(self,x_init,controls,measurements,sensor_ind):
        if measurements.shape[0] < controls.shape[0] + 1:
            ValueError('measurement should be at least 1 step longer')

        if np.any(np.isnan(x_init)):
            return False, np.inf
        
        error_level = self.parameters.consistency_err

        measure_errs = []
        state = x_init
        err = np.linalg.norm(measurements[0] - self.c(state,sensor_ind))
        measure_errs.append(err)
        # propograte along dynamics
        for step in range(controls.shape[0]):
            state = self.dt_dynamics(step,state,controls[step])
            err = np.linalg.norm(measurements[step+1] - self.c(state,sensor_ind))
            measure_errs.append(err)

        measure_errs = np.array(measure_errs)
        max_err = np.max(measure_errs)
        if max_err > error_level:
            # print(f'consistency check fails for sensors {sensor_ind}, max error: {max_err}')
            return False,max_err
        else:
            return True,max_err
    
    # TODO: implement uncertainty propogration in these differential observability maps.
    def observ_map(self,control_data:np.ndarray,measure_data:np.ndarray,sensor_ind:List[int]):
        
        def diffobser_map012(y_hat,y_bound):
            '''
            Return estimated state and state bound from estimated output and its first-order time derivative 
            and output bounds for 1 time step
            
            Inputs:
                y_hat   - (2,p) - y output and its first derivative at a time t
                y_bound - (2,p) - bound for y output and it's first derivative at a time t

            Outputs:
                state_hat - (3,) - estimated state
                state_bound - (3,) - bound for state estimate
            '''
            p1_hat,p2_hat = y_hat[0,0],y_hat[0,1]
            theta_hat = np.arctan2(y_hat[1,1],y_hat[1,0])
            state_hat = np.array([p1_hat,p2_hat,theta_hat])
            return state_hat,None

        def diffobser_map013(y_hat,y_bound):
            p1_hat,p2_hat = y_hat[0,0],y_hat[0,1]
            theta_hat = np.arctan2(y_hat[1,1],y_hat[1,0])
            state_hat = np.array([p1_hat,p2_hat,theta_hat])
            return state_hat,None

        def diffobser_map014(y_hat,y_bound):
            p1_hat,p2_hat = y_hat[0,0],y_hat[0,1]
            theta_hat = y_hat[0,2]
            state_hat = np.array([p1_hat,p2_hat,theta_hat])
            return state_hat,None

        def diffobser_map023(y_hat,y_bound):
            p1_hat,r,beta = y_hat[0,0],y_hat[0,1],y_hat[0,2]
            r_dot,beta_dot = y_hat[1,1],y_hat[1,2]
            # p1_hat = r*np.cos(beta)
            p2_hat = r*np.sin(beta)
            # after some math derivation, we found [p1   p2]  [\dot{p}_1] =  [  r\dot{r}    ]
            #                                      [-p2  p1]  [\dot{p}_2] =  [r^2 \dot{beta}]
            # no need to explicit solve the linear equality,but use the fact that
            # the det(A) = r^2 \geq 0 and inv(A) = 1/det(A) * [[p1 -p2],
            #                                                  [p2  p1]]
            p1_dot_scale = p1_hat*r*r_dot - p2_hat*beta_dot*(r**2)
            p2_dot_scale = p2_hat*r*r_dot + p1_hat*beta_dot*(r**2)
            theta_hat = np.arctan2(p2_dot_scale,p1_dot_scale)
            state_hat = np.array([p1_hat,p2_hat,theta_hat])
            return state_hat,None

        def diffobser_map024(y_hat,y_bound):
            p1_hat,p1_dot = y_hat[0,0],y_hat[1,0]
            r_hat,r_dot = y_hat[0,1],y_hat[1,1]
            theta_hat = y_hat[0,2]
            # from r dot{r} = p1*dot{p1} + p2*dot{p2}
            # unobservable singularity can happen
            # p2_dot = p1_dot*np.tan(theta_hat)
            # p2_hat = ((r_hat*r_dot) - p1_hat*p1_dot)/(p2_dot)
            try:
                p2_hat_mag = np.sqrt(r_hat**2 - p1_hat**2)
                p2p2_dot = r_hat*r_dot - p1_hat*p1_dot
                p2_dot_sign = np.sign(p1_dot*np.tan(theta_hat))
                p2_sign = np.sign(p2p2_dot*p2_dot_sign)
                p2_hat = p2_hat_mag*p2_sign
            except:                
                p2_dot = p1_dot*np.tan(theta_hat)
                p2_hat = ((r_hat*r_dot) - p1_hat*p1_dot)/(p2_dot)
            
            state_hat = np.array([p1_hat,p2_hat,theta_hat])
            return state_hat,None

        def diffobser_map034(y_hat,y_bound):
            p1_hat,beta_hat = y_hat[0,0],y_hat[0,1]
            theta_hat = y_hat[0,2]
            # singluarity can happen
            p2_hat = p1_hat*np.tan(beta_hat)
            state_hat = np.array([p1_hat,p2_hat,theta_hat])
            return state_hat,None

        def diffobser_map123(y_hat,y_bound):
            r,beta = y_hat[0,1],y_hat[0,2]
            r_dot,beta_dot = y_hat[1,1],y_hat[1,2]
            p2_hat = y_hat[0,0]

            p1_hat = r*np.cos(beta)
            # p2_hat = r*np.sin(beta)
            # after some math derivation, we found [p1   p2]  [\dot{p}_1] =  [  r\dot{r}    ]
            #                                      [-p2  p1]  [\dot{p}_2] =  [r^2 \dot{beta}]
            # no need to explicit solve the linear equality,but use the fact that
            # the det(A) = r^2 \geq 0 and inv(A) = 1/det(A) * [[p1 -p2],
            #                                                  [p2  p1]]
            p1_dot_scale = p1_hat*r*r_dot - p2_hat*beta_dot*(r**2)
            p2_dot_scale = p2_hat*r*r_dot + p1_hat*beta_dot*(r**2)
            theta_hat = np.arctan2(p2_dot_scale,p1_dot_scale)
            state_hat = np.array([p1_hat,p2_hat,theta_hat])
            return state_hat,None

        def diffobser_map124(y_hat,y_bound):
            p2_hat,p2_dot = y_hat[0,0],y_hat[1,0]
            r_hat,r_dot = y_hat[0,1],y_hat[1,1]
            theta_hat = y_hat[0,2]
            # from r dot{r} = p1*dot{p1} + p2*dot{p2}
            # unobservable singularity can happen
            # p1_dot = p2_dot/np.tan(theta_hat)
            # p1_hat = ((r_hat*r_dot) - p2_hat*p2_dot)/(p1_dot)
            try:
                p1_hat_mag = np.sqrt(r_hat**2 - p2_hat**2)
                p1p1_dot = r_hat*r_dot - p2_hat*p2_dot
                p1_dot_sign = np.sign(p2_dot*np.tan(theta_hat))
                p1_sign = np.sign(p1p1_dot*p1_dot_sign)
                p1_hat = p1_hat_mag*p1_sign
            except ValueError as e:
                print(f"Error: sqrt of negative value {r_hat**2 - p2_hat**2}")               
                p1_dot = p2_dot/np.tan(theta_hat)
                p1_hat = ((r_hat*r_dot) - p2_hat*p2_dot)/(p1_dot)

            state_hat = np.array([p1_hat,p2_hat,theta_hat])
            return state_hat,None

        def diffobser_map134(y_hat,y_bound):
            p2_hat,beta_hat = y_hat[0,0],y_hat[0,1]
            # singluarity can happen
            p1_hat = p2_hat/np.tan(beta_hat)
            theta_hat = y_hat[0,2]
            state_hat = np.array([p1_hat,p2_hat,theta_hat])
            return state_hat,None

        def diffobser_map234(y_hat,y_bound):
            r,beta,theta_hat = y_hat[0,0],y_hat[0,1],y_hat[0,2]
            p1_hat = r*np.cos(beta)
            p2_hat = r*np.sin(beta)

            state_hat = np.array([p1_hat,p2_hat,theta_hat])
            return state_hat,None

        def choose_correct_map(sensor_ind):
            # creat a dictionary to establish the correspondence
            mapping_dict = {
                (0,1,2): diffobser_map012,
                (0,1,3): diffobser_map013,
                (0,1,4): diffobser_map014,
                (0,2,3): diffobser_map023,
                (0,2,4): diffobser_map024,
                (0,3,4): diffobser_map034,
                (1,2,3): diffobser_map123,
                (1,2,4): diffobser_map124,
                (1,3,4): diffobser_map134,
                (2,3,4): diffobser_map234
            }
            #  lists can't be used as dictionary keys because they're mutable and unhashable.
            return mapping_dict.get(tuple(sensor_ind),False)

        # generate the output and its derivative estimate
        if len(sensor_ind) != len(self.parameters.sensor_dict.keys()) - self.parameters.attack_count:
            ValueError('only implemented for 3 sensors right now')

        t_seq = np.arange(measure_data.shape[0])*self.parameters.sample_time
        l = self.parameters.data_length - 1

        m1 = measure_data[:,0]
        m2 = measure_data[:,1]
        m3 = measure_data[:,2]
        # Upper bound for derivatives of measurements
        deri_max = [0, 1, 10]


        # polynomial fit
        # Noise_Mag = 0.001
        # m1_hat_seq, m1_bound_seq = get_poly_estimates(t_seq, m1, 1, 1, deri_max, l, delay=0, E=Noise_Mag)
        # m2_hat_seq, m2_bound_seq = get_poly_estimates(t_seq, m2, 1, 1, deri_max, l, delay=0, E=Noise_Mag)
        # m3_hat_seq, m3_bound_seq = get_poly_estimates(t_seq, m3, 1, 1, deri_max, l, delay=0, E=Noise_Mag)

        # Guassian fit
        m1_hat_seq, m1_bound_seq = get_gauss_estimates(t_seq, m1, 1, 3, deri_max, l, delay=0, sigma=2, reg=0)
        m2_hat_seq, m2_bound_seq = get_gauss_estimates(t_seq, m2, 1, 3, deri_max, l, delay=0, sigma=2, reg=0)
        m3_hat_seq, m3_bound_seq = get_gauss_estimates(t_seq, m3, 1, 3, deri_max, l, delay=0, sigma=2, reg=0)
        
        y_hat_init   = np.column_stack([m1_hat_seq[:,l],m2_hat_seq[:,l],m3_hat_seq[:,l]]) # output and derivative estimate at time l
        y_hat_curr   = np.column_stack([m1_hat_seq[:,-1],m2_hat_seq[:,-1],m3_hat_seq[:,-1]]) # output and derivative estimate at time 2l
        y_bound_init = np.column_stack([m1_bound_seq[:,l],m2_bound_seq[:,l],m3_bound_seq[:,l]]) # output and derivative estimate at time l
        y_bound_curr = np.column_stack([m1_bound_seq[:,-1],m2_bound_seq[:,-1],m3_bound_seq[:,-1]]) # output and derivative estimate at time l
        
        # print(f'y_hat_curr:{y_hat_curr}')
        # generate initial and current state estimation
        diffobser_map = choose_correct_map(sensor_ind)

        x_hat_init,_ = diffobser_map(y_hat_init,y_bound_init)
        x_hat_curr,_ = diffobser_map(y_hat_curr,y_bound_curr)

        
        return x_hat_init,x_hat_curr
    
    @classmethod
    def remove_duplicate_states(cls,possible_states:np.ndarray) -> np.ndarray:
        state_lst = [possible_states[0]]
        for state in possible_states[1:]:
            if not any(cls.is_same_state(state,st) for st in state_lst):
                state_lst.append(state)
        return np.array(state_lst)

    @staticmethod
    def is_same_state(st1:np.ndarray,st2:np.ndarray):
        error = st1.flatten() - st2.flatten()
        # print(f'error norm: {linalg.norm(error)}')
        if np.linalg.norm(error)<0.1:
            return True
        return False