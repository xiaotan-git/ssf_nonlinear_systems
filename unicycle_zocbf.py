import numpy as np
from scipy.optimize import minimize
import logging

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ZOCBF():
    def __init__(self,dt_dynamics,parameters):
        self.dynamics = dt_dynamics
        self.parameters = parameters
        self.u_guess = []
        pass
    
    # wrap to (-pi, pi]
    def wrap_to_pi(self,angle):
        wrapped = (angle + np.pi) % (2 * np.pi) - np.pi
        wrapped[wrapped == -np.pi] = np.pi
        return wrapped


    # ZOCBF safety function
    def h(self,x):
        r = self.parameters.radius
        fre = self.parameters.frequency
        amp = self.parameters.amplitude
        pha_shift = self.parameters.phase_shift
        y_shift = self.parameters.y_shift

        h_value = r**2 - (x[1] - (amp*(np.sin(fre*x[0] + pha_shift))+ y_shift))**2

        return h_value
    

    
    def safety_filter(self,t,x_set:np.ndarray,u_nom):
        dyn = self.dynamics
        gamma = self.parameters.gamma
        eps = self.parameters.eps
        control_bounds =  self.parameters.input_bounds

        def constraint_onestate(u,x):
            # constraint function, h(\phi(T;x,u)) - h(x) \geq - \gamma(h(x)) + \delta
            h_current = self.h(x)
            if h_current < 0:
                logger.warning('System is unsafe at current state: h_current: %f', h_current)
            x_next = dyn(t,x,u)
            h_next = self.h(x_next)
            return h_next - h_current + gamma*h_current - eps

        
        def objective(u):
            return np.linalg.norm(u - u_nom)**2
    
        # input bound
        initial_guess = self.u_guess.copy() # initial guess
        if x_set.ndim == 1:
            constraints = {'type': 'ineq', 'fun': lambda u: constraint_onestate(u,x_set)}  # 1 constraint
        else:
            # multiple constraints
            # !!!Tricky case!!! Below is wrong due to late binding issue when use lambda function in a loop
            # constraints = tuple({'type': 'ineq', 'fun': lambda u: constraint_onestate(u, row)} for row in x_set)
            # the following is correct by introducing another lambda parameter x and explicitly let it be row in the loop
            constraints = tuple({'type': 'ineq', 'fun': lambda u, x=row: constraint_onestate(u, x)} for row in x_set)

        # solve using 'SLSQP'
        try:
            result = minimize(
                objective,
                initial_guess,
                method='SLSQP',
                bounds=control_bounds,
                constraints=constraints,
                options={'disp': False, 'maxiter': 500}
            )
        except Exception as e:
            logger.error('Optimization failed with exception: %s', e)
            result = None

        if result is not None and result.success:
            u_filtered = result.x
        else:
            u_filtered = u_nom  # nominal control if safety filter fails
            logger.warning('Filtering failed at state %s.', x_set)
            if result is not None:
                logger.warning('Optimization result: %s', result)

        return u_filtered

