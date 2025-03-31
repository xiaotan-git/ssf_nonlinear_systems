import numpy as np

def saturation(input, bounds):
    u_sat = np.copy(input)  # Create a copy to avoid modifying the original
    for i in range(input.shape[0]):
        u_sat[i] = np.clip(input[i], bounds[i, 0], bounds[i, 1])  # Apply the bounds for each control input
    return u_sat

def nominal_control(t,state,xg = [0,0,0],kr = 0.5, ka = 1.0):
    if state.size !=3:
        raise ValueError('the state must be of size 3')

    ## a global convergent stabilizing controller
    # theta = x[2]
    # r = np.sqrt(x[0]**2 + x[1]**2)
    # eta = np.arctan2(x[1],x[0])
    # alpha = theta - eta
    # v = -kr*r*np.cos(alpha)
    # if np.abs(alpha)>1e-2:
    #     mu = -ka*alpha - kr*np.sin(alpha)*np.cos(alpha)*(alpha-eta)/alpha
    # else:
    #     mu = -ka*alpha - kr*np.cos(alpha)*(alpha-eta)


    # v = kr*r*np.cos(theta)
    # mu = -ka*theta

    ## a propotional tracking controller
    x, y, theta = state
    x_g, y_g = xg[:2]

    p = np.array([x, y])
    p_g = np.array([x_g, y_g])

    ev = np.array([[np.cos(theta), np.sin(theta)]]).dot((p_g - p).reshape(-1, 1))
    e_perp_v = np.array([[-np.sin(theta), np.cos(theta)]]).dot((p_g - p).reshape(-1, 1))

    v = kr * ev[0, 0]

    if np.allclose(p, p_g):
        mu = 0
    else:
        mu = ka * np.arctan2(e_perp_v[0, 0], ev[0, 0])  


    ## time-dependent open-loop control signal 
    # a = 4
    # v = a*np.sin(np.pi*t/4)+2.0
    # mu = a*np.cos(2*np.pi*t/4)

    return np.array([v,mu])

def reference_path(t,param):
    x_ref = param.p0 + param.vx*t 
    # y = a*sin(bx+c) + d 
    y_ref = param.a*np.sin(param.b*x_ref+param.c) + param.d
    theta_ref = np.arctan2(float(param.a*np.cos(param.a*x_ref + param.b)),1.0)
    return np.array([x_ref,y_ref,theta_ref])

    # a = 5
    # if t < a:
    #     xg = np.array([10,10,0])
    #     return xg
    # if t < 2*a:
    #     xg = np.array([-10,10,0])
    #     return xg
    # if t < 3*a:
    #     xg = np.array([-10,-10,0])
    #     return xg
    # if t < 4*a:
    #     xg = np.array([10,-10,0])
    #     return xg

    # a = 5
    # x_ref = 4*a*np.sin(t) + 1
    # y_ref = a*np.sin(2*t) + 1
    # return np.array([x_ref,y_ref,0])
    return None

# lane_y = zocbf_param.amplitude*np.sin(zocbf_param.frequency*lane_x)