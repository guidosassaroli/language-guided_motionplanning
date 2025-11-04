# mpc.py
import numpy as np
import cvxpy as cp

def make_double_integrator(dt: float):
    A = np.array([[1,0,dt,0],
                  [0,1,0,dt],
                  [0,0,1, 0],
                  [0,0,0, 1]], dtype=float)
    B = np.array([[0.5*dt*dt, 0],
                  [0, 0.5*dt*dt],
                  [dt, 0],
                  [0, dt]], dtype=float)
    return A, B

def build_mpc(A, B, N=20,
              Qp=np.diag([5.0, 5.0]),
              Qv=np.diag([0.5, 0.5]),
              R =np.diag([0.1, 0.1]),
              Qf=np.diag([20.0, 20.0]),
              u_max=2.0, v_max=2.0):
    n = A.shape[0]  # 4
    m = B.shape[1]  # 2
    x  = cp.Variable((n, N+1))
    u  = cp.Variable((m, N))
    x0 = cp.Parameter(n)
    pref = cp.Parameter((2, N+1))  # positions only

    constr = [x[:,0] == x0]
    cost = 0
    for k in range(N):
        constr += [x[:,k+1] == A @ x[:,k] + B @ u[:,k]]
        constr += [cp.norm_inf(u[:,k]) <= u_max]
        constr += [cp.norm_inf(x[2:4,k]) <= v_max]
        cost   += cp.quad_form(x[0:2,k] - pref[:,k], Qp)
        cost   += cp.quad_form(x[2:4,k], Qv)
        cost   += cp.quad_form(u[:,k], R)
    cost += cp.quad_form(x[0:2,N] - pref[:,N], Qf)

    prob = cp.Problem(cp.Minimize(cost), constr)
    return prob, x0, pref, x, u

def mpc_step(prob, x0_par, pref_par, x_var, u_var, x0_val, pref_window):
    x0_par.value = x0_val
    pref_par.value = pref_window  # shape (2, N+1)
    prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        return None
    u0 = u_var[:,0].value
    return np.array(u0).reshape(-1)
