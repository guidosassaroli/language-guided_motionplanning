# mpc.py
import numpy as np
import cvxpy as cp

def make_double_integrator(dt: float):
    """
    Build discrete-time double-integrator dynamics for 2D motion.
    State x = [px, py, vx, vy]^T
    Input u = [ax, ay]^T  (accelerations)

    Discretization (zero-order hold):
      p_{k+1} = p_k + v_k*dt + 0.5*a*dt^2
      v_{k+1} = v_k + a*dt

    Returns:
      A (4x4), B (4x2)
    """
    A = np.array([[1, 0, dt, 0],   # px' = px + vx*dt
                  [0, 1, 0,  dt],  # py' = py + vy*dt
                  [0, 0, 1,  0],   # vx' = vx
                  [0, 0, 0,  1]],  # vy' = vy
                 dtype=float)

    B = np.array([[0.5*dt*dt, 0],  # px' += 0.5*ax*dt^2
                  [0, 0.5*dt*dt],  # py' += 0.5*ay*dt^2
                  [dt, 0],         # vx' += ax*dt
                  [0, dt]],        # vy' += ay*dt
                 dtype=float)
    return A, B

def build_mpc(A, B, N=20,
              Qp=np.diag([5.0, 5.0]),     # position tracking weight (px, py)
              Qv=np.diag([0.5, 0.5]),     # velocity penalty (vx, vy)
              R =np.diag([0.1, 0.1]),     # control effort penalty (ax, ay)
              Qf=np.diag([20.0, 20.0]),   # terminal position penalty
              u_max=2.0,                  # |ax|, |ay| <= u_max (∞-norm)
              v_max=2.0):                 # |vx|, |vy| <= v_max (∞-norm)
    """
    Build a convex MPC problem for tracking a reference position trajectory.

    Arguments:
      A, B : system matrices (from make_double_integrator)
      N    : horizon length (number of control stages)
      Qp   : weight on position error per stage (2x2)
      Qv   : weight on velocities per stage (2x2)
      R    : weight on control per stage (2x2)
      Qf   : terminal weight on final position error (2x2)
      u_max: bound on input infinity-norm at each step
      v_max: bound on velocity infinity-norm at each step

    Returns:
      prob : cvxpy.Problem (minimize quadratic cost subject to constraints)
      x0   : cvxpy.Parameter (size 4,) initial state parameter
      pref : cvxpy.Parameter (size 2x(N+1)) reference positions
      x    : cvxpy.Variable  (size 4x(N+1)) trajectory states
      u    : cvxpy.Variable  (size 2xN)     trajectory inputs
    """
    n = A.shape[0]  # = 4
    m = B.shape[1]  # = 2

    # Decision variables: state and input trajectories over the horizon
    x  = cp.Variable((n, N+1))
    u  = cp.Variable((m, N))

    # Time-varying parameters (set each MPC step):
    x0 = cp.Parameter(n)           # initial state
    pref = cp.Parameter((2, N+1))  # desired positions for k=0..N

    constr = [x[:, 0] == x0]       # initial condition
    cost = 0

    for k in range(N):
        # Discrete-time dynamics
        constr += [x[:, k+1] == A @ x[:, k] + B @ u[:, k]]

        # Input and velocity bounds via infinity norm
        constr += [cp.norm_inf(u[:, k]) <= u_max]  # |ax|,|ay| ≤ u_max
        constr += [cp.norm_inf(x[2:4, k]) <= v_max]  # |vx|,|vy| ≤ v_max

        # Stage costs: position tracking, velocity penalty, control effort
        cost += cp.quad_form(x[0:2, k] - pref[:, k], Qp)
        cost += cp.quad_form(x[2:4, k],            Qv)
        cost += cp.quad_form(u[:, k],              R)

    # Terminal cost on final position error at k = N
    cost += cp.quad_form(x[0:2, N] - pref[:, N], Qf)

    prob = cp.Problem(cp.Minimize(cost), constr)
    return prob, x0, pref, x, u

def mpc_step(prob, x0_par, pref_par, x_var, u_var, x0_val, pref_window):
    """
    Single MPC iteration:
      - Set the initial state and reference window parameters
      - Solve the QP with warm start
      - Return the first control input u[:,0] (receding horizon)

    Arguments:
      prob         : cvxpy.Problem from build_mpc
      x0_par       : cvxpy.Parameter for initial state (size 4,)
      pref_par     : cvxpy.Parameter for reference positions (2x(N+1))
      x_var, u_var : variables returned by build_mpc
      x0_val       : np.array shape (4,) current state
      pref_window  : np.array shape (2, N+1) reference positions over horizon

    Returns:
      u0 : np.array shape (2,) optimal first control, or None if infeasible.
    """
    x0_par.value = x0_val
    pref_par.value = pref_window  # must match shape (2, N+1)

    # OSQP is a good choice for large sparse QPs; warm_start speeds up repeats
    prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        # Infeasible or solver failed; caller can handle fallback
        return None

    u0 = u_var[:, 0].value
    return np.array(u0).reshape(-1)

