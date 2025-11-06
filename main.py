# main.py
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt

from scene import random_scene, perceive, RELATIONS
from language import parse_dispatch
from planning import choose_goal
from viz import draw_scene, animate_path_grid
from mpc import make_double_integrator, build_mpc, mpc_step

def path_to_refs(path_xy, N: int, robot_pos: np.ndarray):
    if len(path_xy) == 0:
        return np.tile(robot_pos[:2].reshape(2,1), (1, N+1))
    path = np.array(path_xy, dtype=float).T
    diffs = path - robot_pos[:2].reshape(2,1)
    d2 = np.sum(diffs*diffs, axis=0)
    idx = int(np.argmin(d2))
    seg = path[:, idx:idx+N+1]
    if seg.shape[1] < N+1:
        pad = np.tile(seg[:, -1:], (1, N+1 - seg.shape[1]))
        seg = np.concatenate([seg, pad], axis=1)
    return seg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmd", type=str, default="move the robot close to the blue circle")
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--grid", type=int, default=40)
    parser.add_argument("--controller", choices=["grid","mpc"], default="grid")
    parser.add_argument("--parser", choices=["rule","cloud","hybrid"], default="rule")
    parser.add_argument("--llm_provider", choices=["openai","mistral","google"], default="openai")
    parser.add_argument("--llm_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--llm_api_key", type=str, default=None)
    parser.add_argument("--llm_endpoint", type=str, default=None)
    parser.add_argument("--dt", type=float, default=0.15)
    parser.add_argument("--N", type=int, default=20)
    parser.add_argument("--u_max", type=float, default=2.0)
    parser.add_argument("--v_max", type=float, default=2.0)
    args = parser.parse_args()

    objects, occ = random_scene(args.grid, n_objects=5, seed=args.seed)

    C = parse_dispatch(args.cmd, args)

    det = perceive(objects)
    if (C.dst_color, C.dst_shape) not in det:
        raise RuntimeError(
            f"Objects not found in scene: {C.dst_color} {C.dst_shape}. "
            "Try another --seed or modify your command."
        )
    dst_xy = det[(C.dst_color, C.dst_shape)]

    if C.src_is_robot:
        rng = random.Random(args.seed + 999)
        while True:
            candidate = (rng.randint(2, args.grid-3), rng.randint(2, args.grid-3))
            if not occ[candidate[1], candidate[0]]:
                robot_xy0 = np.array([candidate[0], candidate[1]], dtype=float)
                break
        src_xy = (int(robot_xy0[0]), int(robot_xy0[1]))
        src_obj = None
        mover_radius = 2
    else:
        if (C.src_color, C.src_shape) not in det:
            missing = f"{C.src_color} {C.src_shape}"
            raise RuntimeError(f"Objects not found in scene: {missing}. Try another --seed or modify your command.")
        src_xy = det[(C.src_color, C.src_shape)]
        from scene import Obj  # for type
        src_obj = next(o for o in objects if (o.color,o.shape)==(C.src_color,C.src_shape))
        mover_radius = src_obj.radius

    goal, path = choose_goal(occ, src_xy, dst_xy, C.relation, radius=mover_radius)
    if path is None:
        print("No feasible goal path found. Try a different seed or command.")
        return

    fig, ax = plt.subplots(figsize=(6,6))
    draw_scene(ax, occ, objects, path=path, src=path[0], goal=goal)
    plt.title(f'Cmd: "{args.cmd}"'); plt.pause(0.5)

    if args.controller == "grid" or not C.src_is_robot:
        if src_obj is None:
            for p in path:
                robot_xy0 = np.array([p[0], p[1]], dtype=float)
                draw_scene(ax, occ, objects, path=None, src=path[0], goal=goal, robot_xy=robot_xy0)
                plt.pause(0.03)
        else:
            animate_path_grid(fig, ax, occ, objects, path, src_obj, goal)
        plt.title("Done"); plt.show(); return

    # MPC
    dt = args.dt
    A, B = make_double_integrator(dt)
    prob, x0_par, pref_par, x_var, u_var = build_mpc(A, B, N=args.N, u_max=args.u_max, v_max=args.v_max)
    state = np.array([robot_xy0[0], robot_xy0[1], 0.0, 0.0], dtype=float)

    goal_reached = False
    close_radius = 1.25
    steps = 0
    while steps < 600:
        refs = path_to_refs(path, args.N, state)
        u = mpc_step(prob, x0_par, pref_par, x_var, u_var, state, refs)
        if u is None:
            print("MPC infeasible. Stopping."); break
        state = A @ state + B @ u
        draw_scene(ax, occ, objects, path=path, src=path[0], goal=goal, robot_xy=state[:2], refs=refs)
        plt.pause(0.03)
        if np.linalg.norm(state[:2] - np.array(goal, dtype=float)) <= close_radius:
            goal_reached = True; break
        steps += 1

    plt.title("Done" if goal_reached else "Stopped")
    plt.show()

if __name__ == "__main__":
    main()
