# mvp_mpc.py
import argparse
import math
import random
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional
import re

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from scipy.ndimage import maximum_filter

from mpc import make_double_integrator, build_mpc, mpc_step

# ---------------------------
# Scene & Objects
# ---------------------------
COLORS = ["red", "blue", "green", "yellow"]
SHAPES = ["square", "circle", "triangle"]
RELATIONS = ["next to", "left of", "right of", "above", "below", "close to"]

@dataclass
class Obj:
    name: str
    color: str
    shape: str
    xy: Tuple[int, int]
    radius: int = 2  # footprint radius in grid cells

def random_scene(grid_size=40, n_objects=3, seed=0):
    random.seed(seed); np.random.seed(seed)
    objs = []
    placed = set()
    for i in range(n_objects):
        color = COLORS[i % len(COLORS)]
        shape = SHAPES[i % len(SHAPES)]
        # sample a free spot
        while True:
            xy = (random.randint(4, grid_size-5), random.randint(4, grid_size-5))
            if xy not in placed:
                placed.add(xy); break
        objs.append(Obj(name=f"{color}_{shape}", color=color, shape=shape, xy=xy))
    # obstacles: some random walls/blocks
    occ = np.zeros((grid_size, grid_size), dtype=bool)
    for _ in range(4):
        x = random.randint(5, grid_size-6)
        y0 = random.randint(5, grid_size-15)
        h = random.randint(5, 10)
        occ[y0:y0+h, x] = True
    return objs, occ

# ---------------------------
# Language Parser
# ---------------------------
STOPWORDS = {"move", "put", "place", "the", "a", "an", "to"}
alias_shapes = {"squares": "square", "circles": "circle", "triangles": "triangle"}

@dataclass
class Command:
    # If src_is_robot=True, the source is the robot (no color/shape)
    src_is_robot: bool
    src_color: Optional[str]
    src_shape: Optional[str]
    relation: str
    dst_color: str
    dst_shape: str

def _tokenize(phrase: str):
    toks = re.findall(r"[a-z]+", phrase.lower())
    return [t for t in toks if t not in STOPWORDS]

def _extract_obj(cs: str):
    tokens = _tokenize(cs)
    if "robot" in tokens:
        return ("__robot__", "__robot__")
    color = next((t for t in tokens if t in COLORS), None)
    shape = None
    for t in tokens:
        t2 = alias_shapes.get(t, t)
        if t2 in SHAPES:
            shape = t2; break
    if ("robot" not in tokens) and (not color or not shape):
        raise ValueError(f"Could not parse color/shape from: {cs.strip()!r}")
    return color, shape

def parse_command(cmd: str) -> Command:
    s = cmd.lower().strip().replace("next-to", "next to").replace("close-to","close to")
    rel = next((r for r in RELATIONS if r in s), None)
    if rel is None:
        raise ValueError(f"Relation not found. Use one of: {RELATIONS}")
    left, right = s.split(rel, 1)
    src_c, src_s = _extract_obj(left)
    dst_c, dst_s = _extract_obj(right)
    src_is_robot = (src_c == "__robot__")
    if src_is_robot:
        src_c = None; src_s = None
    return Command(src_is_robot, src_c, src_s, rel, dst_c, dst_s)

# ---------------------------
# Perception (synthetic)
# ---------------------------
def perceive(objects: List[Obj]) -> Dict[Tuple[str, str], Tuple[int,int]]:
    return {(o.color, o.shape): o.xy for o in objects}

# ---------------------------
# Relations → Goal candidates
# ---------------------------
def neighbor_positions(target_xy, rel: str) -> List[Tuple[int,int]]:
    x, y = target_xy
    if rel == "next to":
        return [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]
    elif rel == "left of":
        return [(x-1,y),(x-2,y),(x-3,y)]
    elif rel == "right of":
        return [(x+1,y),(x+2,y),(x+3,y)]
    elif rel == "above":
        return [(x,y-1),(x,y-2),(x,y-3)]
    elif rel == "below":
        return [(x,y+1),(x,y+2),(x,y+3)]
    else:
        return []

def close_to_candidates(target_xy, radius: int) -> List[Tuple[int,int]]:
    x0, y0 = target_xy
    cands = []
    for r in range(1, radius+1):
        ring = [(x0+r,y0),(x0-r,y0),(x0,y0+r),(x0,y0-r)]
        cands.extend(ring)
    # unique, keep order
    uniq = []
    for p in cands:
        if p not in uniq: uniq.append(p)
    return uniq

# ---------------------------
# Planner (A*)
# ---------------------------
def inflate_obstacles(occ: np.ndarray, radius: int) -> np.ndarray:
    k = 2*radius+1
    return maximum_filter(occ.astype(np.uint8), size=(k,k)).astype(bool)

def astar(occ: np.ndarray, start: Tuple[int,int], goal: Tuple[int,int]) -> Optional[List[Tuple[int,int]]]:
    H,W = occ.shape
    def inb(p): return 0<=p[0]<W and 0<=p[1]<H and not occ[p[1],p[0]]
    def h(a,b): return abs(a[0]-b[0]) + abs(a[1]-b[1])
    openset = {start}
    came = {}
    g = {start:0}
    f = {start:h(start,goal)}
    while openset:
        cur = min(openset, key=lambda p:f.get(p,1e9))
        if cur==goal:
            path=[cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            return list(reversed(path))
        openset.remove(cur)
        for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nb=(cur[0]+dx,cur[1]+dy)
            if not inb(nb): continue
            tentative = g[cur]+1
            if tentative < g.get(nb,1e9):
                came[nb]=cur
                g[nb]=tentative
                f[nb]=tentative+h(nb,goal)
                openset.add(nb)
    return None

def choose_goal(occ: np.ndarray, src_xy, dst_xy, rel, radius=2):
    inflated = inflate_obstacles(occ, radius)
    if rel == "close to":
        # try a few rings around the target
        for rr in [2,3,4]:
            for g in close_to_candidates(dst_xy, rr):
                if 0<=g[0]<occ.shape[1] and 0<=g[1]<occ.shape[0] and not inflated[g[1],g[0]]:
                    path = astar(inflated, src_xy, g)
                    if path: return g, path
        return None, None
    else:
        for g in neighbor_positions(dst_xy, rel):
            if 0<=g[0]<occ.shape[1] and 0<=g[1]<occ.shape[0] and not inflated[g[1],g[0]]:
                path = astar(inflated, src_xy, g)
                if path: return g, path
        return None, None

# ---------------------------
# Viz / Animation
# ---------------------------
def draw_scene(ax, occ, objects, path=None, src=None, goal=None, robot_xy=None, refs=None):
    H,W = occ.shape
    ax.clear()
    ax.set_xlim(-1,W); ax.set_ylim(-1,H); ax.set_aspect('equal'); ax.invert_yaxis()
    ax.set_xticks([]); ax.set_yticks([])
    # obstacles
    ys,xs = np.where(occ)
    ax.scatter(xs, ys, s=10, c="k", alpha=0.35, label="obstacles")
    # objects
    for o in objects:
        if o.shape=="square":
            ax.add_patch(patches.Rectangle((o.xy[0]-0.5,o.xy[1]-0.5),1,1,color=o.color,alpha=0.8))
        elif o.shape=="circle":
            ax.add_patch(patches.Circle((o.xy[0],o.xy[1]),0.5,color=o.color,alpha=0.8))
        else: # triangle
            tri = np.array([[0, -0.6],[0.52,0.3],[-0.52,0.3]]) + np.array(o.xy)
            ax.add_patch(patches.Polygon(tri,closed=True,color=o.color,alpha=0.8))
    # path
    if path:
        xs=[p[0] for p in path]; ys=[p[1] for p in path]
        ax.plot(xs, ys, linewidth=2)
    # refs (MPC window)
    if refs is not None:
        ax.plot(refs[0,:], refs[1,:], linestyle='--')
    # start/goal markers
    if src: ax.add_patch(patches.Circle((src[0],src[1]),0.3,fill=False,linewidth=2))
    if goal: ax.add_patch(patches.Circle((goal[0],goal[1]),0.35,fill=False,linewidth=2,linestyle='--'))
    # robot
    if robot_xy is not None:
        ax.add_patch(patches.RegularPolygon((robot_xy[0], robot_xy[1]), numVertices=3, radius=0.35, orientation=math.pi/2, color="black"))
    ax.legend(loc='upper right', fontsize=8)

def animate_path_grid(fig, ax, occ, objects, path, src_obj, goal):
    for p in path:
        src_obj.xy = p
        draw_scene(ax, occ, objects, path=None, src=path[0], goal=goal)
        plt.pause(0.03)

# ---------------------------
# Helpers for MPC mode
# ---------------------------
def path_to_refs(path_xy: List[Tuple[int,int]], N: int, robot_pos: np.ndarray):
    """
    Build a (2, N+1) reference window starting from the closest point along the path
    to the current robot position.
    """
    if len(path_xy) == 0:
        return np.tile(robot_pos[:2].reshape(2,1), (1, N+1))

    path = np.array(path_xy, dtype=float).T  # shape (2, L)
    # find closest index
    diffs = path - robot_pos[:2].reshape(2,1)
    d2 = np.sum(diffs*diffs, axis=0)
    idx = int(np.argmin(d2))
    seg = path[:, idx:idx+N+1]
    if seg.shape[1] < N+1:  # pad with last
        pad = np.tile(seg[:, -1:], (1, N+1 - seg.shape[1]))
        seg = np.concatenate([seg, pad], axis=1)
    return seg

# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmd", type=str, default="move the robot close to the blue circle")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--grid", type=int, default=40)
    parser.add_argument("--controller", type=str, choices=["grid","mpc"], default="grid",
                        help="grid = previous stepper, mpc = linear MPC on double-integrator")
    parser.add_argument("--dt", type=float, default=0.15)
    parser.add_argument("--N", type=int, default=20)
    parser.add_argument("--u_max", type=float, default=2.0)
    parser.add_argument("--v_max", type=float, default=2.0)
    args = parser.parse_args()

    # scene
    objects, occ = random_scene(args.grid, n_objects=3, seed=args.seed)

    # parse
    C = parse_command(args.cmd)

    # perceive
    det = perceive(objects)
    # destination must exist
    if (C.dst_color, C.dst_shape) not in det:
        raise RuntimeError(
            f"Objects not found in scene: {C.dst_color} {C.dst_shape}. "
            "Try another --seed or modify your command."
        )

    dst_xy = det[(C.dst_color, C.dst_shape)]

    # Choose source:
    if C.src_is_robot:
        # Create a robot start position in a free cell
        rng = random.Random(args.seed + 999)
        while True:
            candidate = (rng.randint(2, args.grid-3), rng.randint(2, args.grid-3))
            if not occ[candidate[1], candidate[0]]:
                robot_xy0 = np.array([candidate[0], candidate[1]], dtype=float)
                break
        src_xy = (int(robot_xy0[0]), int(robot_xy0[1]))
        src_obj = None  # no moving object; we move the robot marker
        mover_radius = 2
    else:
        # Move a specific object (legacy behavior)
        if (C.src_color, C.src_shape) not in det:
            missing = f"{C.src_color} {C.src_shape}"
            raise RuntimeError(f"Objects not found in scene: {missing}. Try another --seed or modify your command.")
        src_xy = det[(C.src_color, C.src_shape)]
        src_obj = next(o for o in objects if (o.color,o.shape)==(C.src_color,C.src_shape))
        mover_radius = src_obj.radius

    # plan
    goal, path = choose_goal(occ, src_xy, dst_xy, C.relation, radius=mover_radius)
    if path is None:
        print("No feasible goal path found. Try a different seed or command.")
        return

    fig, ax = plt.subplots(figsize=(6,6))
    draw_scene(ax, occ, objects, path=path, src=path[0], goal=goal)
    plt.title(f'Cmd: "{args.cmd}"')
    plt.pause(0.5)

    if args.controller == "grid" or not C.src_is_robot:
        # legacy animation: move object cell-by-cell
        if src_obj is None:
            # If command said "robot" but controller=grid, emulate grid motion of robot marker
            for p in path:
                robot_xy0 = np.array([p[0], p[1]], dtype=float)
                draw_scene(ax, occ, objects, path=None, src=path[0], goal=goal, robot_xy=robot_xy0)
                plt.pause(0.03)
        else:
            animate_path_grid(fig, ax, occ, objects, path, src_obj, goal)
        plt.title("Done")
        plt.show()
        return

    # ---------------------------
    # MPC mode (robot-only)
    # ---------------------------
    dt = args.dt
    A, B = make_double_integrator(dt)
    prob, x0_par, pref_par, x_var, u_var = build_mpc(
        A, B, N=args.N, u_max=args.u_max, v_max=args.v_max
    )

    # initial state
    state = np.array([robot_xy0[0], robot_xy0[1], 0.0, 0.0], dtype=float)

    # Sim loop
    goal_reached = False
    close_radius = 1.25  # continuous distance threshold
    steps = 0
    while steps < 600:
        # build reference window from path
        refs = path_to_refs(path, args.N, state)
        # solve MPC
        u = mpc_step(prob, x0_par, pref_par, x_var, u_var, state, refs)
        if u is None:
            print("MPC infeasible. Stopping.")
            break
        # forward simulate
        state = A @ state + B @ u
        # draw
        draw_scene(ax, occ, objects, path=path, src=path[0], goal=goal,
                   robot_xy=state[:2], refs=refs)
        plt.pause(0.03)
        # check goal
        if np.linalg.norm(state[:2] - np.array(goal, dtype=float)) <= close_radius:
            goal_reached = True
            break
        steps += 1

    plt.title("Done" if goal_reached else "Stopped")
    plt.show()

if __name__ == "__main__":
    main()
