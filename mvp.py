import argparse
import math
import random
import re
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

# ---------------------------
# Scene & Objects
# ---------------------------
COLORS = ["red", "blue", "green", "yellow"]
SHAPES = ["square", "circle", "triangle"]
RELATIONS = ["next to", "left of", "right of", "above", "below"]
STOPWORDS = {"move", "put", "place", "the", "a", "an", "to"}

@dataclass
class Obj:
    name: str
    color: str
    shape: str
    xy: Tuple[int, int]
    radius: int = 2  # footprint radius in grid cells

def random_scene(grid_size=40, n_objects=3, seed=0):
    random.seed(seed)
    np.random.seed(seed)
    objs = []
    placed = set()
    for i in range(n_objects):
        color = COLORS[i % len(COLORS)]
        shape = SHAPES[i % len(SHAPES)]
        # sample a free spot
        while True:
            xy = (random.randint(4, grid_size-5), random.randint(4, grid_size-5))
            if xy not in placed:
                placed.add(xy)
                break
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
# Simple Language Parser
# ---------------------------
@dataclass
class Command:
    src_color: str
    src_shape: str
    relation: str
    dst_color: str
    dst_shape: str


def _tokenize(phrase: str):
    # keep only a–z letters, split into tokens
    toks = re.findall(r"[a-z]+", phrase.lower())
    return [t for t in toks if t not in STOPWORDS]

def _extract_obj(cs: str):
    tokens = _tokenize(cs)
    # color
    color = next((t for t in tokens if t in COLORS), None)

    # shape (with plural aliases)
    alias = {"squares": "square", "circles": "circle", "triangles": "triangle"}
    shape = None
    for t in tokens:
        t2 = alias.get(t, t)
        if t2 in SHAPES:
            shape = t2
            break

    if not color or not shape:
        raise ValueError(f"Could not parse color/shape from: {cs.strip()!r}")
    return color, shape

def parse_command(cmd: str) -> Command:
    s = cmd.lower().strip().replace("next-to", "next to")
    # find the first matching relation (prefer longest phrases if you add more)
    rel = next((r for r in RELATIONS if r in s), None)
    if rel is None:
        raise ValueError(f"Relation not found. Use one of: {RELATIONS}")

    # split once around the relation
    left, right = s.split(rel, 1)
    src_color, src_shape = _extract_obj(left)
    dst_color, dst_shape = _extract_obj(right)
    return Command(src_color, src_shape, rel, dst_color, dst_shape)

# ---------------------------
# Perception (synthetic)
# ---------------------------
def perceive(objects: List[Obj]) -> Dict[Tuple[str, str], Tuple[int,int]]:
    """Return mapping (color,shape) -> (x,y) using ground truth."""
    return {(o.color, o.shape): o.xy for o in objects}

# ---------------------------
# Relation → Goal Pose
# ---------------------------
def neighbor_positions(target_xy, rel: str) -> List[Tuple[int,int]]:
    x, y = target_xy
    # define candidate goal cells ordered by relation
    if rel == "next to":
        cands = [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]
    elif rel == "left of":
        cands = [(x-1,y),(x-2,y),(x-3,y)]
    elif rel == "right of":
        cands = [(x+1,y),(x+2,y),(x+3,y)]
    elif rel == "above":
        cands = [(x,y-1),(x,y-2),(x,y-3)]
    elif rel == "below":
        cands = [(x,y+1),(x,y+2),(x,y+3)]
    else:
        cands = []
    return cands

# ---------------------------
# Planner (A*)
# ---------------------------
def inflate_obstacles(occ: np.ndarray, radius: int) -> np.ndarray:
    from scipy.ndimage import maximum_filter
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

# ---------------------------
# Execute: choose best goal, plan, animate
# ---------------------------
def choose_goal(occ: np.ndarray, src_xy, dst_xy, rel, radius=2):
    inflated = inflate_obstacles(occ, radius)
    for g in neighbor_positions(dst_xy, rel):
        if 0<=g[0]<occ.shape[1] and 0<=g[1]<occ.shape[0] and not inflated[g[1],g[0]]:
            path = astar(inflated, src_xy, g)
            if path: return g, path
    return None, None

# ---------------------------
# Viz / Animation
# ---------------------------
def draw_scene(ax, occ, objects, path=None, src=None, goal=None):
    H,W = occ.shape
    ax.clear()
    ax.set_xlim(-1,W); ax.set_ylim(-1,H); ax.set_aspect('equal'); ax.invert_yaxis()
    ax.set_xticks([]); ax.set_yticks([])
    # obstacles
    ys,xs = np.where(occ)
    ax.scatter(xs, ys, s=10, c="k", alpha=0.4, label="obstacles")
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
    # start/goal markers
    if src: ax.add_patch(patches.Circle((src[0],src[1]),0.3,fill=False,linewidth=2))
    if goal: ax.add_patch(patches.Circle((goal[0],goal[1]),0.35,fill=False,linewidth=2,linestyle='--'))
    ax.legend(loc='upper right', fontsize=8)

def animate_path(fig, ax, occ, objects, path, src_obj, goal):
    # move the source object along the path (simple “pick-and-move”)
    frames = []
    for p in path:
        src_obj.xy = p
        draw_scene(ax, occ, objects, path=None, src=path[0], goal=goal)
        plt.pause(0.03)
        frames.append(p)
    return frames

# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmd", type=str, default="move the red square next to the blue circle")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--grid", type=int, default=40)
    args = parser.parse_args()

    # scene
    objects, occ = random_scene(args.grid, n_objects=3, seed=args.seed)

    # parse
    C = parse_command(args.cmd)

    # perceive
    det = perceive(objects)
    if (C.src_color, C.src_shape) not in det or (C.dst_color, C.dst_shape) not in det:
        raise RuntimeError("Requested objects not in scene. Change --seed or command.")
    src_xy = det[(C.src_color, C.src_shape)]
    dst_xy = det[(C.dst_color, C.dst_shape)]

    # pick source object handle
    src_obj = next(o for o in objects if (o.color,o.shape)==(C.src_color,C.src_shape))

    # plan
    goal, path = choose_goal(occ, src_xy, dst_xy, C.relation, radius=src_obj.radius)
    if path is None:
        print("No feasible goal path found. Try a different seed or command.")
        return

    # animate
    fig, ax = plt.subplots(figsize=(6,6))
    draw_scene(ax, occ, objects, path=path, src=path[0], goal=goal)
    plt.title(f'Cmd: "{args.cmd}"')
    plt.pause(0.7)
    animate_path(fig, ax, occ, objects, path, src_obj, goal)
    plt.title("Done")
    plt.show()

if __name__ == "__main__":
    main()
