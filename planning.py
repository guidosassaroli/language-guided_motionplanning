# planning.py
from typing import Tuple, List, Optional
import numpy as np
from scipy.ndimage import maximum_filter

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
    uniq = []
    for p in cands:
        if p not in uniq: uniq.append(p)
    return uniq

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
