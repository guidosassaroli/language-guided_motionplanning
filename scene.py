# scene.py
import random
from dataclasses import dataclass
from typing import Tuple, Dict, List
import numpy as np

# Global catalogs
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

def random_scene(grid_size=40, n_objects=5, seed=0):
    random.seed(seed); np.random.seed(seed)
    objs: List[Obj] = []
    placed = set()
    for i in range(n_objects):
        color = COLORS[i % len(COLORS)]
        shape = SHAPES[i % len(SHAPES)]
        while True:
            xy = (random.randint(4, grid_size-5), random.randint(4, grid_size-5))
            if xy not in placed:
                placed.add(xy); break
        objs.append(Obj(name=f"{color}_{shape}", color=color, shape=shape, xy=xy))
    # obstacles
    occ = np.zeros((grid_size, grid_size), dtype=bool)
    for _ in range(4):
        x = random.randint(5, grid_size-6)
        y0 = random.randint(5, grid_size-15)
        h = random.randint(5, 10)
        occ[y0:y0+h, x] = True
    return objs, occ

def perceive(objects: List[Obj]) -> Dict[Tuple[str, str], Tuple[int,int]]:
    return {(o.color, o.shape): o.xy for o in objects}
