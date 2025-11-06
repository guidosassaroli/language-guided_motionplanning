# viz.py
import math
from typing import Tuple, List, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

def draw_scene(ax, occ, objects, path: Optional[List[Tuple[int,int]]]=None,
               src: Optional[Tuple[int,int]]=None, goal: Optional[Tuple[int,int]]=None,
               robot_xy: Optional[Tuple[float,float]]=None, refs=None):
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
        else:
            tri = np.array([[0,-0.6],[0.52,0.3],[-0.52,0.3]]) + np.array(o.xy)
            ax.add_patch(patches.Polygon(tri,closed=True,color=o.color,alpha=0.8))
    # path
    if path:
        xs=[p[0] for p in path]; ys=[p[1] for p in path]
        ax.plot(xs, ys, linewidth=2)
    # refs (MPC window)
    if refs is not None:
        ax.plot(refs[0,:], refs[1,:], linestyle='--')
    # start/goal markers
    if src:  ax.add_patch(patches.Circle((src[0],src[1]),0.3,fill=False,linewidth=2))
    if goal: ax.add_patch(patches.Circle((goal[0],goal[1]),0.35,fill=False,linewidth=2,linestyle='--'))
    # robot
    if robot_xy is not None:
        ax.add_patch(patches.RegularPolygon((robot_xy[0], robot_xy[1]), numVertices=3,
                                            radius=0.35, orientation=math.pi/2, color="black"))
    ax.legend(loc='upper right', fontsize=8)

def animate_path_grid(fig, ax, occ, objects, path, src_obj, goal):
    for p in path:
        src_obj.xy = p
        draw_scene(ax, occ, objects, path=None, src=path[0], goal=goal)
        plt.pause(0.03)
