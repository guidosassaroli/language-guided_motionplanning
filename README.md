# Language-Guided Motion Planning

One-liner: A tiny simulator that turns plain English into goal-directed robot actions: parse → perceive → plan → execute.

## Demo

To install the requirements run:

`pip install -r requirements.txt`

To run the demo just execute this:

`python mvp.py --cmd "move the robot close to the blue circle" --controller grid`

To run the demo with the MPC controller use:
`python mvp.py --cmd "move the robot close to the blue circle" --controller mpc`

The system:
- Parses the sentence (color + shape + relation)
- “Perceives” a simple synthetic scene (colored shapes on a grid)
- Plans a collision-free path (A*) to the goal pose that satisfies the relation
- Animates the motion

#### Reminder for me: 

To actviate the virtual environment:

`cd Documents/py3theker/`

`source bin/activate`

## Why this is interesting

This project was developed as a compact demonstration of language-guided reasoning and control. The goal is not to build a full manipulation pipeline, but to show how a robot can interpret natural-language commands, perceive its environment, and plan feasible actions within it. By deliberately keeping everything in 2D and simulation-only, the focus stays on the integration of perception, language understanding, and motion planning, rather than on hardware details.

From a research perspective, this miniature environment captures the essential loop behind autonomous behavior: understand → plan → act → verify. The system fails gracefully when requested objects are not in the scene, mimicking how a real robot would detect inconsistencies between a user’s instruction and its perception.

## Technical Overview

The project is implemented entirely in Python to remain accessible and reproducible, with a modular structure that mirrors a real robotic pipeline.
Natural-language parsing is handled through a lightweight, rule-based system using regular expressions and token filtering — simple yet transparent, ideal for mapping linguistic cues (color, shape, relation) to symbolic world elements without requiring large models.
The synthetic perception layer emulates a vision module by generating and interpreting a randomized 2D environment, while NumPy provides efficient numerical operations for grid and geometry handling.
Path planning is implemented with the A* algorithm on a discrete occupancy map, chosen for its clarity and guaranteed optimality on small grids, while SciPy’s morphological filters are used to inflate obstacles and simulate robot footprint constraints.
Finally, Matplotlib handles visualization and live animation, making it easy to inspect each stage — perception, planning, and motion — in real time.

This combination of simplicity, transparency, and modularity makes the project a start point to prototype ideas in language grounding, perceptual reasoning, and motion planning, all without requiring any hardware or external datasets.

## Features

Natural-language parser for: {move} {color} {shape} [next to | left of | right of | above | below] {color} {shape}
Synthetic perception (ground-truth scene) + optional OpenCV color segmentation
A* grid planner with obstacle inflation
Animated execution with live overlays (start/goal/path)