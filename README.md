# Language-Guided Motion Planning

A tiny simulator that turns sentences into goal-directed robot actions: parse → perceive → plan → execute.

## Demo

To install the requirements run:

`pip install -r requirements.txt`

To run the demo just execute this:

`python main.py --cmd "move the robot close to the blue circle" --controller grid --parser rule`

To run the demo with the MPC controller use:

`python main.py --cmd "move the robot close to the blue circle" --controller mpc --parser rule`

The system:
- Parses the sentence (color + shape + relation)
- “Perceives” a simple synthetic scene (colored shapes on a grid)
- Plans a collision-free path (A*) to the goal pose that satisfies the relation
- Animates the motion

### LLM run

LLM run is still a work in progress.

```
python main.py \
  --cmd "move the robot close to the blue circle" \
  --controller mpc \
  --parser cloud \
  --llm_provider openai \
  --llm_model gpt-4o-mini \
  --llm_api_key OPENAI_API_KEY
```

## Why this is interesting

This project was developed as a compact demonstration of language-guided reasoning and control. The goal is to show how a robot can interpret natural-language commands, perceive its environment, and plan feasible actions within it. By deliberately keeping everything in 2D and simulation-only, the focus stays on the integration of perception, language understanding, and motion planning, rather than on hardware details.

From a research perspective, this miniature environment captures the essential loop behind autonomous behavior: understand → plan → act → verify. 

Remarl: The system fails gracefully when requested objects are not in the scene, mimicking how a real robot would detect inconsistencies between a user’s instruction and its perception.

## Technical Overview

The project is implemented entirely in Python to remain accessible and reproducible, with a modular structure that mirrors a real robotic pipeline.
Natural-language parsing is handled through a lightweight, rule-based system using regular expressions and token filtering — simple yet transparent, ideal for mapping linguistic cues (color, shape, relation) to symbolic world elements without requiring large models.
The synthetic perception layer emulates a vision module by generating and interpreting a randomized 2D environment, while NumPy provides efficient numerical operations for grid and geometry handling.
Path planning is implemented with the A* algorithm on a discrete occupancy map, chosen for its clarity and guaranteed optimality on small grids, while SciPy’s morphological filters are used to inflate obstacles and simulate robot footprint constraints.
Finally, Matplotlib handles visualization and live animation, making it easy to inspect each stage — perception, planning, and motion — in real time.

## Features

Natural-language parser for: {move} {color} {shape} [next to | left of | right of | above | below] {color} {shape}
Synthetic perception (ground-truth scene) + optional OpenCV color segmentation
A* grid planner with obstacle inflation
Animated execution with live overlays (start/goal/path)