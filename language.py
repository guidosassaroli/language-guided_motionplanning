# language.py
import re
from dataclasses import dataclass
from typing import Optional
from scene import COLORS, SHAPES, RELATIONS
from llm_cloud import parse_cloud  # cloud parser hook

STOPWORDS = {"move", "put", "place", "the", "a", "an", "to"}
alias_shapes = {"squares": "square", "circles": "circle", "triangles": "triangle"}

@dataclass
class Command:
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
    s = cmd.lower().strip().replace("next-to","next to").replace("close-to","close to")
    rel = next((r for r in RELATIONS if r in s), None)
    if rel is None:
        raise ValueError(f"Relation not found. Use one of: {RELATIONS}")
    left, right = s.split(rel, 1)
    src_c, src_s = _extract_obj(left)
    dst_c, dst_s = _extract_obj(right)
    src_is_robot = (src_c == "__robot__")
    if src_is_robot: src_c = None; src_s = None
    return Command(src_is_robot, src_c, src_s, rel, dst_c, dst_s)

def parse_dispatch(cmd, args) -> Command:
    if args.parser == "rule":
        return parse_command(cmd)

    # cloud or hybrid
    try:
        data = parse_cloud(args.llm_provider, cmd, model=args.llm_model,
                           api_key=args.llm_api_key, endpoint=args.llm_endpoint)
        src_is_robot = (data["src"] == "robot")
        return Command(
            src_is_robot=src_is_robot,
            src_color=None if src_is_robot else data.get("src_color"),
            src_shape=None if src_is_robot else data.get("src_shape"),
            relation=data["relation"],
            dst_color=data["dst_color"],
            dst_shape=data["dst_shape"],
        )
    except Exception:
        if args.parser == "hybrid":
            return parse_command(cmd)
        raise
