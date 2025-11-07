# language.py
import re
from dataclasses import dataclass
from typing import Optional
from scene import COLORS, SHAPES, RELATIONS
from llm_cloud import parse_cloud

# Words to ignore during tokenization (keeps parser tolerant to phrasing)
STOPWORDS = {"move", "put", "place", "the", "a", "an", "to"}

# Map plural shape tokens to their singular canonical names
alias_shapes = {"squares": "square", "circles": "circle", "triangles": "triangle"}

@dataclass
class Command:
    """
    A normalized, structured representation of a natural language command.

    Fields:
      src_is_robot: True if the source (thing being moved) is "the robot".
      src_color:    Color of the source object, or None if the robot is the source.
      src_shape:    Shape of the source object, or None if the robot is the source.
      relation:     Spatial relation keyword (must be one of RELATIONS).
      dst_color:    Color of the destination/anchor object.
      dst_shape:    Shape of the destination/anchor object.
    """
    src_is_robot: bool
    src_color: Optional[str]
    src_shape: Optional[str]
    relation: str
    dst_color: str
    dst_shape: str

def _tokenize(phrase: str):
    """
    Lowercase the phrase, extract alphabetic tokens, and drop stopwords.

    Examples:
      "Move the red square to the blue circle" -> ["red", "square", "blue", "circle"]
    """
    toks = re.findall(r"[a-z]+", phrase.lower())
    return [t for t in toks if t not in STOPWORDS]

def _extract_obj(cs: str):
    """
    Parse a 'color + shape' object (or 'robot') from a chunk of text.

    Returns:
      - ("__robot__", "__robot__") if the text mentions 'robot'
      - (color, shape) where:
          color is one of COLORS or None,
          shape is one of SHAPES or None.

    Raises:
      ValueError if the text is not 'robot' and we cannot find both color and shape.
    """
    tokens = _tokenize(cs)

    # Special case: robot as a logical source object
    if "robot" in tokens:
        return ("__robot__", "__robot__")

    # Extract first token that matches a known color
    color = next((t for t in tokens if t in COLORS), None)

    # Extract first token that matches a known (possibly plural) shape
    shape = None
    for t in tokens:
        t2 = alias_shapes.get(t, t)  # normalize plural to singular if needed
        if t2 in SHAPES:
            shape = t2
            break

    # If it's not robot and we lack either color or shape, fail early
    if ("robot" not in tokens) and (not color or not shape):
        raise ValueError(f"Could not parse color/shape from: {cs.strip()!r}")

    return color, shape

def parse_command(cmd: str) -> Command:
    """
    Rule-based parser.
    - Normalizes 'next-to'/'close-to' to spaced forms.
    - Finds the first relation present from RELATIONS.
    - Splits the command into 'source' (left of relation) and 'destination' (right).
    - Parses both sides into structured fields.

    Raises:
      ValueError if no relation is found or objects cannot be parsed.
    """
    # Normalize spelling variants for easier relation matching
    s = cmd.lower().strip().replace("next-to","next to").replace("close-to","close to")

    # Find the first recognized relation keyword present anywhere in s
    rel = next((r for r in RELATIONS if r in s), None)
    if rel is None:
        raise ValueError(f"Relation not found. Use one of: {RELATIONS}")

    # Split once on the matched relation into left (src) and right (dst)
    left, right = s.split(rel, 1)

    # Parse each side into (color, shape) or robot sentinel
    src_c, src_s = _extract_obj(left)
    dst_c, dst_s = _extract_obj(right)

    # If the src is the robot, clear color/shape and mark the flag
    src_is_robot = (src_c == "__robot__")
    if src_is_robot:
        src_c = None
        src_s = None

    return Command(src_is_robot, src_c, src_s, rel, dst_c, dst_s)

def parse_dispatch(cmd, args) -> Command:
    """
    Dispatch parser:
      - If args.parser == "rule": always use rule-based parse_command.
      - Else: try cloud parser first (parse_cloud), and if it fails:
          - if args.parser == "hybrid": fall back to rule-based parse_command
          - otherwise: bubble up the exception

    The cloud parser is expected to return a dict like:
      {
        "src": "robot" | "object",
        "src_color": "...",
        "src_shape": "...",
        "relation": "...",
        "dst_color": "...",
        "dst_shape": "..."
      }
    """
    if args.parser == "rule":
        return parse_command(cmd)

    # Try cloud or hybrid strategy first
    try:
        data = parse_cloud(
            args.llm_provider, cmd,
            model=args.llm_model,
            api_key=args.llm_api_key,
            endpoint=args.llm_endpoint
        )
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
        # If hybrid, try rule-based fallback; otherwise re-raise
        if args.parser == "hybrid":
            return parse_command(cmd)
        raise

