"""Standalone geometry-only cycle assay, without model/schema/I/O dependencies.
Regions are normalized xyxy boxes indexed like nodes; edges connect i to i+1.
"""
from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import math
import os
import random
import shutil
import sys
import tempfile
from io import BytesIO
from pathlib import Path

COLORS = ("red", "blue", "green", "yellow", "purple", "orange")
SHAPES = ("triangle", "circle", "square")
SPLITS = ("engineering", "discovery", "pilot_validation", "confirmation")
_RGB = ("#df3038", "#1674d1", "#179c45", "#f2cf23", "#8942b6", "#f48624")
_FACTORS = tuple((e, d, p) for e in (1, -1) for d in (1, -1) for p in (0, 1))
_NAME = "reading-or-routing-v1"
_MAX_METADATA, _MAX_PNG = 16 * 1024**2, 8 * 1024**2


def _integer(value, low, high, name):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")


def _settings(seed, counts, size):
    _integer(seed, 0, 2**63 - 1, "seed")
    _integer(size, 128, 1024, "image_size")
    if type(counts) is not dict or set(counts) != set(SPLITS):
        raise ValueError("Exactly four split counts are required")
    for value in counts.values():
        _integer(value, 0, 1024, "split count")
    _integer(sum(counts.values()), 1, 1024, "total families")


def _absolute(path):
    path = Path(path).absolute()
    if ".." in path.parts or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Traversal and symlink paths are forbidden")
    return path


def _read(path, limit):
    if not path.is_file() or not 0 < path.stat().st_size <= limit:
        raise ValueError("Missing, empty, or oversized artifact")
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError("Oversized artifact")
    return data


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _unique_object(pairs):
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("Duplicate JSON key")
    return result


def _write(path, data):
    """Publish a flushed file using an exclusive hard link, never replacement."""
    fd, temporary = tempfile.mkstemp(prefix=".publish-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _rename_directory(source, destination):
    """Use native exclusive rename; unsupported filesystems fail closed."""
    if os.name == "nt":
        os.rename(source, destination)
        return
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        operation = getattr(libc, "renamex_np", None)
        arguments = (os.fsencode(source), os.fsencode(destination), 4)  # RENAME_EXCL
        types = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
    else:
        operation = getattr(libc, "renameat2", None)
        arguments = (-100, os.fsencode(source), -100, os.fsencode(destination), 1)
        types = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    if operation is None:
        raise OSError(errno.ENOTSUP, "Atomic no-clobber directory rename unavailable")
    operation.argtypes, operation.restype = types, ctypes.c_int
    if operation(*arguments) != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(destination))


def _image_id(family_id, edge, palette):
    return f"{family_id}-e{'p' if edge == 1 else 'm'}-p{palette}"


def _geometry(nodes, i):
    a, b = nodes[i], nodes[(i + 1) % 3]
    dx, dy = b["x"] - a["x"], b["y"] - a["y"]
    length = math.hypot(dx, dy)
    u, v = dx / length, dy / length
    start = (a["x"] + .125 * u, a["y"] + .125 * v)
    end = (b["x"] - .125 * u, b["y"] - .125 * v)
    middle = ((a["x"] + b["x"]) / 2, (a["y"] + b["y"]) / 2)
    return start, end, middle, u, v


def _family(seed, split, index):
    scene_seed = int.from_bytes(hashlib.sha256(f"{_NAME}:{seed}:{split}:{index}".encode())
                                .digest()[:8], "big") & (2**63 - 1)
    rng = random.Random(scene_seed)  # Never modify the process-wide RNG.
    shapes, colors = list(SHAPES), list(COLORS)
    rng.shuffle(shapes)
    rng.shuffle(colors)
    cx, cy = .5 + rng.uniform(-.015, .015), .47 + rng.uniform(-.015, .015)
    radius, rotation = rng.uniform(.29, .31), rng.uniform(-.14, .14)
    nodes = [{"shape": shape,
              "x": round(cx + radius * math.cos(-math.pi / 2 + rotation + i*2*math.pi/3), 8),
              "y": round(cy + radius * math.sin(-math.pi / 2 + rotation + i*2*math.pi/3), 8)}
             for i, shape in enumerate(shapes)]
    regions = {"nodes": [], "chips": [], "edges": [], "arrowheads": []}
    for i, node in enumerate(nodes):
        x, y = node["x"], node["y"]
        # Include the resampling halo even at the smallest supported resolution.
        regions["nodes"].append([x-.10, y-.10, x+.10, y+.15])
        regions["chips"].append([x-.075, y+.052, x+.075, y+.148])
        start, end, (mx, my), _, _ = _geometry(nodes, i)
        regions["edges"].append([min(start[0], end[0])-.06, min(start[1], end[1])-.06,
                                 max(start[0], end[0])+.06, max(start[1], end[1])+.06])
        regions["arrowheads"].append([mx-.075, my-.075, mx+.075, my+.075])
    family = {"family_id": f"f{index:06d}-{scene_seed:016x}", "split": split,
              "index": index, "anchor": index % 3, "nodes": nodes,
              "palettes": [colors[:3], colors[3:]], "regions": regions,
              "program": {"renderer": _NAME, "seed": scene_seed, "supersample": 4}}
    family["images"] = [{"image_id": _image_id(family["family_id"], e, p),
                         "relative_path": f"images/{_image_id(family['family_id'], e, p)}.png",
                         "edge": e, "palette": p} for e in (1, -1) for p in (0, 1)]
    family["cases"] = [make_case(family, edge=e, direction=d, palette=p) for e, d, p in _FACTORS]
    family["cases"] += [make_case(family, edge=1, direction=d, palette=0, task=task)
                        for task, d in (("color_readout", 1), ("edge_readout", 1),
                                        ("text_binding", 1), ("text_binding", -1))]
    return family


def make_case(family: dict, *, edge: int, direction: int, palette: int,
              task: str = "binding", anchor: int | None = None) -> dict:
    """Compute a canonical question/answer, including alternate-anchor donors."""
    for value in (edge, direction):
        _integer(value, -1, 1, "direction factor")
        if value == 0:
            raise ValueError("Direction factors must be +1 or -1")
    _integer(palette, 0, 1, "palette")
    anchor = family["anchor"] if anchor is None else anchor
    _integer(anchor, 0, 2, "anchor")
    if task not in ("binding", "color_readout", "edge_readout", "text_binding"):
        raise ValueError("Unknown task")
    target = (anchor + edge * direction) % 3
    shapes = [node["shape"] for node in family["nodes"]]
    traversal = "forward" if direction == 1 else "backward"
    question = (f"Starting at the {shapes[anchor]}, follow one arrow {traversal}. "
                "What color is the destination chip? Reply with one color name.")
    if task == "color_readout":
        question = f"What color is the chip at the {shapes[target]}? Reply with one color name."
    elif task == "edge_readout":
        neighbor = (anchor + 1) % 3
        target = neighbor if edge == 1 else anchor
        question = (f"Look only at the arrow between the {shapes[anchor]} and the {shapes[neighbor]}. "
                    "Which socket does this arrow point toward? Reply with one shape name.")
    elif task == "text_binding":
        facts = "; ".join(f"{shape} has a {family['palettes'][palette][i]} chip"
                          for i, shape in enumerate(shapes))
        arrows = "; ".join(f"{shape} to {shapes[(i + edge) % 3]}"
                           for i, shape in enumerate(shapes))
        question = f"Text-only control. Chips: {facts}. Arrows: {arrows}. {question}"
    return {"case_id": f"{family['family_id']}-{task}-a{anchor}-e{edge:+d}-d{direction:+d}-p{palette}",
            "family_id": family["family_id"], "split": family["split"], "task": task,
            "question": question, "gold": shapes[target] if task == "edge_readout"
            else family["palettes"][palette][target],
            "candidates": list(SHAPES if task == "edge_readout" else COLORS),
            "image_id": None if task == "text_binding"
            else _image_id(family["family_id"], edge, palette),
            "edge": edge, "direction": direction, "palette": palette,
            "anchor": anchor, "target": target}


def binding_case(family: dict, edge: int, direction: int, palette: int) -> dict:
    """Return the canonical binding member without mutating stored cases."""
    return make_case(family, edge=edge, direction=direction, palette=palette)


def representative_case(family: dict) -> dict:
    """Balance all eight e/d/p combinations over consecutive family indices."""
    return binding_case(family, *_FACTORS[family["index"] % 8])


def _render(family, edge, palette, size):
    from PIL import Image, ImageDraw

    scale = size * 4
    image = Image.new("RGB", (scale, scale), "white")
    draw = ImageDraw.Draw(image)
    for i in range(3):
        start, end, (mx, my), u, v = _geometry(family["nodes"], i)
        draw.line([tuple(t * scale for t in start), tuple(t * scale for t in end)],
                  fill="black", width=round(.008 * scale))
        points = [(mx+.028*edge*u, my+.028*edge*v),
                  (mx-.028*edge*u-.022*v, my-.028*edge*v+.022*u),
                  (mx-.028*edge*u+.022*v, my-.028*edge*v-.022*u)]
        draw.polygon([(x*scale, y*scale) for x, y in points], fill="black")
    for i, node in enumerate(family["nodes"]):
        x, y = node["x"] * scale, node["y"] * scale
        options = {"fill": "white", "outline": "black", "width": round(.006 * scale)}
        if node["shape"] == "triangle":
            draw.polygon([(x, y-.07*scale), (x+.07*scale, y+.055*scale),
                          (x-.07*scale, y+.055*scale)], **options)
        elif node["shape"] == "circle":
            draw.ellipse((x-.065*scale, y-.065*scale, x+.065*scale, y+.065*scale), **options)
        else:
            draw.rectangle((x-.06*scale, y-.06*scale, x+.06*scale, y+.06*scale), **options)
        draw.rounded_rectangle((x-.042*scale, y+.084*scale, x+.042*scale, y+.116*scale),
                               radius=.007*scale, fill=_RGB[COLORS.index(family["palettes"][palette][i])],
                               outline="black", width=round(.004*scale))
    return image.resize((size, size), Image.Resampling.LANCZOS)


def _png(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG", compress_level=9)
    return buffer.getvalue()


def _check_regions(images, regions, size):
    from PIL import ImageChops

    for a, b, kind in ((0, 2, "arrowheads"), (1, 3, "arrowheads"),
                       (0, 1, "chips"), (2, 3, "chips")):
        difference = ImageChops.difference(images[a], images[b])
        if difference.getbbox() is None:
            raise ValueError(f"Factor edit did not change {kind} pixels")
        for box in regions[kind]:
            difference.paste((0, 0, 0), (math.floor(box[0]*size), math.floor(box[1]*size),
                                        math.ceil(box[2]*size), math.ceil(box[3]*size)))
        if difference.getbbox() is not None:
            raise ValueError(f"Factor edit changed pixels outside {kind} regions")


def generate_dataset(output: Path, *, seed: int = 20261031, engineering: int = 24,
                     discovery: int = 40, pilot_validation: int = 60,
                     confirmation: int = 200, image_size: int = 448) -> dict:
    """Atomically publish a NEW directory; clean staging on any failure.
    Total families: 1..1024; seed: unsigned 63-bit; image size: 128..1024.
    """
    counts = dict(zip(SPLITS, (engineering, discovery, pilot_validation, confirmation), strict=True))
    _settings(seed, counts, image_size)
    output = _absolute(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    dataset = {"schema_version": 1, "name": _NAME, "seed": seed,
               "image_size": image_size, "split_counts": counts, "families": []}
    try:
        (staging / "images").mkdir()
        for split, count in counts.items():
            for _ in range(count):
                family = _family(seed, split, len(dataset["families"]))
                rendered = []
                for item in family["images"]:
                    image = _render(family, item["edge"], item["palette"], image_size)
                    rendered.append(image)
                    encoded = _png(image)
                    item["sha256"] = hashlib.sha256(encoded).hexdigest()
                    _write(_absolute(staging / item["relative_path"]), encoded)
                _check_regions(rendered, family["regions"], image_size)
                dataset["families"].append(family)
        metadata = _json(dataset)
        if len(metadata) > _MAX_METADATA:
            raise ValueError("Oversized metadata")
        _write(staging / "dataset.json", metadata)
        _rename_directory(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return dataset


def load_dataset(root: Path, *, verify_images: bool = True) -> dict:
    """Validate canonical metadata/paths and, by default, compare decoded scene pixels.
    Checksums cover original bytes; PNG codec differences do not invalidate pixels.
    False imports neither Pillow nor torch, but still verifies every image checksum.
    """
    root = _absolute(root)
    try:
        raw = json.loads(_read(_absolute(root / "dataset.json"), _MAX_METADATA),
                         object_pairs_hook=_unique_object)
        _json(raw)  # Reject NaN/Infinity, including overflowing JSON float literals.
    except (ValueError, RecursionError) as exc:
        raise ValueError("Invalid dataset JSON") from exc
    if type(raw) is not dict or set(raw) != {
            "schema_version", "name", "seed", "image_size", "split_counts", "families"}:
        raise ValueError("Invalid dataset fields")
    _settings(raw["seed"], raw["split_counts"], raw["image_size"])
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 1 or raw["name"] != _NAME:
        raise ValueError("Unsupported dataset")
    if type(raw["families"]) is not list or len(raw["families"]) != sum(raw["split_counts"].values()):
        raise ValueError("Wrong family count")
    splits = [split for split in SPLITS for _ in range(raw["split_counts"][split])]
    for index, split in enumerate(splits):
        family, expected = raw["families"][index], _family(raw["seed"], split, index)
        if type(family) is not dict or type(family.get("images")) is not list or len(family["images"]) != 4:
            raise ValueError("Each family requires four images")
        for item, canonical in zip(family["images"], expected["images"], strict=True):
            digest = item.get("sha256") if type(item) is dict else None
            if type(digest) is not str or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("Invalid image checksum")
            canonical["sha256"] = digest
        # Exact comparison also rejects duplicate IDs and wrong factorial coverage.
        if _json(family) != _json(expected):
            raise ValueError("Noncanonical family, cases, coordinates, IDs, or regions")
        rendered = []
        for item in family["images"]:
            encoded = _read(_absolute(root / item["relative_path"]), _MAX_PNG)
            if hashlib.sha256(encoded).hexdigest() != item["sha256"]:
                raise ValueError("Image checksum mismatch")
            if verify_images:
                from PIL import Image

                image = _render(family, item["edge"], item["palette"], raw["image_size"])
                with Image.open(BytesIO(encoded)) as stored:
                    if (stored.format != "PNG" or stored.mode != "RGB" or stored.size != image.size
                            or stored.tobytes() != image.tobytes()):
                        raise ValueError("Image pixels or scene-program mismatch")
                rendered.append(image)
        if verify_images:
            _check_regions(rendered, family["regions"], raw["image_size"])
    return raw


def contact_sheet(root: Path, output: Path, *, limit: int = 8) -> Path:
    """Publish a labeled preview only; labels never enter experiment images."""
    from PIL import Image, ImageDraw

    _integer(limit, 1, 32, "limit")
    output, root = _absolute(output), _absolute(root)
    if output.exists():
        raise FileExistsError(output)
    dataset = load_dataset(root)
    families = dataset["families"][:limit]
    sheet = Image.new("RGB", (960, 280 * len(families)), "white")
    draw = ImageDraw.Draw(sheet)
    for row, family in enumerate(families):
        y = row * 280
        draw.text((8, y+4), f"{family['family_id']} / {family['split']}", fill="black")
        for col, item in enumerate(family["images"]):
            with Image.open(_absolute(root / item["relative_path"])) as image:
                sheet.paste(image.resize((224, 224), Image.Resampling.LANCZOS), (col*240+8, y+22))
            forward = binding_case(family, item["edge"], 1, item["palette"])["gold"]
            backward = binding_case(family, item["edge"], -1, item["palette"])["gold"]
            draw.text((col*240+8, y+250), f"e={item['edge']:+d}, p={item['palette']} "
                      f"forward:{forward} backward:{backward}", fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    _write(output, _png(sheet))
    return output