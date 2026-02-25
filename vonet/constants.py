CLASS_NAMES = ("front", "rear", "side", "front-side", "rear-side")

VIEWPOINT_TO_CLASS_NAME = {
    1: "front",
    2: "rear",
    3: "side",
    4: "front-side",
    5: "rear-side",
}

CLASS_NAME_ALIASES = {
    "front": ("front",),
    "rear": ("rear",),
    "side": ("side",),
    "front-side": ("front-side", "front_side", "frontside"),
    "rear-side": ("rear-side", "rear_side", "rearside"),
}

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
