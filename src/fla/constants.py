FPS: int = 60
PX_PER_MM: float = 37.4

TRIPOD_STANCES: frozenset = frozenset({(0, 1, 0, 1, 0, 1), (1, 0, 1, 0, 1, 0)})
TETRAPOD_STANCES: frozenset = frozenset({
    (0, 0, 1, 1, 0, 0), (0, 1, 0, 0, 0, 1), (1, 0, 0, 0, 1, 0),
    (1, 0, 0, 0, 0, 1), (0, 0, 1, 0, 1, 0), (0, 1, 0, 1, 0, 0),
})

# Column order used throughout the package: (hindR, midR, foreR, hindL, midL, foreL)
LEG_NAMES: list = ["hindlegR1", "midlegR1", "forelegR1", "hindlegL1", "midlegL1", "forelegL1"]
LEG_LABELS: list = ["RH", "RM", "RF", "LH", "LM", "LF"]
