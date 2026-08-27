# Action space matches libriichi: 0-33=discard tile, 37=riichi(unused),
# 38-40=chi, 41=pon, 42=kan, 43=agari, 44=ryukyoku, 45=pass
ACTION_SPACE = 46

# Observation shape: (OBS_ROWS, OBS_COLS) float32
# Modelled after Mortal v3 encoding minus riichi/furiten/aka/SP/EV features
# OBS_ROWS depends on USE_SHANTEN_FEATURES in obs_encoder.py.
# Use obs_encoder._obs_rows() for the dynamic value, or these constants directly.
OBS_ROWS_WITH_SHANTEN    = 216
OBS_ROWS_WITHOUT_SHANTEN = 210
OBS_ROWS = OBS_ROWS_WITH_SHANTEN  # default (use_shanten_features=true)
OBS_COLS = 34

# Tile indices 0-33 (deaka):
# 0-8  = 1m-9m (Man/Characters)
# 9-17 = 1p-9p (Pin/Circles)
# 18-26= 1s-9s (Sou/Bamboo)
# 27=E, 28=S, 29=W, 30=N, 31=P(Haku), 32=F(Hatsu), 33=C(Chun)
TILE_COUNT = 34

# Wind tile indices
WIND_EAST  = 27
WIND_SOUTH = 28
WIND_WEST  = 29
WIND_NORTH = 30

# mjai wind string → tile index
WIND_INDEX = {"E": 27, "S": 28, "W": 29, "N": 30}

# Starting score per player (confirmed: 30000, but stored scaled in logs as 500 for quick games)
# The obs encoder uses the actual score values from the log directly.

# Action indices
ACTION_DISCARD_BASE = 0   # 0-33: discard tile index
ACTION_RIICHI       = 37  # unused in Northeast rules
ACTION_CHI_LOW      = 38
ACTION_CHI_MID      = 39
ACTION_CHI_HIGH     = 40
ACTION_PON          = 41
ACTION_KAN          = 42
ACTION_AGARI        = 43
ACTION_RYUKYOKU     = 44
ACTION_PASS         = 45
