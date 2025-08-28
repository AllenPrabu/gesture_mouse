# config.py
# This file contains the essential settings for the gesture control application.

# --- MOUSE CONTROL ---
SMOOTHING_FACTOR = 3     # Lower value = more responsive, higher = smoother. (2-5 is a good range)
SENSITIVITY_PADDING = 200 # Pixels from the edge of the frame to ignore. Lower value = more sensitive.
SCROLL_SPEED = 0.05  # Lower = slower scroll, Higher = faster scroll (try 0.1 to 1.0)


# --- GESTURE THRESHOLDS ---
CLICK_DISTANCE_THRESHOLD = 30.0 # The max distance between finger tips to register a pinch/click.
ACTION_COOLDOWN = 1.0           # Seconds to wait between performing discrete actions (like clicks or shortcuts).

# --- SWIPE GESTURE ---
SWIPE_DISTANCE_RATIO = 0.10     # The swipe must cover at least 25% of the screen width.
FIST_LOCK_DURATION = 1.0        # How long swipe mode stays active after making a fist.
SWIPE_NAV_DISTANCE_RATIO = 0.03 # How far to move to NAVIGATE left/right within the switcher.