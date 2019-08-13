import vizdoom as vzd
from parameters import state_size, stack_size
import numpy as np
from skimage import transform
from collections import deque
import itertools as it


# Creates and initializes ViZDoom environment.
def initialize_vizdoom(config_file_path):
    print("Initializing doom...")
    game = vzd.DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(False)  # Only for training purposes
    game.set_mode(vzd.Mode.PLAYER)
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.init()
    print("Doom initialized.")
    return game


def init_watching_environment(configuration):
    print("Initializing doom...")
    game = vzd.DoomGame()
    game.load_config(configuration)
    game.set_window_visible(True)
    game.set_mode(vzd.Mode.ASYNC_PLAYER)
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    n = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=n)]
    game.init()
    print("Doom initialized. It's time to watch!")
    return game, actions


# Processes Doom screen image to produce cropped and resized image.
def preprocess_frame(frame):
    # The frame is already grayscaled in vizdoom config
    # [Up: Down, Left: right]
    cropped_frame = frame[15:-5, 20:-20]   # DEADLY CORRIDOR TUNING
    # Normalize Pixel Values
    normalized_frame = cropped_frame / 255.0
    # Resize
    # preprocessed_frame = transform.resize(normalized_frame, [100, 160])
    preprocessed_frame = transform.resize(normalized_frame, [state_size[0], state_size[1]])
    return preprocessed_frame


def stack_frames(stacked_frames, state, is_new_episode):
    # Preprocess frame
    frame = preprocess_frame(state)
    # frame = transform.resize(state, [state_size[0], state_size[1]])
    if is_new_episode:
        # Clear our stacked_frames
        stacked_frames = deque([np.zeros((state_size[0], state_size[1]), dtype=np.int) for i in range(stack_size)],
                               maxlen=4)
        for i in range(3):
            stacked_frames.append(frame)
        # Stack the frames
        stacked_state = np.stack(stacked_frames, axis=2)
    else:
        # Append frame to deque, automatically removes the oldest frame
        stacked_frames.append(frame)
        # Build the stacked state (first dimension specifies different frames)
        stacked_state = np.stack(stacked_frames, axis=2)
    return stacked_state, stacked_frames
