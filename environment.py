import skimage
import vizdoom as vzd
from parameters import state_size, stack_size
import numpy as np
from skimage import transform
from collections import deque
import itertools as it


def initialize_vizdoom(config_file_path, is_rgb):
    print("Initializing doom...")
    game = vzd.DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(False)
    game.set_mode(vzd.Mode.PLAYER)
    if not is_rgb:
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


def preprocess_frame(frame):
    cropped_frame = frame[15:-5, 20:-20]   # DEADLY CORRIDOR TUNING
    normalized_frame = cropped_frame / 255.0
    preprocessed_frame = transform.resize(normalized_frame, [state_size[0], state_size[1]])
    return preprocessed_frame


def preprocess_frame_v2(frame, size):
    frame = np.rollaxis(frame, 0, 3)
    frame = skimage.transform.resize(frame, size)
    frame = skimage.color.rgb2gray(frame)
    return frame


def stack_frames(stacked_frames, state, is_new_episode):
    frame = preprocess_frame(state)
    if is_new_episode:
        stacked_frames = deque([np.zeros((state_size[0], state_size[1]), dtype=np.int) for i in range(stack_size)],
                               maxlen=4)
        for i in range(3):
            stacked_frames.append(frame)
        stacked_state = np.stack(stacked_frames, axis=2)
    else:
        stacked_frames.append(frame)
        stacked_state = np.stack(stacked_frames, axis=2)
    return stacked_state, stacked_frames
