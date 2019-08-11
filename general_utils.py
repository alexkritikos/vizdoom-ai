from datetime import datetime as dt
import constants as const
from skimage import transform
import os
from parameters import *
import numpy as np
from collections import deque


def get_current_timestamp():
    return dt.now().strftime("%d-%b-%Y-%H%M%S")


def validate_scenario_input():
    user_scenario = input("Specify scenario configuration: ")
    while user_scenario not in const.scenarios_constants:
        user_scenario = input("This config doesn't exist. Try again: ")
    return const.scenarios_constants[user_scenario]


def get_scenario_name(config):
    for conf in const.scenarios_constants:
        if config == const.scenarios_constants[conf]:
            return conf.lower()


def should_save_or_load():
    valid_options = ('y', 'n')
    user_input = input("Save model after training?(y/n): ")
    while user_input not in valid_options:
        user_input = input("Type \'y\' or \'n\': ")
    if user_input == 'y':
        return True, False, False
    else:
        return False, True, True


def load_by_scenario(is_load, scenario, saver, session):
    if is_load:
        load_path = const.SAVE_PATH + scenario + "/"
        if os.path.exists(load_path):
            print("Loading model from: ", load_path)
            saver.restore(session, load_path)
            return True
        else:
            print("Couldn't restore a model for the "
                  "specified scenario. A new model "
                  "will be saved after training.")
            return False
    else:
        return False


# Processes Doom screen image to produce cropped and resized image.
def preprocess_frame(frame):
    # The frame is already grayscaled in vizdoom config
    # x = np.mean(frame,-1)

    # Crop the screen (remove the roof because it contains no information)
    # [Up: Down, Left: right]
    cropped_frame = frame[40:, :]
    # Normalize Pixel Values
    normalized_frame = cropped_frame / 255.0
    # Resize
    # preprocessed_frame = transform.resize(normalized_frame, [100, 160])
    preprocessed_frame = transform.resize(normalized_frame, [30, 45])
    return preprocessed_frame


def stack_frames(stacked_frames, state, is_new_episode):
    # Preprocess frame
    frame = preprocess_frame(state)
    if is_new_episode:
        # Clear our stacked_frames
        stacked_frames = deque([np.zeros((30, 45), dtype=np.int) for i in range(stack_size)], maxlen=4)
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
