from datetime import datetime as dt
import constants as const
from skimage import transform


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
            print(conf.lower())
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
