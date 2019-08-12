from datetime import datetime as dt
import constants as const
import os


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


def load_file_simple(config):
    return const.SAVE_PATH + get_scenario_name(config) + "/"
