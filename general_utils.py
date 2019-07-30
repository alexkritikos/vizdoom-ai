from datetime import datetime
import constants


class DateUtils:
    def __init__(self):
        pass

    @staticmethod
    def get_current_timestamp():
        return datetime.now().strftime("%d-%b-%Y-%H%M%S")


class InputValidator:
    def __init__(self):
        pass

    def validate_scenario_input():
        user_scenario = input("Specify scenario configuration: ")
        while user_scenario not in constants.scenarios_constants:
            user_scenario = input("Specify scenario configuration: ")
        return constants.scenarios_constants[user_scenario]

    def should_save_or_load():
        valid_options = ('y', 'n')
        user_input = input("Save model after training?(y/n): ")
        while user_input not in valid_options:
            user_input = input("Save model after training?(y/n): ")
        if user_input == 'y':
            return True, False, False
        else:
            return False, True, True
