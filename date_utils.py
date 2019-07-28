from datetime import datetime


class DateUtils:
    def __init__(self):
        pass

    @staticmethod
    def get_current_timestamp():
        return datetime.now().strftime("%d-%b-%Y-%H%M%S")
