from constants import *

# network hyperparameters
state_size = [30, 45, 4]
stack_size = 4

# training hyperparameters
learning_rate = 0.00025
# learning_rate = 0.0001
discount_factor = 0.99
epochs = 2
learning_steps_per_epoch = 2000
replay_memory_size = 10000
batch_size = 64
test_episodes_per_epoch = 100
frame_repeat = 12
# resolution = (100, 160)
resolution = (30, 45)
episodes_to_watch = 10

DEFAULT_MODEL_SAVEFILE = "savefiles/scenario-BASIC/"
DEFAULT_CONFIG = scenarios_constants['BASIC']
