# input hyperparameters
state_size = [30, 45, 4]  # Our input is a stack of 4 frames hence 30x45x4 (Width, height, channels)
stack_size = 4

# training hyperparameters
learning_rate = 0.00025  # Initial value. Used on optimizer to minimize the loss
# learning_rate = 0.0001
# learning_rate =  0.0002
epochs = 20
learning_steps_per_epoch = 2000
batch_size = 64



# Q learning hyperparameters
gamma = 0.99  # Initial Discounting rate
# gamma = 0.95



# Memory hyperparameters
pretrain_length = batch_size  # Number of experiences stored in the Memory when initialized for the first time
# replay_memory_size = 1000000  # PRODUCES VM CRASH. REQUIRES GPU USAGE
replay_memory_size = 100000  # Number of experiences the Memory can keep
# replay_memory_size = 10000



# Test parameters
test_episodes_per_epoch = 100
frame_repeat = 12
# resolution = (100, 160)
resolution = (30, 45)

# post-training parameters
episodes_to_watch = 10

