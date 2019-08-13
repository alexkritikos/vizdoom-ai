# input hyperparameters
state_size = [100, 120, 4]  # Our input is a stack of 4 frames hence 100x120x4 (Width, height, channels)
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
decay_step = 0  # Initialize the decay rate (that will use to reduce epsilon)
tau = 0  # Every tau step we update the target network



# Memory hyperparameters
# pretrain_memory_size = 100000  # Number of experiences stored in the Memory when initialized for the first time
pretrain_memory_size = 10000
# Number of experiences the Memory can keep
# replay_memory_size = 1000000  # PRODUCES VM CRASH. REQUIRES GPU USAGE
# replay_memory_size = 100000
replay_memory_size = 10000  # Used for deadly corridor where the state array will be huge
PER_e = 0.01  # Parameter that we use to avoid some experiences to have 0 probability of being taken
PER_a = 0.6  # Parameter that we use to make a tradeoff between taking only exp with high priority and sampling randomly
# PER_a = 0.7
PER_b = 0.4  # Importance-sampling, from initial value increasing to 1
PER_b_increment_per_sampling = 0.001
absolute_error_upper = 1.  # clipped abs error



# Test parameters
test_episodes_per_epoch = 100
frame_repeat = 12
# resolution = (100, 160)
resolution = (100, 120)

# post-training parameters
episodes_to_watch = 10

