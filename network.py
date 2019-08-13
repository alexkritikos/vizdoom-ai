from parameters import learning_rate
import tensorflow as tf


# Initial implementation without stacked input functionality
class DuelingDoubleDQN:
    def __init__(self, state_size, available_actions_count, name):
        self.state_size = state_size
        self.available_actions_count = available_actions_count
        self.name = name

        with tf.variable_scope(self.name):
            self.states_ = tf.compat.v1.placeholder(tf.float32, [None, *state_size], name="InputStates")
            self.w_ = tf.compat.v1.placeholder(tf.float32, [None, 1], name="Weights")
            self.a_ = tf.compat.v1.placeholder(tf.float32, [None, available_actions_count], name="Action")
            self.target_q_ = tf.compat.v1.placeholder(tf.float32, [None, self.available_actions_count], name="TargetQ")

            self.conv1 = tf.layers.conv2d(inputs=self.states_, filters=32, kernel_size=[8, 8],
                                          strides=[4, 4], padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="convnet1")
            self.conv1_out = tf.nn.elu(self.conv1, name="convnet1_output")
            self.conv2 = tf.layers.conv2d(inputs=self.conv1_out, filters=64, kernel_size=[4, 4],
                                          strides=[2, 2], padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="convnet2")
            self.conv2_out = tf.nn.elu(self.conv2, name="convnet2_output")
            self.conv3 = tf.layers.conv2d(inputs=self.conv2_out, filters=128, kernel_size=[4, 4],
                                          strides=[2, 2], padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="convnet3")
            self.conv3_out = tf.nn.elu(self.conv3, name="convnet3_output")

            self.flatten = tf.layers.flatten(self.conv3_out)

            # Calculates V(s)
            self.value_fully_connected = tf.layers.dense(inputs=self.flatten, units=512, activation=tf.nn.elu,
                                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                         name="value_fully_connected")
            self.value = tf.layers.dense(inputs=self.value_fully_connected, units=1, activation=None,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name="value_output")

            # Calculates A(s, a)
            self.advantage_fully_connected = tf.layers.dense(inputs=self.flatten, units=512, activation=tf.nn.elu,
                                                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                             name="advantage_fully_connected")
            self.advantage = tf.layers.dense(inputs=self.advantage_fully_connected, units=self.available_actions_count,
                                             activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                             name="advantage_output")

            # Q(s,a) = V(s) + (A(s,a) - 1/|A| * sum A(s',a))
            self.output_layer = self.value + tf.subtract(self.advantage, tf.reduce_mean(self.advantage, axis=1,
                                                                                        keepdims=True))
            self.predictedQ = tf.reduce_sum(tf.multiply(self.output_layer, self.a_), axis=1)

            self.absolute_errors = tf.abs(self.target_q_ - self.predictedQ)
            self.loss = tf.reduce_mean(self.w_ * tf.squared_difference(self.target_q_, self.predictedQ))
            self.optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate)
            self.train_step = self.optimizer.minimize(self.loss)


def write_to_tensorboard(loss):
    writer = tf.summary.FileWriter("/tensorboard/dddqn/1")
    tf.summary.scalar("Loss", loss)
    write_op = tf.summary.merge_all()


def update_target_graph():
    # Get the parameters of our DQNNetwork
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "DuelingDoubleDQN")
    # Get the parameters of our Target_network
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "TargetDuelingDoubleDQN")
    op_holder = []
    # Update our target_network parameters with DQNNetwork parameters
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder





# Initial network implementation
def create_network(session, available_actions_count):
    from parameters import state_size
    # Create the input variables
    s1_ = tf.compat.v1.placeholder(tf.float32, [None] + list(state_size), name="State")
    a_ = tf.compat.v1.placeholder(tf.int32, [None], name="Action")
    target_q_ = tf.compat.v1.placeholder(tf.float32, [None, available_actions_count], name="TargetQ")

    # Add 2 convolutional layers with ReLu activation
    conv1 = tf.contrib.layers.convolution2d(s1_, num_outputs=8, kernel_size=[6, 6], stride=[3, 3],
                                            activation_fn=tf.nn.relu,
                                            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                            biases_initializer=tf.constant_initializer(0.1))
    conv2 = tf.contrib.layers.convolution2d(conv1, num_outputs=8, kernel_size=[3, 3], stride=[2, 2],
                                            activation_fn=tf.nn.relu,
                                            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                            biases_initializer=tf.constant_initializer(0.1))
    conv2_flat = tf.contrib.layers.flatten(conv2)
    fc1 = tf.contrib.layers.fully_connected(conv2_flat, num_outputs=128, activation_fn=tf.nn.relu,
                                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                                            biases_initializer=tf.constant_initializer(0.1))

    q = tf.contrib.layers.fully_connected(fc1, num_outputs=available_actions_count, activation_fn=None,
                                          weights_initializer=tf.contrib.layers.xavier_initializer(),
                                          biases_initializer=tf.constant_initializer(0.1))
    best_a = tf.argmax(q, 1)

    loss = tf.compat.v1.losses.mean_squared_error(q, target_q_)

    optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate)
    # Update the parameters according to the computed gradient using RMSProp.
    train_step = optimizer.minimize(loss)

    def function_learn(s1, target_q):
        feed_dict = {s1_: s1, target_q_: target_q}
        l, _ = session.run([loss, train_step], feed_dict=feed_dict)
        return l

    def function_get_q_values(state):
        return session.run(q, feed_dict={s1_: state})

    def function_get_best_action(state):
        return session.run(best_a, feed_dict={s1_: state})

    def function_simple_get_best_action(state):
        return function_get_best_action(state.reshape([1, state_size[0], state_size[1], state_size[2]]))[0]

    return function_learn, function_get_q_values, function_simple_get_best_action
