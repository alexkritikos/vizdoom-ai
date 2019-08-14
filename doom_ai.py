#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
import itertools as it
from random import randint, random, choice
from time import time, sleep
from tqdm import trange
import vizdoom as vzd
from argparse import ArgumentParser

from general_utils import *
import os
from network import *
from memory import ReplayMemory, PERMemory
from environment import *
from parameters import *


def learn_from_memory():
    """ Learns from a single transition (making use of replay memory).
    s2 is ignored if s2_isterminal """

    # Get a random minibatch from the replay memory and learns from it.
    if memory.size > batch_size:
        s1, a, s2, isterminal, r = memory.get_sample(batch_size)

        q2 = np.max(get_q_values(s2), axis=1)
        target_q = get_q_values(s1)
        # target differs from q only for the selected action. The following means:
        # target_Q(s,a) = r + gamma * max Q(s2,_) if not isterminal else r
        target_q[np.arange(target_q.shape[0]), a] = r + gamma * (1 - isterminal) * q2
        learn(s1, target_q)


def perform_learning_step(epoch, stacked_frames):
    """ Makes an action according to eps-greedy policy, observes the result
    (next state, reward) and learns from the transition"""

    def exploration_rate(epoch):
        """# Define exploration rate change over time"""
        start_eps = 1.0
        end_eps = 0.1
        const_eps_epochs = 0.1 * epochs  # 10% of learning time
        eps_decay_epochs = 0.6 * epochs  # 60% of learning time

        if epoch < const_eps_epochs:
            return start_eps
        elif epoch < eps_decay_epochs:
            # Linear decay
            return start_eps - (epoch - const_eps_epochs) / \
                               (eps_decay_epochs - const_eps_epochs) * (start_eps - end_eps)
        else:
            return end_eps

    # s1 = preprocess_frame(game.get_state().screen_buffer)
    s1 = game.get_state().screen_buffer
    s1, stacked_frames = stack_frames(stacked_frames, s1, True)

    # With probability eps make a random action.
    eps = exploration_rate(epoch)
    if random() <= eps:
        a = randint(0, len(actions) - 1)
    else:
        # Choose the best action according to the network.
        a = get_best_action(s1)
    reward = game.make_action(actions[a], frame_repeat)
    isterminal = game.is_episode_finished()
    # s2 = preprocess_frame(game.get_state().screen_buffer) if not isterminal else None
    if not isterminal:
        s2 = game.get_state().screen_buffer
        s2, stacked_frames = stack_frames(stacked_frames, s2, True)
    else:
        s2 = np.zeros(s1.shape)
    # Remember the transition that was just experienced.
    # experience = s1, a, reward, s2, isterminal
    # memory.store(experience)

    learn_from_memory()


"""
This function will do the part
With ϵ select a random action atat, otherwise select at=argmaxaQ(st,a)
"""


def predict_action(explore_start, explore_stop, decay_rate, decay_step, session, state):
    ## EPSILON GREEDY STRATEGY
    # Choose action a from state s using epsilon greedy.
    ## First we randomize a number
    tradeoff = np.random.rand()  # Exploration - exploitation tradeoff
    # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
    explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
    if explore_probability > tradeoff:
        # Make a random action (exploration)
        action = randint(0, len(actions) - 1)
    else:
        # Get action from Q-network (exploitation)
        # Estimate the Qs values state
        predictions = session.run(network.output_layer, feed_dict={network.states_: state.reshape((1, *state.shape))})
        # Take the biggest Q value (= the best action)
        choice = np.argmax(predictions)
        action = actions[int(choice)]
    return action, explore_probability


def train_agent(s1, stack, tau, decay_step, session, epoch, target_graph):
    tau += 1
    decay_step += 1
    a, explore_prob = predict_action(explore_start, explore_stop, decay_rate, decay_step, session, stack)
    r = game.make_action(actions[a])
    isterminal = game.is_episode_finished()
    if not isterminal:
        s2 = game.get_state().screen_buffer
        s2, stack = stack_frames(stack, s2, True)
        experience = s1, a, r, s2, isterminal
        memory.store(experience)
        s1 = s2
    else:
        s2 = np.zeros(shape=(state_size[0], state_size[1]), dtype=np.int)
        s2, stack = stack_frames(stack, s2, False)
        experience = s1, a, r, s2, isterminal
        memory.store(experience)
    tree_index, mem_batch, samplingWeights = memory.sample(batch_size)
    # Get the experiences data from batch
    states_batch = np.array([each[0][0] for each in mem_batch], ndmin=3)
    actions_batch = np.array([each[0][1] for each in mem_batch])
    rewards_batch = np.array([each[0][2] for each in mem_batch])
    next_states_batch = np.array([each[0][3] for each in mem_batch], ndmin=3)
    terminals_batch = np.array([each[0][4] for each in mem_batch])
    target_Qs_batch = []

    ### DOUBLE DQN Logic
    # Use DQNNetwork to select the action to take at next_state (a') (action with the highest Q-value)
    # Use TargetNetwork to calculate the Q_val of Q(s',a')
    # Get Q values for next_state
    q_next_state = session.run(network.output_layer, feed_dict={network.states_: next_states_batch})
    # Calculate Qtarget for all actions that state
    q_target_next_state = session.run(targetNetwork.output_layer, feed_dict={targetNetwork.states_: next_states_batch})
    for i in range(0 , len(mem_batch)):
        terminal = terminals_batch[i]
        # We got a'
        a = np.argmax(q_next_state)
        if terminal:
            target_Qs_batch.append(rewards_batch[i])
        else:
            targetQ = rewards_batch[i] + gamma * q_target_next_state[i][a]
            target_Qs_batch.append(targetQ)
    targets_batch = np.array([each for each in target_Qs_batch])
    optimizer, loss, absolute_errors = session.run([network.optimizer, network.loss, network.absolute_errors],
                                                   feed_dict={network.states_: states_batch,
                                                              network.target_q_: targets_batch,
                                                              network.w_: samplingWeights,
                                                              network.a_: actions_batch})
    # Update priority
    memory.batch_update(tree_index, absolute_errors)
    # Write TF Summaries
    writer, writer_operation = get_tensorboard_writer(loss)
    summary = session.run(writer_operation, feed_dict={network.states_: states_batch,
                                                       network.target_q_: targets_batch,
                                                       network.w_: samplingWeights,
                                                       network.a_: actions_batch})
    writer.add_summary(summary, epoch)
    writer.flush()

    if tau > max_tau:
        # Update the parameters of our TargetNetwork with DQN_weights
        target_graph = update_target_graph()
        session.run(target_graph)
        tau = 0
        print("Target Network Updated")


if __name__ == '__main__':
    user_scenario = validate_scenario_input()
    save_model, load_model, skip_learning = should_save_or_load()
    parser = ArgumentParser("ViZDoom example showing how to train a simple agent using simplified DQN.")
    parser.add_argument(dest="config",
                        default=user_scenario,
                        nargs="?",
                        help="Path to the configuration file of the scenario."
                             " Please see "
                             "../../maps/*cfg for more maps.")
    parser.add_argument(dest="save",
                        default=save_model,
                        nargs="?",
                        help="Flag for saving the network weights to a file.")
    parser.add_argument(dest="load",
                        default=load_model,
                        nargs="?",
                        help="Flag for loading a pre-trained model.")
    parser.add_argument(dest="skip_learning",
                        default=skip_learning,
                        nargs="?",
                        help="Flag for skip_learning.")

    args = parser.parse_args()

    # Create Doom instance
    game = initialize_vizdoom(args.config)

    # Initialize stacked frames
    stacked_frames = deque([np.zeros((state_size[0], state_size[1]), dtype=np.int) for i in range(stack_size)],
                           maxlen=4)
    # Action = which buttons are pressed
    n = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=n)]

    # Create replay memory which will store the transitions
    # memory = ReplayMemory(capacity=replay_memory_size)

    memory = PERMemory(e=PER_e, a=PER_a, b=PER_b, b_increment_per_sampling=PER_b_increment_per_sampling,
                       abs_error_upper=absolute_error_upper, capacity=replay_memory_size)
    sess = tf.compat.v1.Session()

    network = DuelingDoubleDQN(state_size, len(actions), name="DuelingDoubleDQN")
    targetNetwork = DuelingDoubleDQN(state_size, len(actions), name="TargetDuelingDoubleDQN")

    target_updated = update_target_graph()

    # learn, get_q_values, get_best_action = create_network(sess, len(actions))
    saver = tf.compat.v1.train.Saver()
    if not load_by_scenario(args.load, get_scenario_name(user_scenario), saver, sess):
        args.skip_learning = False
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)
    print("Starting the training!")

    time_start = time()
    if not args.skip_learning:
        for epoch in range(epochs):
            print("\nEpoch %d\n-------" % (epoch + 1))
            train_episodes_finished = 0
            train_scores = []

            print("Training...")
            game.new_episode()
            initial_state = game.get_state().screen_buffer
            initial_state, stacked_frames = stack_frames(stacked_frames, initial_state, True)
            for step in trange(learning_steps_per_epoch, leave=False):
                # perform_learning_step(epoch, stacked_frames)
                train_agent(initial_state, stacked_frames, tau, decay_step, sess, epoch, target_updated)
                if game.is_episode_finished():
                    # Monte Carlo Approach: rewards are only received at the end of the game.
                    score = game.get_total_reward()
                    train_scores.append(score)
                    game.new_episode()
                    train_episodes_finished += 1

            print("%d training episodes played." % train_episodes_finished)

            train_scores = np.array(train_scores)
            print(train_scores)

            print("Results: mean: %.1f±%.1f," % (train_scores.mean(), train_scores.std()), \
                  "min: %.1f," % train_scores.min(), "max: %.1f," % train_scores.max())

            # print("\nTesting...")
            # stacked_frames = deque([np.zeros((state_size[0], state_size[1]), dtype=np.int) for i in range(stack_size)],
            #                        maxlen=4)
            # test_episode = []
            # test_scores = []
            # for test_episode in trange(test_episodes_per_epoch, leave=False):
            #     game.new_episode()
            #     while not game.is_episode_finished():
            #         state = game.get_state().screen_buffer
            #         state, stacked_frames = stack_frames(stacked_frames, state, True)
            #         best_action_index = get_best_action(state)
            #         game.make_action(actions[best_action_index], frame_repeat)
            #     r = game.get_total_reward()
            #     test_scores.append(r)
            #
            # test_scores = np.array(test_scores)
            # print(test_scores)
            #
            # print("Results: mean: %.1f±%.1f," % (
            #     test_scores.mean(), test_scores.std()), "min: %.1f" % test_scores.min(),
            #       "max: %.1f" % test_scores.max())

            save_dir = "savefiles/scenario-" + get_scenario_name(user_scenario) + "/"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            print("Saving the network weigths to:", save_dir)
            saver.save(sess, save_dir)

            print("Total elapsed time: %.2f minutes" % ((time() - time_start) / 60.0))

    game.close()
    print("======================================")
    print("Training finished. It's time to watch!")

    # Reinitialize the game with window visible
    game.set_window_visible(True)
    game.set_mode(vzd.Mode.ASYNC_PLAYER)
    game.init()
    stacked_frames = deque([np.zeros((state_size[0], state_size[1]), dtype=np.int) for i in range(stack_size)],
                           maxlen=4)
    for _ in range(episodes_to_watch):
        game.new_episode()
        while not game.is_episode_finished():
            state = game.get_state().screen_buffer
            state, stacked_frames = stack_frames(stacked_frames, state, True)
            # best_action_index = get_best_action(state)
            exp_exp_tradeoff = np.random.rand()
            explore_probability = 0.01
            if explore_probability > exp_exp_tradeoff:
                watch_action = choice(actions)
            else:
                qVals = sess.run(network.output_layer, feed_dict={network.states_: state.reshape((1, *state.shape))})
                best = np.argmax(qVals)
                watch_action = actions[int(best)]
            # Instead of make_action(a, frame_repeat) in order to make the animation smooth
            game.set_action(watch_action)
            for _ in range(frame_repeat):
                game.advance_action()
        # Sleep between episodes
        sleep(1.0)
        score = game.get_total_reward()
        print("Total score: ", score)
