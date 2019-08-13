#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
import itertools as it
from random import randint, random
from time import time, sleep
from tqdm import trange
import vizdoom as vzd
from argparse import ArgumentParser

from general_utils import *
import os
from network import *
from memory import ReplayMemory, PERMemory
from environment import *


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
    experience = s1, a, reward, s2, isterminal
    memory.store(experience)

    # learn_from_memory()


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
    session = tf.compat.v1.Session()

    # network = DuelingDoubleDQN(session, len(actions), name="DuelingDoubleDQN")
    # targetNetwork = DuelingDoubleDQN(session, len(actions), name="TargetDuelingDoubleDQN")

    learn, get_q_values, get_best_action = create_network(session, len(actions))
    saver = tf.compat.v1.train.Saver()
    if not load_by_scenario(args.load, get_scenario_name(user_scenario), saver, session):
        args.skip_learning = False
        init = tf.compat.v1.global_variables_initializer()
        session.run(init)
    print("Starting the training!")

    time_start = time()
    if not args.skip_learning:
        for epoch in range(epochs):
            print("\nEpoch %d\n-------" % (epoch + 1))
            train_episodes_finished = 0
            train_scores = []

            print("Training...")
            game.new_episode()
            for step in trange(pretrain_memory_size, leave=False):
                perform_learning_step(epoch, stacked_frames)
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

            print("\nTesting...")
            stacked_frames = deque([np.zeros((state_size[0], state_size[1]), dtype=np.int) for i in range(stack_size)],
                                   maxlen=4)
            test_episode = []
            test_scores = []
            for test_episode in trange(test_episodes_per_epoch, leave=False):
                game.new_episode()
                while not game.is_episode_finished():
                    state = game.get_state().screen_buffer
                    state, stacked_frames = stack_frames(stacked_frames, state, True)
                    best_action_index = get_best_action(state)
                    game.make_action(actions[best_action_index], frame_repeat)
                r = game.get_total_reward()
                test_scores.append(r)

            test_scores = np.array(test_scores)
            print(test_scores)

            print("Results: mean: %.1f±%.1f," % (
                test_scores.mean(), test_scores.std()), "min: %.1f" % test_scores.min(),
                  "max: %.1f" % test_scores.max())

            save_dir = "savefiles/scenario-" + get_scenario_name(user_scenario) + "/"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            print("Saving the network weigths to:", save_dir)
            saver.save(session, save_dir)

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
            best_action_index = get_best_action(state)

            # Instead of make_action(a, frame_repeat) in order to make the animation smooth
            game.set_action(actions[best_action_index])
            for _ in range(frame_repeat):
                game.advance_action()

        # Sleep between episodes
        sleep(1.0)
        score = game.get_total_reward()
        print("Total score: ", score)
