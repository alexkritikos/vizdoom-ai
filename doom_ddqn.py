#!/usr/bin/env python
from __future__ import print_function
import os
import random
import numpy as np
from collections import deque
from keras import backend as K
import tensorflow as tf

from environment import initialize_vizdoom, preprocess_frame_v2
from network import dqn


class DoubleDQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.99
        self.learning_rate = 0.0001
        self.epsilon = 1.0
        self.initial_epsilon = 1.0
        self.final_epsilon = 0.0001
        self.batch_size = 32
        self.observe = 5000
        self.explore = 50000
        self.frame_per_action = 4
        self.update_target_freq = 3000
        self.timestep_per_train = 100
        self.max_iterations = 300000
        self.memory = deque()
        self.max_memory = 50000
        self.model = None
        self.target_model = None
        self.stats_window_size = 50
        self.mavg_score = []
        self.var_score = []
        self.mavg_ammo_left = []
        self.mavg_kill_counts = []

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            action_idx = random.randrange(self.action_size)
        else:
            q = self.model.predict(state)
            action_idx = np.argmax(q)
        return action_idx

    def shape_reward(self, r_t, misc, prev_misc, t):
        if misc[0] > prev_misc[0]:
            r_t = r_t + 1
        if misc[1] < prev_misc[1]:
            r_t = r_t - 0.1
        if misc[2] < prev_misc[2]:
            r_t = r_t - 0.1
        return r_t

    def replay_memory(self, s_t, action_idx, r_t, s_t1, is_terminated, t):
        self.memory.append((s_t, action_idx, r_t, s_t1, is_terminated))
        if self.epsilon > self.final_epsilon and t > self.observe:
            self.epsilon -= (self.initial_epsilon - self.final_epsilon) / self.explore
        if len(self.memory) > self.max_memory:
            self.memory.popleft()
        if t % self.update_target_freq == 0:
            self.update_target_model()

    def train_replay(self):
        num_samples = min(self.batch_size * self.timestep_per_train, len(self.memory))
        replay_samples = random.sample(self.memory, num_samples)

        update_input = np.zeros(((num_samples,) + self.state_size))
        update_target = np.zeros(((num_samples,) + self.state_size))
        action, reward, done = [], [], []

        for i in range(num_samples):
            update_input[i, :, :, :] = replay_samples[i][0]
            action.append(replay_samples[i][1])
            reward.append(replay_samples[i][2])
            update_target[i, :, :, :] = replay_samples[i][3]
            done.append(replay_samples[i][4])

        target = self.model.predict(update_input)
        target_val = self.model.predict(update_target)
        target_val_ = self.target_model.predict(update_target)

        for i in range(num_samples):
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                a = np.argmax(target_val[i])
                target[i][action[i]] = reward[i] + self.gamma * (target_val_[i][a])
        loss = self.model.fit(update_input, target, batch_size=self.batch_size, epochs=1, verbose=0)
        return np.max(target[-1]), loss.history['loss']

    def save_model(self, name):
        self.model.save_weights(name)

    def load_model(self, name):
        self.model.load_weights(name)


if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)
    
    game = initialize_vizdoom("maps/config/defend_the_center.cfg", True)

    game.new_episode()
    game_state = game.get_state()
    misc = game_state.game_variables
    prev_misc = misc

    action_size = game.get_available_buttons_size()

    img_rows, img_cols = 64, 64
    img_channels = 4

    state_size = (img_rows, img_cols, img_channels)
    agent = DoubleDQNAgent(state_size, action_size)

    agent.model = dqn(state_size, action_size, agent.learning_rate)
    agent.target_model = dqn(state_size, action_size, agent.learning_rate)

    x_t = game_state.screen_buffer
    x_t = preprocess_frame_v2(x_t, size=(img_rows, img_cols))
    s_t = np.stack(([x_t] * 4), axis=2)
    s_t = np.expand_dims(s_t, axis=0)

    is_terminated = game.is_episode_finished()

    epsilon = agent.initial_epsilon
    GAME = 0
    t = 0
    max_life = 0
    life = 0

    life_buffer, ammo_buffer, kills_buffer = [], [], []

    while not game.is_episode_finished():
        loss = 0
        Q_max = 0
        r_t = 0
        a_t = np.zeros([action_size])

        action_idx = agent.get_action(s_t)
        a_t[action_idx] = 1

        a_t = a_t.astype(int)
        game.set_action(a_t.tolist())
        skiprate = agent.frame_per_action
        game.advance_action(skiprate)

        game_state = game.get_state()
        is_terminated = game.is_episode_finished()

        r_t = game.get_last_reward()

        if is_terminated:
            if life > max_life:
                max_life = life
            GAME += 1
            life_buffer.append(life)
            ammo_buffer.append(misc[1])
            kills_buffer.append(misc[0])
            print("Episode Finish ", misc)
            game.new_episode()
            game_state = game.get_state()
            misc = game_state.game_variables
            x_t1 = game_state.screen_buffer

        x_t1 = game_state.screen_buffer
        misc = game_state.game_variables

        x_t1 = preprocess_frame_v2(x_t1, size=(img_rows, img_cols))
        x_t1 = np.reshape(x_t1, (1, img_rows, img_cols, 1))
        s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)

        r_t = agent.shape_reward(r_t, misc, prev_misc, t)

        if is_terminated:
            life = 0
        else:
            life += 1

        prev_misc = misc

        agent.replay_memory(s_t, action_idx, r_t, s_t1, is_terminated, t)

        if t > agent.observe and t % agent.timestep_per_train == 0:
            Q_max, loss = agent.train_replay()

        s_t = s_t1
        t += 1

        if t % 10000 == 0:
            if not os.path.exists("savefiles/ddqn"):
                print("Creating model directory")
                os.makedirs("savefiles/ddqn")
            print("Saving model to: savefiles/ddqn/ddqn.h5")
            agent.model.save_weights("savefiles/ddqn/ddqn.h5", overwrite=True)

        state = ""
        if t <= agent.observe:
            state = "observe"
        elif agent.observe < t <= agent.observe + agent.explore:
            state = "explore"
        else:
            state = "train"

        if is_terminated:
            print("TIME", t, "/ GAME", GAME, "/ STATE", state, \
                  "/ EPSILON", agent.epsilon, "/ ACTION", action_idx, "/ REWARD", r_t, \
                  "/ Q_MAX %e" % np.max(Q_max), "/ LIFE", max_life, "/ LOSS", loss)

            if GAME % agent.stats_window_size == 0 and t > agent.observe:
                print("Update Rolling Statistics")
                agent.mavg_score.append(np.mean(np.array(life_buffer)))
                agent.var_score.append(np.var(np.array(life_buffer)))
                agent.mavg_ammo_left.append(np.mean(np.array(ammo_buffer)))
                agent.mavg_kill_counts.append(np.mean(np.array(kills_buffer)))

                life_buffer, ammo_buffer, kills_buffer = [], [], []

                if not os.path.exists("statistics/ddqn"):
                    print("Creating statistics directory")
                    os.makedirs("statistics/ddqn")

                with open("statistics/ddqn/ddqn_stats.txt", "w") as stats_file:
                    stats_file.write('Game: ' + str(GAME) + '\n')
                    stats_file.write('Max Score: ' + str(max_life) + '\n')
                    stats_file.write('mavg_score: ' + str(agent.mavg_score) + '\n')
                    stats_file.write('var_score: ' + str(agent.var_score) + '\n')
                    stats_file.write('mavg_ammo_left: ' + str(agent.mavg_ammo_left) + '\n')
                    stats_file.write('mavg_kill_counts: ' + str(agent.mavg_kill_counts) + '\n')

        if t >= agent.max_iterations:
            print("Training finished")
            break

