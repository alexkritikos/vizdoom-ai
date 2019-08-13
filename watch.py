from collections import deque
from time import sleep
from environment import stack_frames, init_watching_environment
from general_utils import load_file_simple, validate_scenario_input
import numpy as np
from network import create_network
from parameters import state_size, stack_size, episodes_to_watch, frame_repeat
import tensorflow as tf


def watch_doom_game(configuration):
    game, actions = init_watching_environment(configuration)
    load_file = load_file_simple(configuration)
    session = tf.compat.v1.Session()
    learn, get_q_values, get_best_action = create_network(session, len(actions))
    saver = tf.compat.v1.train.Saver()
    saver.restore(session, load_file)
    session.run(tf.compat.v1.global_variables_initializer())
    stacked_frames = deque([np.zeros((state_size[0], state_size[1]), dtype=np.int) for i in range(stack_size)],
                           maxlen=4)
    run_episodes(game, get_best_action, actions, stacked_frames)


def run_episodes(game, best_action, actions, stacked_frames):
    for _ in range(episodes_to_watch):
        game.new_episode()
        while not game.is_episode_finished():
            state = game.get_state().screen_buffer
            state, stacked_frames = stack_frames(stacked_frames, state, True)
            best_action_index = best_action(state)
            # Instead of make_action(a, frame_repeat) in order to make the animation smooth
            game.set_action(actions[best_action_index])
            for _ in range(frame_repeat):
                game.advance_action()
        # Sleep between episodes
        sleep(1.0)
        print("Total score: ", game.get_total_reward())


if __name__ == '__main__':
    config = validate_scenario_input()
    watch_doom_game(config)

