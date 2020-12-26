import argparse
import datetime
import h5py
from encoder.base import get_encoder_by_name
from agent.policyAgent import PolicyAgent
from rl.experience import load_experience
import os
import tensorflow as tf
from tensorflow.keras.optimizers import SGD

def train_agent():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent-out', required=True)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--bs', type=int, default=512)
    parser.add_argument('experience_dir', nargs='+')

    args = parser.parse_args()



    learning_agent = agent.load_policy_agent(h5py.File(args.learning_agent))

    experience_list = os.listdir(args.experience_dir)

    for exp_filename in experience_list:
        print('Training with %s...' % exp_filename)
        exp_buffer = load_experience(h5py.File(exp_filename))
        learning_agent.train(exp_buffer, lr=args.lr, batch_size=args.bs)

    with h5py.File(args.agent_out, 'w') as updated_agent_outf:
        learning_agent.serialize(updated_agent_outf)

def train_model():
    parser = argparse.ArgumentParser()

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.00015)
    parser.add_argument('--bs', type=int, default=2048)
    parser.add_argument('--experience_dir',  default="experience_data/")

    args = parser.parse_args()

    model = tf.keras.models.load_model('saved_model/layer_20_model')
    encoder = get_encoder_by_name('layer_20_encoder', (8, 8))
    learning_agent = PolicyAgent(model, encoder, 1)

    experience_list = os.listdir(args.experience_dir)

    for exp_filename in [args.experience_dir + file for file in experience_list]:
        print('Training with %s...' % exp_filename)
        exp_buffer = load_experience(h5py.File(exp_filename))
        learning_agent.train(exp_buffer, lr=args.lr, batch_size=args.bs)

    learning_agent.save(version="2_00015_rate")

if __name__ == '__main__':
    train_model()