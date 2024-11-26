import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

class R2D2Net(Model):
    def __init__(self, in_dim, hid_dim, out_dim, num_lstm_layers=2):
        super(R2D2Net, self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_lstm_layers = num_lstm_layers

        self.net = tf.keras.Sequential([
            layers.Dense(hid_dim, activation='relu')
        ])
        self.lstm = layers.LSTM(
            hid_dim,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='orthogonal',
            recurrent_activation='sigmoid',
        )
        self.fc_v = layers.Dense(1)
        self.fc_a = layers.Dense(out_dim)

    def get_initial_state(self, batch_size):
        h0 = tf.zeros([self.num_lstm_layers, batch_size, self.hid_dim])
        c0 = tf.zeros([self.num_lstm_layers, batch_size, self.hid_dim])
        return {"h0": h0, "c0": c0}

    def duel(self, v, a, legal_move):
        legal_a = a * legal_move
        q = v + legal_a - tf.reduce_mean(legal_a, axis=2, keepdims=True)
        return q

    def call(self, s, legal_move, hid):
        x = self.net(s)
        o, h, c = self.lstm(x, initial_state=[hid["h0"], hid["c0"]])
        v = self.fc_v(o)
        a = self.fc_a(o)
        q = self.duel(v, a, legal_move)
        return q, {"h0": h, "c0": c}

    def act(self, s, legal_move, hid):
        x = self.net(tf.expand_dims(s, axis=0))  # Add time dimension
        o, h, c = self.lstm(x, initial_state=[hid["h0"], hid["c0"]])
        a = self.fc_a(o)
        a = tf.squeeze(a, axis=0)  # Remove time dimension
        return a, {"h0": h, "c0": c}


class R2D2Agent(Model):
    def __init__(self, multi_step, gamma, eta, in_dim, hid_dim, out_dim):
        super(R2D2Agent, self).__init__()
        self.online_net = R2D2Net(in_dim, hid_dim, out_dim)
        self.target_net = R2D2Net(in_dim, hid_dim, out_dim)
        self.multi_step = multi_step
        self.gamma = gamma
        self.eta = eta

    def get_initial_state(self, batch_size):
        return self.online_net.get_initial_state(batch_size)

    def sync_target_with_online(self):
        self.target_net.set_weights(self.online_net.get_weights())

    def greedy_act(self, s, legal_move, hid):
        q, new_hid = self.online_net.act(s, legal_move, hid)
        legal_q = (1 + q - tf.reduce_min(q, axis=1, keepdims=True)) * legal_move
        greedy_action = tf.argmax(legal_q, axis=1)
        return greedy_action, new_hid

    def compute_priority(self, obs, action, reward, bootstrap, next_obs, hid, next_hid):
        q, _ = self.online_net(obs['s'], obs['legal_move'], hid)
        q = tf.squeeze(q, axis=0)
        online_qa = tf.gather(q, action, batch_dims=1)

        next_q, _ = self.target_net(next_obs['s'], next_obs['legal_move'], next_hid)
        next_q = tf.squeeze(next_q, axis=0)
        bootstrap_q = tf.gather(next_q, tf.argmax(next_q, axis=1), batch_dims=1)

        target = reward + bootstrap * (self.gamma ** self.multi_step) * bootstrap_q
        priority = tf.abs(target - online_qa)
        return priority

    def loss(self, batch):
        err = self.compute_priority(
            batch['obs'],
            batch['action'],
            batch['reward'],
            batch['bootstrap'],
            batch['next_obs'],
            batch['hid'],
            batch['next_hid'],
        )
        loss = tf.keras.losses.Huber(reduction='none')(err, tf.zeros_like(err))
        return tf.reduce_sum(loss, axis=0), tf.reduce_mean(err)
