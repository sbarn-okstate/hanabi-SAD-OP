"""
Code based on Python PyTorch code from https://github.com/codeaudit/hanabi_SAD/blob/master/pyhanabi/vdn_r2d2.py
"""

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

class R2D2Net(tf.keras.Model):
    def __init__(self, in_dim, hid_dim, out_dim, num_lstm_layers=2):
        super(R2D2Net, self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_lstm_layers = num_lstm_layers

        #Network architecture
        self.net = tf.keras.Sequential([
            layers.Dense(self.hid_dim, activation='relu')
        ])
        self.lstm_layers = []
        for i in range(self.num_lstm_layers):
            return_sequences = i < self.num_lstm_layers - 1  #True for all but last layer
            self.lstm_layers.append(tf.keras.layers.LSTM(
                self.hid_dim, return_state=True, return_sequences=return_sequences
            ))

        self.lstm = tf.keras.Sequential(self.lstm_layers)

        self.fc_v = layers.Dense(1)
        self.fc_a = layers.Dense(self.out_dim)

    def get_h0(self, batch_size):
        #Initialize LSTM hidden states (h0, c0)
        shape = (self.num_lstm_layers, batch_size, self.hid_dim)
        return {
            "h0": tf.zeros(shape, dtype=tf.float32),
            "c0": tf.zeros(shape, dtype=tf.float32)
        }

    def duel(self, v, a, legal_move):
        legal_a = a * legal_move
        q = v + legal_a - tf.reduce_mean(legal_a, axis=2, keepdims=True)
        return q

    @tf.function
    def act(self, s, legal_move, hid):
        assert len(s.shape) == 2, f"should be 2 [batch, dim], get {len(s.shape)}"
        s = tf.expand_dims(s, axis=0)  #Adding sequence dimension
        o = self.net(s)
        for i in range(self.num_lstm_layers):
          o, h, c = self.lstm_layers[i](o, initial_state=[hid["h0"][i], hid["c0"][i]])
        a = self.fc_a(o)
        #a = tf.squeeze(a, axis=0)
        return a, {"h0": h, "c0": c}

    def call(self, s, legal_move, action, hid):
        seq_len, batch_size, num_player, dim = s.shape
        s = tf.reshape(s, (seq_len, batch_size * num_player, dim))
        legal_move = tf.reshape(legal_move, (seq_len, batch_size * num_player, self.out_dim))
        action = tf.reshape(action, (seq_len, batch_size * num_player))
        initial_state=[hid["h0"], hid["c0"]]

        o = self.net(s)
        if not hid:
            o, h, c = self.lstm(o)
        else:
          for i in range(self.num_lstm_layers):
            o, h, c = self.lstm_layers[i](o, initial_state=[hid["h0"][i], hid["c0"][i]])

        a = self.fc_a(o)
        v = self.fc_v(o)
        q = self.duel(v, a, legal_move)

        qa = tf.gather(q, action[:, :, None], batch_dims=2)
        qa = tf.squeeze(qa, axis=2)
        qa = tf.reshape(qa, (seq_len, batch_size, num_player))
        sum_q = tf.reduce_sum(qa, axis=2)

        legal_q = (1 + q - tf.reduce_min(q)) * legal_move
        greedy_action = tf.argmax(legal_q, axis=2)
        greedy_action = tf.reshape(greedy_action, (seq_len, batch_size, num_player))
        return sum_q, greedy_action

class R2D2Agent(tf.keras.Model):
    def __init__(self, multi_step, gamma, eta, in_dim, hid_dim, out_dim):
        super(R2D2Agent, self).__init__()
        self.multi_step = multi_step
        self.gamma = gamma
        self.eta = eta
        self.online_net = R2D2Net(in_dim, hid_dim, out_dim)
        self.target_net = R2D2Net(in_dim, hid_dim, out_dim)

    def call(self, inputs):
        raise NotImplementedError("Use specific methods like `act` or `compute_priority`")

    def get_h0(self, batch_size):
        return self.online_net.get_h0(batch_size)

    def clone(self):
        cloned_agent = R2D2Agent(
            self.multi_step,
            self.gamma,
            self.eta,
            self.online_net.in_dim,
            self.online_net.hid_dim,
            self.online_net.out_dim,
        )
        cloned_agent.online_net.set_weights(self.online_net.get_weights())
        cloned_agent.target_net.set_weights(self.target_net.get_weights())
        return cloned_agent

    def sync_target_with_online(self):
        self.target_net.set_weights(self.online_net.get_weights())

    @tf.function
    def greedy_act(self, s, legal_move, hid):
        q, new_hid = self.online_net.act(s, legal_move, hid)
        legal_q = (1 + q - tf.reduce_min(q)) * legal_move
        greedy_action = tf.argmax(legal_q, axis=1)
        return greedy_action, new_hid

    @tf.function
    def act(self, obs, hid):
        """
        Perform epsilon-greedy action selection.
        obs: Dict with "s" (state), "legal_move", and "eps" (exploration probability)
        """
        s = tf.reshape(obs["s"], [-1, obs["s"].shape[-1]])
        legal_move = tf.reshape(obs["legal_move"], [-1, obs["legal_move"].shape[-1]])
        eps = tf.reshape(obs["eps"], [-1])
        greedy_action, new_hid = self.greedy_act(s, legal_move, hid)

        random_action = tf.squeeze(tf.random.categorical(tf.math.log(legal_move), num_samples=1), axis=1)
        rand = tf.random.uniform(tf.shape(greedy_action), dtype=tf.float32)
        tf.debugging.assert_equal(tf.shape(rand), tf.shape(eps), message="Shapes of rand and eps do not match")
        rand = tf.cast(rand < eps, tf.int32)

        #Combine greedy and random actions based on the random decision
        action = tf.cast(greedy_action, tf.int32) * (1 - rand) + tf.cast(random_action, tf.int32) * rand
        action = tf.reshape(action, [obs["s"].shape[0], -1])
        
        return {"a": action, "greedy_a": tf.reshape(greedy_action, action.shape)}, new_hid

    @tf.function
    def compute_priority(
        self, obs, action, reward, terminal, bootstrap, next_obs, hid, next_hid
    ):
        """
        Compute priority for a batch.
        """
        s = tf.expand_dims(obs["s"], axis=0)  #Addingsequence dimension
        legal_move = tf.expand_dims(obs["legal_move"], axis=0)
        action = tf.expand_dims(action["a"], axis=0)

        online_q = self.online_net(s, legal_move, action, hid)[0]

        #Compute next_q with double Q-learning
        next_s = next_obs["s"]
        next_legal_move = next_obs["legal_move"]
        online_next_a, _ = self.greedy_act(
            tf.reshape(next_s, [-1, next_s.shape[-1]]),
            tf.reshape(next_legal_move, [-1, next_legal_move.shape[-1]]),
            next_hid,
        )
        online_next_a = tf.expand_dims(tf.reshape(online_next_a, [1, -1]), axis=0)
        bootstrap_q = self.target_net(
            tf.expand_dims(next_s, axis=0), tf.expand_dims(next_legal_move, axis=0), online_next_a, next_hid
        )[0]

        target = reward + bootstrap * (self.gamma ** self.multi_step) * bootstrap_q
        priority = tf.abs(target - online_q)
        return priority

    def aggregate_priority(self, priority, seq_len):
        """
        Aggregate priority values.
        """
        mask = tf.cast(
            tf.range(priority.shape[0])[:, None] < seq_len[None, :], tf.float32
        )
        priority *= mask

        p_mean = tf.reduce_sum(priority, axis=0) / tf.cast(seq_len, tf.float32)
        p_max = tf.reduce_max(priority, axis=0)
        agg_priority = self.eta * p_max + (1.0 - self.eta) * p_mean
        return agg_priority

    @tf.function
    def loss(self, batch):
        err = self._err(
            batch["obs"],
            batch["h0"],
            batch["action"],
            batch["reward"],
            batch["terminal"],
            batch["bootstrap"],
            batch["seq_len"],
        )
        loss = tf.keras.losses.Huber()(tf.zeros_like(err), err)
        loss = tf.reduce_sum(loss, axis=0)

        priority = self.aggregate_priority(tf.abs(err), batch["seq_len"])
        return loss, priority

    def _err(self, obs, hid, action, reward, terminal, bootstrap, seq_len):
        #Compute temporal-difference error for the agent.
        max_seq_len = obs["s"].shape[0]

        s = obs["s"]
        legal_move = obs["legal_move"]
        action = action["a"]

        online_qas, target_as = self.online_net(s, legal_move, action, hid)
        with tf.GradientTape() as tape:
            target_qas, _ = self.target_net(s, legal_move, target_as, hid)

        errs = []
        for i in range(max_seq_len):
            target_i = i + self.multi_step
            target_qa = tf.zeros_like(target_qas[i])
            if target_i < max_seq_len:
                target_qa = target_qas[target_i]
            bootstrap_qa = (self.gamma ** self.multi_step) * target_qa
            target = reward[i] + bootstrap[i] * bootstrap_qa
            should_padding = i >= seq_len
            err = (target - online_qas[i]) * (1 - tf.cast(should_padding, tf.float32))
            errs.append(err)
        return tf.stack(errs, axis=0)
