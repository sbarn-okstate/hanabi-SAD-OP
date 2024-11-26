import tensorflow as tf
import threading
import random
import numpy as np
import queue

class PrioritizedReplay:
    def __init__(self, capacity, seed, alpha, beta, prefetch):
        self.alpha_ = alpha
        self.beta_ = beta
        self.prefetch_ = prefetch
        self.capacity_ = capacity
        self.storage_ = ConcurrentQueue(int(1.25 * capacity))
        self.numAdd_ = 0
        self.rng_ = random.Random(seed)
        self.sampledIds_ = []
        self.futures_ = []

    def add(self, sample, priority):
        weights = tf.pow(priority, self.alpha_)
        self.storage_.blockAppend(sample, weights)
        self.numAdd_ += priority.shape[0]

    def sample(self, batchsize, device):
        if self.prefetch_ == 0:
            return self.sample_(batchsize, device)

        if self.futures_:
            batch, priority, sampledIds = self.futures_.pop(0).result()
        else:
            batch, priority, sampledIds = self.sample_(batchsize, device)

        while len(self.futures_) < self.prefetch_:
            f = threading.Thread(target=self.sample_, args=(batchsize, device))
            self.futures_.append(f)

        return batch, priority

    def sample_(self, batchsize, device):
        with self.storage_.m_:
            sum_ = 0
            size = self.storage_.safeSize()
            segment = sum_ / batchsize
            dist = tf.random.uniform([batchsize], minval=0.0, maxval=segment, dtype=tf.float32)

            samples = []
            weights = tf.zeros(batchsize, dtype=tf.float32)
            ids = []
            accSum = 0
            nextIdx = 0
            for i in range(batchsize):
                rand = dist[i].numpy() + i * segment
                rand = min(sum_ - 0.1, rand)

                while nextIdx <= size:
                    if accSum >= rand:
                        element = self.storage_.getElementAndMark(nextIdx - 1)
                        samples.append(element)
                        ids.append(nextIdx)
                        weights[i] = accSum
                        break
                    w = self.storage_.getWeight(nextIdx)
                    accSum += w
                    nextIdx += 1

            weights = weights / sum_
            weights = tf.pow(len(weights) * weights, -self.beta_)
            weights /= tf.reduce_max(weights)

            batch = tf.stack(samples)
            return batch, weights, ids

    def updatePriority(self, priority):
        with self.storage_.m_:
            weights = tf.pow(priority, self.alpha_)
            self.storage_.update(self.sampledIds_, weights)
        self.sampledIds_ = []
