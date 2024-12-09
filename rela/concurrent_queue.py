"""
Code based on C++ PyTorch code from https://github.com/codeaudit/hanabi_SAD/blob/master/rela/prioritized_replay.h
"""

import tensorflow as tf
import threading

class ConcurrentQueue:
    def __init__(self, capacity):
        self.capacity = capacity
        self.head_ = 0
        self.tail_ = 0
        self.size_ = 0
        self.safeTail_ = 0
        self.safeSize_ = 0
        self.sum_ = 0
        self.evicted_ = [False] * capacity
        self.elements_ = [None] * capacity
        self.weights_ = [0.0] * capacity
        
        self.m_ = threading.Lock()
        self.cvSize_ = threading.Condition(self.m_)
        self.cvTail_ = threading.Condition(self.m_)

    def safeSize(self):
        with self.m_:
            return self.safeSize_

    def size(self):
        with self.m_:
            return self.size_

    def blockAppend(self, block, weights):
        blockSize = len(block)
        
        with self.m_:
            while self.size_ + blockSize > self.capacity:
                self.cvSize_.wait()
            
            start = self.tail_
            end = (self.tail_ + blockSize) % self.capacity
            self.tail_ = end
            self.size_ += blockSize
            self.checkSize(self.head_, self.tail_, self.size_)

        sum_ = 0
        for i in range(blockSize):
            idx = (start + i) % self.capacity
            self.elements_[idx] = block[i]
            self.weights_[idx] = weights[i]
            sum_ += weights[i]

        with self.m_:
            while self.safeTail_ != start:
                self.cvTail_.wait()
            self.safeTail_ = end
            self.safeSize_ += blockSize
            self.sum_ += sum_
            self.checkSize(self.head_, self.safeTail_, self.safeSize_)

        self.cvTail_.notify_all()

    def blockPop(self, blockSize):
        diff = 0
        head = self.head_
        for i in range(blockSize):
            diff -= self.weights_[head]
            self.evicted_[head] = True
            head = (head + 1) % self.capacity
        
        with self.m_:
            self.sum_ += diff
            self.head_ = head
            self.safeSize_ -= blockSize
            self.size_ -= blockSize
            self.checkSize(self.head_, self.safeTail_, self.safeSize_)

        self.cvSize_.notify_all()

    def update(self, ids, weights):
        diff = 0
        for i, id in enumerate(ids):
            if self.evicted_[id]:
                continue
            diff += (weights[i] - self.weights_[id])
            self.weights_[id] = weights[i]

        with self.m_:
            self.sum_ += diff

    def getElementAndMark(self, idx):
        id = (self.head_ + idx) % self.capacity
        self.evicted_[id] = False
        return self.elements_[id]

    def getWeight(self, idx):
        id = (self.head_ + idx) % self.capacity
        return self.weights_[id]

    def checkSize(self, head, tail, size):
        if size == 0:
            assert tail == head
        elif tail > head:
            assert tail - head == size
        else:
            assert tail + self.capacity - head == size
