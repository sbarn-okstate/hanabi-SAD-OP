"""
Code based on C++ PyTorch code from https://github.com/codeaudit/hanabi_SAD/blob/master/rela/thread_loop.h
"""

import threading

class ThreadLoop:
    def __init__(self):
        self._terminated = False
        self._paused = False
        self._pause_event = threading.Event()
        self._pause_event.set()  #Inits to not paused

    def terminate(self):
        self._terminated = True

    def pause(self):
        #Pauses the thread
        self._paused = True
        self._pause_event.clear()

    def resume(self):
        #Resumes the thread
        self._paused = False
        self._pause_event.set()

    def wait_until_resume(self):
        #Blocks the thread until it's resumed
        self._pause_event.wait()

    def terminated(self):
        #Returns whether the loop is terminated
        return self._terminated

    def paused(self):
        #Returns whether the loop is paused
        return self._paused

    def main_loop(self):
        #Main loop to be overridden in the subclasses
        raise NotImplementedError("Subclasses must implement `main_loop`.")
