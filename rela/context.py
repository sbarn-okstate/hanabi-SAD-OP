import threading
import time
from typing import List, Optional
from rela.thread_loop import ThreadLoop

class Context:
    """
    Context manages and coordinates multiple threads running `ThreadLoop` tasks.
    """

    def __init__(self):
        self._started = False
        self._num_terminated_thread = 0
        self._loops: List[ThreadLoop] = []
        self._threads: List[threading.Thread] = []

    def pushThreadLoop(self, env: ThreadLoop) -> int:
        """
        Add a new `ThreadLoop` to the context.

        Args:
            env (ThreadLoop): A ThreadLoop instance.

        Returns:
            int: Index of the added loop.
        """
        if self._started:
            raise Exception("Cannot add threads after starting")
        self._loops.append(env)
        return len(self._loops)

    def start(self):
        """
        Start all threads in the context.
        """
        for i, loop in enumerate(self._loops):
            thread = threading.Thread(target=loop.main_loop)
            self._threads.append(thread)
            thread.start()
        self._started = True

    def pause(self):
        """
        Pause all running threads.
        """
        for loop in self._loops:
            loop.pause()

    def resume(self):
        """
        Resume all paused threads.
        """
        for loop in self._loops:
            loop.resume()

    def terminate(self):
        """
        Terminate all threads.
        """
        for loop in self._loops:
            loop.terminate()

    def terminated(self) -> bool:
        """
        Check if all threads have been terminated.

        Returns:
            bool: True if all threads are terminated, False otherwise.
        """
        return self._num_terminated_thread == len(self._loops)

    def join(self):
        """
        Wait for all threads to finish.
        """
        for thread in self._threads:
            thread.join()