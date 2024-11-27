import threading

class ThreadLoop:
    def __init__(self):
        self._terminated = False
        self._paused = False
        self._pause_event = threading.Event()
        self._pause_event.set()  # Initially, not paused.

    def terminate(self):
        self._terminated = True

    def pause(self):
        """Pauses the thread."""
        self._paused = True
        self._pause_event.clear()  # Clear the event to block execution.

    def resume(self):
        """Resumes the thread."""
        self._paused = False
        self._pause_event.set()  # Set the event to allow the thread to continue.

    def wait_until_resume(self):
        """Blocks the thread until it's resumed."""
        self._pause_event.wait()  # Wait until the event is set (resumed).

    def terminated(self):
        """Returns whether the loop is terminated."""
        return self._terminated

    def paused(self):
        """Returns whether the loop is paused."""
        return self._paused

    def main_loop(self):
        """Main loop to be overridden in the subclasses."""
        raise NotImplementedError("Subclasses must implement `main_loop`.")
