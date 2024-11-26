import threading
import time


class ThreadLoop:
    def __init__(self):
        self._running = False
        self._paused = False
        self._terminate = False
        self._lock = threading.Lock()

    def _main_loop(self):
        """The main loop that runs in the thread."""
        while not self._terminate:
            if self._paused:
                # Pause the loop, wait until resumed
                time.sleep(0.1)
                continue

            # Simulate work being done in the loop
            print("ThreadLoop is running...")
            time.sleep(1)  # Simulate time-consuming work

    def start(self):
        """Starts the loop in a separate thread."""
        with self._lock:
            if not self._running:
                self._terminate = False
                self._paused = False
                self._running = True
                self._thread = threading.Thread(target=self._main_loop)
                self._thread.start()
                print("ThreadLoop started.")
            else:
                print("ThreadLoop is already running.")

    def pause(self):
        """Pauses the loop."""
        with self._lock:
            if self._running and not self._paused:
                self._paused = True
                print("ThreadLoop paused.")
            elif not self._running:
                print("ThreadLoop is not running, cannot pause.")
            else:
                print("ThreadLoop is already paused.")

    def resume(self):
        """Resumes the loop."""
        with self._lock:
            if self._running and self._paused:
                self._paused = False
                print("ThreadLoop resumed.")
            elif not self._running:
                print("ThreadLoop is not running, cannot resume.")
            else:
                print("ThreadLoop is already running.")

    def terminate(self):
        """Terminates the loop and the thread."""
        with self._lock:
            if self._running:
                self._terminate = True
                self._thread.join()  # Wait for the thread to finish
                self._running = False
                print("ThreadLoop terminated.")
            else:
                print("ThreadLoop is not running, cannot terminate.")

    def is_running(self):
        """Returns whether the loop is running."""
        return self._running


# Example usage:

if __name__ == "__main__":
    loop = ThreadLoop()
    loop.start()  # Start the loop
    time.sleep(3)  # Let the loop run for a few seconds
    loop.pause()  # Pause the loop
    time.sleep(2)  # Let it stay paused
    loop.resume()  # Resume the loop
    time.sleep(3)  # Let the loop run again
    loop.terminate()  # Terminate the loop
