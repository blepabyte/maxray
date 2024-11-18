from maxray.nodes import NodeContext
from maxray.inators.core import R, S, Ray
from maxray.inators.display import SetVisible, SetStatus, DumpTraceback
from maxray.runner import ExecInfo, RunCompleted, RunErrored, RunAborted, Break

from contextlib import contextmanager


class BaseInator:
    def __init__(self):
        """
        Args:
        - name (str): Descriptive name for the program. Only used for visualization and logging.
        """
        self._name = type(self).__name__

    def __repr__(self):
        return f"{self._name}()"

    def __call__(self, x, ray: Ray):
        while True:
            try:
                self.xray(x, ray)
                x = self.maxray(x, ray)
                break
            except Break:
                R.DisplayChannel.push(SetStatus("PAUSED", "[violet]"))
                self.wait_and_reload()
            except Exception as e:
                # Capture and show traceback
                R.DisplayChannel.push(SetStatus("PAUSED", "[violet]"))
                R.DisplayChannel.push(DumpTraceback(e, e.__traceback__))
                self.wait_and_reload()

        return x

    def xray(self, x, ray: Ray):
        """
        Override to implement equivalent of @xray
        """
        pass

    def maxray(self, x, ray: Ray):
        """
        Override to implement equivalent of @maxray
        """
        return x

    def runner(self):
        raise NotImplementedError()

    def wait_and_reload(self):
        # Patched in at runtime
        raise NotImplementedError()

    @contextmanager
    def _handle_reload(self):
        """
        Provides control over what happens if an error is encountered while reloading itself.
        """
        try:
            yield
        except Exception as e:
            # Capture and show traceback
            R.DisplayChannel.push(SetStatus("PAUSED", "[violet]"))
            R.DisplayChannel.push(DumpTraceback(e, e.__traceback__))

    @contextmanager
    def enter_session(self, xi: ExecInfo):
        try:
            yield
        finally:
            pass

    @contextmanager
    def hide_display(self):
        try:
            R.DisplayChannel.push(SetVisible(False))
            yield
        finally:
            R.DisplayChannel.push(SetVisible(True))
