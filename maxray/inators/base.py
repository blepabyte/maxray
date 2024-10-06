from maxray.nodes import NodeContext
from maxray.inators.core import S, Ray
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
        S.display.update_context(ray.ctx)

        while True:
            try:
                self.xray(x, ray)
                x = self.maxray(x, ray)
                break
            except Break:
                S.display.update_status("[violet]PAUSED")
                self.wait_and_reload()
            except Exception as e:
                # Capture and show traceback
                S.display.update_status("[violet]PAUSED")
                S.display.render_traceback(e, e.__traceback__)
                self.wait_and_reload()

        match x:
            case RunCompleted() | RunAborted() | RunErrored():
                self.update_display_state(x)
            case _:
                S.display.update_status("[yellow]Running...")
                S.display.render_maybe()
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
            S.display.update_status("[violet]PAUSED")
            S.display.render_traceback(e, e.__traceback__)

    @contextmanager
    def enter_session(self, xi: ExecInfo):
        try:
            yield
        finally:
            S.display.live.stop()

    def update_display_state(self, state: RunCompleted | RunAborted | RunErrored):
        match state:
            case RunCompleted():
                S.display.update_status("[green]Completed")
                S.display.render()
            case RunAborted(exception=exception):
                S.display.update_status(f"[cyan]Aborted ({type(exception).__name__})")
                S.display.render()
            case RunErrored(exception=exception, traceback=traceback):
                S.display.update_status("[red]Errored")
                S.display.render_traceback(exception, traceback)
