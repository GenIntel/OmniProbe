import sys

from collections.abc import Iterable

from tqdm import tqdm

_INTERACTIVE = sys.stderr.isatty()


class _Progress(tqdm):
    # On a non-interactive stream (e.g. SLURM), set_postfix/set_description
    # default to refresh=True, which redraws on every call and bypasses
    # mininterval -- flooding the log with one line per step. Defer those to
    # the iteration-driven refresh that mininterval throttles; stay live on a
    # terminal.
    def set_postfix(self, *args, **kwargs):
        kwargs.setdefault("refresh", _INTERACTIVE)
        super().set_postfix(*args, **kwargs)

    def set_description(self, *args, **kwargs):
        kwargs.setdefault("refresh", _INTERACTIVE)
        super().set_description(*args, **kwargs)


def progress(iterable: Iterable, **kwargs):
    kwargs.setdefault("dynamic_ncols", True)
    # Fast live refresh on a terminal; sparse periodic lines when stderr is
    # captured to a file (e.g. SLURM), so the .err log isn't flooded.
    kwargs.setdefault("mininterval", 0.1 if _INTERACTIVE else 30.0)
    return _Progress(iterable, **kwargs)
