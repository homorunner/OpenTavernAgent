"""Turn native unwind exceptions into errors multiprocessing can return to its parent."""


def worker_call(function, context, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except BaseException as error:
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        # PyO3 PanicException inherits BaseException, which Pool workers do not catch.
        # Let the parent abort the batch; never silently discard a failed game.
        raise RuntimeError(f"{context}: {type(error).__name__}: {error}") from error
