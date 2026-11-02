import pickle

import pytest
from tavernagent.worker import worker_call


def test_native_base_exception_becomes_picklable_worker_error():
    class NativePanic(BaseException):
        pass

    def fail():
        raise NativePanic("index out of bounds")

    with pytest.raises(
        RuntimeError, match="seed=1390079: NativePanic: index out of bounds"
    ) as caught:
        worker_call(fail, "self-play seed=1390079")
    restored = pickle.loads(pickle.dumps(caught.value))
    assert str(restored) == str(caught.value)


@pytest.mark.parametrize("exception", [KeyboardInterrupt, SystemExit])
def test_worker_control_exceptions_still_propagate(exception):
    def fail():
        raise exception()

    with pytest.raises(exception):
        worker_call(fail, "interrupted")


def test_successful_worker_result_is_unchanged():
    result = object()
    assert worker_call(lambda: result, "ok") is result


def test_invalid_checkpoint_fails_before_evaluation_workers_start(tmp_path, monkeypatch):
    import numpy as np

    from tavernagent import eval as evaluation

    model = tmp_path / "bad.npz"
    np.savez(model, invalid=np.zeros(1))

    def forbidden(*args, **kwargs):
        pytest.fail("invalid checkpoint reached Pool creation")

    monkeypatch.setattr(evaluation.mp, "get_context", forbidden)
    with pytest.raises(ValueError, match="unsupported checkpoint"):
        evaluation.evaluate({"network": model}, games=2, workers=2)
