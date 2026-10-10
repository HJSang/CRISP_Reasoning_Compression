"""The transport must bind complete Boolean results to the exact response tape."""

import hashlib
import json

import pytest

from tools.opsd.code_sandbox import _merge_results, _queued_grade


def _result(draw=0):
    return dict(id="example", draw=draw, correct=True, parse_failure=False, grader_timeout=False, test_failure=False)


@pytest.mark.parametrize("results", [[], [_result(), _result()], [_result(draw=1)], [_result() | {"correct": 1}]])
def test_missing_duplicate_wrong_draw_or_non_boolean_results_are_rejected(results):
    with pytest.raises(ValueError):
        _merge_results([dict(id="example", draw=0)], results)


@pytest.mark.parametrize("wrong_identity", [False, True])
def test_queue_binds_results_to_the_complete_request(tmp_path, monkeypatch, wrong_identity):
    def complete(_):
        request = next((tmp_path / "pending").glob("*.json"))
        digest = hashlib.sha256(request.read_bytes()).hexdigest()
        assert request.stem == digest
        result = dict(request_sha256="wrong" if wrong_identity else digest, results=[_result()])
        (tmp_path / "done" / request.name).write_text(json.dumps(result))

    monkeypatch.setattr("tools.opsd.code_sandbox.time.sleep", complete)
    rows = [dict(id="example", draw=0, text="candidate")]
    if wrong_identity:
        with pytest.raises(ValueError, match="different response tape"):
            _queued_grade(rows, {"example": {}}, tmp_path)
    else:
        _queued_grade(rows, {"example": {}}, tmp_path)
        assert rows[0]["correct"] is True
