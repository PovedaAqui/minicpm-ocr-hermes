"""Error taxonomy: status mapping, retry hints, and payload shape."""

from __future__ import annotations

from minicpm_ocr.core import errors


def test_to_dict_includes_status_and_detail():
    exc = errors.AuthError("bad key", status=401, detail='{"error": "bad"}')
    assert exc.to_dict() == {
        "code": "auth",
        "message": "bad key",
        "retryable": False,
        "status": 401,
        "detail": '{"error": "bad"}',
    }


def test_retryable_override():
    exc = errors.OcrError("transient", retryable=True)
    assert exc.to_dict()["retryable"] is True


def test_status_mapping():
    assert isinstance(errors.from_status(401, b""), errors.AuthError)
    assert isinstance(errors.from_status(403, b""), errors.AuthError)
    assert isinstance(errors.from_status(413, b""), errors.PayloadTooLarge)
    assert isinstance(errors.from_status(400, b""), errors.RequestRejected)
    assert isinstance(errors.from_status(418, b""), errors.RequestRejected)
    assert isinstance(errors.from_status(500, b""), errors.BackendUnavailable)
    assert isinstance(errors.from_status(503, b""), errors.BackendUnavailable)


def test_rate_limit_keeps_retry_after():
    exc = errors.from_status(429, b'{"error":{"message":"slow down"}}', headers={"Retry-After": "7"})
    assert isinstance(exc, errors.RateLimited)
    assert exc.retry_after == 7.0
    assert exc.message == "slow down"
    assert exc.retryable is True


def test_invalid_retry_after_is_ignored():
    exc = errors.from_status(429, b"", headers={"Retry-After": "later"})
    assert isinstance(exc, errors.RateLimited)
    assert exc.retry_after is None


def test_message_from_json_and_plain_body():
    assert errors._message_from_body('{"error": {"message": "invalid api key"}}') == "invalid api key"
    assert errors._message_from_body('{"message": "nope"}') == "nope"
    assert errors._message_from_body("first line\nsecond") == "first line"
    assert errors._message_from_body("") is None


def test_non_json_body_keeps_snippet():
    exc = errors.from_status(502, b"upstream is down")
    assert isinstance(exc, errors.BackendUnavailable)
    assert exc.detail == "upstream is down"
    assert exc.message == "upstream is down"
