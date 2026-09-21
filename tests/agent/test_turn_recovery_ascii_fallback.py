"""Tests for turn-recovery ASCII fallback fix (#117802).

Verifies that UnicodeEncodeError from a non-ASCII API key does NOT trigger
the destructive message-stripping path, and that the actual system encoding
is checked instead of inferring from error text.
"""
import sys
import locale
from unittest.mock import MagicMock, patch

import pytest

from agent.turn_recovery import _recover_unicode_encode_error


def _make_agent(api_key="sk-test"):
    """Create a minimal mock agent for turn-recovery tests."""
    agent = MagicMock()
    agent.api_key = api_key
    agent._client_kwargs = {"api_key": api_key}
    agent.client = MagicMock()
    agent.client.api_key = api_key
    agent._unicode_sanitization_passes = 0
    agent._force_ascii_payload = False
    agent._cached_system_prompt = None
    agent.ephemeral_system_prompt = None
    agent.prefill_messages = None
    agent.tools = None
    agent._buffer_vprint = MagicMock()
    agent._vlines = MagicMock()
    return agent


class TestAsciiCodecDetection:
    """The ASCII-locale detection must not infer from error text."""

    def test_unicode_encode_error_with_ascii_substring_does_not_trigger_strip(self):
        """A UnicodeEncodeError about a non-ASCII API key contains 'ascii'
        but must NOT be treated as an ASCII-only locale."""
        agent = _make_agent(api_key="sk-test\u00abredacted")
        messages = [{"role": "user", "content": "Hello \u4e16\u754c"}]
        api_messages = [{"role": "user", "content": "Hello \u4e16\u754c"}]
        api_kwargs = {"model": "test", "messages": api_messages}

        # Simulate the exact error from httpx when encoding a non-ASCII header
        error = UnicodeEncodeError(
            "ascii", "sk-test\u00abredacted", 7, 8,
            "ordinal not in range(128)"
        )

        recovered, _ = _recover_unicode_encode_error(
            agent, error, messages, api_messages, api_kwargs, None
        )

        # Should recover (retry after sanitizing the API key)
        assert recovered is True
        # Messages must be untouched -- this is the critical fix
        assert messages[0]["content"] == "Hello \u4e16\u754c"
        assert api_messages[0]["content"] == "Hello \u4e16\u754c"
        # API key should be sanitized
        assert agent.api_key == "sk-testredacted"

    def test_actual_ascii_locale_triggers_credential_sanitization(self):
        """When the system IS actually ASCII-only, sanitize the API key
        but never the conversation content."""
        agent = _make_agent(api_key="sk-test\u00abredacted")
        messages = [{"role": "user", "content": "Hello \u4e16\u754c"}]
        api_messages = [{"role": "user", "content": "Hello \u4e16\u754c"}]
        api_kwargs = {"model": "test", "messages": api_messages}

        error = UnicodeEncodeError(
            "ascii", "sk-test\u00abredacted", 7, 8,
            "ordinal not in range(128)"
        )

        with patch.object(sys, 'getdefaultencoding', return_value='ascii'):
            with patch.object(locale, 'getpreferredencoding', return_value='ascii'):
                recovered, _ = _recover_unicode_encode_error(
                    agent, error, messages, api_messages, api_kwargs, None
                )

        # Should recover (ASCII locale detected)
        assert recovered is True
        # API key sanitized
        assert agent.api_key == "sk-testredacted"
        # Messages must NOT be stripped -- this is the critical fix
        assert messages[0]["content"] == "Hello \u4e16\u754c"
        assert api_messages[0]["content"] == "Hello \u4e16\u754c"

    def test_utf8_locale_still_retries_after_key_sanitization(self):
        """Under UTF-8 (the normal case), the ASCII path must never trigger,
        but we still retry after sanitizing a non-ASCII API key."""
        agent = _make_agent(api_key="sk-test\u00abredacted")
        messages = [{"role": "user", "content": "Hello \u4e16\u754c"}]
        api_messages = [{"role": "user", "content": "Hello \u4e16\u754c"}]
        api_kwargs = {"model": "test", "messages": api_messages}

        error = UnicodeEncodeError(
            "ascii", "sk-test\u00abredacted", 7, 8,
            "ordinal not in range(128)"
        )

        # Default UTF-8 locale
        recovered, _ = _recover_unicode_encode_error(
            agent, error, messages, api_messages, api_kwargs, None
        )

        # Should recover (retry after sanitizing the API key)
        assert recovered is True
        # Messages must NOT be stripped
        assert messages[0]["content"] == "Hello \u4e16\u754c"
        # API key should be sanitized
        assert agent.api_key == "sk-testredacted"

    def test_ascii_locale_with_clean_key_enables_force_ascii_payload(self):
        """Under ASCII locale with a clean API key, _force_ascii_payload is set."""
        agent = _make_agent(api_key="sk-test123")
        messages = [{"role": "user", "content": "Hello \u4e16\u754c"}]
        api_messages = [{"role": "user", "content": "Hello \u4e16\u754c"}]
        api_kwargs = {"model": "test", "messages": api_messages}

        error = UnicodeEncodeError(
            "ascii", "sk-test123", 7, 8,
            "ordinal not in range(128)"
        )

        with patch.object(sys, 'getdefaultencoding', return_value='ascii'):
            with patch.object(locale, 'getpreferredencoding', return_value='ascii'):
                recovered, _ = _recover_unicode_encode_error(
                    agent, error, messages, api_messages, api_kwargs, None
                )

        assert recovered is True
        # _force_ascii_payload should be set for the retry path
        assert agent._force_ascii_payload is True
        # Messages must NOT be stripped
        assert messages[0]["content"] == "Hello \u4e16\u754c"


class TestSurrogatePathUnaffected:
    """Surrogate handling must remain unchanged."""

    def test_surrogate_error_still_triggers_recovery(self):
        """Surrogate errors should still trigger the surrogate recovery path."""
        agent = _make_agent()
        messages = [{"role": "user", "content": "Hello \ud800World"}]
        api_messages = [{"role": "user", "content": "Hello \ud800World"}]
        api_kwargs = {"model": "test", "messages": api_messages}

        error = UnicodeEncodeError(
            "utf-8", "Hello \ud800World", 6, 7,
            "surrogates not allowed"
        )

        recovered, _ = _recover_unicode_encode_error(
            agent, error, messages, api_messages, api_kwargs, None
        )

        assert recovered is True
        # Surrogate should be stripped
        assert "\ud800" not in messages[0]["content"]
        assert "\ud800" not in api_messages[0]["content"]


class TestApiKeySanitization:
    """API key sanitization must still work for non-ASCII keys."""

    def test_non_ascii_api_key_is_sanitized(self):
        """Non-ASCII characters in API keys should be stripped."""
        agent = _make_agent(api_key="sk-test\u028bbad")
        messages = [{"role": "user", "content": "Hello"}]
        api_messages = [{"role": "user", "content": "Hello"}]
        api_kwargs = {"model": "test", "messages": api_messages}

        error = UnicodeEncodeError(
            "ascii", "sk-test\u028bbad", 7, 8,
            "ordinal not in range(128)"
        )

        with patch.object(sys, 'getdefaultencoding', return_value='ascii'):
            with patch.object(locale, 'getpreferredencoding', return_value='ascii'):
                recovered, _ = _recover_unicode_encode_error(
                    agent, error, messages, api_messages, api_kwargs, None
                )

        assert recovered is True
        assert agent.api_key == "sk-testbad"
        assert agent._client_kwargs["api_key"] == "sk-testbad"
        assert agent.client.api_key == "sk-testbad"

    def test_ascii_only_api_key_unchanged(self):
        """An ASCII-only API key should not be modified."""
        agent = _make_agent(api_key="sk-test123")
        messages = [{"role": "user", "content": "Hello"}]
        api_messages = [{"role": "user", "content": "Hello"}]
        api_kwargs = {"model": "test", "messages": api_messages}

        error = UnicodeEncodeError(
            "ascii", "sk-test123", 7, 8,
            "ordinal not in range(128)"
        )

        with patch.object(sys, 'getdefaultencoding', return_value='ascii'):
            with patch.object(locale, 'getpreferredencoding', return_value='ascii'):
                recovered, _ = _recover_unicode_encode_error(
                    agent, error, messages, api_messages, api_kwargs, None
                )

        assert recovered is True
        assert agent.api_key == "sk-test123"
