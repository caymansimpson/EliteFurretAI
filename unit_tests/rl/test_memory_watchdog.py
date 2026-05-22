# -*- coding: utf-8 -*-
"""Tests for the memory watchdog in rl_utils.

The watchdog runs in a daemon thread, samples combined Pss across the
trainer + its process tree, and sets the shutdown event when usage
crosses a configurable threshold. We test the contract — not psutil's
internals — by stubbing the Pss sampler.
"""

import threading
from unittest.mock import patch


def test_watchdog_disabled_when_threshold_none():
    """threshold_gb=None returns None and does not start a thread."""
    from elitefurretai.rl.rl_utils import start_memory_watchdog

    event = threading.Event()
    thread = start_memory_watchdog(event, threshold_gb=None)
    assert thread is None
    assert not event.is_set()


def test_watchdog_disabled_when_threshold_zero():
    """threshold_gb=0 disables the watchdog (treated the same as None)."""
    from elitefurretai.rl.rl_utils import start_memory_watchdog

    event = threading.Event()
    thread = start_memory_watchdog(event, threshold_gb=0)
    assert thread is None
    assert not event.is_set()


def test_watchdog_fires_event_when_rss_exceeds_threshold():
    """When combined Pss goes above the threshold, the event is set.

    We mock the Pss sampler so the first sample is already above the
    threshold; the daemon thread should set the event and exit promptly.
    """
    from elitefurretai.rl import rl_utils

    event = threading.Event()
    # Threshold is 5 GB; report 6 GB combined. Bytes-not-GB return value.
    over_threshold_bytes = 6 * 1024**3
    sample = (over_threshold_bytes, {"trainer": over_threshold_bytes})

    with patch.object(rl_utils, "sum_process_tree_pss_bytes", return_value=sample):
        thread = rl_utils.start_memory_watchdog(
            event, threshold_gb=5.0, poll_interval_s=0.01
        )
        assert thread is not None
        # Watchdog should set the event within a couple of poll intervals.
        assert event.wait(timeout=2.0), "watchdog did not set the event"
        thread.join(timeout=1.0)
        assert not thread.is_alive()


def test_watchdog_does_not_fire_when_below_threshold_and_exits_on_shutdown():
    """Below-threshold samples leave the event clear; setting the event
    externally must wake the watchdog out of its poll-wait promptly."""
    from elitefurretai.rl import rl_utils

    event = threading.Event()
    under_threshold_bytes = 1 * 1024**3  # 1 GB, well below 5 GB threshold
    sample = (under_threshold_bytes, {"trainer": under_threshold_bytes})

    with patch.object(rl_utils, "sum_process_tree_pss_bytes", return_value=sample):
        thread = rl_utils.start_memory_watchdog(
            event, threshold_gb=5.0, poll_interval_s=0.05
        )
        assert thread is not None
        # Give the watchdog a couple of polls — it must not trip on its own.
        assert not event.wait(timeout=0.2)
        # Now request shutdown externally; watchdog should exit on its
        # Event.wait, not after a full poll_interval_s of dead time.
        event.set()
        thread.join(timeout=1.0)
        assert not thread.is_alive()
