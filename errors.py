"""Exceptions for the accounting layer."""

from __future__ import annotations


class AccountingError(ValueError):
    """Raised when an accounting structure is malformed."""
