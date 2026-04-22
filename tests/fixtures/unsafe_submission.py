"""Unsafe submission fixture — fails the security scanner.

Triggers multiple critical findings: blocked import, blocked attribute,
blocked builtin, and an unknown import.
"""

import os
import subprocess                # blocked_import
import mysterious_unknown_lib    # unknown_import

from polybench import FLAT, Model


class ModelSubmission(Model):
    def on_tick(self, tick):
        # blocked_attr: os.system
        os.system("echo pwned")
        # blocked_call: exec
        exec("print('pwned')")
        return FLAT
