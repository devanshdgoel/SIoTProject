"""Let only one collector run at a time, and say what the other one is.

Running `main.py collect` twice used to half-work, which is worse than failing.
The second process would start its own copy of every collector thread and its own
analysis loop against the same database. Three things then went wrong, none of
them loudly:

  - Both tried to bind the ingest port. The loser raised inside its thread, that
    thread died, and the rest of the process carried on looking healthy - so a
    node posting readings reached whichever server won, and the other process
    quietly never received anything.
  - Both ran `learn()` every five minutes. learn() clears a room's parameters and
    rewrites them, so two passes interleaving leaves room_params holding whichever
    subset the second DELETE did not catch. No exception, no log, just a room that
    has mysteriously lost its source term.
  - Both called the same rate-limited APIs on the same schedule, for nothing.

The lock is a loopback TCP socket rather than a lock file, because a lock file has
to answer "is the process that wrote this still alive?", and on Windows there is no
clean way to ask. A bound socket needs no such question: the operating system
releases it the moment the process ends, however it ends - Ctrl-C, a crash, a
closed terminal, a pulled plug. There is no stale state to clean up and no way to
be locked out by a process that no longer exists.

The port is derived from the database path, so two different databases (a test's
tmp_path, say) do not block each other, while two collectors on the SAME database -
which is the actual hazard - always collide.
"""

import os
import socket
import zlib

# A high, unprivileged range well clear of the ingest server and the dashboard.
_LOCK_PORT_BASE = 38000
_LOCK_PORT_SPAN = 2000

_held = []          # keeps the socket alive for the life of the process


def lock_port(key):
    """The port that stands for `key`. Stable across runs, distinct across keys."""
    digest = zlib.crc32(str(key).encode("utf-8"))
    return _LOCK_PORT_BASE + digest % _LOCK_PORT_SPAN


def acquire(key):
    """Take the lock for `key`, or return None if someone else holds it.

    SO_REUSEADDR is deliberately NOT set. On Windows it would let a second process
    bind the same port and both would think they had the lock, which is the exact
    failure this exists to prevent.
    """
    port = lock_port(key)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", port))
        s.listen(1)
    except OSError:
        s.close()
        return None
    _held.append(s)
    return s


def release(sock):
    """Give the lock up early. Not required - the OS does it on exit."""
    if sock is None:
        return
    try:
        _held.remove(sock)
    except ValueError:
        pass
    try:
        sock.close()
    except OSError:
        pass


def describe(key):
    """A line about who holds the lock, for a message to a person.

    The holder's pid is recorded beside the database when it takes the lock. It is
    a courtesy, not the lock itself: if the file is missing or stale the socket is
    still authoritative, so this never decides anything, it only explains.
    """
    note = _note_path(key)
    try:
        with open(note, encoding="utf-8") as fh:
            return fh.read().strip()
    except OSError:
        return "another process on this machine"


def _note_path(key):
    return str(key) + ".collector.pid"


def record(key, what):
    """Leave a note saying who we are, for the next process's error message."""
    try:
        with open(_note_path(key), "w", encoding="utf-8") as fh:
            fh.write(f"pid {os.getpid()}, {what}")
    except OSError:
        pass                      # a note we cannot write is not worth failing over
