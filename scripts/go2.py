"""Run a command on the Go2 over SSH.

The robot uses password auth and its login shell asks
``ros:foxy(1) noetic(2) ?`` before giving a prompt, so a plain
``ssh host 'cmd'`` can hang waiting for that answer. This wraps paramiko and
answers it, so commands come back cleanly.

    python scripts/go2.py "uname -a"
    python scripts/go2.py --file localscript.sh        # run a local script remotely
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import os

# Robot addresses and login come from outside the repo -- environment first,
# then ~/.fr_robots.env. Nothing here carries a real address or password, so
# the file is safe to publish; see .fr_robots.env.example for the keys.
#
# The robot has two addresses, wifi (wlan0) and ethernet (eth0), and which one
# is reachable changes with how the laptop is attached -- hence a lookup rather
# than one value.
ENV_FILE = Path.home() / ".fr_robots.env"


def _settings() -> dict:
    """KEY=VALUE pairs from ~/.fr_robots.env, overridden by the environment."""
    values = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            values[key.strip()] = val.strip().strip("'\"")
    values.update({k: v for k, v in os.environ.items() if k.startswith("FR_")})
    return values


_S = _settings()
HOSTS = {"wifi": _S.get("FR_HOST_GO2", ""),
         "ethernet": _S.get("FR_HOST_GO2_ETH", ""),
         "g1": _S.get("FR_HOST_G1", "")}
HOST = os.environ.get("GO2_HOST") or HOSTS["wifi"]
# The vendor ships every robot with this account name; it is documentation,
# not a secret. The password is a secret and has no default on purpose.
USER = _S.get("FR_USER", "unitree")
PASSWORD = _S.get("FR_PASSWORD", "")
ROS_CHOICE = "1"  # answer to the foxy/noetic prompt


def _require_credentials(host: str) -> None:
    """Fail with instructions rather than a paramiko authentication error."""
    if PASSWORD and host:
        return
    missing = []
    if not host:
        missing.append("robot address (FR_HOST_G1 / FR_HOST_GO2)")
    if not PASSWORD:
        missing.append("password (FR_PASSWORD)")
    raise SystemExit(
        "Missing %s.\n"
        "These are deliberately not in the repo. Create %s from\n"
        ".fr_robots.env.example, or export them:\n"
        "    FR_HOST_G1=...  FR_USER=...  FR_PASSWORD=...\n"
        "Only the remote modes (--g1 / --wifi / --ethernet) need this; running\n"
        "on the robot itself with --local does not."
        % (" and ".join(missing), ENV_FILE)
    )


def run(command: str, timeout: int = 120, host: str = HOST) -> tuple[int, str, str]:
    import paramiko

    _require_credentials(host)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, username=USER, password=PASSWORD, timeout=15,
                   banner_timeout=30, auth_timeout=30)
    try:
        # A non-login, non-interactive shell skips the ROS prompt entirely.
        stdin, stdout, stderr = client.exec_command(command, timeout=timeout,
                                                    get_pty=False)
        # Answer the ros:foxy(1)/noetic(2) prompt if it appears. A short command
        # can finish and close the channel before we get here, which is fine --
        # there was no prompt to answer.
        try:
            stdin.write(ROS_CHOICE + "\n")
            stdin.flush()
        except OSError:
            pass
        out = stdout.read().decode("utf-8", "replace")
        err = stderr.read().decode("utf-8", "replace")
        return stdout.channel.recv_exit_status(), out, err
    finally:
        client.close()


def launch(command: str, host: str = HOST) -> None:
    """Start a long-running command and return immediately.

    A backgrounded process inherits the SSH channel's stdout, so reading it
    blocks until that process exits -- which for a server is never. This fires
    the command with every stream detached and closes without reading.
    """
    import paramiko

    _require_credentials(host)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, username=USER, password=PASSWORD, timeout=15)
    try:
        client.exec_command(command, timeout=10)
        import time

        time.sleep(2)  # let the remote shell actually spawn it before we close
    finally:
        client.close()


def put(local: Path, remote: str, host: str = HOST) -> None:
    """Copy a file to the robot atomically, and verify it arrived intact.

    Writing straight to the destination means anyone reading the file mid-copy
    sees a truncated one -- which produced a `BadZipFile` when the robot loaded
    a half-written embeddings.npz. So upload to a temporary name and rename:
    rename within a filesystem is atomic, so a reader sees either the old file
    or the complete new one, never a partial.

    The size is checked afterwards because sftp.put can return without having
    written everything if the connection drops.
    """
    import paramiko

    _require_credentials(host)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, username=USER, password=PASSWORD, timeout=15)
    try:
        sftp = client.open_sftp()
        tmp = f"{remote}.part"
        try:
            sftp.put(str(local), tmp)
            expected = local.stat().st_size
            actual = sftp.stat(tmp).st_size
            if actual != expected:
                raise IOError(
                    f"{local.name}: transferred {actual} of {expected} bytes"
                )
            sftp.posix_rename(tmp, remote)
        except Exception:
            try:
                sftp.remove(tmp)
            except Exception:
                pass
            raise
        finally:
            sftp.close()
    finally:
        client.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", nargs="?", default="echo connected")
    ap.add_argument("--file", type=Path, help="run a local shell script on the robot")
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--host", default=HOST,
                    help=f"robot address, or a name from {list(HOSTS)}")
    args = ap.parse_args()

    host = HOSTS.get(args.host, args.host)
    cmd = args.file.read_text() if args.file else args.command
    try:
        code, out, err = run(cmd, timeout=args.timeout, host=host)
    except Exception as exc:
        print(f"SSH failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if out:
        print(out, end="")
    if err.strip():
        print("--- stderr ---\n" + err, end="", file=sys.stderr)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
