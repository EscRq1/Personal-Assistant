"""Offline installer command. This script runs only inside DockerExecutor."""
from __future__ import annotations

import shlex


def install_command(packages: list[str]) -> str:
    script = r'''
import os, pathlib, subprocess, sys, venv
root = pathlib.Path('/workspace/.venv')
if root.is_symlink() or (root.exists() and not root.is_dir()):
    sys.exit('Invalid .venv; remove /workspace/.venv and retry sandbox_install.')
if not root.exists():
    # Keep image-provided scientific packages visible; installs are local to
    # this venv and cannot modify the read-only image environment.
    venv.EnvBuilder(with_pip=True, system_site_packages=True).create(root)
elif not (root / 'pyvenv.cfg').is_file() or not (root / 'bin/python').is_file():
    sys.exit('Incomplete .venv; remove /workspace/.venv and retry sandbox_install.')
env = {k: v for k, v in os.environ.items() if not k.startswith(('PIP_', 'PYTHON'))}
env.update(PIP_CONFIG_FILE='/dev/null', PIP_NO_INDEX='1',
           PIP_DISABLE_PIP_VERSION_CHECK='1', PYTHONNOUSERSITE='1')
command = [str(root / 'bin/python'), '-I', '-m', 'pip', '--disable-pip-version-check',
           '--no-input', '--no-cache-dir', 'install', '--no-index',
           '--find-links=/workspace/.packages', '--only-binary=:all:', *PACKAGES]
code = subprocess.call(command, env=env)
if code == 0:
    print('Environment: /workspace/.venv. Use python -m pytest for tests.', flush=True)
sys.exit(code)
'''
    # -I -S and an image-owned interpreter prevent workspace modules/.pth from
    # affecting bootstrap. Package code still runs ONLY inside offline Docker.
    return "/usr/local/bin/python -I -S -c " + shlex.quote("PACKAGES=" + repr(packages) + "\n" + script)
