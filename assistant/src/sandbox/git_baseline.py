"""Trusted initialization script, executed INSIDE a disposable container only."""
from __future__ import annotations
import shlex
from .config import SandboxConfig


def init_command(config: SandboxConfig, imported_names: list[str] | None = None) -> str:
    # -I prevents imports of workspace-provided Python modules. Explicit git
    # paths avoid option/pathspec injection and bounded batches avoid ARG_MAX.
    script = '''
import os, pathlib, subprocess
root = pathlib.Path('/workspace')
skip = {'.git', '.venv', 'venv', 'node_modules', '__pycache__', '.pytest_cache', '.mypy_cache', 'build', 'dist'}
extensions = {'.gguf','.safetensors','.pt','.pth','.onnx','.bin','.npy','.npz','.parquet','.arrow','.csv','.tsv','.h5','.hdf5','.zip','.tar','.gz','.xz','.7z','.pdf','.png','.jpg','.jpeg','.mp4','.so','.o','.a','.exe','.dll'}
env = dict(os.environ, GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL='/dev/null',
           GIT_AUTHOR_NAME='sandbox', GIT_AUTHOR_EMAIL='sandbox@local',
           GIT_COMMITTER_NAME='sandbox', GIT_COMMITTER_EMAIL='sandbox@local')
def git(*args):
    subprocess.run(['git','-c','core.hooksPath=/dev/null',*args], cwd=root, env=env, check=True,
                   stdout=subprocess.DEVNULL)
git('init','-b','main')
selected, ignored, total = [], [], 0
for directory, dirs, files in os.walk(root, followlinks=False):
    dirs[:] = sorted(d for d in dirs if d not in skip and not (pathlib.Path(directory)/d).is_symlink())
    for name in sorted(files):
        p = pathlib.Path(directory)/name
        rel = str(p.relative_to(root))
        if IMPORTED is not None and p.relative_to(root).parts[0] not in IMPORTED:
            continue
        if p.is_symlink() or not p.is_file():
            continue
        size = p.stat().st_size
        if size > MAX_FILE or total + size > MAX_TOTAL or p.suffix.lower() in extensions:
            ignored.append('/' + rel.replace('\\\\','\\\\\\\\').replace('[','\\\\[').replace('*','\\\\*').replace('?','\\\\?'))
            continue
        if subprocess.run(['git','check-ignore','-q','--',rel],cwd=root,env=env).returncode == 0:
            continue
        selected.append(rel)
        total += size
with (root/'.git/info/exclude').open('a') as f:
    f.write('\\n' + '\\n'.join(d+'/' for d in sorted(skip- {'.git'})) + '\\n')
    f.write('\\n'.join('*'+e for e in sorted(extensions)) + '\\n')
    f.write('\\n'.join(x for x in ignored if '\\n' not in x) + '\\n')
for start in range(0,len(selected),100):
    git('--literal-pathspecs','add','--',*selected[start:start+100])
# Commit only these imported paths; preserve unrelated staged model edits.
if selected:
    import tempfile
    with tempfile.NamedTemporaryFile() as paths:
        for rel in selected:
            paths.write(os.fsencode(rel) + bytes([0]))
        paths.flush()
        git('--literal-pathspecs','commit','--allow-empty','--only',
            '--pathspec-from-file='+paths.name,'--pathspec-file-nul','-m','sandbox imported baseline')
else:
    git('commit','--allow-empty','--only','-m','sandbox imported baseline')
'''
    script = f"IMPORTED={imported_names!r}\nMAX_FILE={config.git_max_file_bytes}\nMAX_TOTAL={config.git_max_total_bytes}\n" + script
    return "python -I -c " + shlex.quote(script)
