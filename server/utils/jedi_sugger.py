import jedi
import sys
import os
import tempfile
from pathlib import Path

class JediEnvManager:

    def __init__(self):
        self._temp_dirs = []

    def create_env_with_extra_paths(self, extra_paths, livecodes={}):
        temp_dir = tempfile.TemporaryDirectory()
        self._temp_dirs.append(temp_dir)
        temp_root = Path(temp_dir.name)
        for rel_path, code in livecodes.items():
            full_path = temp_root / rel_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(code, encoding='utf-8')
        env = jedi.get_default_environment()
        original_sys_path = env.get_sys_path()
        combined_path = [os.path.abspath(p) for p in extra_paths if os.path.isdir(p)]
        combined_path = [str(temp_root)] + combined_path + original_sys_path
        env._original_get_sys_path = env.get_sys_path
        env.get_sys_path = lambda: combined_path
        env._temp_dir_holder = temp_dir
        return env

    def _suggest(self, maincode, line, column, extpkgs=[], livecodes={}):
        script = jedi.Script(code=maincode, environment=self.create_env_with_extra_paths(extpkgs, livecodes))
        completions = script.complete(line=line, column=column)
        sugs = [{'name': c.name, 'type': c.type, 'doc': c.docstring()} for c in completions]
        return sugs

    def cleanup_all(self):
        for td in self._temp_dirs:
            try:
                td.cleanup()
            except Exception:
                pass
        self._temp_dirs.clear()

    def suggest(self, maincode, line, column, extpkgs=[], livecodes={}):
        try:
            return self._suggest(maincode, line, column, extpkgs=extpkgs, livecodes=livecodes)
        finally:
            self.cleanup_all()