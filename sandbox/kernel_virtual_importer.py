import sys
import ast
import traceback
from importlib.util import spec_from_loader, module_from_spec
from importlib.abc import MetaPathFinder, SourceLoader
from kernel_cache_handler import _disp_to_cache
from kernel_namespaces import default_namespace
import json
import re
from kernel_error_handling import ClientParsedError
import linecache

class VirtualModuleLoader(SourceLoader, MetaPathFinder):

    def __init__(self, code_dict, rootpath, run_id, cascns={}, vars_tracking={}):
        self.code_dict = code_dict
        self.module_cache = {}
        self.rootpath = rootpath
        self.run_id = run_id
        self.cascns = cascns
        self.vars_tracking = vars_tracking
        _disp_to_cache(10008, self.code_dict, node_id='<debug-VirtualModuleLoader-code_dict>')

    def find_spec(self, fullname, path, target=None):

        def deduce(package_dir):
            init_path = f'{package_dir}/__init__.py'
            if init_path in self.code_dict:
                spec = spec_from_loader(fullname, self, is_package=True)
                if spec is not None:
                    spec.loader = self
                    spec.submodule_search_locations = [package_dir]
                return spec
            filename = package_dir + '.py'
            if filename in self.code_dict:
                spec = spec_from_loader(fullname, self)
                if spec is not None:
                    spec.loader = self
                return spec
            return '<UNK>'
        if self.rootpath:
            package_dir = f"{self.rootpath}/{fullname.replace('.', '/')}"
            spec = deduce(package_dir)
            if spec != '<UNK>':
                return spec
        package_dir = fullname.replace('.', '/')
        spec = deduce(package_dir)
        if spec != '<UNK>':
            return spec
        if path:
            for p in path:
                if isinstance(p, str):
                    base = p.rsplit('/', 1)[0] if '/' in p and (not p.startswith('<')) else p
                    rel_path = f'{base}/{fullname}.py'.lstrip('/')
                    if rel_path in self.code_dict:
                        spec = spec_from_loader(fullname, self)
                        if spec is not None:
                            spec.loader = self
                        return spec
        return None

    def get_filename(self, fullname):

        def _get(package_dir):
            init_path = f'{package_dir}/__init__.py'
            if init_path in self.code_dict:
                return init_path
            filename = f'{package_dir}.py'
            if filename in self.code_dict:
                return filename
            return None
        filename = None
        if self.rootpath:
            package_dir = f"{self.rootpath}/{fullname.replace('.', '/')}"
            filename = _get(package_dir)
        if filename is None:
            package_dir = fullname.replace('.', '/')
            filename = _get(package_dir)
        if filename is None:
            filename = fullname + '.py'
        return filename

    def get_data(self, path):
        raise NotImplementedError()

    def get_source(self, fullname):
        filename = f"{fullname.replace('.', '/')}.py"
        for path, code in self.code_dict.items():
            if path == filename:
                return code
        if '.' in fullname:
            pkg, mod = fullname.rsplit('.', 1)
            base_path = pkg.replace('.', '/')
            target = f'{base_path}/{mod}.py'
            for path, code in self.code_dict.items():
                if path == target:
                    return code
        return None

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        fullname = module.__name__

        def _exec(module_path):
            maybe_init_path = f'{module_path}/__init__.py'
            if maybe_init_path in self.code_dict:
                filename = maybe_init_path
            else:
                filename = f'{module_path}.py'
            code = None
            for path, src in self.code_dict.items():
                if path == filename:
                    code = src
                    break
            return code
        code = None
        if self.rootpath:
            module_path = f"{self.rootpath}/{fullname.replace('.', '/')}"
            code = _exec(module_path)
        if code is None:
            module_path = fullname.replace('.', '/')
            code = _exec(module_path)
        if code is None:
            _disp_to_cache(0, 'code is None', node_id='<debug-exec_module-codeIsNone>')
            raise ImportError(f'Cannot find module {fullname}')
        self.module_cache[fullname] = module
        fake_filename = f'<virtual:{fullname}>'
        lines = code.splitlines(keepends=True)
        linecache.cache[fake_filename] = (len(code), None, lines, fake_filename)
        module.__file__ = fake_filename
        for k, v in default_namespace.items():
            if not k in module.__dict__:
                module.__dict__[k] = v
        module.__dict__['_run_id'] = self.run_id
        module.__dict__['_cascns'] = self.cascns
        module.__dict__['_vars_tracking'] = self.vars_tracking
        try:
            compiled_code = compile(code, fake_filename, 'exec')
            exec(compiled_code, module.__dict__)
            _disp_to_cache(10001, f'module {fullname} __dict__: {module.__dict__.keys()}', node_id='<debug-modified-__dict__-keys>')
        except Exception as e:
            traceback.print_exc()
            _disp_to_cache(10001, e, node_id='<debug-exec_module-exception>')
            del self.module_cache[fullname]
            raise ClientParsedError(traceback.format_exc(), self.code_dict)
import_count = 0

def install_virtual_importer(code_dict, rootpath, run_id, cascns={}, vars_tracking={}):
    global import_count
    loader = VirtualModuleLoader(code_dict, rootpath, run_id, cascns=cascns, vars_tracking=vars_tracking)
    _disp_to_cache(10004, sys.meta_path[:], node_id=f'<debug-sys-meta_path-{import_count}>')
    import_count = import_count + 1
    for finder in sys.meta_path[:]:
        if isinstance(finder, VirtualModuleLoader):
            sys.meta_path.remove(finder)
    sys.meta_path.insert(0, loader)
    return loader
if __name__ == '__main__':
    code_dict = {'workdir/utils.py': 'def add(x, y): return x + y', 'workdir/main.py': 'from utils import add; disp(add(1,2))'}
    loader = VirtualModuleLoader(code_dict)
    for finder in sys.meta_path[:]:
        if isinstance(finder, VirtualModuleLoader):
            sys.meta_path.remove(finder)
    sys.meta_path.insert(0, loader)