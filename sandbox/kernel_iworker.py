import asyncio
import linecache
pickle = None
import sys
sys.path.insert(1, '../')
import sys
import traceback
from kernel_cache_handler import LoguruToCache, _disp_to_cache, _send_vars_to_cache, cache, OutToCache, ErrToCache, plt_show2cache
import json
from kernel_virtual_importer import install_virtual_importer
from _sbconsts import RUN_FINISH_LABEL, RUN_START_LABEL, NAMESPACE_PICKLE_FILE
import re
from kernel_namespaces import clean_unpicklables, default_namespace
from kernel_error_handling import KernelInterrupted, parse_client_error, ClientParsedError
import copy
from loguru import logger
builtin_modules = list(sys.modules.keys())
original_stdout = sys.stdout

def extract_node_ids(code):
    uids = re.findall('<<NODE-UID>>(.*?)<</NODE-UID>>', code, re.DOTALL)
    uids = [uid.strip().split('\n')[0].strip() for uid in uids]
    return uids

def extract_lineno_from_trace(trace):
    linenos = re.findall('File "<string>", line (.*?), in <module>', trace, re.DOTALL)
    real_linenos = []
    for lno in linenos:
        lno = lno.strip()
        if lno.isdigit():
            real_linenos.append(int(lno))
    return real_linenos
plot_importer = "\ntry:\n    import matplotlib\n    matplotlib.use('Agg')\n    import matplotlib.pyplot as plt\nexcept:\n    pass\n"

async def main():
    namespace = default_namespace.copy()
    cascaded_namespace = {}
    namespace['_cascns'] = cascaded_namespace
    current_loader = None
    main_code = None
    quiterr = 0
    builtin_names = set()
    orig_syspaths = copy.deepcopy(sys.path)
    exec(plot_importer, namespace)
    while True:
        run_id = -1
        try:
            try:
                _disp_to_cache(0, 'START', node_id='<debug-iworker-start>')
            except:
                traceback.print_exc()
                break
            codedata = input()
            if codedata.strip() == '__EXIT__':
                break
            if codedata.strip() == '__CLEAR__':
                namespace = default_namespace.copy()
                if current_loader:
                    sys.meta_path.remove(current_loader)
                    current_loader = None
                continue
            sys.path = orig_syspaths.copy()
            _disp_to_cache(0, codedata, node_id='<debug-iworker-codedata>')
            assert codedata.startswith('<run_id>'), codedata
            (run_id_part, codedata) = codedata.split('</run_id>', 1)
            run_id = int(run_id_part[8:])
            _disp_to_cache(0, run_id, node_id='<debug-iworker-run_id>')
            data = json.loads(codedata)
            builtin_names = builtin_names | set(data.get('builtins', []))
            project_code = data['files']
            reloads = data.get('reloads', ['<ALL>'])
            sys.path = sys.path[:1] + (data.get('external_pkgs') or []) + sys.path[1:] if sys.path else data.get('external_pkgs') or []
            if reloads == ['<ALL>']:
                reloads = list(project_code.keys())
            maybe_awrapper = data.get('async_wrapper')
            cnskey = data.get('cnskey', '<UNDEFINED>')
            vartrack = data.get('untrack_vars') or {}
            entry_point = data['entry']
            if entry_point not in project_code:
                print(f'Kernel Error: Entry point {entry_point} not found', file=sys.__stderr__)
                continue
            rootpath = entry_point.rsplit('/', 1)[-2] if '/' in entry_point else ''

            def format_mod(modname):
                rets = []
                assert not '>' in modname and (not '^' in modname), modname
                rootpkg = rootpath.replace('/', '.')
                modname = modname.replace('/', '.')
                if modname.endswith('.py'):
                    modname = modname[:-3]
                rets.append(modname)
                if modname.startswith(rootpkg + '.'):
                    rets.append(modname[len(rootpkg) + 1:])
                elif not rootpkg:
                    rets.append(modname)
                return rets
            reloads = [format_mod(r) for r in reloads]
            reloads = [item for sublist in reloads for item in sublist]
            last_modules = list(sys.modules.keys())
            _disp_to_cache(10006, rootpath, node_id='<debug-iworker-rootpath>')
            _disp_to_cache(10006, last_modules, node_id='<debug-iworker-last_modules>')
            _disp_to_cache(10006, reloads, node_id='<debug-sys-reloads>')
            _disp_to_cache(10006, builtin_modules, node_id='<debug-builtin_modules>')
            try:
                dels = []
                for modname in last_modules:
                    if modname in reloads:
                        if not modname in builtin_modules:
                            dels.append(modname)
                            del sys.modules[modname]
                        else:
                            _disp_to_cache(run_id, f'[WARNING] A module name overlaps with a built-in module and reloading is rejected: {modname}', node_id='0')
                    elif not modname in builtin_modules:
                        sys.modules[modname].__dict__['_run_id'] = run_id
                        sys.modules[modname].__dict__['_vars_tracking'] = vartrack
                _disp_to_cache(20000, dels, node_id='<info-iworker-deled-modules>')
            except Exception as e:
                traceback.print_exc()
                print(f'Kernel Error: Failed to clean import cache:{e}', file=sys.__stderr__)
            for k in project_code.keys():
                project_code[k] = project_code[k].replace('<<<LINE-CHANGE>>>', '\n')
            module_name = entry_point.replace('/', '.')
            if module_name.endswith('.py'):
                module_name = module_name[:-3]
            assert module_name, module_name
            package_name = module_name.rsplit('.', 1)[-2] if '.' in module_name else ''
            file_name = f'<virtual:{module_name}>'
            namespace.update({'__file__': file_name, '__name__': module_name, '__package__': package_name, '_run_id': run_id, '_vars_tracking': vartrack})
            if not '_cascns' in namespace:
                namespace['_cascns'] = {}
            localns = namespace['_cascns'].get(cnskey, {})
            namespace.update(localns)
            _disp_to_cache(10007, (file_name, module_name, package_name, run_id), node_id='<debug-__file__-__name__-__package__-run_id>')
            current_loader = install_virtual_importer(project_code, rootpath, run_id, cascns=namespace['_cascns'], vars_tracking=vartrack)
            main_code = project_code[entry_point]
            lines = main_code.splitlines(keepends=True)
            linecache.cache[file_name] = (len(main_code), None, lines, file_name)
            printer = OutToCache(run_id)
            sys.stdout = printer
            sys.stderr = ErrToCache(run_id)
            logger.remove()
            loguru_cache = LoguruToCache(run_id)
            logger.add(loguru_cache.write, level=data.get('loglevel') or 'TRACE', format='[{level}] {time:YY-MM-DD HH:mm:ss.SSS} {file}:{line} | {message}', enqueue=False, backtrace=True, diagnose=True, catch=True)
            if 'plt' in namespace and 'matplotlib' in namespace:
                print('劫持plt.show', file=sys.__stderr__)

                def _pltshow(*args, **kwargs):
                    plt_show2cache(namespace['plt'], run_id)
                namespace['plt'].show = _pltshow
            try:
                exec(compile(main_code, file_name, 'exec'), namespace)
                if maybe_awrapper:
                    assert f'async def {maybe_awrapper}' in main_code, f'Async wrapper {maybe_awrapper} notified but not seen in main code.'
                    await namespace[maybe_awrapper]()
            except ClientParsedError as cpe:
                raise cpe
            except Exception as e:
                raise ClientParsedError(traceback.format_exc(), project_code)
            finally:
                if cnskey and cnskey in namespace['_cascns']:
                    namespace['_cascns'][cnskey].clear()
        except (EOFError, BrokenPipeError):
            quiterr = 1
            break
        except KernelInterrupted as ie:
            quiterr = 1
            print(f'[Inputer interrupted]', file=sys.__stderr__)
            _disp_to_cache(run_id, ie, node_id='0', content_type='error')
        except ClientParsedError as e:
            try:
                print(f'ModuleErrorParsed from imported module: {type(e).__name__}: {e}', file=sys.__stderr__)
                errors = json.loads(str(e))
                for err in errors:
                    _disp_to_cache(run_id, err, node_id=err['node_id'] if err['node_prop'] == 'task' else '0', content_type='error')
            except (BrokenPipeError, OSError):
                break
            except Exception as e:
                print(f'Kernel Error: Failed to handle client error: {type(e).__name__}: {e}', file=sys.__stderr__)
                traceback.print_exc()
        except Exception as e:
            try:
                traceback.print_exc()
                print(f'Error at runtime: {type(e).__name__}: {e}', file=sys.__stderr__)
                errors = parse_client_error(traceback.format_exc())
                for err in errors:
                    _disp_to_cache(run_id, err, node_id=err['node_id'] if err['node_prop'] == 'task' else '0', content_type='error')
                sys.stderr.flush()
            except (BrokenPipeError, OSError):
                print(f'(BrokenPipeError, OSError)', file=sys.__stderr__)
                quiterr = 1
                break
            except Exception as e:
                print(f'Kernel Error: Failed when handling client error: {type(e).__name__}: {e}', file=sys.__stderr__)
                traceback.print_exc()
        finally:
            sys.stdout = original_stdout
            cache.clean_sigkill()
            if run_id > 0:
                _disp_to_cache(run_id, RUN_FINISH_LABEL, node_id=RUN_FINISH_LABEL)
                _disp_to_cache(run_id, '', node_id=RUN_FINISH_LABEL, content_type='node_runned')
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except:
        pass
if __name__ == '__main__':
    asyncio.run(main())