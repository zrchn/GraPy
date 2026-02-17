import sys
import traceback
from kernel_namespaces import reset_pickled_namespace
import asyncio
import sys
import time
from kernel_basic.configer import configer
import copy
import subprocess
from loguru import logger
from kernel_cache_handler import accept_user_input, cache
import json
from kernel_basic.background_runner import AsyncTaskRunner
from starlette.websockets import WebSocketDisconnect
from _sbconsts import BUILTIN_NAMES, CONSOLE_INPUTER, RUN_FINISH_LABEL, CONSOLE_PRINTER, VARSEND_FUNC, EXTRA_BUILTIN_MODS
import json5
TIMEOUT = configer.grapy.execution_timeout

def enum_parents(dir):
    sects = dir.split('/')
    dirs = ['/'.join(sects[:i]) for i in range(1, len(sects))]
    return dirs

def makeup_initers(codedict):
    dirs = set()
    codedict_new = codedict.copy()
    added = []
    for path in codedict.keys():
        parents = enum_parents(path)
        dirs = dirs.union(parents)
    for dir in dirs:
        initdir = dir + '/__init__.py'
        if not initdir in codedict_new:
            codedict_new[initdir] = ''
            added.append(initdir)
    logger.debug('加入省略的__init__.py：{}', added)
    return codedict_new

class CodesRunner(AsyncTaskRunner):

    def __init__(self):
        super().__init__()
        reset_pickled_namespace()
        self.reset_worker()
        self.do_kill = False

    def reset_builtin_modules(self):
        self.builtin_modules = list(sys.modules.keys())

    def reset_worker(self):
        subpath = 'kernel_iworker.py'

        def reopen(subpath):
            self.process = subprocess.Popen(['python', subpath], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        reopen(subpath)

    def safe_terminate(self, graceful_timeout=8, force_timeout=8):
        self.do_kill = True
        if self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=graceful_timeout)
            logger.info('进程已优雅杀停')
            return
        except subprocess.TimeoutExpired:
            logger.warning('进程无法优雅杀停，尝试强制杀停')
            pass
        self.process.kill()
        try:
            self.process.wait(timeout=force_timeout)
            logger.info('进程已强制杀停')
        except subprocess.TimeoutExpired:
            logger.error('警告：进程无法终止!')

    def rebirth(self):
        self.safe_terminate()
        del self.process
        self.reset_worker()

    def reset_iworker_namespace(self):
        self.process.stdin.write('__CLEAR__\n')
        reset_pickled_namespace()

    def run_codes(self, codedata, run_id):
        self.do_kill = False
        cache.prepare_for_new_run()
        if self.process.poll() is not None:
            logger.info('子进程挂了，重启。run_id: {}', run_id)
            self.reset_worker()
        codedata = copy.deepcopy(codedata)
        for k in codedata['files'].keys():
            if 1:
                codedata['files'][k] = codedata['files'][k].replace(f'{CONSOLE_INPUTER}(', '_input_via_cache(_run_id, ')
                codedata['files'][k] = codedata['files'][k].replace(f'{CONSOLE_PRINTER}(', '_disp_to_cache(_run_id, ')
                codedata['files'][k] = codedata['files'][k].replace(f'{VARSEND_FUNC}(', '_send_vars_to_cache(_run_id, ')
            codedata['files'][k] = codedata['files'][k].replace('\n', '<<<LINE-CHANGE>>>')
        codedata['files'] = makeup_initers(codedata['files'])
        fullcode = json.dumps(codedata, ensure_ascii=False)
        fullcode = f'<run_id>{run_id}</run_id>' + fullcode
        logger.debug('fullcode: {}', fullcode)
        try:
            self.process.stdin.write(fullcode + '\n')
            self.process.stdin.flush()
        except BrokenPipeError:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
        logger.debug('客户代码组提交运行：{}', run_id)

    async def arun_codes(self, codedata, run_id):
        self.run_codes(codedata, run_id)

    async def start_running_codes(self, codedata, run_id):
        if not self.monitor_started:
            self.start_monitor()
        dorun = asyncio.create_task(self.arun_codes(codedata, run_id))
        self.submit_task(dorun)

    def poll_outputs(self, run_id):
        cache.reset_olds(run_id)
        time0 = time.time()
        while True:
            time.sleep(0.3)
            time1 = time.time()
            if time1 - time0 > TIMEOUT:
                logger.warning(f'Execution timeout. run_id: {run_id}')
                break
            news = cache._get_news(run_id)
            if news:
                yield news
            if RUN_FINISH_LABEL in news.keys():
                logger.info(f'运行完毕。run_id: {run_id}')
        cache.reset_olds(run_id)

    def delvar(self, name, cnskey=None):
        cnsdeler = ''
        if cnskey:
            cnsdeler = f"\ntry:\n    del _cascns['{cnskey}']['{name}']\nexcept:\n    pass\n"
        deler = f'\ntry:\n    del {name}\nexcept:\n    pass\n{cnsdeler}\n'
        codedata = {'files': {'_del_var.py': deler}, 'entry': '_del_var.py', 'reloads': ['_del_var.py'], 'builtins': [], 'external_pkgs': []}
        self.run_codes(codedata, 10)

    def get_kernel_var_infos(self, cnskey=None, include_values=False, name_filter='lambda x: True'):
        maybe_value = '' if not include_values else ",'repr':value"
        fake_node_id = '_get_kernel_var_infos'
        if not cnskey:
            getter = f"\n_name_filter = {name_filter}\n_global_var_infos = [\n    {{'name':_var_name,'type':type(value).__name__{maybe_value},'belong':'global'}}\n        for _var_name, value in globals().items()\n        if _name_filter(_var_name) and not _var_name.startswith('__') and not _var_name in {str(BUILTIN_NAMES)}\n]\n"
        else:
            getter = f"\n_name_filter = {name_filter}\n_global_var_infos = [\n    {{'name':_var_name,'type':type(value).__name__{maybe_value}}}\n        for _var_name, value in globals().items()\n        if _name_filter(_var_name) and not _var_name.startswith('__') and not _var_name in {str(BUILTIN_NAMES)}\n]\n_global_vnames = [v['name'] for v in _global_var_infos]\n_local_var_infos = [\n    {{'name':_var_name,'type':type(value).__name__{maybe_value}}}\n        for _var_name, value in _cascns.get('{cnskey}',{{}}).items()\n        if _name_filter(_var_name) and not _var_name.startswith('__') and not _var_name in {str(BUILTIN_NAMES)}\n]\n_local_vnames = [v['name'] for v in _local_var_infos]\n# 合并：以local的优先\nfor _global_var in _global_var_infos:\n    if not _global_var['name'] in _local_vnames:\n        _global_var['belong'] = 'global'\n    else:\n        _global_var['belong'] = 'global&local'\nfor _local_var in _local_var_infos: # local里有、global里没有的也挂进来\n    if not _local_var['name'] in _global_vnames:\n        _local_var['belong'] = 'local'\n        _global_var_infos.insert(0,_local_var)\n"
        getter = getter + f"\n_global_var_infos = sorted(_global_var_infos, key=lambda x: x['name'])\n{VARSEND_FUNC}('{fake_node_id}',{{'_global_var_infos':_global_var_infos}})\n"
        codedata = {'files': {'_get_kernel_var_infos.py': getter}, 'entry': '_get_kernel_var_infos.py', 'reloads': ['_get_kernel_var_infos.py'], 'builtins': [], 'external_pkgs': []}
        self.run_codes(codedata, 10)
        for i in range(50):
            time.sleep(0.2)
            glovars = cache.get_all_by_run_id(10, choice='vars')
            if not glovars:
                continue
            assert list(glovars[fake_node_id].values())[0]['content']['name'] == '_global_var_infos'
            glovars = list(glovars[fake_node_id].values())[0]['content']['repr']
            glovars = eval(glovars)
            return glovars
        raise TimeoutError('Cannot retrieve kernel vars.')

    def get_kernel_var_value(self, name, cnskey=None):
        fake_node_id = 'get_kernel_var_value'
        getter = f"{VARSEND_FUNC}('{fake_node_id}',{{'{name}':{name}}})"
        if cnskey:
            getter = f"\nif '{name}' in _cascns.get('{cnskey}',{{}}):\n    {VARSEND_FUNC}('{fake_node_id}',{{'{name}':_cascns.get('{cnskey}',{{}})['{name}']}})\nelse:\n    {getter}\n"
        codedata = {'files': {'_get_kernel_var_value.py': getter}, 'entry': '_get_kernel_var_value.py', 'reloads': ['_get_kernel_var_value.py'], 'builtins': [], 'external_pkgs': []}
        self.run_codes(codedata, 10)
        for i in range(50):
            time.sleep(0.2)
            repr = cache.get_all_by_run_id(10, choice='vars')
            if not repr:
                continue
            assert list(repr[fake_node_id].values())[0]['content']['name'] == name
            cont = list(repr[fake_node_id].values())[0]['content']
            repr = cont['repr']
            dtype = cont['type']
            return (repr, dtype)
        raise TimeoutError('Cannot retrieve kernel var value.')

    def get_kernel_modules(self):
        fake_node_id = '_get_kernel_modules'
        getter = f"\nimport sys\n_mods = [\n    {{'name':_mod_name,'type':'module'}}\n        for _mod_name in list(sys.modules.keys())\n        if not _mod_name in {list(self.builtin_modules) + EXTRA_BUILTIN_MODS}\n]\n_mods = sorted(_mods, key=lambda x: x['name'])\n{VARSEND_FUNC}('{fake_node_id}',{{'_mods':_mods}})\n        "
        codedata = {'files': {'_get_kernel_modules.py': getter}, 'entry': '_get_kernel_modules.py', 'reloads': ['_get_kernel_modules.py'], 'builtins': [], 'external_pkgs': []}
        self.run_codes(codedata, 10)
        for i in range(50):
            time.sleep(0.2)
            mods = cache.get_all_by_run_id(10, choice='vars')
            if not mods:
                continue
            assert list(mods[fake_node_id].values())[0]['content']['name'] == '_mods'
            mods = list(mods[fake_node_id].values())[0]['content']['repr']
            mods = eval(mods)
            return mods
        raise TimeoutError('Cannot retrieve kernel modules.')

    def del_kernel_modules(self, modnames):
        code = f'\nimport sys\nfor _mod_name in {modnames}:\n    try:\n        del sys.modules[_mod_name]\n    except:\n        pass\n'
        codedata = {'files': {'_del_kernel_modules.py': code}, 'entry': '_del_kernel_modules.py', 'reloads': ['_del_kernel_modules.py'], 'builtins': [], 'external_pkgs': []}
        self.run_codes(codedata, 10)

    def refresh_kernel_modules(self, modnames):
        fake_node_id = '_refresh_kernel_modules'
        code = f"""\nimport sys\n_mod_errs = ''\nfor _mod_name in {modnames}:\n    try:\n        del sys.modules[_mod_name]\n        exec("import "+_mod_name)\n    except Exception as _mod_e:\n        _mod_errs += str(_mod_e)+'\\n'\n{VARSEND_FUNC}('{fake_node_id}',{{'_errs':_mod_errs}})\n"""
        codedata = {'files': {'_refresh_kernel_modules.py': code}, 'entry': '_refresh_kernel_modules.py', 'reloads': ['_refresh_kernel_modules.py'], 'builtins': [], 'external_pkgs': []}
        self.run_codes(codedata, 10)
        for i in range(125):
            time.sleep(0.5)
            _errs = cache.get_all_by_run_id(10, choice='vars')
            if not _errs:
                continue
            assert list(_errs[fake_node_id].values())[0]['content']['name'] == '_errs'
            _errs = list(_errs[fake_node_id].values())[0]['content']['repr']
            _errs = str(_errs).strip()
            if _errs:
                logger.error('刷新mods失败：{}', _errs)
            return _errs
        raise TimeoutError('Refresh kernel modules exceeded 60s. Its state is unknown.')

    def get_sugs(self, objpart):
        fake_node_id = '_get_sugs'
        code = f"\nimport inspect\ndef _autocomplete(pattern: str, namespace:dict = None):\n    import inspect\n    if namespace is None:\n        namespace = globals()\n    else:\n        pass\n    if '.' not in pattern:\n        # 顶层变量补全，如 'ob' → ['obj', 'other']\n        return [{{'name':name,'type':'variable'}} for name in namespace if name.startswith(pattern)]\n    else:\n        pass\n    # 分割为 base + suffix\n    last_dot = pattern.rfind('.')\n    base_path = pattern[:last_dot]\n    suffix = pattern[last_dot + 1:]\n    parts = base_path.split('.')\n    print('base_path:', base_path)\n    # 获取 base 对象\n    try:\n        obj = namespace[parts[0]]\n        for part in parts[1:]:\n            obj = getattr(obj, part)\n            print('obj = getattr(obj, part) obj:', obj)\n    except:\n        traceback.print_exc()\n        return []\n    # 获取 obj 的所有属性\n    try:\n        attrs = dir(obj)\n    except:\n        return []\n    # 过滤以 suffix 开头的，并拼接完整路径\n    matches = []\n    for attr in attrs:\n        _temp_attr_startswith_value = attr.startswith(suffix)\n        if _temp_attr_startswith_value:\n            matches.append({{'name':attr,'type':'variable'}})\n        else:\n            pass\n    return matches\n_autocompleted = _autocomplete('{objpart}')\n{VARSEND_FUNC}('{fake_node_id}',{{'_autocompleted':_autocompleted}})\n"
        codedata = {'files': {'_get_sugs.py': code}, 'entry': '_get_sugs.py', 'reloads': ['_get_sugs.py'], 'builtins': [], 'external_pkgs': []}
        self.run_codes(codedata, 10)
        for i in range(50):
            time.sleep(0.2)
            sugs = cache.get_all_by_run_id(10, choice='vars')
            if not sugs:
                continue
            assert list(sugs[fake_node_id].values())[0]['content']['name'] == '_autocompleted'
            sugs = list(sugs[fake_node_id].values())[0]['content']['repr']
            try:
                sugs = json5.loads(sugs)
            except:
                sugs = eval(sugs)
            return sugs
        raise TimeoutError('Cannot retrieve suggestions.')

    def get_params(self, funcpart):
        fake_node_id = '_get_params'
        code = f'''\ndef _get_function_arg_names(full_name: str, namespace=None):\n    """\n    安全获取函数/方法的参数名列表。\n    \n    Args:\n        full_name: 函数路径，如 'func' 或 'obj.method' 或 'mod.sub.func'\n        namespace: 命名空间字典，默认为 globals()\n    \n    Returns:\n        参数名列表（不包含 *args, **kwargs 的特殊符号），如果无效则返回空列表。\n    """\n    assert isinstance(full_name,str), full_name\n    if namespace is None:\n        namespace = globals()\n    else:\n        pass\n    # 分割路径\n    parts = full_name.split('.')\n    obj = namespace\n    # 逐级获取对象\n    try:\n        for part in parts:\n            if isinstance(obj, dict):\n                if not part in obj:\n                    return []\n                obj = obj[part]\n            else:\n                obj = getattr(obj, part)\n    except:\n        traceback.print_exc()\n        return []\n    # 检查是否可调用\n    if not callable(obj):\n        return []\n    else:\n        pass\n    # 尝试获取签名\n    try:\n        sig = inspect.signature(obj=obj)\n        # 过滤掉 *args, **kwargs 等，只保留普通参数名\n        params = []\n        for (name, param) in sig.parameters.items():\n            if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):\n                params.append(name)\n            elif param.kind == inspect.Parameter.VAR_POSITIONAL:\n                params.append('*'+name)\n            elif param.kind == inspect.Parameter.VAR_KEYWORD:\n                params.append('**'+name)\n        return params\n    except:\n        # inspect.signature() 对某些 built-in 函数会失败（如 len, print）\n        return []\n_function_arg_names = _get_function_arg_names(\'{funcpart}',globals())\n{VARSEND_FUNC}(\'{fake_node_id}',{{'_function_arg_names':_function_arg_names}})\n'''
        codedata = {'files': {'_get_params.py': code}, 'entry': '_get_params.py', 'reloads': ['_get_params.py'], 'builtins': [], 'external_pkgs': []}
        self.run_codes(codedata, 10)
        for i in range(50):
            time.sleep(0.2)
            sugs = cache.get_all_by_run_id(10, choice='vars')
            if not sugs:
                continue
            assert list(sugs[fake_node_id].values())[0]['content']['name'] == '_function_arg_names'
            sugs = list(sugs[fake_node_id].values())[0]['content']['repr']
            sugs = eval(sugs)
            return sugs
        raise TimeoutError('Cannot retrieve params.')

    async def apoll_outputs(self, run_id, choice='all'):
        cache.reset_olds(run_id)
        time0 = time.time()
        try:
            while True:
                if self.do_kill:
                    logger.debug('杀停输出轮询')
                    self.do_kill = False
                    break
                await asyncio.sleep(0.3)
                time1 = time.time()
                if time1 - time0 > TIMEOUT:
                    logger.warning(f'Execution timeout. run_id: {run_id}')
                    yield '<<<EXECUTION-TIMEOUT>>>'
                    break
                prompt_buffer = []
                if choice == 'stateonly':
                    news = cache._get_new_nodes_states(run_id)
                    for node_id, item in news.items():
                        if item.get('prompts'):
                            pieces = [{'run_id': run_id, 'node_id': node_id, 'prompt': p} for p in item['prompts']]
                            prompt_buffer = prompt_buffer + pieces
                            logger.info(f'发现有input prompt，加入buffer:{pieces}')
                else:
                    news = cache._get_news(run_id)
                    for node_id, item in news.items():
                        for subitem in item.values():
                            if subitem['content_type'] == 'prompt':
                                piece = {'run_id': run_id, 'node_id': node_id, 'prompt': subitem['content']}
                                prompt_buffer.append(piece)
                                logger.info(f"发现有input prompt，run_id={run_id}, node_id={node_id},prompt={subitem['content']},加入buffer")
                if news:
                    yield {'event': 'output', 'content': {k: v for k, v in news.items() if not k == RUN_FINISH_LABEL}, 'run_id': run_id}
                if prompt_buffer:
                    for prompt_info in prompt_buffer:
                        inp = (yield {'event': 'prompt', 'content': prompt_info['prompt'], 'run_id': run_id, 'node_id': prompt_info['node_id']})
                        logger.info(f'收到用户输入，在run_id={run_id}, node_id={node_id}:', inp)
                        if not (inp['run_id'] == prompt_info['run_id'] and inp['node_id'] == prompt_info['node_id'] and (inp['prompt'] == prompt_info['prompt']['content'])):
                            logger.error(f'有bug，收到的input有东西对不上：{inp} vs {prompt_info}')
                        accept_user_input(prompt_info['run_id'], prompt_info['node_id'], prompt_info['prompt']['content'], inp['inputed'])
                if RUN_FINISH_LABEL in news.keys():
                    logger.info(f'运行完毕。run_id: {run_id}')
                    yield '<<<FINISH-EXECUTION>>>'
                    break
        finally:
            cache.reset_olds(run_id)
            print('runner cleared cache.olds:', cache.olds)
            try:
                cache.postrun_clean(run_id)
            except Exception as e:
                traceback.print_exc()
                logger.error('有bug，postrun_clean失败。run_id={}, e:{}', run_id, e)

    async def astream_outputs(self, run_id, streamer, choice='all'):
        cache.reset_olds(run_id)
        try:
            apollor = self.apoll_outputs(run_id, choice=choice)
            output = await anext(apollor)
            while True:
                logger.trace('output: {}', output)
                if output == '<<<FINISH-EXECUTION>>>':
                    await streamer.send_text(json.dumps({'event': 'end', 'run_id': run_id}, ensure_ascii=False))
                    break
                elif output == '<<<EXECUTION-TIMEOUT>>>':
                    await streamer.send_text(json.dumps({'event': 'end', 'msg': 'timeout', 'run_id': run_id}, ensure_ascii=False))
                    break
                elif output['event'] == 'output':
                    await streamer.send_text(json.dumps(output, ensure_ascii=False))
                    output = await anext(apollor)
                elif output['event'] == 'prompt':
                    await streamer.send_text(json.dumps(output, ensure_ascii=False))
                    user_rsp = await streamer.receive_text()
                    user_rsp = json.loads(user_rsp)
                    output = await apollor.asend(user_rsp)
                else:
                    logger.error(f'无法识别的poll输出：{output}')
                    output = await anext(apollor)
        except WebSocketDisconnect:
            logger.warning(f'客户端主动关闭连接')
            cache.sigkill_inputer()
        except Exception as e:
            await streamer.send_text(json.dumps({'event': 'error', 'msg': str(e), 'run_id': run_id}, ensure_ascii=False))
            logger.error(f'运行出错：{e}')
            traceback.print_exc()
        finally:
            await streamer.close()
runner = CodesRunner()
if __name__ == '__main__':
    logger.add(sys.stdout, level='TRACE')
    runner = CodesRunner()
    project1 = {'files': {'workdir/zz/foo/__init__.py': '', 'workdir/__init__.py': '', 'workdir/zz/__init__.py': '', 'workdir/zz/xv.py': 'v=5', 'workdir/zz/utils.py': 'from ..zz.xv import v\nimport time\ntime.sleep(1)\ndef add(x, y):    return x + y\ndef x():    return v', 'workdir/zz/foo/main.py': "from ..utils import add, x;disp('fff',219);import time;time.sleep(1); disp(add(x(), 2),220); y = 43"}, 'entry': 'workdir/zz/foo/main.py'}
    project2 = {'files': {'wd2/bar/div/__init__.py': '', 'wd2/bar/div/test.py': 'b=9', 'wd2/bar/__init__.py': '', 'wd2/bar/c.py': 'c=-2', 'wd2/xxx/__init__.py': '', 'wd2/xxx/nn.py': 'from bar.c import c \ndef operate(a,b): \n    return a*b*c', 'wd2/main.py': 'from bar.div.test import b\nfrom bar.c import c \nfrom xxx.nn import operate\nv=operate(y,b)\ndisp(v,223)'}, 'entry': 'wd2/main.py'}

    async def main():
        await runner.start_running_codes(project1, 302)
        await asyncio.sleep(4)
    asyncio.run(main())