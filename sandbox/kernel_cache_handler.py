import base64
from concurrent.futures import ThreadPoolExecutor
import io
import traceback
from _sbconsts import INPUT_TIMEOUT, REDIS_LIMIT_MAX, KEEP_HISTORY_RUNS, UID_COMMENT_LEFTLABEL, UID_COMMENT_RIGHTLABEL
import redis
from kernel_basic.configer import configer
from loguru import logger
import json
import json5
import orjson
import sys
from kernel_disp import disp
from _sbutils import df_to_safe_dict, idgen, safe_default, series_to_safe_dict, stringify_dict_keys
import time
from kernel_error_handling import KernelInterrupted
from _sbutils import statics
import inspect
from rich.console import Console
from rich.text import Text
import pandas as pd
REDIS_HOST = configer.grapy.redis_host
REDIS_PORT = int(configer.grapy.redis_port)
NODE_MAX_RECORDS = int(configer.grapy.node_max_prints)
assert NODE_MAX_RECORDS > 1 and NODE_MAX_RECORDS < 5000, f'node_max_prints must be between 1 and 5000, got {NODE_MAX_RECORDS}'
SKIP_BLANK_PRINTS = configer.grapy.skip_blank_prints
ENRICH = configer.grapy.enrich_vars_display == 'eager'
TRACK_VARS_MAXLEN = configer.grapy.track_vars_maxlen
PRINT_MAXLEN = configer.grapy.print_maxlen
assert TRACK_VARS_MAXLEN > 6
assert PRINT_MAXLEN > 6
rich_console = Console(file=io.StringIO(), width=100, force_jupyter=False)

def _pretty_format_rich(obj):
    if not ENRICH:
        try:
            if type(obj).__name__ in ('dict', 'list', 'tuple', 'DataFrame', 'Series'):
                if type(obj).__name__ == 'DataFrame':
                    try:
                        obj = df_to_safe_dict(obj)
                    except Exception as e:
                        print('df df_to_safe_dict 失败', file=sys.__stderr__)
                        pass
                elif type(obj).__name__ == 'Series':
                    try:
                        obj = series_to_safe_dict(obj)
                    except Exception as e:
                        print('series_to_safe_dict 失败', file=sys.__stderr__)
                        pass
                return orjson.dumps(obj, default=safe_default, option=orjson.OPT_NON_STR_KEYS | orjson.OPT_SERIALIZE_NUMPY).decode('utf-8')
            else:
                return str(obj)
        except Exception as e:
            print('[WARNING] 转json失败:', e, file=sys.__stderr__)
            return str(obj)
        return str(obj)
    with rich_console.capture() as capture:
        rich_console.print(obj, highlight=False, soft_wrap=False)
    return capture.get().rstrip()

def to_dumpable(data):
    if data is None:
        return data
    flats = (int, str, float, bool)
    if type(data) in flats:
        return data
    if type(data) in (set, list, tuple):
        newdata = []
        for subdata in data:
            newdata.append(to_dumpable(subdata))
        return newdata
    if type(data) == dict:
        newdata = {}
        for (k, subdata) in data.items():
            newdata[k] = to_dumpable(subdata)
        return newdata
    try:
        return str(data)
    except:
        return '[UNREPRESENTABLE]'

def tostr(data, restrict=49999):
    try:
        if isinstance(data, str):
            if len(data) > restrict and restrict > 0:
                data = data[:restrict - 6] + '......'
        ret = _pretty_format_rich(data)
    except:
        ret = str(data)
    if len(ret) > restrict and restrict > 0:
        ret = ret[:restrict - 6] + '......'
    return ret
beyond312 = False
if int(sys.version_info.minor) >= 12:
    beyond312 = True
    logger.info(f'You can ignore the SyntaxWarnings about escape sequences that may arise. These escapes are intentional to align with redis grammar.')

class CacheHandler:

    def __init__(self, node_max_records=None):
        redis_pool = redis.ConnectionPool(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        self.cache = redis.Redis(connection_pool=redis_pool, decode_responses=True)
        self.node_max_records = node_max_records or NODE_MAX_RECORDS
        self.olds = {}
        self.store_counts = {}
        try:
            self.cache.execute_command('FT.CREATE', 'idx:output', 'ON', 'JSON', 'PREFIX', '1', 'output:', 'SCHEMA', '$.run_id', 'AS', 'run_id', 'NUMERIC', 'SORTABLE', '$.node_id', 'AS', 'node_id', 'TAG', '$.record_id', 'AS', 'record_id', 'NUMERIC', 'SORTABLE', '$.content_type', 'AS', 'content_type', 'TAG')
            logger.info(f'output index created')
        except redis.exceptions.ResponseError as e:
            if e.__repr__() == "ResponseError('Index already exists')":
                logger.info(f'using existing output index')
            else:
                raise RuntimeError(e)

    def prepare_for_new_run(self):
        self.store_counts = {}
        self.olds = {}
        self.clean_sigkill()
        self.forget_all_before(100000, lbound=-10)

    def postrun_clean(self, run_id):
        self.trunk_to_n_runs(KEEP_HISTORY_RUNS)

    def periodic_clean(self, alive_node_ids):
        pass

    def sigkill_inputer(self):
        run_id = (self.get_all_available_run_ids() + [0])[0]
        self.store(run_id, 'ghost', '', content_type='kill')

    def check_kill(self) -> bool:
        result = self.cache.execute_command('FT.SEARCH', 'idx:output', '@content_type:{kill}', 'SORTBY', 'record_id', 'DESC', 'LIMIT', '0', str(REDIS_LIMIT_MAX), 'RETURN', '0')
        if result[0] == 0:
            return False
        return True

    def clean_sigkill(self):
        result = self.cache.execute_command('FT.SEARCH', 'idx:output', '@content_type:{kill}', 'SORTBY', 'record_id', 'DESC', 'LIMIT', '0', str(REDIS_LIMIT_MAX), 'RETURN', '0')
        if result[0] == 0:
            return
        cleanups = result[1:]
        self.cache.delete(*cleanups)

    def reset_olds(self, run_id):
        if run_id in self.olds:
            del self.olds[run_id]

    def store(self, run_id, node_id, content, content_type='text', cleanup_ratio=1, dump=True, record_id=None, forgetive=True):
        run_id = int(run_id)
        node_id = str(node_id)
        record_id = int(record_id) if record_id is not None else idgen.generate(return_type='int')
        record = {'run_id': int(run_id), 'content': [to_dumpable(content)] if dump else content, 'content_type': content_type, 'node_id': node_id, 'record_id': record_id}
        self.cache.json().set(f'output:{run_id}:{node_id}:{record_id}', '$', record)
        if forgetive:
            runcount = self.store_counts.get(node_id, 0)
            runcount = runcount + 1
            if runcount % int(self.node_max_records / cleanup_ratio) == 0:
                self.node_outputs_limit_n(run_id, node_id, self.node_max_records, choice='text')
                self.node_clean_repeat_vars(run_id, node_id)
                if self.check_kill():
                    raise KernelInterrupted('execution interrupted.')
            if runcount % int(REDIS_LIMIT_MAX / 2) == 0:
                self.node_outputs_limit_n(run_id, node_id, int(REDIS_LIMIT_MAX / 2), choice='node_runned')
            self.store_counts[node_id] = runcount

    def get_n_outputs(self, run_id, node_id, n=None, choice='prints', dolog=True) -> dict[int, dict[str, str | dict]]:
        n = n or self.node_max_records
        run_id = int(run_id)
        node_id = str(node_id)
        node_id = node_id.replace('<', '\\<').replace('>', '\\>').replace('-', '\\-').replace('.', '\\.')
        typeselector = ''
        if choice == 'prints':
            typeselector = ' @content_type:{text|error|node_runned|inputed|prompt}'
        elif choice == 'vars':
            typeselector = ' @content_type:{var}'
        elif choice == 'stateonly':
            typeselector = ' @content_type:{node_runned|error}'
        elif choice == 'all':
            typeselector = ''
        elif choice == 'prompt':
            typeselector = ' @content_type:{prompt}'
        elif choice == 'inputed':
            typeselector = ' @content_type:{inputed}'
        result = self.cache.execute_command('FT.SEARCH', 'idx:output', f'@run_id:[{run_id} {run_id}] @node_id:{{{node_id}}} {typeselector}', 'SORTBY', 'record_id', 'DESC', 'LIMIT', '0', str(n), 'RETURN', '3', '$.content', 'record_id', '$.content_type')
        if result == [0] and dolog:
            logger.info('output记录查不到，run_id={}, node_id={},choice={}', run_id, node_id, choice)
            return {}
        retdata = {}
        for i in range(2, len(result), 2):
            record_id = result[i][1]
            content = result[i][3]
            content_type = result[i][5]
            try:
                content = json.loads(content)
                content = content[0]
            except:
                logger.info('content无法用json解析，尝试json5，run_id={}, node_id={}', run_id, node_id)
                print('buggy content:', content)
                try:
                    content = json5.loads(content)
                    content = content[0]
                except Exception as e:
                    logger.error('json5也无法解析，run_id={}, node_id={}，error:{}', run_id, node_id, e)
            retdata[record_id] = {'content': content, 'content_type': content_type}
        return retdata

    def consume_input(self, record_id):
        record_id = int(record_id)
        result = self.cache.execute_command('FT.SEARCH', 'idx:output', f'@record_id:[{record_id} {record_id}]', 'LIMIT', '0', '1', 'RETURN', '5', '$.content', '$.record_id', '$.content_type', '$.run_id', '$.node_id')
        if result == [0]:
            print(f'[WARN] consume_input没找到record_id: {record_id}', file=sys.__stderr__)
            return
        if result[0] > 1:
            print(f'[ERROR] 有bug，consume_input找到多条相同record_id: {result}', file=sys.__stderr__)
        result = result[1:]
        deldexs = [result[i] for i in range(0, len(result), 2)]
        content = json.loads(result[1][1])[0]
        if not result[1][5] == 'inputed':
            print(f'[ERROR] 有bug，consume_input找到的content_type不是inputed。record_id: {result}, content_type:{result[1][5]}', file=sys.__stderr__)
            return
        content['consumed'] = 1
        self.cache.delete(*deldexs)
        run_id = int(result[1][7])
        node_id = result[1][9]
        record = {'run_id': run_id, 'content': [content], 'content_type': 'inputed', 'node_id': node_id, 'record_id': record_id}
        self.cache.json().set(f'output:{run_id}:{node_id}:{record_id}', '$', record)

    def get_all_by_run_id(self, run_id, n_per_node=None, choice='prints') -> dict[str, dict[int, dict[str, str | dict]]]:
        n_per_node = n_per_node or self.node_max_records
        run_id = int(run_id)
        typeselector = ''
        if choice == 'prints':
            typeselector = ''
        elif choice == 'vars':
            typeselector = ' @content_type:{var}'
        elif choice == 'stateonly':
            typeselector = ' @content_type:{node_runned|error|prompt}'
        result = self.cache.execute_command('FT.SEARCH', 'idx:output', f'@run_id:[{run_id} {run_id}]{typeselector}', 'SORTBY', 'record_id', 'DESC', 'LIMIT', '0', str(REDIS_LIMIT_MAX), 'RETURN', '4', '$.node_id', '$.content', '$.record_id', '$.content_type')
        data = {}
        if result == [0]:
            return data
        for i in range(2, len(result), 2):
            node_id = result[i][1]
            record_id = result[i][5]
            content = result[i][3]
            content_type = result[i][7]
            if choice == 'prints' and content_type == 'var':
                continue
            try:
                content = json.loads(content)
                content = content[0]
            except:
                logger.info('content无法用json解析，尝试json5，run_id={}, node_id={}', run_id, node_id)
                print('buggy content:', content)
                try:
                    content = json5.loads(content)
                    content = content[0]
                except Exception as e:
                    logger.warning('json5也无法解析，run_id={}, node_id={}，error:{}', run_id, node_id, e)
            if not node_id in data:
                data[node_id] = {}
            data[node_id][record_id] = {'content': content, 'content_type': content_type}
        return data

    def get_all_available_run_ids(self, node_id=None, lbound=0):
        selector = '*'
        if node_id:
            selector = f'@node_id:{{{node_id}}}'.replace('<', '\\<').replace('>', '\\>').replace('-', '\\-').replace('.', '\\.')
        all_run_ids = []
        loopcnt = 0
        last_run_id = None
        while True:
            if loopcnt > 0:
                add_selector = f'@run_id:[{lbound} {last_run_id}]'
                if selector != '*':
                    selector = add_selector + ' ' + selector
                else:
                    selector = add_selector
            result = self.cache.execute_command('FT.AGGREGATE', 'idx:output', selector, 'GROUPBY', '1', '@run_id', 'REDUCE', 'COUNT', '0', 'SORTBY', '2', '@run_id', 'DESC')
            if result == [0]:
                break
            loopcnt = loopcnt + 1
            unique_run_ids = [int(r[1]) for r in result[1:]]
            assert sum(unique_run_ids) >= 0, unique_run_ids
            all_run_ids = all_run_ids + unique_run_ids
            last_run_id = unique_run_ids[-1] - 1
        return all_run_ids

    def trunk_to_n_runs(self, n, lbound=100000):
        exists = self.get_all_available_run_ids()
        exists = [x for x in exists if x > lbound]
        if len(exists) <= n:
            logger.debug('想保留{}条，但是记录才{}条，不用删', n, len(exists))
            return
        logger.debug('想保留{}条，记录共{}条，删一波', n, len(exists))
        trunk_at = exists[n]
        self.forget_except_lasts(trunk_at, lbound=lbound)

    def node_clean_repeat_vars(self, run_id, node_id):
        selector = f'@run_id:[{run_id} {run_id}] @node_id:{{{node_id}}} @content_type:{{var}}'
        try:
            result = self.cache.execute_command('FT.SEARCH', 'idx:output', selector.replace('<', '\\<').replace('>', '\\>').replace('-', '\\-').replace('.', '\\.'), 'SORTBY', 'record_id', 'DESC', 'LIMIT', '0', str(REDIS_LIMIT_MAX), 'RETURN', '1', '$.content')
        except Exception as e:
            print('!!! buggy REDIS_LIMIT_MAX, selector:', REDIS_LIMIT_MAX, selector, file=sys.__stderr__)
            traceback.print_exc()
            raise e
        needrecur = False
        if result[0] >= REDIS_LIMIT_MAX - 10:
            needrecur = True
        result = result[1:]
        varnames = set()
        dels = []
        for i in range(0, len(result), 2):
            vardic = json.loads(result[i + 1][1])[0]
            if vardic['name'] in varnames:
                dels.append(result[i])
            else:
                varnames.add(vardic['name'])
        if len(varnames) >= REDIS_LIMIT_MAX * 0.8:
            raise ValueError(f'Detected over {int(REDIS_LIMIT_MAX * 0.8)} vars on a single node ({node_id}), which is overwhelming. Try split into multiple nodes.')
        if dels:
            self.cache.delete(*dels)
        if needrecur:
            self.node_clean_repeat_vars(run_id, node_id)

    def node_outputs_limit_n(self, run_id, node_id, n, choice='text&var'):
        assert n <= REDIS_LIMIT_MAX / 2
        if choice == 'text&var':
            self.node_outputs_limit_n(run_id, node_id, n, choice='text')
            self.node_outputs_limit_n(run_id, node_id, n, choice='var')
            return
        assert choice in ('text', 'var', 'node_runned')
        selector = f'@run_id:[{run_id} {run_id}] @node_id:{{{node_id}}} @content_type:{{{choice}}}'
        result = self.cache.execute_command('FT.SEARCH', 'idx:output', selector.replace('<', '\\<').replace('>', '\\>').replace('-', '\\-').replace('.', '\\.'), 'SORTBY', 'record_id', 'DESC', 'LIMIT', '0', str(REDIS_LIMIT_MAX), 'RETURN', '2', 'record_id', '$.content_type')
        total_docs = result[0]
        docs = result[1:]
        if total_docs <= n:
            return
        todels = docs[2 * n:]
        todels = [todels[i] for i in range(0, len(todels), 2) if todels[i + 1][3] == choice]
        if todels:
            self.cache.delete(*todels)
        if total_docs >= REDIS_LIMIT_MAX * 0.9:
            self.node_outputs_limit_n(run_id, node_id, n, choice=choice)

    def forget_except_lasts(self, run_id, lbound=0):
        node_latests_getter = ['FT.AGGREGATE', 'idx:output', '*', 'GROUPBY', '1', '@node_id', 'REDUCE', 'MAX', '1', '@run_id', 'AS', 'latest_run_id']
        nodesresult = self.cache.execute_command(*node_latests_getter)
        nodeids = [n[1] for n in nodesresult[1:]]
        runids = [n[3] for n in nodesresult[1:]]

        def get_last_data(arun_id, anode_id):
            got = self.cache.execute_command('FT.SEARCH', 'idx:output', f'@run_id:[{arun_id} {arun_id}] @node_id:{{{anode_id}}}'.replace('<', '\\<').replace('>', '\\>').replace('-', '\\-').replace('.', '\\.'), 'SORTBY', 'record_id', 'DESC', 'LIMIT', '0', str(REDIS_LIMIT_MAX), 'RETURN', '5', '$.run_id', '$.node_id', '$.record_id', '$.content_type', '$.content')
            adata = [{'run_id': g[1], 'node_id': g[3], 'record_id': g[5], 'content_type': g[7], 'content': g[9]} for g in got[1:] if isinstance(g, list)]
            return adata
        with ThreadPoolExecutor(max_workers=configer.grapy.max_workers) as executor:
            bkps = list(executor.map(get_last_data, runids, nodeids))
        bkps = [item for sub in bkps for item in sub]
        self.forget_all_before(run_id, lbound=lbound)
        postnodes = self.cache.execute_command(*node_latests_getter)
        postnodeids = [n[1] for n in postnodes[1:]]
        gones = [n for n in nodeids if not n in postnodeids]
        putbacks = [b for b in bkps if b['node_id'] in gones]
        if len(putbacks) < REDIS_LIMIT_MAX / 3:
            logger.debug('为完全清空的{}个节点恢复{}条记录', len(gones), len(putbacks))
            with ThreadPoolExecutor(max_workers=configer.grapy.max_workers) as executor:
                executor.map(lambda x: self.store(x['run_id'], x['node_id'], x['content'], content_type=x['content_type'], dump=False, record_id=x['record_id'], forgetive=False), putbacks)
        else:
            logger.debug('完全清空的有{}个节点、{}条记录，记录数量过多，放弃恢复，一波带走', len(gones), len(putbacks))

    def forget_all_before(self, run_id, lbound=0):

        def _forget_all_before(run_id, lbound=0):
            run_id = int(run_id)
            index_name = 'idx:output'
            result = self.cache.execute_command('FT.SEARCH', index_name, f'@run_id:[{lbound} {run_id}]', 'LIMIT', '0', str(REDIS_LIMIT_MAX), 'RETURN', '0')
            total = result[0]
            keys_to_delete = result[1:]
            if not keys_to_delete:
                logger.debug('删除时已无匹配记录 (run_id <= {})', run_id)
                return
            deleted_count = len(keys_to_delete)
            self.cache.delete(*keys_to_delete)
            logger.debug('删除 {} 条记录 (run_id <= {})。将自递归到查不到符合删除标准的记录为止。注意后续可能为完全清空的节点恢复部分记录。', deleted_count, run_id)
            _forget_all_before(run_id, lbound=lbound)
        _forget_all_before(run_id, lbound=lbound)

    def _get_news(self, run_id, choice='prints') -> dict[str, dict[int, dict[str, str | dict]]]:
        if not run_id in self.olds:
            self.olds[run_id] = []
        allouts = self.get_all_by_run_id(run_id, choice=choice)
        news = {}
        for (node_id, records) in allouts.items():
            for (record_id, data) in records.items():
                if not record_id in self.olds[run_id]:
                    self.olds[run_id].append(record_id)
                    if not node_id in news:
                        news[node_id] = {}
                    news[node_id][record_id] = data
        return news

    def _get_new_nodes_states(self, run_id):
        news = self._get_news(run_id, choice='stateonly')
        retdic = {}
        for (nodeid, record) in news.items():
            runcount = [1 for v in record.values() if v['content_type'] == 'node_runned']
            runcount = len(runcount)
            errs = [v for v in record.values() if v['content_type'] == 'error']
            prompts = [v for v in record.values() if v['content_type'] == 'prompt']
            retdic[nodeid] = {'runned': runcount, 'errors': errs, 'prompts': prompts}
        return retdic

    def suicide(self):
        try:
            self.cache.execute_command('FT.DROPINDEX', 'idx:output')
            logger.info(f'自杀成功')
        except Exception as e:
            logger.error(f'自杀失败：{e}')
cache = CacheHandler()

def _disp_to_cache(run_id, *content, node_id=None, end=' ', content_type='text'):
    if node_id is None:
        raise TypeError(f'Missing node_id, which is mandatory')
    if content_type == 'text':
        toprint = end.join([str(c) for c in content])
    else:
        assert len(content) == 1, f'非text模式只支持一个content，因为必须保留格式。收到：{content}'
        toprint = to_dumpable(content[0])
    cache.store(run_id, node_id, toprint, content_type=content_type)

def _send_vars_to_cache(run_id, node_id, varsdic, cascns=None, cnskey=None, vars_tracking={}):
    cache.store(run_id, node_id, '', content_type='node_runned')
    restrict = 0 if run_id == 10 else TRACK_VARS_MAXLEN
    varinfo = vars_tracking.get(cnskey)
    option = 'untrack'
    varnames = []
    if varinfo:
        option = varinfo.get('option', 'untrack')
        varnames = varinfo.get('vars') or []
    for (k, v) in varsdic.items():
        if k == '_BUG_':
            print('[WARNING] 存变量出现_BUG_：', run_id, cnskey, node_id, v, file=sys.__stderr__)
        try:
            if cnskey is not None and isinstance(cascns, dict) and (not '.' in k):
                if not cnskey in cascns:
                    cascns[cnskey] = {}
                cascns[cnskey][k] = v
            if option == 'untrack':
                dotrack = True
                if '<ALL>' in varnames or k in varnames:
                    dotrack = False
                elif any([k.startswith(vn + '.') for vn in varnames]):
                    dotrack = False
            else:
                dotrack = False
                if '<ALL>' in varnames or k in varnames:
                    dotrack = True
                elif any([k.startswith(vn + '.') for vn in varnames]):
                    dotrack = True
            if dotrack:
                cache.store(run_id, node_id, {'name': k, 'type': type(v).__name__, 'repr': tostr(v, restrict=restrict)}, content_type='var')
        except Exception as e:
            print('[WARNING] 存变量失败：', run_id, node_id, k, e, file=sys.__stderr__)
            cache.store(run_id, node_id, {'name': k, 'type': '[unknown]', 'repr': '[unknown]'}, content_type='var')

def _input_via_cache(run_id, *prompt, node_id=None):
    if node_id is None:
        raise TypeError(f'Missing node_id, which is mandatory')
    prompt = ' '.join([str(p) for p in prompt])
    cache.store(run_id, node_id, prompt, content_type='prompt')
    time0 = time.time()
    while True:
        maybe_kill = cache.check_kill()
        if maybe_kill:
            raise KernelInterrupted(f'Inputer interrupted.')
        inputs = cache.get_n_outputs(run_id, node_id, choice='inputed', dolog=False)
        for (record_id, inpdic) in inputs.items():
            if inpdic['content']['prompt'] == prompt and inpdic['content']['consumed'] == 0:
                cache.consume_input(record_id)
                return inpdic['content']['input']
        time1 = time.time()
        if time1 - time0 > INPUT_TIMEOUT:
            logger.error('Input not received from user within {}s. Prompt: {}', INPUT_TIMEOUT, prompt)
            raise TimeoutError(f"Input timeout: At run_id {run_id} and node_id {node_id}, with prompt '{prompt}', no matching input received in {INPUT_TIMEOUT} seconds.")
        time.sleep(0.5)

def accept_user_input(run_id, node_id, prompt, inputed):
    cache.store(run_id, node_id, {'prompt': prompt, 'input': inputed, 'consumed': 0}, content_type='inputed')

def read_node_id():
    nodeid = None
    stack = inspect.stack()
    stack.reverse()
    codes = []
    for frame in stack:
        cline = frame.code_context
        codes.append(cline)
        if not cline:
            continue
        if isinstance(cline, list):
            cline = '\n'.join(cline)
        if not UID_COMMENT_LEFTLABEL in cline:
            continue
        nodeid = cline.split(UID_COMMENT_LEFTLABEL)[-1].split(UID_COMMENT_RIGHTLABEL)[0].strip()
        break
    return nodeid

class OutToCache:

    def __init__(self, run_id):
        self.run_id = run_id

    def write(self, text, nodeid=None):
        try:
            text = str(text)
            if SKIP_BLANK_PRINTS and (not text.strip()):
                return
            if len(text) > PRINT_MAXLEN:
                text = text[:PRINT_MAXLEN - 6] + '......'
            nodeid = nodeid or read_node_id()
            if not nodeid:
                print(f'[WARNING] 有bug或者源代码没挂好uid，stdout源头用户节点uid没找到。text:{text}', file=sys.__stderr__)
                return
            cache.store(self.run_id, nodeid, text, content_type='text')
        except Exception as e:
            print(f'[ERROR] 打印到缓存出错。text:{text}, e:{e}, {traceback.format_exc()}', file=sys.__stderr__)

    def flush(self):
        pass

class ErrToCache:

    def __init__(self, run_id):
        self.run_id = run_id
        self.printer = OutToCache(run_id)

    def write(self, text):
        try:
            if not text.strip():
                return
            if beyond312:
                parts = text.split(':')
                if len(parts) >= 4:
                    if parts[2].strip() == 'SyntaxWarning':
                        if parts[3].strip().startswith('invalid escape sequence'):
                            return
            text = '[ERROR]' + str(text)
            self.printer.write(text)
        except Exception as e:
            print(f'[ERROR] 打印报错到缓存出错。text:{text}', file=sys.__stderr__)

    def flush(self):
        pass

    def isatty(self):
        return False

class LoguruToCache:

    def __init__(self, run_id):
        self.run_id = run_id
        self.printer = OutToCache(run_id)

    def write(self, text):
        print(f'LoguruToCache text:{text}', file=sys.__stderr__)
        try:
            if not text.strip():
                return
            self.printer.write(text)
        except Exception as e:
            print(f'[ERROR] loguru到缓存出错。text:{text}', file=sys.__stderr__)

    def flush(self):
        pass

def plt_show2cache(plt, run_id):
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    markdown_img = f'![plot](data:image/png;base64,{img_base64})'
    nodeid = read_node_id()
    cache.store(run_id, nodeid, markdown_img, content_type='text')
    plt.clf()