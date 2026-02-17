import math
import requests
from consts import vbs
import numpy as np
from loguru import logger
import traceback
import datetime
from importlib.metadata import distributions
import json5
from rich.console import Console
from rich.text import Text
import io
import pandas as pd
import json

class Shared:
    tooldess: dict = {}
    dialogs: list = []
shared = Shared()

def gid_startswith(gid1, gid2):
    if not gid1.startswith(gid2):
        return False
    if gid1 == gid2:
        return True
    reminder = gid1[len(gid2):]
    if not reminder[0] == '.':
        return False
    return True

def sort_gids(chapter_numbers):

    def custom_sort_key(chapter_number):
        parts = [int(part) for part in chapter_number.split('.')]
        return parts
    sorted_chapters = sorted(chapter_numbers, key=custom_sort_key)
    return sorted_chapters
import copy

def remove_decisions(dag):
    origlen = len(dag)
    dag = copy.deepcopy(dag)
    dag_dict = {atask['task_id']: atask for atask in dag}
    dag_nodeci = []
    nodeci_invrela = {}
    for atask in dag:
        orig_tid = atask['task_id']
        atask = copy.deepcopy(atask)
        if atask['type'] == 'decision':
            continue
        if atask['parent_id'] in dag_dict.keys():
            if dag_dict[atask['parent_id']]['type'] == 'decision':
                atask['parent_id'] = dag_dict[atask['parent_id']]['parent_id']
        atask['parent_id'] = dag_dict[atask['parent_id']]['stable_id'] if atask['parent_id'] in dag_dict.keys() else atask['parent_id']
        atask['task_id'] = atask['stable_id']
        dependent_task_ids = []
        for adepid in atask['dependent_task_ids']:
            if adepid in dag_dict.keys():
                if not dag_dict[adepid]['type'] == 'decision':
                    dependent_task_ids.append(dag_dict[adepid]['stable_id'])
        atask['dependent_task_ids'] = dependent_task_ids
        dag_nodeci.append(atask)
        nodeci_invrela[atask['task_id']] = orig_tid
    return (dag_nodeci, origlen - len(dag_nodeci), nodeci_invrela)

def remove_comments(json_str, consider_multiline=False):
    json_str = re.sub('//.*', '', json_str)
    if consider_multiline:
        json_str = re.sub('/\\*.*?\\*/', '', json_str, flags=re.DOTALL)
    return json_str.strip()

def time14_to_readable(t: int) -> str:
    seconds = t // 10000
    tenth_millis = t % 10000
    microseconds = tenth_millis * 100
    local_tz = datetime.datetime.now().astimezone().tzinfo
    dt = datetime.datetime.fromtimestamp(seconds, tz=local_tz).replace(microsecond=microseconds)
    return dt.strftime('%Y%m%d-%H:%M:%S')[2:] + f'.{tenth_millis:04d}'

def safe_default(obj):
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
    if isinstance(obj, np.floating):
        if np.isnan(obj) or np.isinf(obj):
            return None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return repr(obj)
import time
import threading
import re

def estimate_tokens(sentence: str) -> int:
    sentence = str(sentence)
    chn_pattern = '[\\u4e00-\\u9fa5\\s.,?!+]+'
    eng_pattern = '[^\\u4e00-\\u9fa5\\s.,?!+]'
    remain_eng = re.sub(chn_pattern, ' ', sentence).strip()
    remain_chn = re.sub(eng_pattern, '', sentence).strip()
    remain_chn = re.sub('\\s+', '', remain_chn)
    len_remain_eng = 0 if remain_eng.strip() == '' else len(remain_eng.strip().split(' '))
    return len_remain_eng + len(remain_chn.strip())

def recursive_text_splitter(text, max_length, overlap=0):
    if overlap >= max_length:
        raise ValueError('Overlap must be less than max_length')
    separators = [('\n+', '换行'), ('[。！？]', '句号'), (',', '逗号'), ('[^\\w\\s]', '其他标点')]

    def split_by_separators(text, separators, current_separator_index=0, previous_overlap=''):
        if len(text) <= max_length:
            return [previous_overlap + text]
        if current_separator_index >= len(separators):
            return [previous_overlap + text[i:i + max_length - len(previous_overlap)] for i in range(0, len(text), max_length - len(previous_overlap))]
        separator_pattern, separator_name = separators[current_separator_index]
        parts = re.split(f'({separator_pattern})', text)
        result = []
        current_chunk = previous_overlap
        for part in parts:
            if not part:
                continue
            if re.match(separator_pattern, part):
                if current_chunk and len(current_chunk) + len(part) <= max_length:
                    current_chunk = current_chunk + part
                else:
                    if current_chunk:
                        result.append(current_chunk)
                    result.append(part)
                    current_chunk = ''
            elif len(current_chunk) + len(part) <= max_length:
                current_chunk = current_chunk + part
            else:
                sub_parts = split_by_separators(part, separators, current_separator_index + 1, current_chunk[-overlap:] if overlap else '')
                if current_chunk:
                    result.append(current_chunk)
                result.extend(sub_parts[:-1])
                current_chunk = sub_parts[-1]
        if current_chunk:
            result.append(current_chunk)
        return result
    chunks = split_by_separators(text, separators, previous_overlap='')
    final_chunks = []
    for i, chunk in enumerate(chunks):
        if i > 0 and overlap > 0:
            final_chunks.append(chunks[i - 1][-overlap:] + chunk)
        else:
            final_chunks.append(chunk)
    return final_chunks

def suppress_tokens(context, limit):
    assert limit >= 0
    if isinstance(context, list):
        if len(context) > 3:
            while estimate_tokens(str(context)) > limit:
                context = context[:int(len(context) / 1.2)]
                if len(context) < 3:
                    break
    toolout = str(context)
    origlen = estimate_tokens(toolout)
    if origlen <= limit:
        return context
    if limit < 6:
        return ''
    if limit < 50:
        return toolout[:limit] + ' ...略'
    step = int(limit / 10)
    toolout_left = toolout[:int(len(toolout) / 2)]
    toolout_right = toolout[int(len(toolout) / 2):]
    while estimate_tokens(toolout_left) + estimate_tokens(toolout_right) > limit:
        toolout_left = toolout_left[:-step]
        toolout_right = toolout_right[step:]
    return toolout_left[:-5] + ' ...略... ' + toolout_right[4:]

def extract_section(text: str, key: str) -> str | None:
    escaped_key = re.escape(key)
    pattern = f'\\[<{escaped_key}>\\]:\\s*(.*?)(?=\\[<[^>\\]]+>\\]:|$)'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        content = match.group(1)
        return content.rstrip()
    return None

def custout2dict(text, lower=False):
    argnames = re.findall('\\[<[^>\\]]+>\\]', text, re.DOTALL)
    argnames = [a[2:-2] for a in argnames]
    outdic = {}
    for a in argnames:
        v = extract_section(text, a)
        outdic[a] = v
    if lower:
        outdic = {a.lower(): v for a, v in outdic.items()}
    return outdic

def list_installed_packages():
    packages = []
    for dist in distributions():
        packages.append({'name': dist.metadata['Name'], 'version': dist.version, 'summary': dist.metadata.get('Summary', '')})
    return sorted(packages, key=lambda x: x['name'].lower())

def format_pkgs_for_llm():
    pkgs = list_installed_packages()
    pkgstr = ''
    for pkg in pkgs:
        pkgstr = pkgstr + f"{pkg['name']}=={pkg['version']}; "
    return pkgstr

class SnowflakeGenerator:

    def __init__(self, datacenter_id, worker_id):
        self.datacenter_id = datacenter_id
        self.worker_id = worker_id
        self.sequence = 0
        self.last_timestamp = -1
        self.datacenter_id_bits = 5
        self.worker_id_bits = 5
        self.sequence_bits = 12
        self.max_datacenter_id = -1 ^ -1 << self.datacenter_id_bits
        self.max_worker_id = -1 ^ -1 << self.worker_id_bits
        self.max_sequence = -1 ^ -1 << self.sequence_bits
        self.worker_id_shift = self.sequence_bits
        self.datacenter_id_shift = self.sequence_bits + self.worker_id_bits
        self.timestamp_shift = self.sequence_bits + self.worker_id_bits + self.datacenter_id_bits
        self.lock = threading.Lock()

    def _current_milliseconds(self):
        return int(time.time() * 1000)

    def _til_next_millis(self, last_timestamp):
        timestamp = self._current_milliseconds()
        while timestamp <= last_timestamp:
            timestamp = self._current_milliseconds()
        return timestamp

    def generate_id(self):
        with self.lock:
            timestamp = self._current_milliseconds()
            if timestamp < self.last_timestamp:
                raise ValueError('Clock moved backwards. Refusing to generate id.')
            if timestamp == self.last_timestamp:
                self.sequence = self.sequence + 1 & self.max_sequence
                if self.sequence == 0:
                    timestamp = self._til_next_millis(self.last_timestamp)
            else:
                self.sequence = 0
            self.last_timestamp = timestamp
            return timestamp - 1288834974657 << self.timestamp_shift | self.datacenter_id << self.datacenter_id_shift | self.worker_id << self.worker_id_shift | self.sequence

def generate_unique_id(prefix: str, datacenter_id: int, worker_id: int) -> str:
    generator = SnowflakeGenerator(datacenter_id, worker_id)
    snowflake_id = generator.generate_id()
    return int(snowflake_id)

def get_unused_name(name: str, exnames: list, remember=True, unusables=['if', 'else', 'for', 'while', 'and', 'or', 'index', 'item']):
    i = 1
    newname = name
    while newname in exnames + unusables:
        newname = name + f'_{i}'
        i = i + 1
    if remember:
        exnames.append(newname)
    return newname

def repair_condinfo(oldinfo):
    newconds = [{'nodes': [k], 'condition': v} for k, v in oldinfo.items()]
    logic = list(oldinfo.keys())
    logic = [str(l) for l in list(range(len(logic)))]
    logic = ' && '.join(logic)
    return {'conditions': newconds, 'logic': logic}

def upgrade_condition_infos(dag):
    dag = copy.deepcopy(dag)
    for node in dag:
        node['condition_info'] = repair_condinfo(node['condition_info'])
    return dag

def dict_try_del(dic, keys):
    dic = copy.deepcopy(dic)
    for k in keys:
        if k in dic:
            del dic[k]
    return dic

def table_get_record(table: list[dict], condition: dict, ensure_unique=False, need_indices=False):
    ret = []
    indices = []
    for i, record in enumerate(table):
        met = True
        for k, v in condition.items():
            if record.get(k) != v:
                met = False
                break
        if met:
            ret.append(copy.deepcopy(record))
            indices.append(i)
    if ensure_unique:
        assert len(ret) == 1, (ret, condition)
    if not need_indices:
        return ret
    else:
        return (ret, indices)

def table_multi_get(table: list[dict], conditions: list[dict], ensure_unique=False, need_indices=False):
    got = []
    indices = []
    for condition in conditions:
        subgot, subids = table_get_record(table, condition, ensure_unique=ensure_unique, need_indices=True)
        got = got + subgot
        indices = indices + subids
    if len(indices) > 1:
        sorted_pairs = sorted(zip(indices, got), key=lambda pair: pair[0])
        indices, got = zip(*sorted_pairs)
    if need_indices:
        return (got, indices)
    return got

def table_unique_get(table: list[dict], condition: dict, return_pointer=False):
    got, idx = table_get_record(table, condition, ensure_unique=True, need_indices=True)
    got = got[0]
    idx = idx[0]
    if not return_pointer:
        return got
    else:
        return table[idx]

def table_lambda_get(table: list[dict], condfunc: callable, return_pointers=False):
    met = []
    if not return_pointers:
        table = copy.deepcopy(table)
    for record in table:
        try:
            if condfunc(record):
                met.append(record)
        except:
            pass
    return met

def is_expr(formation):
    varpattern = re.compile('\\$\\{[^}]+\\}')
    if not isinstance(formation, str):
        return False
    try:
        formation = re.sub(varpattern, lambda match: '0', formation)
    except Exception as e:
        raise ValueError(e)
    isexpr = False
    try:
        eval(formation)
        isexpr = True
    except SyntaxError:
        isexpr = False
    except NameError:
        isexpr = False
    except TypeError:
        isexpr = True
    except Exception:
        isexpr = False
    logger.debug(f'{formation} isexpr: {isexpr}')
    return isexpr

def remove_common_indents(code):
    basedent = len(code.lstrip('\n')) - len(code.lstrip())
    if basedent == 0:
        return code
    clines = code.split('\n')
    newlines = []
    for l in clines:
        if not l[:basedent].strip():
            newlines.append(l[basedent:])
        else:
            newlines.append(l)
    code = '\n'.join(newlines)
    return code

def redact_quoted_strings(text):
    q3 = "'''(?:[^'\\\\]|\\\\.)*?'''"
    pattern = f'''\n        (?:\n            """(?:[^"\\\\]|\\\\.)*?"""   # 三双引号字符串（支持内部转义）\n            |\n            {q3}\n            |\n            "(?:[^"\\\\]|\\\\.)*?"       # 双引号字符串\n            |\n            '(?:[^'\\\\]|\\\\.)*?'       # 单引号字符串\n        )\n    '''

    def replacer(match):
        return ''.join(['\n' if c == '\n' else 'X' for c in match.group(0)])
    return re.sub(pattern, replacer, text, flags=re.DOTALL | re.VERBOSE)

def parse_for_sugs(code, row, col, sugtype):
    codelines = code.split('\n')
    focusline = codelines[row]
    partitioners = [' ', '\n', '(', ')', '[', ']', '{', '}', '+', '-', '*', '/', '%', '=', ',']
    already_params = []
    targrow = row
    if sugtype == 'objs' or sugtype == 'kernel_objs':
        lastpart = focusline[:col]
    elif sugtype == 'params' or sugtype == 'kernel_params':
        cleancode = redact_quoted_strings(code)
        cleanlines = cleancode.split('\n')
        fulllines = cleanlines[:row]
        targlinepart = cleanlines[row][:col]
        left = '\n'.join(fulllines) + '\n' + targlinepart
        if '(' in left:
            left, argpart = left.rsplit('(', 1)
            targrow = left.count('\n')
            left = left.strip()
            for p in partitioners:
                left = left.split(p)[-1]
            lastpart = left.strip()
            paramparts = argpart.split('=')[:-1]
            for aparam in paramparts:
                for p in partitioners[1:]:
                    aparam = aparam.split(p)[-1]
                already_params.append(aparam.strip())
        else:
            lastpart = '<UNDEFINED>'
    else:
        raise
    for p in partitioners:
        lastpart = lastpart.split(p)[-1]
    return (lastpart, targrow, already_params)

def requests_post(url, data):
    rsp = requests.post(url=url, json=data)
    if rsp.status_code == 200:
        rsp = rsp.json()
    else:
        raise RuntimeError(f'rsp returned status code {rsp.status_code}')
    if isinstance(rsp, str):
        rsp = json5.loads(rsp)
    return rsp

def pretty_repr(s: str, indent=4) -> str:
    if not s.strip():
        return s
    result = []
    i = 0
    depth = 0
    in_single = False
    in_double = False
    in_triple_single = False
    in_triple_double = False

    def current_indent():
        return ' ' * (depth * indent)
    while i < len(s):
        c = s[i]
        if not in_triple_single and (not in_triple_double):
            if s[i:i + 3] == "'''":
                in_triple_single = True
                result.append("'''")
                i = i + 3
                continue
            elif s[i:i + 3] == '"""':
                in_triple_double = True
                result.append('"""')
                i = i + 3
                continue
        if in_triple_single:
            if s[i:i + 3] == "'''":
                in_triple_single = False
                result.append("'''")
                i = i + 3
                continue
            else:
                result.append(c)
                i = i + 1
                continue
        if in_triple_double:
            if s[i:i + 3] == '"""':
                in_triple_double = False
                result.append('"""')
                i = i + 3
                continue
            else:
                result.append(c)
                i = i + 1
                continue
        if not in_single and (not in_double):
            if c == "'":
                in_single = True
                result.append(c)
                i = i + 1
                continue
            elif c == '"':
                in_double = True
                result.append(c)
                i = i + 1
                continue
        else:
            if in_single:
                if c == '\\' and i + 1 < len(s):
                    result.append(c + s[i + 1])
                    i = i + 2
                    continue
                elif c == "'":
                    in_single = False
                result.append(c)
                i = i + 1
                continue
            if in_double:
                if c == '\\' and i + 1 < len(s):
                    result.append(c + s[i + 1])
                    i = i + 2
                    continue
                elif c == '"':
                    in_double = False
                result.append(c)
                i = i + 1
                continue
        if c in '{[(':
            result.append(c)
            depth = depth + 1
            result.append('\n' + current_indent())
            i = i + 1
        elif c in '}])':
            depth = max(0, depth - 1)
            result.append('\n' + current_indent() + c)
            i = i + 1
        elif c == ',':
            result.append(c)
            result.append('\n' + current_indent())
            i = i + 1
        elif c in ' \t\r\n':
            i = i + 1
        else:
            token = ''
            while i < len(s):
                ch = s[i]
                if ch in '{[()]}.,:' and (not (in_single or in_double or in_triple_single or in_triple_double)):
                    break
                if ch in ' \t\r\n' and (not (in_single or in_double or in_triple_single or in_triple_double)):
                    pass
                token = token + ch
                i = i + 1
            if token.strip():
                result.append(token)
    output = ''.join(result)
    return output.rstrip()
rich_console = Console(file=io.StringIO(), width=100, force_jupyter=False)

def _pretty_format_rich(obj):
    with rich_console.capture() as capture:
        rich_console.print(obj, highlight=False, soft_wrap=False)
    return capture.get().rstrip()

def enrich_by_type(data, dtype='dict', enrichable_len=99999):
    if not isinstance(data, str):
        return data
    if len(data) > enrichable_len:
        logger.info('超长数据跳过enrich。dtype={}, 长度达到{}', dtype, len(data))
        return data
    if dtype in ('dict', 'list', 'tuple', 'DataFrame', 'Series'):
        try:
            data = json.loads(data)
            if dtype == 'DataFrame':
                data = pd.DataFrame(data['data'], columns=data['columns'], index=data.get('index'))
            elif dtype == 'Series':
                data = pd.Series(data['values'], index=data['index'], name=data['name'])
            logger.debug('格式化特定类型的data：{}', dtype)
        except Exception as e:
            logger.warning('格式化data出错。data: {}', data)
            traceback.print_exc()
    return tostr(data)

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
        for k, subdata in data.items():
            newdata[k] = to_dumpable(subdata)
        return newdata
    try:
        return str(data)
    except:
        return '[UNREPRESENTABLE]'

def tostr(data, restrict=-1):
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

class CountedLogger:

    def __init__(self, retry_fail_thres=100):
        self.error_count = 0
        self.i = -1
        self.max_exception_tries = 0
        self.max_error_tries = 0
        self.log_depth = 3
        self._log = []
        self.logs = []
        self.presults = []
        self.error_count = 0
        self.error_counts = []
        self.completions = []
        self.current_retry_closed = True
        self.retry_fail_thres = retry_fail_thres

    def close_retry(self, presult, completed):
        assert not self.current_retry_closed
        self.presults.append(copy.deepcopy(presult))
        self.logs.append(copy.deepcopy(self._log))
        self.error_counts.append(self.error_count)
        if self.error_count > self.retry_fail_thres:
            completed = False
        self.completions.append(completed)
        print('———————————————————— CountedLogger close_retry() metadata ————————————————————')
        print('round:', self.i)
        print('all historical completeds:', self.completions)
        print('log:', self._log)
        print('all historical error_counts:', self.error_counts)
        print('——————————————————————————————————————————————————————————————————————————————')
        self.reset_state_vars()
        self.current_retry_closed = True

    def create_new_retry(self):
        assert self.current_retry_closed
        self.error_count = 0
        self._log = []
        self.current_retry_closed = False
        self.i = self.i + 1

    def trace(self, msg):
        logger.trace(self.wrap(msg), depth=self.log_depth)

    def debug(self, msg):
        logger.debug(self.wrap(msg), depth=self.log_depth)

    def info(self, msg):
        logger.info(self.wrap(msg), depth=self.log_depth)

    def warning(self, msg, weight=0):
        logger.warning(self.wrap(msg), depth=self.log_depth)
        self.error_count = self.error_count + weight
        self._log.append({'level': 'warning', 'msg': msg})

    def error(self, msg, weight=1):
        logger.error(self.wrap(msg), depth=self.log_depth)
        self.error_count = self.error_count + weight
        self._log.append({'level': 'error', 'msg': msg})

    def wrap(self, msg):
        msg = '【retry' + str(self.i) + '/(' + str(self.max_exception_tries) + '+' + str(self.max_error_tries) + ')】 ' + str(msg)
        return msg.replace('{', '{{').replace('}', '}}')

    def reset_state_vars(self):
        self.error_count = 0
        self._log = []

    def export_logs(self):
        return copy.deepcopy(self.logs)

    def export_presults(self):
        return copy.deepcopy(self.presults)

    def get_best_result(self):
        assert self.current_retry_closed, '等本轮重试跑完了才能取结果'
        assert len(self.error_counts) == len(self.presults) and len(self.completions) == len(self.presults)
        if not True in self.completions:
            logger.critical(f'由于没有任何一轮重试成功跑完，无法取得可用的结果')
            raise RuntimeError(f'由于没有任何一轮重试成功跑完，无法取得可用的结果')
        errcounts = [self.error_counts[i] if self.completions[i] == True else 999999 for i in range(len(self.completions))]
        bestdex = np.argmin(self.error_counts)
        return self.presults[bestdex]
from functools import wraps

def retry(max_exception_tries=3, max_error_tries=2, retry_fail_thres=100):

    def decorator(func):

        @wraps(func)
        async def wrapper(*args, **kwargs):
            counted_logger = CountedLogger(retry_fail_thres=retry_fail_thres)
            counted_logger.max_exception_tries = max_exception_tries
            counted_logger.max_error_tries = max_error_tries
            kwargs = {**kwargs, **{'logger': counted_logger}}
            exception_count = 0
            has_error_count = 0
            for i in range(max(max_exception_tries, max_error_tries)):
                logs = counted_logger.export_logs()
                presults = counted_logger.export_presults()
                print(f'### retry() decorator i={i}, last logs: {logs}')
                counted_logger.create_new_retry()
                try:
                    rsp = await func(*args, **{**kwargs, **{'presults': presults, 'logs': logs}})
                    if counted_logger.error_count == 0:
                        logger.info(f'未发现error，直接返回结果')
                        counted_logger.close_retry(rsp, True)
                        break
                    has_error_count = has_error_count + 1
                    if has_error_count == max_error_tries:
                        logger.error(f'重试第{has_error_count}次还是有{counted_logger.error_count}个error，放弃重试')
                        counted_logger.close_retry(rsp, True)
                        break
                    else:
                        logger.warning(f'重试第{has_error_count}次还是有{counted_logger.error_count}个error，继续重试')
                        counted_logger.close_retry(rsp, True)
                except Exception as e:
                    exception_count = exception_count + 1
                    counted_logger.close_retry(None, False)
                    if exception_count == max_exception_tries:
                        logger.critical(f'【非包】重试失败第{exception_count}次，放弃执行：{e}')
                        raise RuntimeError(f'重试失败第{exception_count}次，放弃执行：{e}')
                        traceback.print_exc()
                    else:
                        logger.error(f'【非包】重试失败第{exception_count}次：{e}')
                        traceback.print_exc()
            bestry = counted_logger.get_best_result()
            return bestry
        return wrapper
    return decorator

class Bouncer:

    def __init__(self, taggenerator=lambda x: 'defaulttag'):
        self.lastdatas = {}
        self.lastrets = {}
        self.taggen = taggenerator

    def debounce(self, keys, expected_ret_types, compare_func=lambda x1, x2: x1 == x2, skipper=lambda *args, **kwargs: False, intercepted_kws=[]):

        def decorator(func):

            @wraps(func)
            async def wrapper(*args, **kwargs):
                thisdata = {}
                tag = self.taggen(*args, **kwargs)
                assert type(tag) in (int, str), tag
                for k in keys:
                    if isinstance(k, int):
                        if k >= len(args):
                            logger.error('key超过了args长度：{}', k, len(args))
                            continue
                        thisdata[k] = args[k]
                    elif isinstance(k, str):
                        if not k in kwargs:
                            logger.debug('key不在kwargs中：{}', k, kwargs.keys())
                            continue
                        thisdata[k] = kwargs[k]
                    else:
                        logger.error('key必须是int或str: {}', k)
                do_update = True
                issame = False
                if callable(skipper):
                    if skipper(*args, **kwargs):
                        logger.debug('满足跳过防抖的条件，强制调用目标函数。tag={}', tag)
                        issame = False
                        do_update = False
                if do_update:
                    if not tag in self.lastdatas or not tag in self.lastrets:
                        logger.trace('bouncer发现新tag：{}', tag)
                        issame = False
                    else:
                        try:
                            compare_kwargs = kwargs.get('compare_kwargs', {})
                            issame = compare_func(self.lastdatas.get(tag), thisdata, **compare_kwargs)
                            print('issame,tag,thisdata, lastdata:', issame, tag, '\n', thisdata, '\n', self.lastdatas.get(tag))
                        except Exception as e:
                            logger.warning(f'不支持比较, tag={tag}：{e}')
                            print('tag, thisdata, lastdata:\n', tag, '\n', thisdata, '\n', self.lastdatas.get(tag))
                            traceback.print_exc()
                            issame = False
                if issame:
                    if not type(self.lastrets.get(tag)) in expected_ret_types:
                        logger.info(f'注意上轮返回格式不在本次预设的{expected_ret_types}之中，不使用上轮记录。tag={tag}')
                        issame = False
                if issame:
                    logger.debug('检测到跟上次一样的输入，直接返回上次结果。tag={}', tag)
                    return self.lastrets.get(tag)
                else:
                    ret = await func(*args, **{k: v for k, v in kwargs.items() if not k in ['compare_kwargs'] + intercepted_kws})
                    if type(ret) in expected_ret_types:
                        if do_update:
                            logger.trace('跟上次不一样的输入，执行成功后更新上轮记录. tag={}', tag)
                            self.lastdatas[tag] = thisdata
                            self.lastrets[tag] = ret
                        else:
                            logger.trace('跟上次不一样的输入，但由于skipper，执行成功后也不更新上轮记录. tag={}', tag)
                    else:
                        logger.debug(f'注意返回格式不在预设的{expected_ret_types}之中，不覆盖上轮记录。tag={tag}')
                    return ret
            return wrapper
        return decorator