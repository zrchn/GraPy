import math
import sys
import re
import ast
import json
import json5
import numpy
import pandas
import copy
import traceback
import re
import builtins
from loguru import logger
import asyncio
import numpy as np
import random
import string
import time
import threading
import pandas as pd

def table_get_record(table: list[dict], condition: dict, ensure_unique=False, need_indices=False):
    ret = []
    indices = []
    for (i, record) in enumerate(table):
        met = True
        for (k, v) in condition.items():
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
        (subgot, subids) = table_get_record(table, condition, ensure_unique=ensure_unique, need_indices=True)
        got = got + subgot
        indices = indices + subids
    if len(indices) > 1:
        sorted_pairs = sorted(zip(indices, got), key=lambda pair: pair[0])
        (indices, got) = zip(*sorted_pairs)
    if need_indices:
        return (got, indices)
    return got

def table_unique_get(table: list[dict], condition: dict, return_pointer=False):
    (got, idx) = table_get_record(table, condition, ensure_unique=True, need_indices=True)
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

def get_unused_name(name: str, exnames: list, remember=True, unusables=['if', 'else', 'for', 'while', 'and', 'or', 'index', 'item']):
    i = 1
    newname = name
    while newname in exnames + unusables:
        newname = name + f'_{i}'
        i = i + 1
    if remember:
        exnames.append(newname)
    return newname

def list_endswith(path, comparator):
    assert isinstance(comparator, list)
    if len(comparator) == 0:
        return True
    if len(path) < len(comparator):
        return False
    tail = path[-len(comparator):]
    ignoredexs = [i for i in range(len(comparator)) if comparator[i] == set()]
    tail = [tail[i] if not i in ignoredexs else set() for i in range(len(tail))]
    if tail == comparator:
        return True
    return False

def get_keychain_by_cond(node, condfunc, blocked_chains=[], advance_blockers=[]):
    if not node:
        return []
    assert type(node) in (list, dict, set), node
    node = copy.deepcopy(node)
    found_chains = []
    try:
        if condfunc(node):
            found_chains.append([])
    except:
        pass

    def _peek(node, current_chain):
        if current_chain in blocked_chains:
            return
        for ab in advance_blockers:
            try:
                if ab['cond'](node):
                    if ab['block'] == []:
                        return
                    if not access_nested_data(node, ab['block'], nonexist=None) is None:
                        set_nested_data(node, ab['block'], None, op='=', inplace=True)
            except Exception as e:
                logger.info(f'高级屏蔽cond抛出异常:{e}。虽然可能不影响结果，但会导致速度变慢。')
                traceback.print_exc()
        nonlocal found_chains

        def _peek_one(i, item, current_chain, found_chains):
            subchain = copy.deepcopy(current_chain)
            subchain.append(i)
            try:
                if condfunc(item):
                    if not subchain in blocked_chains:
                        found_chains.append(copy.deepcopy(subchain))
            except Exception as e:
                logger.info(f'condfunc抛出异常:{e}。虽然可能不影响结果，但会导致速度变慢。')
                traceback.print_exc()
            return subchain
        if isinstance(node, list) or isinstance(node, set):
            for (i, item) in enumerate(node):
                subchain = _peek_one(i, item, current_chain, found_chains)
                _peek(item, subchain)
        elif isinstance(node, dict):
            for (i, item) in node.items():
                subchain = _peek_one(i, item, current_chain, found_chains)
                _peek(item, subchain)
    _peek(node, [])
    return found_chains

def access_nested_data(data, keychain, nonexist='raise'):
    if not keychain:
        return data
    current_level = copy.deepcopy(data)
    for key in keychain:
        if isinstance(current_level, dict):
            if key in current_level.keys():
                current_level = current_level[key]
            elif nonexist != 'raise':
                return nonexist
            else:
                raise ValueError('access_nested_dict() received invalid key')
        elif isinstance(current_level, list) or isinstance(current_level, set):
            if isinstance(key, int):
                if len(current_level) > key:
                    current_level = current_level[key]
                elif nonexist != 'raise':
                    return nonexist
                else:
                    raise ValueError('access_nested_dict() received invalid index')
            elif nonexist != 'raise':
                return nonexist
            else:
                raise ValueError('access_nested_dict() received invalid key')
        elif nonexist != 'raise':
            return nonexist
        else:
            raise ValueError('data structure no longer recursible')
    return current_level

def brutal_gets(node, condfunc, blocked_chains=[], advance_blockers=[]):
    if not node:
        return ([], [])
    keychains = get_keychain_by_cond(node, condfunc, blocked_chains=blocked_chains, advance_blockers=advance_blockers)
    gots = []
    for keychain in keychains:
        gots.append(access_nested_data(node, keychain))
    return (keychains, gots)

def set_nested_data(data, keychain, value, op='=', inplace=False):
    if not inplace:
        data = copy.deepcopy(data)
    setters = [f"""[{('"' + k + '"' if isinstance(k, str) else k)}]""" for k in keychain]
    lc = '\n'
    lct = '\\n'
    if op == '=':
        setter = f"""data{''.join(setters)} = {(value if not isinstance(value, str) else '"' + value.replace(lc, lct) + '"')}"""
    elif op == 'append':
        setter = f"""data{''.join(setters)}.append({(value if not isinstance(value, str) else '"' + value.replace(lc, lct) + '"')})"""
    elif op == '+':
        setter = f"""data{''.join(setters)} += {(value if not isinstance(value, str) else '"' + value.replace(lc, lct) + '"')}"""
    elif op.startswith('insert:'):
        place = op[7:]
        assert place.isdigit()
        place = int(place)
        setter = f"""data{''.join(setters)}.insert({place},{(value if not isinstance(value, str) else '"' + value.replace(lc, lct) + '"')})"""
    elif op.startswith('hash:'):
        info = op[5:].split(':')
        key = info[0]
        ktype = 'str'
        if len(info) > 1:
            ktype = info[1]
        ktype = ktype.strip()
        setter = f"""data{''.join(setters)}[{(key if ktype in ('int', 'float') else '"' + key + '"')}] = {(value if not isinstance(value, str) else '"' + value.replace(lc, lct) + '"')}"""
    try:
        exec(setter)
    except Exception as e:
        logger.error(f'链式赋值失败：{e}')
        traceback.print_exc()
    if not inplace:
        return data

def remove_sublists(lists):
    lists.sort(key=len, reverse=True)
    result = []
    for current in lists:
        can_keep = True
        for existing in lists:
            if not current == existing:
                if len(current) >= len(existing) and current[:len(existing)] == existing:
                    can_keep = False
                    break
            elif current in result:
                can_keep = False
                break
        if can_keep:
            result.append(current)
    return result

def find_nth(s, substring, n):
    start = 0
    for i in range(n):
        pos = s.find(substring, start)
        if pos == -1:
            return -1
        start = pos + 1
    return pos

def extend_child_hid(parent_hid, used_hids):
    hid = parent_hid + '.001'
    hid = ensure_unique_hid(hid, used_hids)
    return hid

def ensure_unique_hid(hid, used_hids):
    last = hid.split('.')[-1]
    header = '' if not '.' in hid else hid[:hid.rfind('.') + 1]
    for i in range(40000):
        if hid in used_hids or last == 'end' or last.startswith('0') or (last == 'for'):
            if i == 39999:
                raise ValueError(f'节点数量是否过多？无法产生不重复的hid。used_hids数量：{len(used_hids)}')
            last = str(int(last) + 1)
            hid = header + last
        else:
            break
    used_hids.append(hid)
    return hid

def identify_full_id(def_id):
    hatdex = def_id.rfind('^')
    welldex = def_id.rfind('#')
    slashdex = def_id.rfind('/')
    stardex = def_id.rfind('*')
    cates = ['node', 'cond', 'func', 'class']
    dexs = [hatdex, welldex, slashdex, stardex]
    if sum(dexs) == -4:
        cate = 'folder'
    else:
        maxdex = np.argmax(dexs)
        cate = cates[maxdex]
    logger.debug(f'{def_id}指向的种类为{cate}')
    return cate

async def aidle(n=1, default=None, **kwargs):
    if n <= 1:
        return default
    else:
        return (default,) * n

def _recover(data):
    if isinstance(data, str):
        if data == 'true':
            return True
        elif data == 'false':
            return False
        elif data.isdigit():
            return int(data)
        else:
            return data
    elif isinstance(data, dict):
        return {_recover(k): _recover(v) for (k, v) in data.items()}
    elif isinstance(data, list):
        return [_recover(m) for m in data]
    else:
        return data

def loadstr(k):
    if isinstance(k, str):
        if k in ('true', 'True'):
            k = True
        elif k in ('false', 'False'):
            k = False
        elif k.isdigit():
            k = int(k)
    return k

def recover_conds(data, table_name, field):
    if not table_name == 'nodes':
        return data
    if not field in ('cases', 'childs', 'branch'):
        return data
    if field in ('cases', 'childs'):
        newdata = {}
        for (k, v) in data.items():
            k = loadstr(k)
            newdata[k] = v
        return newdata
    else:
        return loadstr(data)

class SimpleIDGenerator:

    def __init__(self):
        self.last_ms = 0
        self.lock = threading.Lock()

    def generate(self, prefix='', return_type='str'):
        with self.lock:
            ms = int(time.time() * 10000)
            while ms <= self.last_ms:
                ms = int(time.time() * 10000)
                ms = ms + 1
            self.last_ms = ms
            if return_type == 'str':
                return f'{prefix}{ms}'
            else:
                assert not prefix
                return ms
idgen = SimpleIDGenerator()

def gen_base36_id(prefix=''):
    id10 = idgen.generate(return_type='int')
    id36 = int_to_base36(id10)
    return f'{prefix}{id36}'

def int_to_base36(n: int) -> str:
    if n < 0:
        raise ValueError('输入必须是非负整数')
    if n == 0:
        return '0'
    chars = '0123456789abcdefghijklmnopqrstuvwxyz'
    result = []
    while n > 0:
        (n, remainder) = divmod(n, 36)
        result.append(chars[remainder])
    return ''.join(reversed(result))

def base36_to_int(s: str) -> int:
    s = s.strip().lower()
    if not s:
        raise ValueError('输入不能为空')
    return int(s, 36)

def is_valid_varname(s: str) -> bool:
    if not s:
        return False
    pattern = '^[a-zA-Z_][a-zA-Z0-9_]*$'
    return bool(re.match(pattern, s)) and (not s in SKIPNAMES)

def gen_case_id(used):
    assert len(used) < 900, f'分支大于900个，太多了'
    i = random.randint(0, 999)
    while i in used:
        i = random.randint(0, 999)
    used.add(i)
    return i

def df_to_safe_dict(df: pd.DataFrame) -> dict:
    return {'columns': df.columns.astype(str).tolist(), 'index': df.index.astype(str).tolist(), 'data': {str(col): df[col].tolist() for col in df.columns}}

def series_to_safe_dict(s: pd.Series) -> dict:
    return {'name': str(s.name) if s.name is not None else None, 'index': s.index.astype(str).tolist(), 'values': s.tolist()}

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

def stringify_dict_keys(obj):
    if isinstance(obj, dict):
        return {str(k): stringify_dict_keys(v) for (k, v) in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [stringify_dict_keys(item) for item in obj]
    else:
        return obj

class Statics:
    pass
statics = Statics()
statics.run_id = 0
if __name__ == '__main__':

    async def gen_id():
        print(idgen.generate('x'))

    async def run():
        await asyncio.gather(*[gen_id(), gen_id(), gen_id(), gen_id(), gen_id(), gen_id(), gen_id(), gen_id(), gen_id(), gen_id(), gen_id(), gen_id(), gen_id(), gen_id(), gen_id(), gen_id()])
    asyncio.run(run())