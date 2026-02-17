
"""
Copyright (c) 2026 Zhiren Chen. Provided as-is for local use only.
"""

import sys
import re
import ast
import json
import numpy
import pandas
import copy
import traceback
import re
import builtins
from loguru import logger
import asyncio
import numpy as np
from core.cft.consts import DOT_REPL, vbs
from basic.db.mysql_handler import isnan
import random
import time
import threading
from core.cft.consts import SKIPNAMES

def table_get_record(n69wspa34p: list[dict], condition: dict, ensure_unique=False, need_indices=False):
    n69wsp9onl = []
    n69wsp9omv = []
    for i, record in enumerate(n69wspa34p):
        met = True
        for k, v in condition.items():
            if record.get(k) != v:
                met = False
                break
        if met:
            n69wsp9onl.append(copy.deepcopy(record))
            n69wsp9omv.append(i)
    if ensure_unique:
        assert len(n69wsp9onl) == 1, (n69wsp9onl, condition)
    if not need_indices:
        return n69wsp9onl
    else:
        return (n69wsp9onl, n69wsp9omv)

def table_multi_get(n69wspa34p: list[dict], conditions: list[dict], ensure_unique=False, need_indices=False):
    n69wsp9otg = []
    n69wsp9omv = []
    for condition in conditions:
        subgot, subids = table_get_record(n69wspa34p, condition, ensure_unique=ensure_unique, need_indices=True)
        n69wsp9otg = n69wsp9otg + subgot
        n69wsp9omv = n69wsp9omv + subids
    if len(n69wsp9omv) > 1:
        n69zpbl4xk = sorted(zip(n69wsp9omv, n69wsp9otg), key=lambda pair: pair[0])
        n69wsp9omv, n69wsp9otg = zip(*n69zpbl4xk)
    if need_indices:
        return (n69wsp9otg, n69wsp9omv)
    return n69wsp9otg

def table_unique_get(n69wspa34p: list[dict], condition: dict, return_pointer=False):
    n69wsp9otg, idx = table_get_record(n69wspa34p, condition, ensure_unique=True, need_indices=True)
    n69wsp9otg = n69wsp9otg[0]
    idx = idx[0]
    if not return_pointer:
        return n69wsp9otg
    else:
        return n69wspa34p[idx]

def table_lambda_get(n69wspa34p: list[dict], b69wsp9mop: callable, return_pointers=False):
    met = []
    if not return_pointers:
        n69wspa34p = copy.deepcopy(n69wspa34p)
    for record in n69wspa34p:
        try:
            if b69wsp9mop(record):
                met.append(record)
        except:
            pass
    return met

def get_unused_name(name: str, exnames: list, remember=True, unusables=['if', 'else', 'for', 'while', 'and', 'or', 'index', 'item']):
    i = 1
    n69wspa2dd = name
    while n69wspa2dd in exnames + unusables:
        n69wspa2dd = name + f'_{i}'
        i = i + 1
    if remember:
        exnames.append(n69wspa2dd)
    return n69wspa2dd

def list_endswith(path, comparator):
    assert isinstance(comparator, list)
    if len(comparator) == 0:
        return True
    if len(path) < len(comparator):
        return False
    n69wsp9p2p = path[-len(comparator):]
    ignoredexs = [i for i in range(len(comparator)) if comparator[i] == set()]
    n69wsp9p2p = [n69wsp9p2p[i] if not i in ignoredexs else set() for i in range(len(n69wsp9p2p))]
    if n69wsp9p2p == comparator:
        return True
    return False

def get_keychain_by_cond(n69wspa2mh, b69wsp9mop, blocked_chains=[], advance_blockers=[]):
    if not n69wspa2mh:
        return []
    if isinstance(n69wspa2mh, tuple):
        n69wspa2mh = [n for n in n69wspa2mh]
    assert type(n69wspa2mh) in (list, dict, set), n69wspa2mh
    n69wspa2mh = copy.deepcopy(n69wspa2mh)
    found_chains = []
    try:
        if b69wsp9mop(n69wspa2mh):
            found_chains.append([])
    except:
        pass

    def _peek(n69wspa2mh, current_chain):
        if current_chain in blocked_chains:
            return
        for ab in advance_blockers:
            try:
                if ab['cond'](n69wspa2mh):
                    if ab['block'] == []:
                        return
                    if not access_nested_data(n69wspa2mh, ab['block'], nonexist=None) is None:
                        set_nested_data(n69wspa2mh, ab['block'], None, op='=', inplace=True)
            except Exception as e:
                traceback.print_exc()
        nonlocal found_chains

        def _peek_one(i, item, current_chain, found_chains):
            n69wsp9ozo = copy.deepcopy(current_chain)
            n69wsp9ozo.append(i)
            try:
                if b69wsp9mop(item):
                    if not n69wsp9ozo in blocked_chains:
                        found_chains.append(copy.deepcopy(n69wsp9ozo))
            except Exception as e:
                traceback.print_exc()
            return n69wsp9ozo
        if isinstance(n69wspa2mh, list) or isinstance(n69wspa2mh, set):
            for i, item in enumerate(n69wspa2mh):
                n69wsp9ozo = _peek_one(i, item, current_chain, found_chains)
                _peek(item, n69wsp9ozo)
        elif isinstance(n69wspa2mh, dict):
            for i, item in n69wspa2mh.items():
                n69wsp9ozo = _peek_one(i, item, current_chain, found_chains)
                _peek(item, n69wsp9ozo)
    _peek(n69wspa2mh, [])
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
        elif isinstance(current_level, list) or isinstance(current_level, set) or isinstance(current_level, tuple):
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

def brutal_gets(n69wspa2mh, b69wsp9mop, blocked_chains=[], advance_blockers=[]):
    if not n69wspa2mh:
        return ([], [])
    keychains = get_keychain_by_cond(n69wspa2mh, b69wsp9mop, blocked_chains=blocked_chains, advance_blockers=advance_blockers)
    gots = []
    for keychain in keychains:
        gots.append(access_nested_data(n69wspa2mh, keychain))
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
        traceback.print_exc()
    if not inplace:
        return data

def remove_sublists(lists):
    lists.sort(key=len, reverse=True)
    n69wsp9oq8 = []
    for current in lists:
        can_keep = True
        for existing in lists:
            if not current == existing:
                if len(current) >= len(existing) and current[:len(existing)] == existing:
                    can_keep = False
                    break
            elif current in n69wsp9oq8:
                can_keep = False
                break
        if can_keep:
            n69wsp9oq8.append(current)
    return n69wsp9oq8

def find_nth(s, substring, n):
    n69wsp9p0q = 0
    for i in range(n):
        pos = s.find(substring, n69wsp9p0q)
        if pos == -1:
            return -1
        n69wsp9p0q = pos + 1
    return pos

def extend_child_hid(n69wspa333, n69wspa2oe):
    n69wsp9p51 = n69wspa333 + '.001'
    n69wsp9p51 = ensure_unique_hid(n69wsp9p51, n69wspa2oe)
    return n69wsp9p51

def x69xm5dtzx(n69wspa2wq):
    n69wspa2vg = n69wspa2wq.rfind('^')
    n69wspa2ox = n69wspa2wq.rfind('#')
    n69wspa2sg = n69wspa2wq.rfind('/')
    n69wspa2lm = n69wspa2wq.rfind('*')
    cates = ['node', 'cond', 'func', 'class']
    dexs = [n69wspa2vg, n69wspa2ox, n69wspa2sg, n69wspa2lm]
    if sum(dexs) == -4:
        n69wspa33b = 'folder'
    else:
        n69wspa2xx = np.argmax(dexs)
        n69wspa33b = cates[n69wspa2xx]
    return n69wspa33b

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
        return {_recover(k): _recover(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_recover(m) for m in data]
    else:
        return data

def loadstr(k):
    if isinstance(k, str):
        k = k.strip()
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
        for k, v in data.items():
            k = loadstr(k)
            newdata[k] = v
        return newdata
    else:
        return loadstr(data)

class SimpleIDGenerator:

    def __init__(self):
        self.last_ms = 0
        self.lock = threading.Lock()

    def generate(self, n69wspa2xv='', return_type='str'):
        with self.lock:
            ms = int(time.time() * 10000)
            while ms <= self.last_ms:
                ms = int(time.time() * 10000)
                ms = ms + 1
            self.last_ms = ms
            if return_type == 'str':
                return f'{n69wspa2xv}{ms}'
            else:
                assert not n69wspa2xv
                return ms
idgen = SimpleIDGenerator()

def gen_base36_id(n69wspa2xv=''):
    id10 = idgen.generate(return_type='int')
    id36 = int_to_base36(id10)
    return f'{n69wspa2xv}{id36}'

def int_to_base36(n: int) -> str:
    if n < 0:
        raise ValueError('输入必须是非负整数')
    if n == 0:
        return '0'
    chars = '0123456789abcdefghijklmnopqrstuvwxyz'
    n69wsp9oq8 = []
    while n > 0:
        n, remainder = divmod(n, 36)
        n69wsp9oq8.append(chars[remainder])
    return ''.join(reversed(n69wsp9oq8))

def base36_to_int(s: str) -> int:
    s = s.strip().lower()
    if not s:
        raise ValueError('输入不能为空')
    return int(s, 36)

def is_valid_varname(s: str) -> bool:
    if not s:
        return False
    n69wsp9omn = '^[a-zA-Z_][a-zA-Z0-9_]*$'
    return bool(re.match(n69wsp9omn, s)) and (not s in SKIPNAMES)

def gen_case_id(used):
    assert len(used) < 900, f'分支大于900个，太多了'
    i = random.randint(0, 999)
    while i in used:
        i = random.randint(0, 999)
    used.add(i)
    return i

def x69xm5du01(n69wspa38w):
    assert '.' in n69wspa38w
    assert not n69wspa38w.endswith('.py')
    sects = n69wspa38w.split('.')
    n69wspa322 = '>'.join(sects[:-1])
    return n69wspa322 + '/' + sects[-1]

def x69xm5dtzw(n69wspa38w):
    if not '.' in n69wspa38w:
        return n69wspa38w
    return x69xm5du01(n69wspa38w)

def to_func_id(n69wspa2wq):
    n69wspa2sg = n69wspa2wq.rfind('/')
    n69wspa2vg = n69wspa2wq.rfind('^')
    n69wspa2lm = n69wspa2wq.rfind('*')
    if n69wspa2sg < 0:
        raise ValueError(f'目标def_id根本不含func：{n69wspa2wq}')
    if n69wspa2vg < n69wspa2sg and n69wspa2lm < n69wspa2sg:
        return n69wspa2wq
    cutdexs = [c for c in (n69wspa2vg, n69wspa2lm) if c > n69wspa2sg]
    cutdex = min(cutdexs)
    return n69wspa2wq[:cutdex]

def to_module_id(n69wspa2wq):
    if '^' in n69wspa2wq:
        return n69wspa2wq.split('^')[0]
    assert x69xm5dtzx(n69wspa2wq) == 'func', n69wspa2wq
    assert n69wspa2wq.count('/') == 1, n69wspa2wq
    return n69wspa2wq

def replace_lastpart(n69wspa2wq, separator, repfrom='.', repto=DOT_REPL):
    if not separator in n69wspa2wq:
        return n69wspa2wq
    left, right = n69wspa2wq.rsplit(separator, 1)
    if repfrom == '.':
        assert not '^' in right, f"不安全的替换'.'：{n69wspa2wq}，可能影响hid。"
    right = right.replace(repfrom, repto)
    return left + separator + right

def x69xm5du02(n69wspa2l4, n69wsp9osv):
    n69wspa2it = n69wspa2l4['def_id']
    n69wspa2ws = {}
    for p in n69wsp9osv['inputs']:
        n69wspa5n7 = {}
        n69wspa5n7['type'] = (p.get('type') or '').strip()
        n69wspa5n7['doc'] = (p.get('doc') or '').strip()
        n69wspa5n7['default'] = (p.get('default') or '').strip()
        n69wspa2ws[p['name']] = n69wspa5n7
    n69wsp9onl = {}
    n69wsp9onl['type'] = (n69wsp9osv['return'].get('type') or '').strip()
    n69wsp9onl['doc'] = (n69wsp9osv['return'].get('doc') or '').strip()
    newfdic = {}
    newfdic['is_async'] = n69wspa2l4['is_async']
    newfdic['doc'] = (n69wspa2l4.get('doc') or '').strip()
    newfdic['inputs'] = n69wspa2ws
    newfdic['return'] = n69wsp9onl
    newfdic['def_id'] = n69wspa2it
    newfdic['source_file'] = (n69wspa2l4.get('source_file') or '').strip()
    newfdic['raw_def_id'] = n69wspa2l4.get('raw_def_id', n69wspa2it)
    return newfdic

def x69xm5du03(n69wspa2l4, n69wsp9osv, need_defid=True, need_srcfile=True, n69wspa2z4=[], rawclass=''):
    n69wspa2it = n69wspa2l4['def_id']
    pdesc = ''
    for p in n69wsp9osv['inputs']:
        pdstr = ''
        if (p.get('type') or '').strip():
            pdstr = pdstr + f"({p['type']}) "
        if (p.get('doc') or '').strip():
            pdstr = pdstr + p['doc']
        if (p.get('default') or '').strip():
            pdstr = pdstr + f" (default={p['default']})"
        if not pdstr:
            pdstr = 'undescribed'
        pdesc = pdesc + f"  {p['name']}: {pdstr}\n"
    if not pdesc:
        pdesc = 'No params.\n'
    rdesc = ''
    if (n69wsp9osv['return'].get('type') or '').strip():
        rdesc = rdesc + f"({n69wsp9osv['return']['type']}) "
    if (n69wsp9osv['return'].get('doc') or '').strip():
        rdesc = rdesc + n69wsp9osv['return']['doc']
    if not rdesc:
        rdesc = 'undescribed'
    is_async = n69wspa2l4['is_async']
    n69wsp9p3c = n69wspa2l4['doc']
    n69wsp9p3c = n69wsp9p3c + '\n' if (n69wsp9p3c or '').strip() else 'This function lacks description.\n'
    n69wsp9p3i = n69wspa2it.split('*')[-1].replace(DOT_REPL, '.')
    if n69wsp9p3i.count('/') == 1:
        n69wsp9oox = n69wsp9p3i.replace('/', '.')
    else:
        n69wsp9oox = n69wsp9p3i.split('/')[-1]
    asinfo = x69xm5du04(n69wspa2l4['def_id'], n69wspa2z4=n69wspa2z4)
    rawinfo = x69xm5du04(n69wspa2l4['raw_def_id'], n69wspa2z4=n69wspa2z4)
    n69wsp9onh = rawinfo['class']
    fstr = rawinfo['func']
    if not rawinfo['class']:
        if asinfo['func'] != rawinfo['func']:
            fstr = fstr + f" (alias as {asinfo['func'].replace(DOT_REPL, '.')})"
    if n69wsp9onh:
        fstr = fstr + ' under'
        fstr = fstr + (' class ' + n69wsp9onh)
        if n69wsp9onh != rawclass:
            fstr = fstr + f' (base of {rawclass})'
    fstr = fstr + (' from module ' + rawinfo['module'])
    fcate = 'Async Function' if is_async else 'Function'
    defidpart = f" at {n69wspa2it.replace(DOT_REPL, '.')}" if need_defid else ''
    n69wspa34o = (n69wspa2l4.get('source_file') or '').strip()
    srcpart = f' from {n69wspa34o}' if need_srcfile and (not n69wspa34o in ('', '<UNK>')) else ''
    n69wspa30m = f'{fcate}: {fstr}{defidpart}{srcpart}\nDesc: {n69wsp9p3c}Params:\n{pdesc}Return: {rdesc}'
    return n69wspa30m

def x69xm5du04(n69wspa2wq, n69wspa2z4):
    assert x69xm5dtzx(n69wspa2wq) in ('func', 'class')
    rmod = '[UNKNOWN]'
    n69wsp9ole = ''
    n69wsp9p3n = ''
    rfid = n69wspa2wq
    if any([rfid.startswith(v + '/') or rfid.startswith(v + '*') for v in n69wspa2z4]):
        rmod = '[LOCAL]'
    else:
        if rfid.startswith('[ENV]'):
            rfid = rfid[6:]
        rmod = rfid.split('^')[0].replace('/', '.').replace('>', '.')
    if '*' in n69wspa2wq:
        right = rfid.split('*')[-1]
        if right.count('/') == 0:
            assert x69xm5dtzx(right) == 'folder'
            n69wsp9ole = right
        elif right.count('/') == 1:
            n69wsp9ole, n69wsp9p3n = right.split('/')
        else:
            n69wsp9ole = ''
            n69wsp9p3n = right.split('/')[-1]
            assert x69xm5dtzx(n69wsp9p3n) == 'folder'
    else:
        n69wsp9p3n = n69wspa2wq.split('/')[-1]
        assert x69xm5dtzx(n69wsp9p3n) == 'folder'
    return {'module': rmod, 'class': n69wsp9ole, 'func': n69wsp9p3n}

def x69xm5du05(n69wspa2l4, need_defid=True, need_srcfile=True, n69wspa2z4=[], rawclass=''):
    return x69xm5du03(n69wspa2l4, {'inputs': [{'name': n69wsp9p3n, **inp} for n69wsp9p3n, inp in n69wspa2l4['inputs'].items()], 'return': n69wspa2l4['return']}, need_defid=need_defid, need_srcfile=need_srcfile, n69wspa2z4=n69wspa2z4, rawclass=rawclass)

def x69xm5du06(n69wspa2us, need_defid=True, n69wspa2z4=[]):
    rawinfo = x69xm5du04(n69wspa2us['raw_def_id'], n69wspa2z4=n69wspa2z4)
    asinfo = x69xm5du04(n69wspa2us['def_id'], n69wspa2z4=n69wspa2z4)
    n69wspa37g = 'class ' + rawinfo['class']
    if not rawinfo['class'] == asinfo['class']:
        n69wspa37g = n69wspa37g + f" (alias as {asinfo['class']})"
    n69wspa37g = n69wspa37g + f" from module {rawinfo['module']}"
    if need_defid:
        n69wspa37g = n69wspa37g + (' at ' + n69wspa2us['def_id'])
    n69wspa37g = n69wspa37g + (': \n' + (n69wspa2us.get('doc') or 'No description.').strip() + '\n')
    if n69wspa2us.get('sitepkg_bases'):
        n69wspa37g = n69wspa37g + 'This class is a child class of imported env classes: '
        sbstrs = []
        for sb in n69wspa2us['sitepkg_bases']:
            sbstrs.append(f"class {sb['classname']} at module {sb['module']}")
        n69wspa37g = n69wspa37g + ('; '.join(sbstrs) + '.\n')
    if n69wspa2us.get('funcs'):
        n69wspa37g = n69wspa37g + 'attr funcs:\n'
    for n69wsp9p3n, n69wspa2l4 in n69wspa2us.get('funcs', {}).items():
        fdesc = x69xm5du05(n69wspa2l4, need_defid=need_defid, need_srcfile=False, n69wspa2z4=n69wspa2z4, rawclass=rawinfo['class'])
        n69wspa37g = n69wspa37g + (fdesc + '\n\n')
    return n69wspa37g.strip()

def all_desc_to_nl(ddic):
    funcdesc = '# ------ Functions ------\n' if ddic.get('funcs') else ''
    for n69wsp9p3n, n69wspa2l4 in ddic.get('funcs', {}).items():
        fstr = '## ' + x69xm5du05(n69wspa2l4, need_defid=False, need_srcfile=False, n69wspa2z4=ddic.get('visibility', []))
        funcdesc = funcdesc + (fstr.strip() + '\n\n')
    classdesc = '# ------ Classes ------\n' if ddic.get('classes') else ''
    for n69wsp9ole, n69wspa2us in ddic.get('classes', {}).items():
        n69wspa37g = '## ' + x69xm5du06(n69wspa2us, need_defid=False, n69wspa2z4=ddic.get('visibility', []))
        classdesc = classdesc + (n69wspa37g.strip() + '\n\n')
    objdesc = '# ------ Objects ------\n' if ddic.get('objs') else ''
    for n69wspa2ce, odic in ddic.get('objs', {}).items():
        objdesc = objdesc + ('## ' + n69wspa2ce + '\ntype: ' + x69xm5du06(odic['class'], need_defid=False, n69wspa2z4=ddic.get('visibility', [])).strip() + '\n\n')
    alldesc = funcdesc + '\n' + classdesc + '\n' + objdesc
    alldesc = alldesc.replace(DOT_REPL, '.')
    return alldesc

def x69xm5du07(n69wspa5m7, filepath, extpkgs):
    if n69wspa5m7.startswith('[ENV]'):
        classpath = filepath or '<UNK>'
        if not any([classpath.startswith(extp) for extp in extpkgs]):
            return True
        if '/python3.' in filepath:
            return True
    return False

def x69xm5du00(n69wsp9p0w, n69wsp9ool):
    if not n69wsp9p0w == 'tool':
        return n69wsp9ool
    else:
        n69wsp9ool = n69wsp9ool.copy()
        for k in n69wsp9ool:
            if isinstance(n69wsp9ool.get(k), str):
                n69wsp9ool[k] = n69wsp9ool[k].replace('.', DOT_REPL)
        return n69wsp9ool

def x69xm5dtzz(n69wsp9p0w, n69wsp9ool):
    if not n69wsp9p0w == 'tool':
        return n69wsp9ool
    else:
        n69wsp9ool = n69wsp9ool.copy()
        for k in n69wsp9ool:
            if isinstance(n69wsp9ool.get(k), str):
                n69wsp9ool[k] = n69wsp9ool[k].replace(DOT_REPL, '.')
        return n69wsp9ool

def trunk_to_func(n69wspa2wq):
    assert '/' in n69wspa2wq and (not n69wspa2wq.endswith('/'))
    left, right = n69wspa2wq.rsplit('/', 1)
    n69wsp9p3n = right.split('*')[0].split('#')[0].split('^')[0]
    return left + '/' + n69wsp9p3n

def x69xm5du08(rootid, n69wspa38w):
    rootid = rootid.strip().replace('>', '.').rstrip('.')
    n69wspa38w = n69wspa38w.strip().replace('>', '.').replace('/', '.').rstrip('.')
    oldmodname = n69wspa38w
    n69wspa38w = n69wspa38w.lstrip('.')
    n69wspa36u = max(len(oldmodname) - len(n69wspa38w) - 1, 0)
    n69wspa2j4 = ''
    if not rootid and n69wspa36u > 0:
        return '<UNDEFINED>'
    if not rootid:
        return n69wspa38w
    if rootid.count('.') < n69wspa36u - 1:
        return '<UNDEFINED>'
    if rootid.count('.') == n69wspa36u - 1:
        n69wspa2j4 = n69wspa38w
    elif n69wspa36u > 0:
        rootsecs = rootid.split('.')[:-n69wspa36u]
        uped_root = '.'.join(rootsecs)
        n69wspa2j4 = uped_root + '.' + n69wspa38w
    else:
        n69wspa2j4 = rootid + '.' + n69wspa38w
    return n69wspa2j4

def x69xm5dtzy(n69wsp9onk, n69wsp9p0f, n69wspa34s):
    imptarg = n69wsp9onk if n69wsp9onk else n69wsp9p0f
    impisclass = True if n69wsp9onk else False
    imporfrom = 'import' if '.' in imptarg else 'from'
    for impline in n69wspa34s.split('\n'):
        if not impline.startswith(imporfrom):
            continue
        if imporfrom == 'from':
            if ' as ' in impline:
                linetarg = impline.split(' as ')[-1]
            else:
                linetarg = impline.split(' import ')[-1]
            if linetarg == imptarg.split('.')[0]:
                return impline
        else:
            if ' as ' in impline:
                linetarg = impline.split(' as ')[-1]
            else:
                linetarg = impline[7:].strip()
            if imptarg == linetarg.rsplit('.', 1)[0]:
                return impline
    return '<NOT_FOUND>'

def x69xm5du09(n69wsp9p12, codeswaps):
    for n65d20cda3, code in codeswaps.items():
        n69wsp9p12.loc[(n69wsp9p12['uid'] == n65d20cda3) & (n69wsp9p12['node_type'] == 'code'), 'code'] = code
if __name__ == '__main__':

    async def gen_id():
        print(idgen.generate('x'))

    async def run():
        await asyncio.gather(*[gen_id(), gen_id(), gen_id(), gen_id(), gen_id(), gen_id(), gen_id(), gen_id(), gen_id(), gen_id(), gen_id(), gen_id(), gen_id(), gen_id(), gen_id(), gen_id()])
    asyncio.run(run())