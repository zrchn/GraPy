
"""
Copyright (c) 2026 Zhiren Chen. Provided as-is for local use only.
"""

import asyncio
import copy
import sys
import traceback
import types
import numpy as np
import pandas as pd
from core.cft.consts import DOT_REPL, SKIPNAMES
import os
import importlib.util
import inspect
from pathlib import Path
from typing import Any, Dict, List, get_origin, get_args, Annotated, Optional
from typing_extensions import Doc
from core.cft.utils import idgen, x69xm5du01, x69xm5dtzx, gen_base36_id
exec(f'from core.cft.processors{sys.version_info.minor} import *')
from loguru import logger

def b69wspa5ab(n69wspa2wq, n69wspa5mk, n69wspa381):
    n69wspa2jx = n69wspa2wq
    n69wspa5n6 = n69wspa2jx.find('/')
    n69wspa2sg = n69wspa2jx.find('/', n69wspa5n6 + 1)
    n69wspa2lm = n69wspa2jx.find('*')
    if n69wspa2sg == -1:
        n69wspa2sg = 99999
    if n69wspa2lm == -1:
        n69wspa2lm = 99999
    if n69wspa2lm > n69wspa2sg:
        n69wspa2xv = '/'
    elif n69wspa2lm < n69wspa2sg:
        n69wspa2xv = '*'
    else:
        return '<UNK>'
    left, right = n69wspa2jx.rsplit(n69wspa2xv, 1)
    if not left.count('^1#_') == 1:
        return '<UNK>'
    n69wspa38w = left.split('^1#_')[0]
    if n69wspa38w.startswith('[ENV]/') or n69wspa38w.startswith('[ENV]>'):
        n69wspa38w = n69wspa38w[6:]
    elif not n69wspa381:
        pass
    elif n69wspa38w.startswith(n69wspa381 + '/') or n69wspa38w.startswith(n69wspa381 + '>'):
        n69wspa38w = n69wspa38w[len(n69wspa381) + 1:]
    else:
        pass
    n69wspa38w = n69wspa38w.replace('>', '.').replace('/', '.')
    n69wspa5mh = n69wspa5mk.get(n69wspa38w)
    if not n69wspa5mh:
        if not '._' in n69wspa38w and (not n69wspa38w.startswith('_')):
            pass
        return '<UNK>'
    right = n69wspa5mh.replace('.', DOT_REPL) + DOT_REPL + right
    n69wspa5mv = left + n69wspa2xv + right
    return n69wspa5mv

def b69wspa5ag(n69wspa2wq, n69wsp9oqn, n69wspa5mk, n69wspa381):
    n69wspa2jx = n69wspa2wq
    if '^' in n69wspa2jx or '*' in n69wspa2jx or n69wspa2jx.count('/') != 1:
        return '<UNK>'
    n69wspa38w = n69wspa2jx
    if n69wspa38w.startswith('[ENV]/') or n69wspa38w.startswith('[ENV]>'):
        n69wspa38w = n69wspa38w[6:]
    elif not n69wspa381:
        pass
    elif n69wspa38w.startswith(n69wspa381 + '/') or n69wspa38w.startswith(n69wspa381 + '>'):
        n69wspa38w = n69wspa38w[len(n69wspa381) + 1:]
    else:
        pass
    n69wspa38w = n69wspa38w.replace('>', '.').replace('/', '.')
    n69wspa5mh = n69wspa5mk.get(n69wspa38w)
    if not n69wspa5mh:
        return '<UNK>'
    n69wsp9onl = n69wspa5mh.replace('.', DOT_REPL) + DOT_REPL + n69wsp9oqn
    return n69wsp9onl

def b69wspa5ah(n69wspa2dr, imports_code, n69wspa381):
    assert not 'from ' in imports_code, 'eim001'
    n69wspa5mk = {}
    n69wspa5m9 = b69wsp9mq1(imports_code)
    for n69wspa2ga in n69wspa5m9['body']:
        n69wspa5mh = n69wspa2ga['names'][0]['asname'] or n69wspa2ga['names'][0]['name']
        n69wspa38w = n69wspa2ga['names'][0]['name']
        n69wspa5mk[n69wspa38w] = n69wspa5mh
    for k in n69wspa2dr:
        if len(n69wspa2dr[k]) == 0:
            continue
        if k != 'objs':
            n69wspa2dr[k]['def_id'] = n69wspa2dr[k]['def_id'].apply(lambda x: b69wspa5ab(x, n69wspa5mk, n69wspa381))
        else:
            n69wspa2dr[k]['name'] = n69wspa2dr[k].apply(lambda n69wspa2xo: b69wspa5ag(n69wspa2xo['def_id'], n69wspa2xo['name'], n69wspa5mk, n69wspa381), axis=1)
    return n69wspa2dr

def b69wspa5am(n69wspa2dr, n69wspa5np=None):
    if not n69wspa5np:
        n69wspa5np = sys.path

    def b69wspa5ai(n69wspa2xo):
        n69wspa2wq = n69wspa2xo['def_id']
        if '*' in n69wspa2wq:
            left, right = n69wspa2wq.rsplit('*', 1)
            n69wspa2jj = n69wspa2wq
            if '/' in right:
                n69wsp9ole, n69wsp9p3n = right.split('/')
                n69wspa5mc = n69wsp9ole.split(DOT_REPL)[-1]
                n69wspa2jj = left + '*' + n69wspa5mc + '/' + n69wsp9p3n
            else:
                n69wspa5mc = right.split(DOT_REPL)[-1]
                n69wspa2jj = left + '*' + n69wspa5mc
            if n69wspa2xo.get('homeclass'):
                if n69wspa5mc != n69wspa2xo.get('homeclass'):
                    n69wspa34o = n69wspa2xo['source_file'].split('.')[0]
                    for sp in n69wspa5np:
                        if n69wspa34o.startswith(sp + '/'):
                            n69wspa34o = n69wspa34o[len(sp) + 1:]
                    n69wspa5l9 = n69wspa34o.split('/')
                    n69wspa34o = '>' + '>'.join(n69wspa5l9[:-1]) + '/' + n69wspa5l9[-1] if len(n69wspa5l9) > 1 else '/' + n69wspa5l9[0]
                    n69wspa2jj = f"[ENV]{n69wspa34o}^1#_*{n69wspa2xo['homeclass']}"
                    if '/' in right:
                        n69wspa2jj = n69wspa2jj + ('/' + right.split('/')[-1])
            return n69wspa2jj
        elif '/' in n69wspa2wq:
            left, right = n69wspa2wq.rsplit('/', 1)
            n69wspa5mi = right.split(DOT_REPL)[-1]
            return left + '/' + n69wspa5mi
        return n69wspa2wq
    for k in ('funcs', 'classes'):
        if len(n69wspa2dr[k]) == 0:
            n69wspa2dr[k]['raw_def_id'] = ''
            continue
        n69wspa2dr[k]['raw_def_id'] = n69wspa2dr[k].apply(b69wspa5ai, axis=1)
    n69wspa2dr['objs']['rawname'] = n69wspa2dr['objs']['name'].apply(lambda x: x.split(DOT_REPL)[-1])
    if None in n69wspa2dr['funcs']['raw_def_id'].tolist() or np.nan in n69wspa2dr['funcs']['raw_def_id'].tolist():
        raise

def b69wspa5a9(n69wspa31q, n69wspa5mo, n69wspa5np=None):
    if not n69wspa5np:
        n69wspa5np = sys.path
    n69wspa32e = {}
    n69wspa2mw = n69wspa5mo.split('\n')
    for n69wsp9ou8 in n69wspa2mw:
        if not n69wsp9ou8.strip():
            continue
        if not ' as ' in n69wsp9ou8:
            continue
        left, right = n69wsp9ou8.split(' as ')
        left = left.split(' import ')[-1].strip()
        right = right.strip()
        n69wspa32e[right] = left

    def b69wspa5ao(n69wspa2xo):
        n69wspa2wq = n69wspa2xo['def_id']
        if '*' in n69wspa2wq:
            left, right = n69wspa2wq.rsplit('*', 1)
            n69wspa2jj = n69wspa2wq
            if '/' in right:
                n69wsp9ole, n69wsp9p3n = right.split('/')
                n69wspa5mc = n69wspa32e.get(n69wsp9ole, n69wsp9ole)
                n69wspa2jj = left + '*' + n69wspa5mc + '/' + n69wsp9p3n
            else:
                n69wspa5mc = n69wspa32e.get(right, right)
                n69wspa2jj = left + '*' + n69wspa5mc
            if n69wspa2xo.get('homeclass'):
                if n69wspa2xo['homeclass'] != n69wspa5mc:
                    n69wspa34o = n69wspa2xo['source_file'].split('.')[0]
                    for sp in n69wspa5np:
                        if n69wspa34o.startswith(sp + '/'):
                            n69wspa34o = n69wspa34o[len(sp) + 1:]
                    n69wspa5l9 = n69wspa34o.split('/')
                    n69wspa34o = '>' + '>'.join(n69wspa5l9[:-1]) + '/' + n69wspa5l9[-1] if len(n69wspa5l9) > 1 else '/' + n69wspa5l9[0]
                    n69wspa2jj = f"[ENV]{n69wspa34o}^1#_*{n69wspa2xo['homeclass']}"
                    if '/' in right:
                        n69wspa2jj = n69wspa2jj + ('/' + right.split('/')[-1])
            if n69wspa2wq == '[ENV]>rag>core/querier^1#_*KBClient/basefunc':
                if n69wspa2jj == '[ENV]>rag>core/querier^1#_*KBClient/basefunc':
                    raise
            return n69wspa2jj
        elif '/' in n69wspa2wq:
            left, right = n69wspa2wq.rsplit('/', 1)
            n69wspa5mi = n69wspa32e.get(right, right)
            return left + '/' + n69wspa5mi
        raise
        return n69wspa2wq
    for k in ('funcs', 'classes'):
        if len(n69wspa31q[k]) == 0:
            n69wspa31q[k]['raw_def_id'] = ''
            continue
        n69wspa31q[k]['raw_def_id'] = n69wspa31q[k].apply(b69wspa5ao, axis=1)
    n69wspa31q['objs']['rawname'] = n69wspa31q['objs']['name'].apply(lambda x: n69wspa32e.get(x, x))

class A69wspa5ap:

    def b69wspa5aj(self, n69wspa5mu: inspect.Parameter) -> dict:
        n69wspa5nh = n69wspa5mu.name
        if n69wspa5mu.kind.name == 'VAR_POSITIONAL':
            n69wspa5nh = '*' + n69wspa5nh
        elif n69wspa5mu.kind.name == 'VAR_KEYWORD':
            n69wspa5nh = '**' + n69wspa5nh
        info = {'name': n69wspa5nh, 'type_annotation': None, 'doc': '', 'default': None}
        if n69wspa5mu.default is not inspect.Parameter.empty:
            info['default'] = f"'{n69wspa5mu.default}'" if isinstance(n69wspa5mu.default, str) else f'{n69wspa5mu.default}'
        n69wspa5mp = n69wspa5mu.annotation
        if n69wspa5mp is inspect.Parameter.empty:
            n69wspa5ng = None
            n69wspa5md = None
        else:
            n69wspa5ng = getattr(n69wspa5mp, '__name__', str(n69wspa5mp))
            if get_origin(n69wspa5mp) is Annotated:
                args = get_args(n69wspa5mp)
                if args:
                    n69wspa5nl = args[0]
                    n69wspa5l7 = args[1:]
                    info['type_annotation'] = self.b69wspa5aa(n69wspa5nl)
                    for meta in n69wspa5l7:
                        if isinstance(meta, Doc):
                            info['doc'] = str(meta.__metadata__[0]) if hasattr(meta, '__metadata__') else str(meta)
                            break
                        elif hasattr(meta, '__doc__'):
                            pass
                        elif isinstance(meta, str):
                            info['doc'] = meta
            else:
                info['type_annotation'] = self.b69wspa5aa(n69wspa5mp)
            if isinstance(info.get('doc'), str):
                if (info['doc'].startswith('Doc("') or info['doc'].startswith("Doc('")) and (info['doc'].endswith('")') or info['doc'].endswith("')")):
                    info['doc'] = info['doc'][5:-2]
        return info

    def b69wspa5aa(self, tp) -> str:
        n69wspa5n1 = ''
        if tp is type(None):
            return 'None'
        elif hasattr(tp, '__name__'):
            try:
                n69wspa5n1 = str(tp)
            except Exception:
                n69wspa5n1 = repr(tp)
        else:
            n69wspa5n1 = repr(tp)
        if n69wspa5n1.startswith('typing.'):
            n69wspa5n1 = n69wspa5n1[7:]
        elif n69wspa5n1.startswith('<class ') and n69wspa5n1.endswith('>'):
            n69wspa5n1 = n69wspa5n1[7:-1].strip().strip('"').strip("'")
        return n69wspa5n1

    def b69wspa5af(self, obj) -> dict:
        try:
            n69wspa5m3 = inspect.signature(obj)
        except Exception as e:
            return ({'inputs': [], 'return': {'type_annotation': None, 'doc': ''}}, (obj.__name__, str(e)))
        parameters = []
        for n69wspa5mu in n69wspa5m3.parameters.values():
            parameters.append(self.b69wspa5aj(n69wspa5mu))
        n69wspa5n3 = n69wspa5m3.return_annotation
        n69wspa5lb = {'type_annotation': None, 'doc': ''}
        if n69wspa5n3 is not inspect.Signature.empty:
            if get_origin(n69wspa5n3) is Annotated:
                args = get_args(n69wspa5n3)
                if args:
                    n69wspa5nl = args[0]
                    n69wspa5l7 = args[1:]
                    n69wspa5lb = {'type_annotation': self.b69wspa5aa(n69wspa5nl), 'doc': ''}
                    for meta in n69wspa5l7:
                        if isinstance(meta, Doc):
                            n69wspa5lb['doc'] = str(meta.__metadata__[0]) if hasattr(meta, '__metadata__') else str(meta)
                            break
                        elif isinstance(meta, str):
                            n69wspa5lb['doc'] = meta
            else:
                n69wspa5lb = {'type_annotation': self.b69wspa5aa(n69wspa5n3), 'doc': ''}
            if isinstance(n69wspa5lb.get('doc'), str):
                if (n69wspa5lb['doc'].startswith('Doc("') or n69wspa5lb['doc'].startswith("Doc('")) and (n69wspa5lb['doc'].endswith('")') or n69wspa5lb['doc'].endswith("')")):
                    n69wspa5lb['doc'] = n69wspa5lb['doc'][5:-2]
        return ({'inputs': parameters, 'return': n69wspa5lb}, ())

    def b69wspa5an(self, module, skip_privates=True, source_file=None) -> List[Dict[str, Any]]:
        n69wspa5ln = []
        n69wspa35k = []
        for name, obj in inspect.getmembers(module):
            if name.startswith('_') and skip_privates:
                continue
            if name in SKIPNAMES:
                continue
            info = {'name': name, 'type': None, 'doc': inspect.getdoc(obj) or '', 'module': getattr(obj, '__module__', str(module.__name__)), 'source_file': source_file, 'methods': []}
            if type(obj).__name__ in ('module', 'class', 'type', 'method', 'function'):
                try:
                    info['source_file'] = inspect.getfile(obj)
                except (TypeError, OSError):
                    pass
            n69wspa5ns = None
            if inspect.isfunction(obj) or inspect.ismethod(obj):
                info['type'] = 'function'
                info['is_async'] = True if inspect.iscoroutinefunction(obj) else False
                info['params'], n69wspa5ns = self.b69wspa5af(obj)
                info['homeclass'] = ''
                n69wspa5ln.append(info)
            elif inspect.isclass(obj):
                info['type'] = 'class'
                info['bases'] = [n69wspa380.__name__ for n69wspa380 in obj.__mro__[1:-1]]
                n69wspa5ne = []
                for m_name, m_obj in inspect.getmembers(obj):
                    if m_name.startswith('_') and m_name not in ('__init__', '__call__') or not inspect.isfunction(m_obj):
                        continue
                    n69wsp9p0t, n69wspa5ns = self.b69wspa5af(m_obj)
                    n69wspa5lc = info['source_file']
                    try:
                        n69wspa5lc = inspect.getfile(m_obj)
                    except:
                        pass
                    n69wspa5n0 = '_UNKNOWN_'
                    for c in [n69wspa380 for n69wspa380 in obj.__mro__[:-1]]:
                        n69wspa5n0 = c.__name__
                        if m_name in c.__dict__:
                            break
                    n69wspa5ne.append({'name': m_name, 'is_async': True if inspect.iscoroutinefunction(m_obj) else False, 'params': n69wsp9p0t, 'doc': inspect.getdoc(m_obj) or '', 'source_file': n69wspa5lc, 'homeclass': n69wspa5n0})
                info['methods'] = n69wspa5ne
                n69wspa5ln.append(info)
            elif callable(obj):
                info['type'] = 'callable'
                info['value_type'] = self.b69wspa5aa(type(obj))
                info['params'], n69wspa5ns = self.b69wspa5af(obj)
                n69wspa5ln.append(info)
            else:
                info['type'] = 'object'
                info['value_repr'] = repr(obj)[:200]
                info['value_type'] = self.b69wspa5aa(type(obj))
                n69wspa5ln.append(info)
            if n69wspa5ns:
                n69wspa35k.append(n69wspa5ns)
        if n69wspa35k:
            pass
        return n69wspa5ln
    import types

    def b69wspa5ae(self, code: str, n69wspa5m6: str='temp_module', skip_privates: bool=True) -> List[Dict]:
        module = types.ModuleType(n69wspa5m6)
        module.__file__ = '<string>'
        module.__name__ = n69wspa5m6
        try:
            exec(code, module.__dict__)
        except Exception as e:
            raise RuntimeError(f'Failed to execute code: {e}') from e
        n69wspa38w = None
        code = code.strip()
        if not '\n' in code:
            if code.startswith('import '):
                n69wspa38w = code[7:].strip()
            elif code.startswith('from '):
                n69wspa38w = code[5:].split(' import ')[0].strip()
        n69wspa5nb = '<UNK>'
        try:
            n69wspa5nb = importlib.util.find_spec(n69wspa38w).n69wspa5nb if n69wspa38w else ''
        except Exception as e:
            pass
        return self.b69wspa5an(module, skip_privates=skip_privates, source_file=n69wspa5nb)

    def b69wspa5a8(self, code, spread_imports=True, skip_privates=True):
        n69wsp9oya = b69wsp9mq1(code, keep_comments=False)['body']
        n69wspa2ej = []
        n69wspa36m = []
        n69wspa5n4 = []
        for n69wspa2mh in n69wsp9oya:
            if len(n69wspa2mh['names']) != 1:
                pass
            if n69wspa2mh['ntype'] == 'ImportFrom':
                if n69wspa2mh['module'].startswith('.'):
                    continue
                if skip_privates:
                    if '._' in n69wspa2mh['module'] or n69wspa2mh['module'].startswith('_') or '._' in (n69wspa2mh['names'][0]['name'] or '') or (n69wspa2mh['names'][0]['name'] or '').startswith('_') or (n69wspa2mh['names'][0]['asname'] or '').startswith('_'):
                        n69wspa5n4.append(n69wspa2mh['code'])
                        continue
                n69wspa5l8 = {'ntype': 'Module', 'body': [n69wspa2mh]}
                _, n69wspa2dv = b65wsp9mrz(n69wspa5l8)
                n69wspa2ej.append(n69wspa2dv)
                n69wspa36m.append(n69wspa2mh['module'])
            elif n69wspa2mh['ntype'] == 'Import':
                if not spread_imports:
                    continue
                if skip_privates:
                    if '._' in (n69wspa2mh['names'][0]['name'] or '') or (n69wspa2mh['names'][0]['name'] or '').startswith('_') or (n69wspa2mh['names'][0]['asname'] or '').startswith('_'):
                        n69wspa5n4.append(n69wspa2mh['code'])
                        continue
                for namedic in n69wspa2mh['names']:
                    if namedic['name'].startswith('.'):
                        pass
                    n69wsp9p59 = {'ntype': 'Module', 'body': [{'ntype': 'ImportFrom', 'module': namedic['name'], 'names': [{'ntype': 'alias', 'code': '*', 'name': '*', 'asname': None}]}]}
                    _, n69wsp9otb = b65wsp9mrz(n69wsp9p59)
                    n69wspa2ej.append(n69wsp9otb)
                    n69wspa36m.append(namedic['name'])
            else:
                continue
        n69wspa5n4 = '\n'.join(list(set(n69wspa5n4)))
        if n69wspa5n4:
            pass
        n69wspa5ld = {}
        assert len(n69wspa2ej) == len(n69wspa36m)
        n69wspa2lc = []
        for i in range(len(n69wspa36m)):
            n69wspa38w = n69wspa36m[i]
            n69wspa5n9 = n69wspa2ej[i]
            try:
                n69wspa5n5 = self.b69wspa5ae(n69wspa5n9, skip_privates=skip_privates)
                if n69wspa38w in n69wspa5ld:
                    n69wspa5ld[n69wspa38w] = n69wspa5ld[n69wspa38w] + n69wspa5n5
                else:
                    n69wspa5ld[n69wspa38w] = n69wspa5n5
            except Exception as e:
                n69wspa2lc.append((n69wspa5n9, str(e)))
                continue
        if n69wspa2lc:
            pass
        return (n69wspa5ld, n69wspa2lc)

    def b69wspa5ak(self, n69wspa5l4: str) -> Dict[str, List[Dict]]:
        n69wspa5l4 = Path(n69wspa5l4).resolve()
        n69wspa5lh = n69wspa5l4
        n69wspa5mr = n69wspa5l4.name
        import sys
        n69wspa5nr = sys.path[:]
        sys.path.insert(0, str(n69wspa5lh))
        n69wspa2me = {}
        try:
            for n69wsp9onc, dirs, n69wspa35u in os.b69wspa0y7(n69wspa5l4):
                dirs[:] = [n69wsp9oq0 for n69wsp9oq0 in dirs if not n69wsp9oq0.startswith('.') and n69wsp9oq0 != '__pycache__']
                for file in n69wspa35u:
                    if file.endswith('.py') and (not file.startswith('__')):
                        n69wspa5l5 = Path(n69wsp9onc).relative_to(n69wspa5lh) / file
                        n69wspa5m6 = str(n69wspa5l5.with_suffix('')).replace(os.sep, '.')
                        try:
                            n69wspa5m2 = importlib.util.spec_from_file_location(n69wspa5m6, Path(n69wsp9onc) / file)
                            if n69wspa5m2 is None:
                                continue
                            module = importlib.util.module_from_spec(n69wspa5m2)
                            n69wspa5m2.loader.exec_module(module)
                            n69wspa5ln = self.b69wspa5an(module)
                            if n69wspa5ln:
                                n69wspa2me[n69wspa5m6] = n69wspa5ln
                        except Exception as e:
                            continue
        finally:
            sys.path[:] = n69wspa5nr
        return n69wspa2me

    def b69wspa5ac(self, data):
        n69wsp9p6q = []
        n69wsp9p3q = []
        n69wspa2nl = []
        n69wsp9osv = []
        n69wspa5m8 = {}
        for n69wspa2ne, items in data.items():
            if '.' in n69wspa2ne:
                n69wspa337 = '[ENV]>' + x69xm5du01(n69wspa2ne)
            else:
                n69wspa337 = '[ENV]/' + n69wspa2ne
            n69wspa5m8[n69wspa2ne] = n69wspa337
            n69wspa5lm = []
            for item in items:
                if item['name'] in n69wspa5lm:
                    continue
                n69wspa5lm.append(item['name'])
                if item['type'] == 'function':
                    n69wspa2qn = n69wspa337 + '^1#_/' + item['name']
                    n69wspa5lj = {'uid': n69wspa2qn, 'def_id': n69wspa2qn, 'globals': [], 'nonlocals': [], 'imports_code': '', 'is_async': item['is_async'], 'deco_expr': '', 'xpos': 0, 'ypos': 0, 'doc': item['doc'], 'ethnic': '[ENV]', 'source_file': item.get('source_file', '<UNK>'), 'homeclass': ''}
                    n69wsp9p3q.append(n69wspa5lj)
                    for pi, n69wspa5mu in enumerate(item['params']['inputs']):
                        n69wspa5n7 = {'name': n69wspa5mu['name'], 'type': n69wspa5mu['type_annotation'], 'doc': n69wspa5mu['doc'], 'default': n69wspa5mu['default'], 'place': pi, 'ctx': 'input', 'def_id': n69wspa2qn, 'source_file': item.get('source_file', '<UNK>')}
                        n69wsp9osv.append(n69wspa5n7)
                    n69wspa5mu = item['params']['return']
                    n69wsp9osv.append({'name': 'return', 'type': n69wspa5mu['type_annotation'], 'doc': n69wspa5mu['doc'], 'default': None, 'place': 0, 'ctx': 'return', 'def_id': n69wspa2qn, 'source_file': item.get('source_file', '<UNK>')})
                elif item['type'] == 'class':
                    n69wspa5n8 = n69wspa337 + '^1#_*' + item['name']
                    n69wspa5m7 = {'bases': [], 'vars': [], 'deco_expr': '', 'def_id': n69wspa5n8, 'uid': n69wspa5n8, 'xpos': 0, 'ypos': 0, 'doc': item['doc'], 'ethnic': '[ENV]', 'source_file': item.get('source_file', '<UNK>')}
                    n69wsp9p6q.append(n69wspa5m7)
                    for func in item['methods']:
                        n69wspa2qn = n69wspa5n8 + '/' + func['name']
                        n69wspa5lj = {'uid': n69wspa2qn, 'def_id': n69wspa2qn, 'globals': [], 'nonlocals': [], 'imports_code': '', 'is_async': func['is_async'], 'deco_expr': '', 'xpos': 0, 'ypos': 0, 'doc': func['doc'], 'ethnic': '[ENV]', 'source_file': func.get('source_file', item.get('source_file', '<UNK>')), 'homeclass': func['homeclass']}
                        n69wsp9p3q.append(n69wspa5lj)
                        for pi, n69wspa5mu in enumerate(func['params']['inputs']):
                            n69wspa5n7 = {'name': n69wspa5mu['name'], 'type': n69wspa5mu['type_annotation'], 'doc': n69wspa5mu['doc'], 'default': n69wspa5mu['default'], 'place': pi, 'ctx': 'input', 'def_id': n69wspa2qn, 'source_file': item.get('source_file', '<UNK>')}
                            n69wsp9osv.append(n69wspa5n7)
                        n69wspa5mu = func['params']['return']
                        n69wsp9osv.append({'name': 'return', 'type': n69wspa5mu['type_annotation'], 'doc': n69wspa5mu['doc'], 'default': None, 'place': 0, 'ctx': 'return', 'def_id': n69wspa2qn, 'source_file': item.get('source_file', '<UNK>')})
                else:
                    n69wspa2og = '_extuid'
                    n69wspa5my = {'name': item['name'], 'from_node': '<EXT>', 'from_def': None, 'ctx': 'out', 'def_id': n69wspa337, 'uid': n69wspa2og, 'type': item['value_type'], 'repr': None, 'value': None, 'ethnic': '[ENV]', 'source_file': item.get('source_file', '<UNK>')}
                    n69wspa2nl.append(n69wspa5my)
        n69wspa5lt = []
        for aobj in n69wspa2nl:
            n69wspa2dq = aobj['type']
            if n69wspa2dq == 'module':
                continue
            if n69wspa2dq in SKIPNAMES:
                aobj['type'] = n69wspa2dq
                continue
            if not '.' in n69wspa2dq:
                continue
            rawmodid, vtype = n69wspa2dq.rsplit('.', 1)
            n69wspa5le = n69wspa5m8.get(rawmodid)
            if not n69wspa5le:
                if not '.' in rawmodid:
                    aobj['type'] = '[ENV]/' + rawmodid + '^1#_*' + vtype
                else:
                    aobj['type'] = '[ENV]>' + x69xm5du01(rawmodid) + '^1#_*' + vtype
                n69wspa5lt.append(aobj)
                continue
            aobj['type'] = n69wspa5le + '^1#_*' + vtype
            n69wspa5lt.append(aobj)
        return {'funcs': n69wsp9p3q, 'params': n69wsp9osv, 'classes': n69wsp9p6q, 'objs': n69wspa5lt}

    def b69wspa5al(self, imports_code, spread_imports=True):
        n69wspa5n5, failed = self.b69wspa5a8(imports_code)
        n69wspa5m1 = self.b69wspa5ac(n69wspa5n5)
        n69wspa5nk = {'funcs': ['uid', 'def_id', 'globals', 'nonlocals', 'imports_code', 'is_async', 'deco_expr', 'doc', 'xpos', 'ypos', 'ethnic', 'source_file', 'homeclass'], 'params': ['name', 'type', 'doc', 'def_id', 'ctx', 'default', 'place', 'source_file'], 'classes': ['bases', 'vars', 'deco_expr', 'def_id', 'uid', 'doc', 'xpos', 'ypos', 'ethnic', 'source_file'], 'objs': ['name', 'from_node', 'from_def', 'ctx', 'def_id', 'uid', 'type', 'repr', 'value', 'ethnic', 'source_file']}
        for k in n69wspa5nk.keys():
            n69wspa5m1[k] = pd.DataFrame(n69wspa5m1[k]) if n69wspa5m1.get(k) else pd.DataFrame(columns=n69wspa5nk[k])
        return (n69wspa5m1, failed)

    def b69wspa5ad(self, imports_code, ext_paths, spread_imports=True, retype='dict', recur_obj_cls=True, xform_imports=True):
        n69wspa5nd = sys.path[:]
        try:
            for p in ext_paths:
                sys.path.insert(0, p)
            if ',' in imports_code:
                _, _, imports_code = b69wsp9mrs(b69wsp9mq1(imports_code), expand=True)
            n69wsp9p31 = imports_code.split('\n')
            n69wspa5mo = '\n'.join([l for l in n69wsp9p31 if l.startswith('from')])
            n69wspa5nv = '\n'.join([l for l in n69wsp9p31 if l.startswith('import')])
            n69wspa31q, ffails = self.b69wspa5al(n69wspa5mo, spread_imports=spread_imports)
            n69wspa2dr, n69wspa5nc = self.b69wspa5al(n69wspa5nv, spread_imports=spread_imports)
            n69wspa5mb = []
            for n69wspa2va in n69wspa5nc:
                f = n69wspa2va[0]
                if not (f.startswith('from ') and f.endswith(' import *')):
                    continue
                n69wspa5mb.append((f'import {f[5:-9].strip()}', n69wspa2va[1]))
            n69wspa5nc = n69wspa5mb
            b69wspa5am(n69wspa2dr, n69wspa5np=sys.path)
            b69wspa5a9(n69wspa31q, n69wspa5mo, n69wspa5np=sys.path)
            if xform_imports:
                n69wspa2dr = b69wspa5ah(n69wspa2dr, n69wspa5nv, '[ENV]')
            n69wspa5nm = {}
            for k in n69wspa31q:
                n69wspa5nm[k] = pd.concat([n69wspa31q[k], n69wspa2dr[k]], ignore_index=True)
                n69wspa2id = 'name' if k == 'objs' else 'def_id'
                n69wsp9p6j = n69wspa5nm[k][n69wspa5nm[k][n69wspa2id] == '<UNK>']
                if len(n69wsp9p6j) > 0:
                    pass
                n69wspa5nm[k] = n69wspa5nm[k][~(n69wspa5nm[k][n69wspa2id] == '<UNK>')]
            n69wspa5nk = {'funcs': ['uid', 'def_id', 'globals', 'nonlocals', 'imports_code', 'is_async', 'deco_expr', 'doc', 'xpos', 'ypos', 'ethnic', 'source_file', 'direct', 'raw_def_id'], 'params': ['name', 'type', 'doc', 'def_id', 'ctx', 'default', 'place', 'source_file', 'direct'], 'classes': ['bases', 'vars', 'deco_expr', 'def_id', 'uid', 'doc', 'xpos', 'ypos', 'ethnic', 'source_file', 'direct', 'raw_def_id'], 'objs': ['name', 'from_node', 'from_def', 'ctx', 'def_id', 'uid', 'type', 'repr', 'value', 'ethnic', 'source_file', 'rawname']}
            for k in n69wspa5nk:
                if k != 'obj':
                    n69wspa5nm[k]['direct'] = True
            if recur_obj_cls:
                n69wspa5ly = ''
                n69wspa5lw = []
                for v in n69wspa5nm['objs'].to_dict(orient='records'):
                    if v['type'] in SKIPNAMES:
                        continue
                    if v['type'] == 'module' or v['type'].startswith('MODULE@'):
                        n69wspa5lw.append((v['def_id'], v['name']))
                        continue
                    if not (v['type'].startswith('[ENV]') and v['type'].count('^') == 1):
                        continue
                    n69wspa2jc, clspart = v['type'][6:].split('^')
                    if not clspart.count('*') == 1:
                        continue
                    n69wspa37l = clspart.split('*')[-1]
                    if not x69xm5dtzx(n69wspa37l) == 'folder':
                        continue
                    n69wspa337 = n69wspa2jc.replace('>', '.').replace('/', '.')
                    n69wspa5na = f'from {n69wspa337} import {n69wspa37l}'
                    if not n69wspa5na + '\n' in n69wspa5ly:
                        n69wspa5ly = n69wspa5ly + (n69wspa5na + '\n')
                if n69wspa5lw:
                    pass
                scanned2, fails2 = self.b69wspa5a8(n69wspa5ly, skip_privates=False)
                if fails2:
                    pass
                n69wspa5me = self.b69wspa5ac(scanned2)
                for k in n69wspa5nk:
                    n69wspa35i = n69wspa5me[k]
                    n69wspa35i = pd.DataFrame(n69wspa35i) if n69wspa35i else pd.DataFrame(columns=n69wspa5nk[k])
                    if k != 'obj':
                        n69wspa35i['direct'] = False
                    n69wspa5me[k] = n69wspa35i
                b69wspa5a9(n69wspa5me, n69wspa5ly, n69wspa5np=sys.path)
                for k in n69wspa5nk:
                    n69wspa5nm[k] = pd.concat([n69wspa5nm[k], n69wspa5me[k]], ignore_index=True)
            n69wspa5nm['classes'] = n69wspa5nm['classes'].groupby(['def_id'], as_index=False).agg({**{n69wspa2id: 'first' for n69wspa2id in n69wspa5nk['classes']}, 'direct': 'any'})
            n69wspa5nm['funcs'] = n69wspa5nm['funcs'].groupby(['def_id'], as_index=False).agg({**{n69wspa2id: 'first' for n69wspa2id in n69wspa5nk['funcs']}, 'direct': 'any'})
            n69wspa5nm['params'] = n69wspa5nm['params'].groupby(['def_id', 'name', 'ctx'], as_index=False).agg({**{n69wspa2id: 'first' for n69wspa2id in n69wspa5nk['params']}, 'direct': 'any'})
            n69wspa5nm['objs'] = n69wspa5nm['objs'].drop_duplicates(['def_id', 'name', 'type'])
            if retype != 'df':
                for k in n69wspa5nm:
                    n69wspa5nm[k] = n69wspa5nm[k].to_dict(orient='records')
            if None in n69wspa5nm['funcs']['raw_def_id'].tolist() or np.nan in n69wspa5nm['funcs']['raw_def_id'].tolist():
                raise
            return (n69wspa5nm, ffails + n69wspa5nc)
        finally:
            sys.path[:] = n69wspa5nd

    async def b69x8ynrdt(self, imports_code, ext_paths, spread_imports=True, retype='dict', recur_obj_cls=True, xform_imports=True):
        n69wspa2xu = asyncio.get_running_loop()
        n69wspa2p9 = await n69wspa2xu.run_in_executor(None, lambda: self.b69wspa5ad(imports_code, ext_paths, spread_imports=spread_imports, retype=retype, recur_obj_cls=recur_obj_cls, xform_imports=xform_imports))
        return n69wspa2p9
n69wspa5l3 = A69wspa5ap()
if __name__ == '__main__':
    pass