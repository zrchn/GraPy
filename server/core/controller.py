
"""
Copyright (c) 2026 Zhiren Chen. Provided as-is for local use only.
"""

from datetime import datetime
import importlib
from pathlib import Path
import sys
import requests
from utils.jedi_sugger import JediEnvManager
import time
import types
from core.cft.consts import DOT_REPL, EMPTY_PASSER, SUGCODE_HOLDER, UNIVERSAL_FAKER, VARSEND_FUNC, vbs, INCOMPLETE_HOLDER, UID_COMMENT_RIGHTLABEL, UID_COMMENT_LEFTLABEL, SECTION_START_LABEL, SECTION_END_LABEL, INSERT_LABEL
from utils.jsonformater import jsonformat
import re
import ast
import json
import json5
import numpy as np
import pandas as pd
import copy
import traceback
from loguru import logger
import asyncio
from concurrent.futures import ThreadPoolExecutor
from sqlalchemy import create_engine, text
import inspect
from sqlalchemy.ext.asyncio import create_async_engine, AsyncConnection
from typing import Dict, List, Any, Literal, Set
from basic.configer import configer
from basic.db.mysql_handler import DBHandler, gen_create_sql
exec(f'from core.cft.processors{sys.version_info.minor} import *')
from core.cft.utils import all_desc_to_nl, x69xm5du07, find_nth, x69xm5du03, x69xm5du02, x69xm5dtzx, aidle, x69xm5du08, recover_conds, loadstr, idgen, replace_lastpart, x69xm5dtzz, x69xm5du00, x69xm5dtzw, x69xm5du09, table_lambda_get, table_unique_get, gen_base36_id, is_valid_varname, x69xm5du01, to_func_id, to_module_id, x69xm5dtzy, trunk_to_func
from functools import partial
from utils.shared import Bouncer, custout2dict, list_installed_packages, parse_for_sugs, remove_common_indents, redact_quoted_strings, requests_post
from agentity.codeplanner.code_agent import Coder
import redis
from core.output_client import client as oclient
from core.import_parser import n69wspa5l3, b69wspa5ab, b69wspa5ag, b69wspa5ah
COPLEX_DB_URL = configer.grapy.db_url
REDIS_HOST = configer.grapy.redis_host
REDIS_PORT = int(configer.grapy.redis_port)
NODESPACINGX = int(configer.ui.node_spacing_x)
NODESPACINGY = int(configer.ui.node_spacing_y)
MAX_REDOS = int(configer.grapy.max_redos)
PRECODE = '\nfrom typing_extensions import Annotated, Doc\nfrom typing import Any, Optional\nfrom functools import reduce as _functools_reduce\nfrom loguru import logger\n'
n69wspa2ha = {'uid': {'VARCHAR(32)'}, 'hid': {'VARCHAR(255)'}, 'branch': {'JSON'}, 'data_providers': {'NoneType', 'list'}, 'node_type': {'VARCHAR(31)'}, 'code': {'NoneType', 'MEDIUMTEXT'}, 'pres': {'NoneType', 'list'}, 'nexts': {'NoneType', 'list'}, 'xpos': {'int'}, 'ypos': {'int'}, 'def_id': {'VARCHAR(512)'}, 'toolcall': {'NoneType', 'dict'}, 'targets': {'NoneType', 'str'}, 'params_map': {'NoneType', 'dict'}, 'comments': {'NoneType', 'VARCHAR(767)'}, 'expr': {'NoneType', 'VARCHAR(767)'}, 'cases': {'NoneType', 'dict'}, 'source': {'NoneType', 'str'}, 'iter': {'NoneType', 'str'}, 'slice': {'NoneType', 'str'}, 'misc': {'NoneType', 'dict'}}
n69wspa2ts = {'uid': {'VARCHAR(64)'}, 'def_id': {'VARCHAR(512)'}, 'globals': {'NoneType', 'list'}, 'nonlocals': {'NoneType', 'list'}, 'imports_code': {'NoneType', 'TEXT'}, 'is_async': {'bool'}, 'deco_expr': {'NoneType', 'str'}, 'doc': {'NoneType', 'TEXT'}, 'xpos': {'NoneType', 'int'}, 'ypos': {'NoneType', 'int'}, 'ethnic': {'NoneType', 'VARCHAR(16)'}}
n69wspa2dn = {'bases': {'list'}, 'vars': {'list'}, 'deco_expr': {'NoneType', 'str'}, 'def_id': {'VARCHAR(512)'}, 'uid': {'VARCHAR(32)'}, 'doc': {'NoneType', 'TEXT'}, 'xpos': {'NoneType', 'int'}, 'ypos': {'NoneType', 'int'}, 'ethnic': {'NoneType', 'VARCHAR(16)'}}
n69wspa2hw = {'name': {'VARCHAR(128)'}, 'from_node': {'NoneType', 'str'}, 'from_def': {'NoneType', 'str'}, 'ctx': {'str'}, 'def_id': {'VARCHAR(512)'}, 'uid': {'VARCHAR(32)'}, 'type': {'NoneType', 'str'}, 'repr': {'NoneType', 'str'}, 'value': {'NoneType', 'str'}, 'ethnic': {'NoneType', 'VARCHAR(16)'}}
n69wspa356 = {'name': {'VARCHAR(128)'}, 'type': {'NoneType', 'str'}, 'doc': {'NoneType', 'TEXT'}, 'def_id': {'VARCHAR(512)'}, 'ctx': {'VARCHAR(32)'}, 'default': {'NoneType', 'str'}, 'place': {'int'}}
n69wspa2fi = {'def_id': {'VARCHAR(512)'}, 'untracked_vars': {'JSON'}}
n69wspa2mc = {'user_id': {'int'}, 'external_pkgs': {'JSON'}}
n69wspa35q = ['uid']
n69wspa2oa = ['uid']
n69wspa2zt = ['uid']
n69wspa2ge = ['def_id', 'uid', 'name']
n69wspa2jd = ['def_id', 'name', 'ctx']
n69wspa2ij = ['user_id']
n69wspa2jl = ['def_id']
n69wspa2ng = [['def_id', 'hid'], ['uid']]
n69wspa33w = ['def_id']
n69wspa2lf = ['def_id']
n69wspa37i = [['def_id', 'ctx', 'place']]
n69wspa2hz = ['def_id']
n69wspa32t = [{'foreigns': 'def_id', 'references': 'funcs(def_id)', 'behaviors': 'ON DELETE CASCADE ON UPDATE CASCADE'}]
n69wspa2fh = [{'foreigns': 'def_id', 'references': 'funcs(def_id)', 'behaviors': 'ON DELETE CASCADE ON UPDATE CASCADE'}]
n69wspa2tq = [{'foreigns': 'uid', 'references': 'nodes(uid)', 'behaviors': 'ON DELETE CASCADE ON UPDATE CASCADE'}]
n69wspa36b = [{'foreigns': 'def_id', 'references': 'funcs(def_id)', 'behaviors': 'ON DELETE CASCADE ON UPDATE CASCADE'}]
n69wspa2u3 = {'funcs': {'fields': n69wspa2ts, 'primes': n69wspa2oa, 'uniques': n69wspa33w}, 'nodes': {'fields': n69wspa2ha, 'primes': n69wspa35q, 'uniques': n69wspa2ng, 'foreigns': n69wspa32t}, 'classes': {'fields': n69wspa2dn, 'primes': n69wspa2zt, 'uniques': n69wspa2lf}, 'vars': {'fields': n69wspa2hw, 'primes': n69wspa2ge, 'foreigns': n69wspa2tq}, 'params': {'fields': n69wspa356, 'primes': n69wspa2jd, 'foreigns': n69wspa2fh}, 'funcmeta': {'fields': n69wspa2fi, 'primes': n69wspa2jl, 'foreigns': n69wspa36b}, 'misc': {'fields': n69wspa2mc, 'primes': n69wspa2ij, 'foreigns': []}}
for dbname, struct in n69wspa2u3.items():
    struct['fields'] = gen_create_sql(dbname, struct['fields'], need_new_mapping=True)[1]
n69wspa2wn = n69wspa2u3
n69wspa38t = {'start': {'width': 400, 'height': 100}, 'if': {'width': 400, 'height': 200}, 'for': {'width': 400, 'height': 200}, 'while': {'width': 400, 'height': 200}, 'with': {'width': 400, 'height': 200}, 'tryhead': {'width': 400, 'height': 200}, 'excepts': {'width': 520, 'height': 200}, 'finally': {'width': 400, 'height': 200}, 'match': {'width': 500, 'height': 200}, 'code': {'width': 560, 'height': 100}, 'tool': {'width': 500, 'height': 200}, 'tool_conc': {'width': 300, 'height': 200}, 'return': {'width': 300, 'height': 100}, 'yield': {'width': 300, 'height': 100}, 'pass': {'width': 300, 'height': 50}, 'continue': {'width': 300, 'height': 50}, 'break': {'width': 300, 'height': 50}, 'endwith': {'width': 240, 'height': 70}, 'endfor': {'width': 240, 'height': 70}, 'endexcepts': {'width': 240, 'height': 70}, 'endif': {'width': 240, 'height': 70}, 'endtryhead': {'width': 240, 'height': 70}, 'endfinally': {'width': 240, 'height': 70}, 'endwhile': {'width': 240, 'height': 70}, 'endmatch': {'width': 240, 'height': 70}}
n69wspa2i9 = {'width': 280, 'height': 200}
n69wspa2tr = {'width': 280, 'height': 200}
nesttypes = {'if', 'for', 'while', 'with', 'tryhead', 'excepts', 'finally', 'match'}
n69wspa2ez = {'tool': 'tool', 'tool_conc': 'tool_conc'}
n69wspa2ko = {'id': None, 'type': None, 'position': {}, 'style': {}, 'data': {'uid': None, 'node_type': None, 'code': '', 'def_id': '', 'toolcall': None, 'targets': None, 'params_map': None, 'comments': None, 'expr': None, 'cases': None, 'iter': None, 'slice': None, 'misc': None, 'handle_in': None, 'handle_outs': None, 'width': None, 'height': None, 'selected': True, 'positionAbsolute': {}, 'dragging': False}}
n69wspa2qf = {'globals': [], 'nonlocals': [], 'imports_code': '', 'is_async': False, 'deco_expr': '', 'xpos': 0, 'ypos': 0, 'doc': ''}
n69wspa2xz = {'bases': [], 'vars': [], 'deco_expr': '', 'xpos': 0, 'ypos': 0, 'doc': ''}
n69wspa2f8 = {'branch': '_', 'data_providers': [], 'code': '', 'pres': [], 'nexts': [], 'xpos': 0, 'ypos': 0}

def b69wspa0yh(*args, **kwargs):
    pass

def b69wspa0xo(n69wspa35c, n69wspa35i):
    try:
        n69wspa2ug = int(n69wspa35c.get('timestamp', 0))
        n69wspa2wf = int(n69wspa35i.get('timestamp', 0))
        if n69wspa2wf > 0:
            if n69wspa2wf < n69wspa2ug:
                return True
    except:
        pass
    n69wspa35c = {k: v for k, v in n69wspa35c.items() if not k in ('timestamp',)}
    n69wspa35i = {k: v for k, v in n69wspa35i.items() if not k in ('timestamp',)}
    return n69wspa35c == n69wspa35i
n69wspa2s9 = b69wspa0xo

def b69wspa0xp(n69wspa35c, n69wspa35i, level=0):
    try:
        n69wspa2ug = int(n69wspa35c.get('timestamp', 0))
        n69wspa2wf = int(n69wspa35i.get('timestamp', 0))
        if n69wspa2wf > 0:
            if n69wspa2wf < n69wspa2ug:
                return True
    except:
        pass
    n69wspa35c = {k: v for k, v in n69wspa35c.items() if not k in ('timestamp',)}
    n69wspa35i = {k: v for k, v in n69wspa35i.items() if not k in ('timestamp',)}
    n69wspa2l6 = {}
    if not n69wspa35c:
        return False
    for e in n69wspa35c.get(2, []):
        n69wspa36d = {k: v for k, v in e.items() if not k == 'id'}
        if not 'source' in n69wspa36d or not 'target' in n69wspa36d:
            return False
        n69wspa2cp = n69wspa36d['source'] + '~' + n69wspa36d['target']
        n69wspa2l6[n69wspa2cp] = n69wspa36d
    n69wspa2zn = {}
    for e in n69wspa35i.get(2, []):
        n69wspa36d = {k: v for k, v in e.items() if not k == 'id'}
        n69wspa2cp = n69wspa36d['source'] + '~' + n69wspa36d['target']
        n69wspa2zn[n69wspa2cp] = n69wspa36d
    n69wspa2l1 = ['vars', 'source', 'vars_in', 'data_providers', 'hid', 'branch', 'vars_out', 'vars_relayed', 'tool_counts', 'pres', 'nexts', 'subWorkflowIds', 'width', 'height', 'selected', 'positionAbsolute', 'dragging']
    if level > 0:
        n69wspa2l1 = n69wspa2l1 + ['data_providers', 'code', 'xpos', 'ypos', 'toolcall', 'targets', 'params_map', 'comments', 'expr', 'source', 'iter', 'slice', 'misc', 'globals', 'nonlocals', 'imports_code', 'deco_expr', 'bases']
    n69wspa2k8 = {n['id']: {subk: subv if subk not in ('cases', 'handle_outs') or not isinstance(subv, dict) else {loadstr(n69wspa2w8): hd for n69wspa2w8, hd in subv.items()} for subk, subv in n['data'].items() if subk not in n69wspa2l1} for n in n69wspa35c[1]}
    n69wspa2wj = {n['id']: {subk: subv if subk not in ('cases', 'handle_outs') or not isinstance(subv, dict) else {loadstr(n69wspa2w8): hd for n69wspa2w8, hd in subv.items()} for subk, subv in n['data'].items() if subk not in n69wspa2l1} for n in n69wspa35i[1]}
    if n69wspa2k8 == n69wspa2wj and n69wspa2l6 == n69wspa2zn:
        return True
    return False

def b69wspa0ye(code):
    n69wsp9oza = code.split('\n')
    n69wsp9ozc = []
    for n69wsp9ou8 in n69wsp9oza:
        if not INCOMPLETE_HOLDER in n69wsp9ou8:
            n69wsp9ozc.append(n69wsp9ou8)
            continue
        n69wspa2n9 = len(n69wsp9ou8) - len(n69wsp9ou8.lstrip())
        if UID_COMMENT_LEFTLABEL in n69wsp9ou8 and UID_COMMENT_RIGHTLABEL in n69wsp9ou8:
            n65d20cda3 = n69wsp9ou8.split(UID_COMMENT_LEFTLABEL)[-1].split(UID_COMMENT_RIGHTLABEL)[0]
            n69wspa2kp = '#' + UID_COMMENT_LEFTLABEL + n65d20cda3 + UID_COMMENT_RIGHTLABEL
        else:
            n69wspa2kp = ''
        n69wsp9p72 = n69wsp9ou8.strip().split(UID_COMMENT_LEFTLABEL)[0].strip().strip('#').strip()
        if not (n69wsp9p72.strip().startswith(f'{INCOMPLETE_HOLDER} =') and n69wsp9p72.strip()[-1] in ("'", '"')):
            n69wsp9ozc.append(n69wsp9ou8)
            continue
        n69wsp9omg = n69wsp9p72[len(INCOMPLETE_HOLDER) + 4:-1]
        n69wspa34m = n69wsp9omg.split('\\n')
        n69wspa34m = [' ' * n69wspa2n9 + rl + n69wspa2kp for rl in n69wspa34m]
        n69wsp9ozc = n69wsp9ozc + n69wspa34m
    return '\n'.join(n69wsp9ozc)

def extract_roi(code, shift_after_varsender=False):
    n69wspa2xb = code
    if SECTION_START_LABEL in code:
        assert SECTION_END_LABEL in code.split(SECTION_START_LABEL)[-1], code
        code = code.split(SECTION_START_LABEL)[-1].split(SECTION_END_LABEL)[0]
        code = code.split(INSERT_LABEL)[-1]
    n69wsp9oza = code.split('\n')
    n69wspa2di = 0
    for n69wspa2di in range(len(n69wsp9oza)):
        if n69wsp9oza[n69wspa2di].strip():
            break
    n69wspa376 = len(n69wsp9oza[n69wspa2di]) - len(n69wsp9oza[n69wspa2di].lstrip())
    code = code.replace('\n' + n69wspa376 * ' ', '\n')
    if code.strip().endswith('#'):
        code = code.rsplit('#', 1)[0]
    if shift_after_varsender and SECTION_END_LABEL in n69wspa2xb:
        n69wspa2p8 = [ol for ol in n69wspa2xb.split('\n') if ol.strip()]
        n69wspa332 = [n69wsp9on8 for n69wsp9on8 in range(len(n69wspa2p8)) if SECTION_END_LABEL in n69wspa2p8[n69wsp9on8]]
        if n69wspa332:
            n69wspa332 = n69wspa332[-1]
            n69wspa31d = n69wspa2p8[n69wspa332 + 1:n69wspa332 + 5]
            n6a0gjsygp = n69wspa2p8[n69wspa332 + 1:n69wspa332 + 2]
            if len(n69wspa31d) == 4:
                if n69wspa31d[0].strip().startswith('try'):
                    if n69wspa31d[1].strip().startswith(VARSEND_FUNC + '_safe'):
                        code = code + '\n' + remove_common_indents('\n'.join(n69wspa31d))
            if len(n6a0gjsygp) == 1:
                if n6a0gjsygp[0].strip().startswith(VARSEND_FUNC):
                    code = code + '\n' + remove_common_indents('\n'.join(n6a0gjsygp))
    if INCOMPLETE_HOLDER in code:
        code = b69wspa0ye(code)
    return code

class A69wspa0yp(Exception):

    def __init__(self, msg):
        super().__init__(msg)

class ClientAbort(Exception):

    def __init__(self, msg):
        super().__init__(msg)
n69wspa2rv = Bouncer(taggenerator=lambda *args, **kwargs: kwargs.get('n69wspa2wq', 'default_def_id'))
n69wspa2jh = Bouncer(taggenerator=lambda *args, **kwargs: args[1])
n69wspa2pp = Bouncer(taggenerator=lambda *args, **kwargs: args[1])

class A69wspa0yq(DBHandler):

    def __init__(self):
        super().__init__(COPLEX_DB_URL, n69wspa2wn)
        self.x69xm5dtzq = False
        self.n69wspa36f = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        self.n69wspa31i = {}
        self.x69xm5dtzr = set()
        self.x69xm5dtzs = {}
        self.x69xm5dtzt = 1
        self.n69wspa2gm = {}
        self.x69xm5dtzu = {}
        self.n69wspa2fo = None
        try:
            self.n69wspa36f.execute_command('FT.DROPINDEX', 'idx:undo')
        except:
            pass
        self.n69wspa36f.execute_command('FT.CREATE', 'idx:undo', 'ON', 'JSON', 'PREFIX', '1', 'undo:', 'SCHEMA', '$.save_id', 'AS', 'save_id', 'NUMERIC', 'SORTABLE', '$.module_id', 'AS', 'module_id', 'TAG', '$.def_id', 'AS', 'def_id', 'TEXT', '$.scope_type', 'AS', 'scope_type', 'TAG', '$.undoed', 'AS', 'undoed', 'TAG')
        self.coder = Coder({'node_id': 'Cody', 'node_type': 'coder'})

    async def b69x8ynntf(self, conn=None, _skiplock=False):
        await self.b69x8ynnt8(conn=conn, _skiplock=_skiplock)

    async def select(self, table_name: str, conds: list[dict]=[], cond_sql: str='', targets=[], avoids=[], post=None, conn=None, _skiplock=False, debuglabel=''):
        n69wsp9oq8 = await super().select(table_name, conds=conds, cond_sql=cond_sql, targets=targets, avoids=avoids, post=post, conn=conn, _skiplock=_skiplock, json_repairer=recover_conds, debuglabel=debuglabel)
        return n69wsp9oq8

    async def b69x8ynnvq(self, n69wspa381='', conn=None, _skiplock=False):

        async def b69wsp9mrq(conn):
            n69wspa357 = f"\n            def_id LIKE '{n69wspa381}%' AND \n            CHAR_LENGTH(def_id) - CHAR_LENGTH(REPLACE(def_id, '/', '')) = 1\n            "
            n69wspa36m = await self.select('funcs', cond_sql=n69wspa357, targets=['def_id'], conn=conn, _skiplock=_skiplock)
            n69wspa36m = n69wspa36m['def_id'].tolist()
            n69wspa37t = []
            for name in n69wspa36m:
                n69wspa35o = name.split('/')[-1]
                if x69xm5dtzx(n69wspa35o) != 'folder':
                    pass
                else:
                    n69wspa37t.append(name)
            n69wspa353 = n69wspa357 = "\n            SELECT count(*) FROM funcs WHERE\n                def_id LIKE '{def_id}^1#_/%' AND\n                (SUBSTRING(def_id, LENGTH('{def_id}^1#_/') + 1) NOT LIKE '%/%'\n                AND SUBSTRING(def_id, LENGTH('{def_id}^1#_/') + 1) NOT LIKE '%#%'\n                AND SUBSTRING(def_id, LENGTH('{def_id}^1#_/') + 1) NOT LIKE '%^%'\n                AND SUBSTRING(def_id, LENGTH('{def_id}^1#_/') + 1) NOT LIKE '%*%')"
            n69wspa2i8 = "\n            SELECT count(*) FROM classes WHERE\n                def_id LIKE '{def_id}^1#_*%' AND\n                (SUBSTRING(def_id, LENGTH('{def_id}^1#_*') + 1) NOT LIKE '%/%'\n                AND SUBSTRING(def_id, LENGTH('{def_id}^1#_*') + 1) NOT LIKE '%#%'\n                AND SUBSTRING(def_id, LENGTH('{def_id}^1#_*') + 1) NOT LIKE '%^%'\n                AND SUBSTRING(def_id, LENGTH('{def_id}^1#_*') + 1) NOT LIKE '%*%')"
            n69wspa2rn = 'SELECT count(*) FROM nodes WHERE def_id = :def_id'
            n69wspa2fj = []
            for n69wspa337 in n69wspa37t:
                n69wspa338 = conn.execute(text(n69wspa2rn), {'def_id': n69wspa337})
                n69wspa2m2 = conn.execute(text(n69wspa353.format(def_id=n69wspa337)))
                n69wspa2q5 = conn.execute(text(n69wspa2i8.format(def_id=n69wspa337)))
                n69wspa2fj.append(asyncio.gather(n69wspa338, n69wspa2m2, n69wspa2q5))
            n69wspa2vh = await asyncio.gather(*n69wspa2fj)
            n69wspa36p = {}
            for i in range(len(n69wspa36m)):
                n69wspa36p[n69wspa36m[i]] = {'dag': n69wspa2vh[i][0].scalar(), 'funcs': n69wspa2vh[i][1].scalar(), 'classes': n69wspa2vh[i][2].scalar()}
            for n69wspa38w, n69wspa2vh in n69wspa36p.items():
                for scope in ('dag', 'funcs', 'classes'):
                    if n69wspa36p[n69wspa38w][scope] is None:
                        n69wspa36p[n69wspa38w][scope] = 0
            return n69wspa36p
        n69wsp9onl = await self._batch_read([b69wsp9mrq], _skiplock=_skiplock, conn=conn)
        n69wsp9onl = n69wsp9onl[0]
        return n69wsp9onl

    async def b69x8ynnv9(self, n69wspa2wq, n69wspa2z4=None, conn=None, _skiplock=False):
        n69wspa2qn = to_func_id(n69wspa2wq)

        async def b69wsp9mrq(conn):
            n69wspa2r3 = await self.b69x8ynnw4(n69wspa2qn, x69xm5dtzv=n69wspa2z4, conn=conn, _skiplock=True)
            n69wspa36m = []
            for n69wsp9ou8 in n69wspa2r3['froms'].split('\n'):
                n69wspa2x4 = n69wsp9ou8[5:].split(' import ')[0].strip()
                n69wspa36m.append(n69wspa2x4)
            for n69wsp9ou8 in n69wspa2r3['imports'].split('\n'):
                n69wspa2x4 = n69wsp9ou8[7:].split(' as ')[0].strip()
                n69wspa36m.append(n69wspa2x4)
            n69wspa2xp = list(set([x69xm5dtzw(mn) for mn in n69wspa36m]))
            n69wspa2kw = [self.select('funcs', conds=[{'def_id': n69wspa2hh}], targets=['def_id'], conn=conn, _skiplock=True) for n69wspa2hh in n69wspa2xp]
            n69wspa2kw = await asyncio.gather(*n69wspa2kw)
            n69wspa2md = [i for i in range(len(n69wspa2kw)) if len(n69wspa2kw[i]) > 0]
            n69wspa391 = [n69wspa2xp[i] for i in range(len(n69wspa2xp)) if i in n69wspa2md]
            n69wspa2ky = ['[ENV]>' + n69wspa2xp[i].lstrip('>') for i in range(len(n69wspa2xp)) if not i in n69wspa2md]
            return (n69wspa391, n69wspa2ky)
        n69wsp9otg = await self._batch_read([b69wsp9mrq], conn=conn, _skiplock=True)
        return n69wsp9otg[0]

    async def b69x8ynnvo(self, n69wsp9p0w, n69wspa2wq, n69wspa2y5, src_handle, src_x, src_y, n69wsp9osu=NODESPACINGX, n69wsp9ozl=NODESPACINGY, save_new_node=True):
        n69wspa2e0 = copy.deepcopy(n69wspa2ko)
        n65d20cda3 = gen_base36_id()
        n69wspa2e0['id'] = n65d20cda3
        n69wspa2e0['data']['uid'] = n65d20cda3
        n69wspa2e0['data']['node_type'] = n69wsp9p0w
        n69wspa2e0['data']['def_id'] = n69wspa2wq
        n69wspa2e0['data']['handle_in'] = '@target'
        n69wsp9p3w = None
        if not n69wsp9p0w in nesttypes:
            n69wspa2e0['type'] = 'plain'
            n69wspa2e0['data']['handle_outs'] = {'_': '@source'}
        else:
            n69wspa2e0['type'] = 'switch'
            if n69wsp9p0w == 'if':
                n69wspa2e0['data']['handle_outs'] = {'true': '#True@source', 'false': '#False@source'}
                n69wspa2e0['data']['subWorkflowIds'] = {'true': {'funcs': f'{n69wspa2wq}^{n65d20cda3}#True:funcs', 'classes': f'{n69wspa2wq}^{n65d20cda3}#True:classes'}, 'false': {'funcs': f'{n69wspa2wq}^{n65d20cda3}#False:funcs', 'classes': f'{n69wspa2wq}^{n65d20cda3}#False:classes'}}
                n69wspa2e0['data']['tool_counts'] = {True: {'funcs': 0, 'classes': 0}, False: {'funcs': 0, 'classes': 0}}
            elif n69wsp9p0w in ('match', 'excepts'):
                n69wspa2e0['data']['handle_outs'] = {0: '#0@source'}
                n69wspa2e0['data']['subWorkflowIds'] = {0: {'funcs': f'{n69wspa2wq}^{n65d20cda3}#0:funcs', 'classes': f'{n69wspa2wq}^{n65d20cda3}#0:classes'}}
                n69wspa2e0['data']['tool_counts'] = {0: {'funcs': 0, 'classes': 0}}
            else:
                n69wspa2e0['data']['handle_outs'] = {'_': '#_@source'}
                n69wspa2e0['data']['subWorkflowIds'] = {'_': {'funcs': f'{n69wspa2wq}^{n65d20cda3}#_:funcs', 'classes': f'{n69wspa2wq}^{n65d20cda3}#_:classes'}}
                n69wspa2e0['data']['tool_counts'] = {'_': {'funcs': 0, 'classes': 0}}
            n69wsp9p3w = self.b69wspa0xu(n69wspa2wq, n65d20cda3, n69wsp9p0w, src_x + n69wsp9osu * 2, src_y)
        n69wspa2ww = {'id': f'{n69wspa2y5}~{n65d20cda3}', 'source': n69wspa2y5, 'sourceHandle': src_handle, 'target': n65d20cda3, 'targetHandle': '@target'}
        n69wspa2mr = src_x + n69wsp9osu
        n69wspa32k = src_y
        if not '#' in src_handle:
            pass
        else:
            assert src_handle.count('#') == 1
            assert src_handle.count('@') == 1
            n69wspa2w8 = src_handle[src_handle.find('#') + 1:src_handle.find('@')]
            if n69wspa2w8 == 'False':
                n69wspa32k = n69wspa32k + n69wsp9ozl
            elif n69wspa2w8.isdigit():
                n69wspa32k = n69wspa32k + int(n69wspa2w8) * n69wsp9ozl
            else:
                pass
        n69wspa2e0['position'] = {'x': n69wspa2mr, 'y': n69wspa32k}
        n69wspa2e0['positionAbsolute'] = {'x': n69wspa2mr, 'y': n69wspa32k}
        n69wspa2e0['style'] = n69wspa38t[n69wsp9p0w]
        match n69wsp9p0w:
            case 'start':
                raise ValueError(f'eh002')
            case 'code':
                pass
            case 'tool':
                n69wspa2e0['type'] = 'tool'
                n69wspa2e0['data']['params_map'] = []
                n69wspa2e0['data']['toolcall'] = {'obj': None, 'func': '', 'class': None}
                n69wspa2e0['data']['misc'] = {'do_await': False}
                n69wspa2e0['data']['targets'] = '_'
            case 'tool_conc':
                n69wspa2e0['type'] = 'tool_conc'
                n69wspa2e0['data']['toolcall'] = {'obj': None, 'func': '', 'class': None}
                n69wspa2e0['data']['params_map'] = []
                n69wspa2e0['data']['targets'] = '_'
                n69wspa2e0['data']['misc'] = {}
            case 'return':
                n69wspa2e0['data']['expr'] = ''
            case 'yield':
                n69wspa2e0['data']['expr'] = ''
            case 'code':
                n69wspa2e0['code'] = ''
            case n if n in ('continue', 'break', 'pass'):
                pass
            case n if n in ('while', 'if'):
                n69wspa2e0['data']['expr'] = ''
            case 'match':
                n69wspa2e0['data']['expr'] = ''
                n69wspa2e0['data']['cases'] = {0: {'expr': '', 'spawn_vars': []}}
            case 'excepts':
                n69wspa2e0['data']['cases'] = {0: {'expr': None, 'exception_var': {'id': None, 'type': None}}}
            case n if n in ('try', 'finally'):
                pass
            case 'with':
                n69wspa2e0['data']['expr'] = ''
                n69wspa2e0['data']['misc'] = {'is_async': False}
            case 'for':
                n69wspa2e0['data']['iter'] = ''
                n69wspa2e0['data']['targets'] = ''
                n69wspa2e0['data']['misc'] = {'is_async': False}
        n69wsp9oya = [n69wspa2e0] if n69wsp9p3w is None else [n69wspa2e0, n69wsp9p3w]
        n69wspa34y = [n69wspa2ww]
        if n69wsp9p0w in nesttypes and save_new_node == True:
            n69wspa2fe = pd.DataFrame([{**n69wspa2e0['data'], 'hid': f"1.{n69wspa2e0['data']['uid']}", 'branch': '_', 'xpos': n69wspa2e0['position']['x'], 'ypos': n69wspa2e0['position']['y']}])
            await self.upsert('nodes', n69wspa2fe)
        return (n69wsp9oya, n69wspa34y)

    def b69wspa0xu(self, n69wspa2wq, n65d20cda3, n69wsp9p0w, x, y):
        n69wsp9p3w = copy.deepcopy(n69wspa2ko)
        n69wsp9p3w['id'] = n65d20cda3 + '-end' + n69wsp9p0w
        n69wsp9p3w['type'] = 'plain'
        n69wsp9p3w['data']['uid'] = n65d20cda3 + '-end' + n69wsp9p0w
        n69wsp9p3w['data']['node_type'] = 'end' + n69wsp9p0w
        n69wsp9p3w['data']['def_id'] = n69wspa2wq
        n69wsp9p3w['data']['handle_in'] = '@target'
        n69wsp9p3w['data']['handle_outs'] = {'_': '@source'}
        n69wsp9p3w['position'] = {'x': x, 'y': y}
        n69wsp9p3w['positionAbsolute'] = {'x': x, 'y': y}
        n69wsp9p3w['style'] = n69wspa38t['end' + n69wsp9p0w]
        return n69wsp9p3w

    async def b69x8ynnvv(self, n69wspa2wq, choice, conn=None, _skiplock=False):
        assert '*' in n69wspa2wq or '/' in n69wspa2wq
        if choice == 'func':
            n69wspa34p = 'funcs'
        elif choice == 'class':
            n69wspa34p = 'classes'
        elif not choice:
            n69wspa2lm = n69wspa2wq.rfind('*')
            n69wspa2sg = n69wspa2wq.rfind('/')
            if n69wspa2lm > n69wspa2sg:
                n69wspa34p = 'classes'
            else:
                n69wspa34p = 'funcs'
        n69wspa357 = f"def_id = '{n69wspa2wq}'"
        if n69wspa34p == 'funcs':
            n69wsp9oya, n69wsp9osv = await asyncio.gather(self.select(n69wspa34p, cond_sql=n69wspa357, conn=conn, _skiplock=_skiplock), self.b69x8ynnt2(n69wspa2wq, conn=conn, _skiplock=_skiplock))
        else:
            n69wsp9oya = await self.select(n69wspa34p, cond_sql=n69wspa357, conn=conn, _skiplock=_skiplock)
        if len(n69wsp9oya) == 0:
            pass
        elif len(n69wsp9oya) > 1:
            pass
        if n69wspa34p == 'funcs':
            n69wsp9ozv = {i: n69wsp9osv['inputs'][i] for i in range(len(n69wsp9osv['inputs']))}
            n69wsp9onl = n69wsp9osv['return']
            n69wsp9oya['inputs'] = [n69wsp9ozv for i in range(len(n69wsp9oya))]
            n69wsp9oya['return'] = [n69wsp9onl for i in range(len(n69wsp9oya))]
        return n69wsp9oya

    async def b69x8ynntu(self, n69wspa2wq, include_self=True, params_style='embedded', conn=None, _skiplock=False):
        assert x69xm5dtzx(n69wspa2wq) == 'func'
        n69wspa34p = 'funcs'
        n69wspa357 = f"def_id LIKE '{n69wspa2wq}^%'"
        if include_self:
            n69wspa357 = f"def_id = '{n69wspa2wq}' OR ({n69wspa357})"
        n69wsp9oya = await self.select(n69wspa34p, cond_sql=n69wspa357, conn=conn, _skiplock=_skiplock)
        n69wspa2gx = n69wsp9oya['def_id'].tolist()
        if include_self:
            if not n69wspa2wq in n69wspa2gx:
                raise RuntimeError(f'eh003{n69wspa2wq}。查到了{n69wspa2gx}')
        if params_style == 'embedded':
            n69wspa35t = [self.b69x8ynnt2(n69wsp9oq0, conn=conn, _skiplock=_skiplock) for n69wsp9oq0 in n69wspa2gx]
            n69wsp9osv = await asyncio.gather(*n69wspa35t)
            n69wsp9ozv = [{i: p['inputs'][i] for i in range(len(p['inputs']))} for p in n69wsp9osv]
            n69wsp9onl = [p['return'] for p in n69wsp9osv]
            n69wsp9oya['inputs'] = n69wsp9ozv
            n69wsp9oya['return'] = n69wsp9onl
            return n69wsp9oya
        else:
            n69wspa2x2 = [f"def_id = '{did}'" for did in n69wspa2gx]
            n69wspa2x2 = ' OR '.join(n69wspa2x2)
            n69wsp9osv = await self.select('params', cond_sql=n69wspa2x2, conn=conn, _skiplock=_skiplock)
            return (n69wsp9oya, n69wsp9osv)

    async def b69x8ynnu2(self, n69wspa2wq, params_style='embedded', targets=[], conn=None, _skiplock=False):
        assert x69xm5dtzx(n69wspa2wq) in ('cond', 'class', 'folder'), n69wspa2wq
        n69wspa34p = 'funcs'
        if n69wspa2wq in (None, '', 'ROOT^1#_'):
            n69wspa357 = "def_id NOT LIKE '%^%'"
        else:
            n69wspa357 = f"def_id LIKE '{n69wspa2wq}/%' AND\n                (SUBSTRING(def_id, LENGTH('{n69wspa2wq}/') + 1) NOT LIKE '%/%'\n                AND SUBSTRING(def_id, LENGTH('{n69wspa2wq}/') + 1) NOT LIKE '%#%'\n                AND SUBSTRING(def_id, LENGTH('{n69wspa2wq}/') + 1) NOT LIKE '%^%'\n                AND SUBSTRING(def_id, LENGTH('{n69wspa2wq}/') + 1) NOT LIKE '%*%')"
        n69wsp9oya = await self.select(n69wspa34p, cond_sql=n69wspa357, targets=targets, conn=conn, _skiplock=_skiplock, debuglabel='get_funcs_under_scope (1)')
        if params_style == 'skip':
            return n69wsp9oya
        n69wspa2gx = n69wsp9oya['def_id'].tolist()
        if params_style == 'embedded':
            n69wspa35t = [self.b69x8ynnt2(n69wsp9oq0, conn=conn, _skiplock=_skiplock) for n69wsp9oq0 in n69wspa2gx]
            n69wsp9osv = await asyncio.gather(*n69wspa35t)
            n69wsp9ozv = [p['inputs'] for p in n69wsp9osv]
            n69wsp9onl = [p['return'] for p in n69wsp9osv]
            n69wsp9oya['inputs'] = n69wsp9ozv
            n69wsp9oya['return'] = n69wsp9onl
            return n69wsp9oya
        else:
            n69wspa2x2 = [f"def_id = '{did}'" for did in n69wspa2gx]
            n69wspa2x2 = ' OR '.join(n69wspa2x2)
            if n69wspa2x2:
                n69wsp9osv = await self.select('params', cond_sql=n69wspa2x2, conn=conn, _skiplock=_skiplock, debuglabel='get_funcs_under_scope (2)')
                n69wsp9osv = pd.concat([sub_df.sort_values(by='place') for _, sub_df in n69wsp9osv.groupby('def_id', sort=False)], ignore_index=True)
            else:
                n69wsp9osv = pd.DataFrame(columns=list(n69wspa356.keys()))
            return (n69wsp9oya, n69wsp9osv)

    async def b69x8ynnuy(self, n69wspa2wq, conn=None, _skiplock=False):
        assert x69xm5dtzx(n69wspa2wq) == 'func'
        n69wspa34p = 'classes'
        n69wspa357 = f"def_id LIKE '{n69wspa2wq}^%'"
        n69wsp9oya = await self.select(n69wspa34p, cond_sql=n69wspa357, conn=conn, _skiplock=_skiplock, debuglabel='get_all_classes_under_func')
        return n69wsp9oya

    async def b69x8ynntl(self, n69wspa2wq, targets=[], conn=None, _skiplock=False):
        assert not n69wspa2wq in (None, '', 'ROOT'), 'eh006'
        assert x69xm5dtzx(n69wspa2wq) == 'cond'
        n69wspa357 = f"def_id LIKE '{n69wspa2wq}*%' AND\n            (SUBSTRING(def_id, LENGTH('{n69wspa2wq}*') + 1) NOT LIKE '%/%'\n            AND SUBSTRING(def_id, LENGTH('{n69wspa2wq}*') + 1) NOT LIKE '%#%'\n            AND SUBSTRING(def_id, LENGTH('{n69wspa2wq}*') + 1) NOT LIKE '%^%'\n            AND SUBSTRING(def_id, LENGTH('{n69wspa2wq}*') + 1) NOT LIKE '%*%')"
        n69wspa34p = 'classes'
        n69wsp9oya = await self.select(n69wspa34p, cond_sql=n69wspa357, targets=targets, conn=conn, _skiplock=_skiplock, debuglabel='get_classes_under_scope')
        return n69wsp9oya

    async def b69x8ynntc(self, n69wspa2wq, conn=None, _skiplock=False):
        assert x69xm5dtzx(n69wspa2wq) == 'func'
        n69wspa34p = 'nodes'
        n69wspa357 = f"def_id = '{n69wspa2wq}' OR def_id LIKE '{n69wspa2wq}^%'"
        n69wsp9oya = await self.select(n69wspa34p, cond_sql=n69wspa357, conn=conn, _skiplock=_skiplock)
        return n69wsp9oya

    async def b69x8ynnt5(self, n69wspa2wq, n69wsp9p51, conn=None, _skiplock=False):
        n69wspa2pl = n69wspa2wq + '^' + n69wsp9p51
        cond_sql = f"def_id LIKE '{n69wspa2pl}#%'"
        n69wspa34c = self.select('nodes', cond_sql=cond_sql, conn=conn, _skiplock=_skiplock)
        n69wsp9p2u = self.select('funcs', cond_sql=cond_sql, conn=conn, _skiplock=_skiplock)
        n69wspa2xf = self.select('classes', cond_sql=cond_sql, conn=conn, _skiplock=_skiplock)
        n69wspa2d3 = self.select('params', cond_sql=cond_sql, conn=conn, _skiplock=_skiplock)
        n69wspa34c, n69wsp9p2u, n69wspa2xf, n69wspa2d3 = await asyncio.gather(n69wspa34c, n69wsp9p2u, n69wspa2xf, n69wspa2d3)
        return (n69wspa34c, n69wsp9p2u, n69wspa2xf, n69wspa2d3)

    async def b69x8ynnvw(self, n69wspa2wq, conn=None, _skiplock=False):
        assert x69xm5dtzx(n69wspa2wq) == 'func'
        n69wspa34p = 'vars'
        n69wspa357 = f"def_id = '{n69wspa2wq}' OR def_id LIKE '{n69wspa2wq}^%'"
        n69wsp9oya = await self.select(n69wspa34p, cond_sql=n69wspa357, conn=conn, _skiplock=_skiplock)
        return n69wsp9oya

    async def b69x8ynnvb(self, n69wspa2wq, conn=None, _skiplock=False):
        assert x69xm5dtzx(n69wspa2wq) in ('func', 'cond'), n69wspa2wq
        if x69xm5dtzx(n69wspa2wq) == 'cond':
            assert n69wspa2wq.endswith('^1#_')
            n69wspa2wq = n69wspa2wq[:-4]
        assert not n69wspa2wq in (None, '', 'ROOT'), 'eh006'
        n69wspa357 = f"def_id = '{n69wspa2wq}'"
        n69wspa34p = 'nodes'
        n69wsp9oya = await self.select(n69wspa34p, cond_sql=n69wspa357, conn=conn, _skiplock=_skiplock)
        return n69wsp9oya

    async def b69x8ynnt2(self, n69wspa2wq, conn=None, _skiplock=False):
        assert x69xm5dtzx(n69wspa2wq) in ('folder', 'func')
        n69wspa352 = f"def_id = '{n69wspa2wq}' AND\n            ctx = 'input'"
        n69wspa32m = f"def_id = '{n69wspa2wq}' AND\n            ctx = 'return'"
        n69wspa34p = 'params'
        n69wspa35t = [asyncio.create_task(self.select(n69wspa34p, cond_sql=n69wspa352, conn=conn, _skiplock=_skiplock)), asyncio.create_task(self.select(n69wspa34p, cond_sql=n69wspa32m, conn=conn, _skiplock=_skiplock))]
        n69wspa2me = await asyncio.gather(*n69wspa35t)
        n69wspa314 = n69wspa2me[0].sort_values(by='place').drop(['place'], axis=1).to_dict(orient='records')
        n69wspa2p9 = n69wspa2me[1].to_dict(orient='records')
        if len(n69wspa2p9) == 0:
            n69wspa2p9 = [{}]
        elif len(n69wspa2p9) > 1:
            pass
        return {'inputs': n69wspa314, 'return': n69wspa2p9[0]}

    async def b69x8ynnuh(self, n69wspa2mh, n69wsp9p0w=None, conn=None, _skiplock=False):
        assert not (n69wspa2mh.get('node_type') and n69wsp9p0w and (n69wspa2mh.get('node_type') != n69wsp9p0w))
        n69wsp9p0w = n69wspa2mh.get('node_type') or n69wsp9p0w
        if not (n69wsp9p0w in nesttypes or n69wsp9p0w in ['func', 'class']):
            return {}
        cases = ['_']
        if n69wsp9p0w == 'if':
            cases = [True, False]
        elif n69wsp9p0w in ('match', 'excepts'):
            cases = list(n69wspa2mh['cases'].keys())
        if n69wsp9p0w == 'func':
            n69wsp9p51 = '1'
        elif n69wsp9p0w == 'class':
            pass
        else:
            n65d20cda3 = n69wspa2mh['uid']
            n69wsp9p51, _ = await self.b69x8ynnvc(n65d20cda3, conn=conn, _skiplock=_skiplock)
        n69wspa2fd = {}

        async def b69x8ynnug(acase):
            nonlocal n69wspa2fd
            n69wspa2cy = {}
            n69wsp9p3x = f"{n69wspa2mh['def_id']}^{n69wsp9p51}#{acase}" if not n69wsp9p0w == 'class' else n69wspa2mh['def_id']
            n69wspa37y = f"def_id LIKE '{n69wsp9p3x}/%' AND\n            (SUBSTRING(def_id, LENGTH('{n69wsp9p3x}/') + 1) NOT LIKE '%/%'\n            AND SUBSTRING(def_id, LENGTH('{n69wsp9p3x}/') + 1) NOT LIKE '%#%'\n            AND SUBSTRING(def_id, LENGTH('{n69wsp9p3x}/') + 1) NOT LIKE '%^%'\n            AND SUBSTRING(def_id, LENGTH('{n69wsp9p3x}/') + 1) NOT LIKE '%*%')"
            n69wspa2qa = f"def_id LIKE '{n69wsp9p3x}*%' AND\n            (SUBSTRING(def_id, LENGTH('{n69wsp9p3x}*') + 1) NOT LIKE '%/%'\n            AND SUBSTRING(def_id, LENGTH('{n69wsp9p3x}*') + 1) NOT LIKE '%#%'\n            AND SUBSTRING(def_id, LENGTH('{n69wsp9p3x}*') + 1) NOT LIKE '%^%'\n            AND SUBSTRING(def_id, LENGTH('{n69wsp9p3x}*') + 1) NOT LIKE '%*%')" if n69wsp9p0w != 'class' else None
            n69wspa2kc = f"def_id = '{n69wspa2mh['def_id']}'" if n69wsp9p0w == 'func' else None
            n69wspa37q = self.select('funcs', cond_sql=n69wspa37y, targets=['uid'], conn=conn, _skiplock=_skiplock)
            n69wspa2i3 = self.select('classes', cond_sql=n69wspa2qa, targets=['uid'], conn=conn, _skiplock=_skiplock) if n69wsp9p0w != 'class' else aidle()
            n69wspa2qb = self.select('nodes', cond_sql=n69wspa2kc, targets=['uid'], conn=conn, _skiplock=_skiplock) if n69wsp9p0w == 'func' else aidle()
            n69wspa37q, n69wspa2i3, n69wspa2qb = await asyncio.gather(n69wspa37q, n69wspa2i3, n69wspa2qb)
            n69wspa2cy = {'funcs': len(n69wspa37q)}
            if n69wsp9p0w != 'class':
                n69wspa2cy['classes'] = len(n69wspa2i3)
            if n69wsp9p0w == 'func':
                n69wspa2cy['nodes'] = len(n69wspa2qb)
            n69wspa2fd[acase] = n69wspa2cy
        await asyncio.gather(*[b69x8ynnug(c) for c in cases])
        return n69wspa2fd

    async def b69x8ynnup(self, n69wspa2ru, conn=None, _skiplock=False, n69wsp9oo8=None):
        n69wspa2kx = n69wspa2ru['uid']
        n69wspa2ci = 'switch' if n69wspa2ru['node_type'] in nesttypes else n69wspa2ez.get(n69wspa2ru['node_type'], 'plain')
        n69wsp9opc = {'id': n69wspa2kx, 'type': n69wspa2ci, 'position': {'x': n69wspa2ru['xpos'], 'y': n69wspa2ru['ypos']}, 'style': n69wspa38t[n69wspa2ru['node_type']], 'data': n69wspa2ru}
        del n69wspa2ru['xpos']
        del n69wspa2ru['ypos']
        n69wspa2ru['handle_in'] = '@target'
        n69wspa2ru['handle_outs'] = {}
        if n69wspa2ru['node_type'] in ('match', 'excepts'):
            n69wspa2ru['handle_outs'] = {c: f'#{c}@source' for c in n69wspa2ru['cases'].keys()}
        elif n69wspa2ru['node_type'] == 'if':
            n69wspa2ru['handle_outs'] = {c: f'#{c}@source' for c in [True, False]}
        elif n69wspa2ru['node_type'] in nesttypes:
            n69wspa2ru['handle_outs'] = {c: f'#{c}@source' for c in ['_']}
        else:
            n69wspa2ru['handle_outs'] = {'_': f'@source'}
        if n69wspa2ru['node_type'] in ['tool', 'tool_conc']:
            if isinstance(n69wspa2ru['params_map'], dict):
                n69wspa2lr = []
                pi = 0
                for k, v in n69wspa2ru['params_map'].items():
                    n69wspa2lr.append([k, v])
                    pi = pi + 1
                n69wspa2ru['params_map'] = n69wspa2lr
        n69wspa393 = []
        if n69wspa2ru['node_type'].startswith('end'):
            assert '-end' in n69wspa2ru['hid']
            n69wspa2lo = n69wspa2ru['hid'].split('-end')[0]
            n69wspa357 = f"def_id = '{n69wspa2ru['def_id']}' AND\n                hid LIKE '{n69wspa2lo}.%' AND\n                (SUBSTRING(hid, LENGTH('{n69wspa2lo}.') + 1) NOT LIKE '%.%') AND\n                NOT (node_type LIKE 'end%')\n                "
            n69wspa34p = 'nodes'
            if n69wsp9oo8 is None:
                n69wsp9otu = (await self.select(n69wspa34p, cond_sql=n69wspa357, conn=conn, _skiplock=_skiplock)).to_dict(orient='records')
            else:
                n69wsp9otu = n69wsp9oo8[n69wsp9oo8['hid'].str.startswith(n69wspa2lo + '.') & ~n69wsp9oo8['hid'].str[len(n69wspa2lo) + 1:].str.contains('\\.') & ~n69wsp9oo8['node_type'].str.contains('end')].to_dict(orient='records')
            n69wspa2yz = []
            for child in n69wsp9otu:
                if not child.get('nexts'):
                    n69wspa2y5 = child['uid']
                    if child['node_type'] in nesttypes:
                        n69wspa2y5 = child['uid'] + '-end' + child['node_type']
                    n69wspa2si = '@source'
                    n69wspa2z9 = n69wspa2ru['handle_in']
                    n69wspa393.append({'id': f"{n69wspa2y5}~{n69wsp9opc['id']}", 'source': n69wspa2y5, 'sourceHandle': n69wspa2si, 'target': n69wsp9opc['id'], 'targetHandle': n69wspa2z9})
        else:
            n69wspa357 = f'''def_id = '{n69wspa2ru['def_id']}' AND\n                JSON_CONTAINS(nexts, '"{n69wspa2ru['hid']}"', '$')\n                '''
            n69wspa34p = 'nodes'
            if n69wsp9oo8 is None:
                n69wsp9orp = (await self.select(n69wspa34p, cond_sql=n69wspa357, conn=conn, _skiplock=_skiplock)).to_dict(orient='records')
            else:
                n69wspa2i6 = n69wsp9oo8['nexts'].apply(lambda x: n69wspa2ru['hid'] in x if isinstance(x, (list, tuple, set)) else False)
                n69wsp9orp = n69wsp9oo8[n69wspa2i6].to_dict(orient='records')
            if not n69wsp9orp:
                if n69wspa2ru['hid'].count('.') == 1:
                    pass
                else:
                    assert '.' in n69wspa2ru['hid']
                    n69wspa36x = n69wspa2ru['hid'][:n69wspa2ru['hid'].rfind('.')]
                    n69wspa357 = f"def_id = '{n69wspa2ru['def_id']}' AND\n                        hid = '{n69wspa36x}'\n                        "
                    n69wspa34p = 'nodes'
                    if n69wsp9oo8 is None:
                        n69wsp9oz5 = (await self.select(n69wspa34p, cond_sql=n69wspa357, conn=conn, _skiplock=_skiplock)).to_dict(orient='records')
                    else:
                        n69wsp9oz5 = n69wsp9oo8[n69wsp9oo8['hid'] == n69wspa36x].to_dict(orient='records')
                    if not len(n69wsp9oz5) == 1:
                        pass
                    else:
                        n69wsp9oz5 = n69wsp9oz5[0]
                        n69wspa2la = n69wsp9oz5['uid']
                        n69wspa2si = f"#{rectify_cond(n69wspa2ru['branch'])}@source"
                        n69wspa2z9 = n69wspa2ru['handle_in']
                        n69wspa393.append({'id': f"{n69wspa2la}~{n69wsp9opc['id']}", 'source': n69wspa2la, 'sourceHandle': n69wspa2si, 'target': n69wsp9opc['id'], 'targetHandle': n69wspa2z9})
            else:
                for n69wsp9os5 in n69wsp9orp:
                    assert n69wsp9os5['branch'] == n69wspa2ru['branch'], f'eh007:{n69wspa2ru},\n{n69wsp9os5}'
                    n69wspa361 = n69wsp9os5['uid']
                    if n69wsp9os5['node_type'] in nesttypes:
                        n69wspa361 = n69wspa361 + '-end' + n69wsp9os5['node_type']
                    n69wspa2si = '@source'
                    n69wspa2z9 = n69wspa2ru['handle_in']
                    n69wspa393.append({'id': f"{n69wspa361}~{n69wsp9opc['id']}", 'source': n69wspa361, 'sourceHandle': n69wspa2si, 'target': n69wsp9opc['id'], 'targetHandle': n69wspa2z9})
        if n69wspa2ru['node_type'] in nesttypes:
            cases = ['_']
            n69wspa2dm = {}
            if n69wspa2ru['node_type'] == 'if':
                cases = [True, False]
            elif n69wspa2ru['node_type'] in ('match', 'excepts'):
                cases = list(n69wspa2ru['cases'].keys())
            for case in cases:
                n69wspa2dm[case] = {'funcs': f"{n69wspa2ru['def_id']}^{n69wspa2ru['uid']}#{case}:funcs", 'classes': f"{n69wspa2ru['def_id']}^{n69wspa2ru['uid']}#{case}:classes"}
            n69wspa2ru['subWorkflowIds'] = n69wspa2dm
        n69wspa2ru['vars'] = []
        return (n69wsp9opc, n69wspa393)

    async def b69x8ynnta(self, n69wsp9p12, scopetype, count_previews=False, conn=None, _skiplock=False):
        right = scopetype
        n69wsp9p12 = n69wsp9p12.replace(np.nan, None)
        n69wsp9oya = n69wsp9p12.to_dict(orient='records')
        n69wspa2tn = []
        n69wspa30n = []
        n69wspa34y = []
        if right == 'dag':
            n69wspa2tg = copy.deepcopy(n69wsp9oya)
            n69wspa35t = [self.b69x8ynnup(n69wspa2ru, n69wsp9oo8=n69wsp9p12, conn=conn, _skiplock=_skiplock) for n69wspa2ru in n69wspa2tg]

            async def b69x8ynnur(n):
                n['tool_counts'] = await self.b69x8ynnuh(n, conn=conn, _skiplock=_skiplock)
            if count_previews:
                n69wspa2zo = asyncio.gather(*[b69x8ynnur(n) for n in n69wspa2tg])
            else:
                n69wspa2zo = aidle()
            n69wspa2kl = await asyncio.gather(n69wspa2zo, *n69wspa35t)
            n69wspa2kl = n69wspa2kl[1:]
            n69wspa2je = [n69wsp9p6g[0] for n69wsp9p6g in n69wspa2kl]
            n69wspa2hx = [n69wsp9p6g[1] for n69wsp9p6g in n69wspa2kl]
            n69wspa34y = [item for sublist in n69wspa2hx for item in sublist]
            n69wspa2tn = n69wspa2je
        elif right == 'classes':
            if count_previews:

                async def b69x8ynnur(n):
                    n['tool_counts'] = await self.b69x8ynnuh(n, n69wsp9p0w='class', conn=conn, _skiplock=_skiplock)
                await asyncio.gather(*[b69x8ynnur(n) for n in n69wsp9oya])
            for n69wspa2ru in n69wsp9oya:
                n69wspa2ru = copy.deepcopy(n69wspa2ru)
                n69wspa2kx = n69wspa2ru['uid']
                n69wspa2ci = 'switch'
                n69wspa2ru['node_type'] = 'class'
                n69wsp9opc = {'id': n69wspa2kx, 'type': n69wspa2ci, 'position': {'x': n69wspa2ru['xpos'], 'y': n69wspa2ru['ypos']}, 'style': n69wspa2tr, 'data': n69wspa2ru}
                del n69wspa2ru['xpos']
                del n69wspa2ru['ypos']
                n69wspa2ru['bases'] = ','.join(n69wspa2ru.get('bases') or [])
                n69wspa2dm = {}
                n69wspa2dm['_'] = {'funcs': f"{n69wspa2ru['def_id']}:funcs"}
                n69wspa2ru['subWorkflowIds'] = n69wspa2dm
                n69wspa2tn.append(n69wsp9opc)
        elif right == 'funcs':
            if count_previews:

                async def b69x8ynnur(n):
                    n['tool_counts'] = await self.b69x8ynnuh(n, n69wsp9p0w='func', conn=conn, _skiplock=_skiplock)
                await asyncio.gather(*[b69x8ynnur(n) for n in n69wsp9oya])
            for n69wspa2ru in n69wsp9oya:
                n69wspa2ru = copy.deepcopy(n69wspa2ru)
                n69wspa2kx = n69wspa2ru['uid']
                n69wspa2ci = 'switch'
                n69wspa2ru['node_type'] = 'func'
                n69wsp9opc = {'id': n69wspa2kx, 'type': n69wspa2ci, 'position': {'x': n69wspa2ru['xpos'], 'y': n69wspa2ru['ypos']}, 'style': n69wspa2i9, 'data': n69wspa2ru}
                del n69wspa2ru['xpos']
                del n69wspa2ru['ypos']
                n69wspa2ru['globals'] = ','.join(n69wspa2ru.get('globals') or [])
                n69wspa2ru['nonlocals'] = ','.join(n69wspa2ru.get('nonlocals') or [])
                n69wspa2dm = {}
                n69wspa2dm['_'] = {'funcs': f"{n69wspa2ru['def_id']}^1#_:funcs", 'classes': f"{n69wspa2ru['def_id']}^1#_:classes", 'dag': f"{n69wspa2ru['def_id']}:dag"}
                n69wspa2ru['subWorkflowIds'] = n69wspa2dm
                n69wspa2tn.append(n69wsp9opc)
        n69wsp9onl = {'nodes': n69wspa2tn, 'edges': n69wspa34y}
        return n69wsp9onl

    async def b69x8ynnvx(self, n69wspa2wq, choice, count_previews=False, to_bouncer=False, _skiplock=False, conn=None):
        if '/' in n69wspa2wq:
            if choice != 'dag':
                assert '^' in n69wspa2wq and '#' in n69wspa2wq, n69wspa2wq
                n69wspa2jo = (n69wspa2wq.rfind('^') + 1, n69wspa2wq.rfind('#'))
                assert n69wspa2jo[0] < n69wspa2jo[1]
                n69wspa379 = n69wspa2wq[n69wspa2jo[0]:n69wspa2jo[1]]
                if not '.' in n69wspa379 and n69wspa379 != '1':
                    last_nodehid, _ = await self.b69x8ynnvc(n69wspa379, _skiplock=_skiplock, conn=conn)
                    n69wspa2hb = n69wspa2wq
                    n69wspa2wq = n69wspa2wq[:n69wspa2jo[0]] + last_nodehid + n69wspa2wq[n69wspa2jo[1]:]
            else:
                pass
        else:
            assert not '^' in n69wspa2wq and (not '#' in n69wspa2wq)
        left = n69wspa2wq.strip()
        right = choice.strip()
        n69wsp9oya = None
        if right == 'funcs':
            n69wsp9oya = await self.b69x8ynnu2(left, _skiplock=_skiplock, conn=conn)
        elif right == 'dag':
            n69wsp9oya = await self.b69x8ynnvb(left, _skiplock=_skiplock, conn=conn)
        elif right == 'classes':
            n69wsp9oya = await self.b69x8ynntl(left, _skiplock=_skiplock, conn=conn)
        else:
            raise ValueError(right)
        n69wsp9onl = await self.b69x8ynnta(n69wsp9oya, right, count_previews=count_previews, _skiplock=_skiplock, conn=conn)
        if to_bouncer:
            if choice == 'dag':
                n69wspa2rv.lastdatas[n69wspa2wq] = {1: n69wsp9onl['nodes'], 2: n69wsp9onl['edges'], 'timestamp': 0}
                n69wspa2rv.lastrets[n69wspa2wq] = {1: n69wsp9onl['nodes'], 2: n69wsp9onl['edges']}
        return n69wsp9onl

    def b69wspa0xi(self, n69wspa2xo, n69wspa2hi, keys):
        for k in keys:
            n69wspa2s7 = []
            if n69wspa2xo[k]:
                for x in n69wspa2xo[k]:
                    if not x in n69wspa2hi:
                        n69wspa2s7.append(x)
                    else:
                        pass
                n69wspa2xo[k] = n69wspa2s7
        return n69wspa2xo

    async def b69x8ynnuu(self, n69wspa2wq, conn=None, _skiplock=False, n69wspa30o=False, skipshell=False):
        n69wspa33b = x69xm5dtzx(n69wspa2wq)
        n69wspa2fm = []
        if n69wspa33b in ('class', 'func'):
            n69wspa2sz = '^'
            if n69wspa33b == 'class':
                if n69wspa30o:
                    pass
                n69wspa30o = False
                n69wspa2sz = '/'
            elif n69wspa33b == 'func':
                pass
            if n69wspa30o:
                n69wspa2rz = f"(def_id LIKE '{n69wspa2wq}^%' OR def_id = '{n69wspa2wq}') AND NOT def_id LIKE '{n69wspa2wq}^1#_/%' AND NOT def_id LIKE '{n69wspa2wq}^1#_*%'"
                n69wspa34u = n69wspa2rz
                n69wspa370 = f"def_id LIKE '{n69wspa2wq}^%' AND NOT def_id LIKE '{n69wspa2wq}^1#_%'"
                n69wspa2sj = n69wspa370
                n69wspa2za = n69wspa370
            elif skipshell:
                n69wspa2rz = f"def_id LIKE '{n69wspa2wq}{n69wspa2sz}%' OR def_id = '{n69wspa2wq}'"
                n69wspa34u = n69wspa2rz
                n69wspa370 = n69wspa2sj = n69wspa2za = f"def_id LIKE '{n69wspa2wq}{n69wspa2sz}%'"
            else:
                n69wspa2rz = f"def_id LIKE '{n69wspa2wq}{n69wspa2sz}%' OR def_id = '{n69wspa2wq}'"
                n69wspa34u = n69wspa370 = n69wspa2sj = n69wspa2za = n69wspa2rz
        elif n69wspa33b == 'node':
            n69wspa2qn = n69wspa2wq[:n69wspa2wq.rfind('^')]
            n69wspa2kt = n69wspa2wq[n69wspa2wq.rfind('^') + 1:]
            n69wspa2rz = f"def_id = '{n69wspa2qn}' AND (hid = '{n69wspa2kt}' OR hid LIKE '{n69wspa2kt}.%' OR hid LIKE '{n69wspa2kt}-%')"
            n69wspa34u = n69wspa2rz
            n69wspa370 = f"def_id LIKE '{n69wspa2wq}#%'"
            n69wspa2sj = n69wspa370
            n69wspa2za = n69wspa370

            async def b69x8ynntw(n69wspa2qn, n69wspa2kt, conn):
                if not '.' in n69wspa2kt:
                    return
                n69wspa333 = n69wspa2kt[:n69wspa2kt.rfind('.')]
                n69wspa2wv = self.select('nodes', cond_sql=f"\n                            def_id = '{n69wspa2qn}' AND hid LIKE '{n69wspa333 + '.'}%'\n                            ", targets=['def_id', 'hid', 'pres', 'nexts', 'data_providers'], _skiplock=True, conn=conn)
                n69wspa2wv = await n69wspa2wv
                n69wspa2wv = n69wspa2wv.apply(lambda n69wspa2xo: self.b69wspa0xi(n69wspa2xo, n69wspa2hi=[n69wspa2kt], keys=['pres', 'nexts', 'data_providers']), axis=1)
                n69wspa2j3 = self.upsert('nodes', n69wspa2wv, _skiplock=True, conn=conn, force_update=4)
                await n69wspa2j3
            n69wspa2fm.append(partial(b69x8ynntw, n69wspa2qn=n69wspa2qn, n69wspa2kt=n69wspa2kt))
        elif n69wspa33b == 'cond':
            pass
        else:
            raise ValueError(f'eh004 {n69wspa2wq} 类型为 {n69wspa33b}，eh028')

        async def b69x8ynnu8(n69wspa2wq, conn):
            await asyncio.gather(self.delete('nodes', cond_sql=n69wspa2rz, conn=conn, _skiplock=True), self.delete('vars', cond_sql=n69wspa34u, conn=conn, _skiplock=True), self.delete('funcs', cond_sql=n69wspa370, conn=conn, _skiplock=True), self.delete('classes', cond_sql=n69wspa2sj, conn=conn, _skiplock=True), self.delete('params', cond_sql=n69wspa2za, conn=conn, _skiplock=True))
        n69wspa2ty = partial(b69x8ynnu8, n69wspa2wq=n69wspa2wq)
        gatherables = [n69wspa2ty, *[b69wsp9mp8(conn=conn) for b69wsp9mp8 in n69wspa2fm]]
        await self._batch_write(gatherables, _skiplock=_skiplock, conn=conn)

    async def b69x8ynnvp(self, n69wspa5m6):
        assert not any([s in n69wspa5m6 for s in ('#', '-', '^', '.', '*')])
        assert n69wspa5m6.count('/') == 1, f'eh009：{n69wspa5m6}'
        assert n69wspa5m6.find('/') > n69wspa5m6.find('>'), f'eh009：{n69wspa5m6}'
        data = [{'uid': idgen.generate('m'), 'def_id': n69wspa5m6, 'globals': [], 'nonlocals': [], 'imports_code': '', 'is_async': 0, 'deco_expr': '', 'xpos': 0, 'ypos': 0}]
        await self.upsert('funcs', pd.DataFrame(data))

    async def b69x8ynnvm(self, n69wsp9p12, n69wsp9omb, n69wsp9oni=None, n69wsp9oxa=None, n69wsp9ot1=None, mutable=True, _skiplock=False, conn=None, assert_funcs=True):
        if not mutable:
            n69wsp9p12 = copy.deepcopy(n69wsp9p12)
            n69wsp9omb = copy.deepcopy(n69wsp9omb)
            n69wsp9oni = copy.deepcopy(n69wsp9oni)
            n69wsp9ot1 = copy.deepcopy(n69wsp9ot1)
            n69wsp9oxa = copy.deepcopy(n69wsp9oxa)
        n69wspa32h = n69wsp9omb['def_id'].tolist()
        n69wspa386 = n69wsp9oni['def_id'].tolist()
        if assert_funcs:
            assert set(n69wsp9p12['def_id'].tolist()) - set(n69wsp9omb['def_id'].tolist()) == set(), f"eh010{set(n69wsp9p12['def_id'].tolist()) - set(n69wsp9omb['def_id'].tolist())}"

        async def b69x8ynnu4(conn):
            n69wspa2q9 = [self.b69x8ynnuu(n69wspa2jx, _skiplock=True, conn=conn) for n69wspa2jx in n69wspa32h]
            n69wspa321 = [self.b69x8ynnuu(cid, _skiplock=True, conn=conn) for cid in n69wspa386]
            await asyncio.gather(*n69wspa2q9, *n69wspa321)
            n69wspa2co = self.upsert('nodes', n69wsp9p12, _skiplock=True, conn=conn)
            n69wspa2cm = self.upsert('funcs', n69wsp9omb, _skiplock=True, conn=conn)
            n69wspa2r9 = self.upsert('classes', n69wsp9oni, _skiplock=True, conn=conn)
            n69wspa2xh = self.upsert('params', n69wsp9oxa, _skiplock=True, conn=conn)
            n69wsp9oql = self.upsert('vars', n69wsp9ot1, _skiplock=True, conn=conn) if n69wsp9ot1 is not None else aidle()
            await asyncio.gather(n69wspa2cm, n69wspa2r9)
            await asyncio.gather(n69wspa2co, n69wspa2xh)
            try:
                await asyncio.gather(n69wsp9oql)
            except Exception as e:
                traceback.print_exc()
        gatherables = [b69x8ynnu4]
        await self._batch_write(gatherables, _skiplock=_skiplock, conn=conn)

    async def b69x8ynnt3(self, n69wspa2qn, needvars=True, include_self=True, _skiplock=False, conn=None):
        assert x69xm5dtzx(n69wspa2qn) == 'func'
        n69wspa2xg = self.b69x8ynntu(n69wspa2qn, include_self=include_self, params_style='separate', _skiplock=_skiplock, conn=conn)
        n69wspa36t = self.b69x8ynnuy(n69wspa2qn, _skiplock=_skiplock, conn=conn)
        n69wspa2cj = self.b69x8ynntc(n69wspa2qn, _skiplock=_skiplock, conn=conn)
        n69wspa2h2 = self.b69x8ynnvw(n69wspa2qn, _skiplock=_skiplock, conn=conn) if needvars else aidle(1)
        funcs_params, n69wsp9p6q, n69wsp9oya, n69wsp9osh = await asyncio.gather(n69wspa2xg, n69wspa36t, n69wspa2cj, n69wspa2h2)
        return (n69wsp9oya, funcs_params[0], n69wsp9p6q, funcs_params[1], n69wsp9osh)

    async def b69x8ynnv0(self, n69wspa2qn, n69wspa2nm=[], external_class_ids=[]):
        n69wspa2qc = self.b69x8ynnt3(n69wspa2qn)
        n69wspa383 = aidle()
        data, vision = await asyncio.gather(n69wspa2qc, n69wspa383)
        n69wsp9oya, n69wsp9p3q, n69wsp9p6q, n69wsp9osv, n69wsp9osh = data
        n69wsp9owq, n69wsp9p0n = b69wsp9mnm(n69wsp9oya, n69wsp9p3q, n69wsp9p6q, n69wsp9osv, n69wspa2qn, n69wspa2in=True, n69wsp9ot1=n69wsp9osh)
        return (n69wsp9owq, n69wsp9p0n)

    def b69wspa0xt(self, n69wspa2el):
        if not '#' in n69wspa2el or not '^' in n69wspa2el:
            return n69wspa2el
        n69wspa2ox = n69wspa2el.rfind('#')
        n69wspa2vg = n69wspa2el.rfind('^') + 1
        assert n69wspa2vg < n69wspa2ox, n69wspa2el
        n69wspa2hr = n69wspa2el[n69wspa2vg:n69wspa2ox]
        n69wsp9orn = n69wspa2hr
        if '.' in n69wsp9orn:
            n69wsp9orn = n69wsp9orn.split('.')[-1]
        n69wspa2el = n69wspa2el[:n69wspa2vg] + n69wsp9orn + n69wspa2el[n69wspa2ox:]
        return n69wspa2el

    async def b69x8ynnvz(self, user_input, n69wspa35p, upsert_base, n69wspa381, n69wspa2zf, n69wspa2nm=[], external_class_ids=[], cached=True, ws=None, conn=None, _skiplock=False):

        async def b69x8ynnvu(x):
            pass

        async def b69x8ynnsy():
            return 'Cannot connect with user.'
        n69wspa360 = ws.send_text if ws else b69x8ynnvu
        n69wspa38v = ws.receive_text if ws else b69x8ynnsy
        n69wspa337 = upsert_base.split('^')[0]

        async def b69x8ynnvt(conn):
            n69wspa2ck, _ = await self.b69x8ynntv(n69wspa35p, 'all', n69wspa2zf=n69wspa2zf, style='pure', conn=None, _skiplock=True)
            n69wspa2pw = await self.b69x8ynntd(n69wspa35p, upsert_base, n69wspa381, conn=None, _skiplock=True)
            n69wspa2ck = '# env_level: 0\n' + n69wspa2ck
            n69wspa30j = all_desc_to_nl(n69wspa2pw)
            n69wspa2jq = None
            n69wspa2rt = copy.deepcopy(n69wspa2zf)
            n69wspa2cq = []
            n69wspa35j = upsert_base
            n69wspa2f0 = 2 if x69xm5dtzx(upsert_base) == 'class' else 1
            n69wspa2u8 = upsert_base.split('*')[-1] if x69xm5dtzx(upsert_base) == 'class' else None
            if n69wspa2zf['mode'] in ('insert', 'replace'):
                if n69wspa2zf['mode'] == 'insert':
                    n69wspa312 = ['after']
                elif n69wspa2zf['mode'] == 'replace':
                    n69wspa312 = ['section', 0]
                if x69xm5dtzx(access_nested_data(n69wspa2zf, n69wspa312)) == 'folder':
                    n69wspa2jq = 'dag'
                    assert n69wspa2f0 == 1
                elif x69xm5dtzx(access_nested_data(n69wspa2zf, n69wspa312)) == 'class':
                    n69wspa2jq = 'classes'
                    n69wspa2rt = {'mode': 'tools', 'scope': 'classes'}
                    if n69wspa2zf['mode'] == 'replace':
                        assert n69wspa2zf['section'][0] == n69wspa2zf['section'][1]
                        n69wspa2cq = [n69wspa2zf['section'][0]]
                        n69wspa2rt['dels'] = n69wspa2cq
                    assert n69wspa2f0 == 1
                elif x69xm5dtzx(access_nested_data(n69wspa2zf, n69wspa312)) == 'func':
                    n69wspa2jq = 'funcs'
                    n69wspa2rt = {'mode': 'tools', 'scope': 'funcs'}
                    if n69wspa2zf['mode'] == 'replace':
                        assert n69wspa2zf['section'][0] == n69wspa2zf['section'][1]
                        n69wspa2cq = [n69wspa2zf['section'][0]]
                        n69wspa2rt['dels'] = n69wspa2cq
            elif n69wspa2zf['mode'] == 'allbelow':
                assert x69xm5dtzx(n69wspa2zf['branch']) == 'cond'
                n69wspa2jq = 'dag'
            elif n69wspa2zf['mode'] == 'append':
                assert x69xm5dtzx(n69wspa2zf['shell_id']) in ('cond', 'class')
                n69wspa2rt = {'mode': 'tools', 'scope': n69wspa2zf['scope_type']}
                n69wspa2jq = n69wspa2zf['scope_type']
                assert n69wspa2zf['shell_id'] == upsert_base
                if x69xm5dtzx(n69wspa2zf['shell_id']) == 'class':
                    assert n69wspa2f0 == 2
            n69wsp9orm = None
            await self.coder.start_duty_cycle()
            n69wspa2lh = [{'role': 'system', 'content': 'userquery'}, {'role': 'user', 'content': {'category': n69wspa2jq, 'mode': n69wspa2zf['mode'], 'existing_code': n69wspa2ck, 'user_input': user_input, 'tools_desc': n69wspa30j, 'class_above': n69wspa2u8}}]
            await self.coder.listen_msgs(n69wspa2lh, session_id=n69wspa337)
            n69wsp9oyy = 0
            n69wspa38p = 0
            async for rsp in self.coder.acquiring_rsps(error_behavior='raise', session_id=n69wspa337):
                n69wsp9oyy = n69wsp9oyy + 1
                assert rsp[0]['role'] == 'assistant'
                try:
                    n69wspa34v = custout2dict(rsp[0]['content'], lower=True)
                    n69wspa2ht = ''
                    if n69wspa34v['action'] != 'orchestrate':
                        if n69wsp9oyy > 20:
                            n69wspa2ht = f'你已查询了{n69wsp9oyy}轮，上限为30轮。请尽快生成代码。'
                        if n69wsp9oyy > 30:
                            await n69wspa360(json.dumps({'event': 'info', 'data': 'Failed. Agent run out of rounds.'}, ensure_ascii=False))
                            raise A69wspa0yp('Failed. Agent run out of rounds.')
                        if n69wspa38p > 5:
                            await n69wspa360(json.dumps({'event': 'info', 'data': 'Failed. Agent keeps outputting illegal response.'}, ensure_ascii=False))
                            raise A69wspa0yp('Failed. Agent keeps outputting unparsable response.')
                    await n69wspa360(json.dumps({'event': 'info', 'data': f"Thought:{n69wspa34v.get('explain')}\n\nAction:{n69wspa34v['action']}"}, ensure_ascii=False))
                    if n69wspa34v['action'] == 'orchestrate':
                        if 'code' in n69wspa34v:
                            n69wsp9orm = n69wspa34v['code'].strip()
                        elif '```python' in rsp[0]['content']:
                            n69wsp9orm = rsp[0]['content'][rsp[0]['content'].rfind('```python'):]
                        else:
                            raise ValueError(f'Code unparsable. The orchestrate action requires a param like: [<CODE>]: ```python\n# your code\n```')
                        if n69wsp9orm.startswith('```python'):
                            n69wsp9orm = n69wsp9orm[9:].strip()
                            if n69wsp9orm.count('```') > 0:
                                n69wsp9orm = n69wsp9orm[:n69wsp9orm.find('```')].strip()
                        await n69wspa360(json.dumps({'event': 'info', 'data': f'Code generated and will be updated into the workflow:\n\n' + n69wsp9orm}, ensure_ascii=False))
                        await n69wspa360(json.dumps({'event': 'msg', 'data': 'Updating the workflow. This may take a moment...'}, ensure_ascii=False))
                        break
                    elif n69wspa34v['action'] == 'check_codes':
                        n69wspa2jw = ''
                        n69wspa35k = ''
                        n69wspa2r2 = await jsonformat(n69wspa34v['selections'])
                        for argdic in n69wspa2r2:
                            try:
                                await n69wspa360(json.dumps({'event': 'info', 'data': f'Agent looking up code: {str(argdic)[1:-1]}'}, ensure_ascii=False))
                                n69wsp9p5m = await self.b69x8ynnul(argdic['module'], argdic.get('class') or '', argdic.get('func') or '', n69wspa2pw['visibility'], n69wspa381, current_space=int(argdic.get('env_level') or 0), conn=conn, _skiplock=True)
                                n69wspa2jw = n69wspa2jw + f"----- {str({k: v for k, v in argdic.items() if k != 'env_level'})[1:-1]} -----\n"
                                n69wspa2jw = n69wspa2jw + n69wsp9p5m
                                n69wsp9p2h = '<NOT_FOUND>' not in n69wsp9p5m[:20]
                                await n69wspa360(json.dumps({'event': 'info', 'data': 'code ' + 'found' if n69wsp9p2h else 'not found'}, ensure_ascii=False))
                            except Exception as e:
                                await n69wspa360(json.dumps({'event': 'warn', 'data': f'Agent looking up code failed: {str(argdic)[1:-1]}'}, ensure_ascii=False))
                                traceback.print_exc()
                                n69wspa35k = n69wspa35k + ('args:' + str(argdic) + ' Error: ' + str(e) + '\n')
                        await self.coder.listen_msgs([{'role': 'system', 'content': 'tool'}, {'role': 'user', 'content': {'action': 'check_codes', 'rsp': n69wspa2jw, 'error': n69wspa35k, 'extra_msg': n69wspa2ht}}], session_id=n69wspa337)
                    elif n69wspa34v['action'] == 'ask_user':
                        n69wspa2ym = n69wspa34v['query']
                        await n69wspa360(json.dumps({'event': 'quest', 'data': f'Question:{n69wspa2ym}'}, ensure_ascii=False))
                        n69wspa2q7 = await n69wspa38v()
                        n69wspa2q7 = json5.loads(n69wspa2q7)
                        n69wspa2q7 = n69wspa2q7['input']
                        await self.coder.listen_msgs([{'role': 'system', 'content': 'tool'}, {'role': 'user', 'content': {'action': n69wspa34v['action'], 'rsp': n69wspa2q7, 'extra_msg': n69wspa2ht}}], session_id=n69wspa337)
                    elif n69wspa34v['action'] == 'check_pkg_exist':
                        n69wspa2fv = n69wspa34v['pkg_name'].strip('"').strip("'")
                        n69wspa2lj = types.ModuleType('tmd')
                        n69wspa2lj.__file__ = '<string>'
                        n69wspa2lj.__name__ = 'tmd'
                        try:
                            exec(f'import {n69wspa2fv}', n69wspa2lj.__dict__)
                            n69wspa2g8 = list_installed_packages()
                            n69wspa349 = 'unknown'
                            n69wspa2rs = 'none'
                            for ep in n69wspa2g8:
                                if ep['name'] == n69wspa2fv:
                                    n69wspa349 = ep['version']
                                    n69wspa2rs = ep.get(n69wspa2rs) or 'none'
                                    break
                            n69wspa2gy = f'package {n69wspa2fv} exists. Version: {n69wspa349}, summary: {n69wspa2rs}'
                        except:
                            n69wspa2gy = f'package {n69wspa2fv} does not exist.'
                        await n69wspa360(json.dumps({'event': 'info', 'data': f'System: {n69wspa2gy}'}, ensure_ascii=False))
                        await self.coder.listen_msgs([{'role': 'system', 'content': 'tool'}, {'role': 'user', 'content': {'action': n69wspa34v['action'], 'rsp': n69wspa2gy, 'extra_msg': n69wspa2ht}}], session_id=n69wspa337)
                    elif n69wspa34v['action'] == 'fail_to_generate':
                        await n69wspa360(json.dumps({'event': 'info', 'data': f"Agent aborted the generation. Explain: {n69wspa34v.get('explain')}"}, ensure_ascii=False))
                        raise A69wspa0yp('Failed. Agent intentionally aborted the generation.')
                    else:
                        n69wspa38p = n69wspa38p + 1
                        await n69wspa360(json.dumps({'event': 'warn', 'data': f"Agent selected non-exist option: {n69wspa34v['action']}"}, ensure_ascii=False))
                        await self.coder.listen_msgs([{'role': 'system', 'content': 'illegal'}], session_id=n69wspa337)
                except A69wspa0yp as e:
                    traceback.print_exc()
                    raise RuntimeError(f'Failed. Agent aborted due to: {e}.')
                except Exception as e:
                    traceback.print_exc()
                    if 'Cannot call "send" once a close message' in str(e):
                        raise ClientAbort('Failed due to client abort.')
                    n69wspa38p = n69wspa38p + 1
                    await n69wspa360(json.dumps({'event': 'warn', 'data': f'Agent output unparsable: {e}. Retrying...'}, ensure_ascii=False))
                    if n69wspa38p > 3:
                        await n69wspa360(json.dumps({'event': 'info', 'data': 'Failed. Agent keeps outputting illegal response.'}, ensure_ascii=False))
                        raise RuntimeError('Failed. Agent keeps outputting unparsable response.')
                    await self.coder.listen_msgs([{'role': 'system', 'content': 'illegal'}, {'role': 'system', 'content': str(e)}], session_id=n69wspa337)
            if n69wspa2f0 == 1:
                n69wsp9onl = await self.b69x8ynnt6(n69wsp9orm, n69wspa35j, n69wspa2nm=[], external_class_ids=[], n69wspa2da=n69wspa2rt, cached=True, n69wspa381=n69wspa381, sustainable=True, n69znp79nl=False, _skiplock=True, conn=conn, tolerance=1)
            else:
                n69wspa2g4 = b69wsp9mq1(n69wsp9orm, def_cutoff=False)
                n69wsp9oxz, maybe_badclass = brutal_gets(n69wspa2g4, lambda x: x.get('ntype') == 'ClassDef' and x.get('name') == n69wspa2u8 if isinstance(x, dict) else False)
                if n69wsp9oxz:
                    n69wspa2zk = [len(p) for p in n69wsp9oxz]
                    n69wspa2na = maybe_badclass[np.argmin(n69wspa2zk)]
                    n69wspa31e = {'ntype': 'Module', 'body': n69wspa2na['body']}
                    _, n69wsp9orm = b65wsp9mrz(n69wspa31e)
                n69wsp9onl = await self.b69x8ynntj(n69wsp9orm, n69wspa35j, n69wspa2nm=[], external_class_ids=[], del_funcs=n69wspa2cq, n69wspa381=n69wspa381, n69znp79nl=False, cached=True, _skiplock=True, conn=conn, tolerance=2)
            return n69wsp9onl
        n69wsp9onl = await self._batch_write([b69x8ynnvt], _skiplock=_skiplock, conn=conn)
        n69wsp9onl = n69wsp9onl[0]
        return n69wsp9onl

    def b69wspa0xm(self, n69wsp9onk, import_data):
        n69wspa2gg = {}
        n69wsp9p2h = False
        for n69wsp9ole, n69wspa2us in import_data.get('classes', {}).items():
            if n69wsp9ole == n69wsp9onk:
                n69wspa2gg = n69wspa2us
                n69wsp9p2h = True
                break
        if not n69wsp9p2h:
            for n69wspa2ce, odic in import_data.get('objs', {}).items():
                if n69wsp9onk == odic['type']:
                    n69wspa2gg = odic['class']
                    n69wsp9p2h = True
                    break
        if not n69wsp9p2h:
            for n69wsp9ole, n69wspa2us in import_data.get('classes', {}).items():
                if n69wsp9ole == n69wspa2us['rawname']:
                    n69wspa2gg = n69wspa2us
                    n69wsp9p2h = True
                    break
        return (n69wspa2gg, n69wsp9p2h)

    async def b69x8ynnt9(self, conn, n69wsp9onk, n69wsp9p0f, n69wspa2z4):
        assert n69wsp9onk or n69wsp9p0f
        if n69wsp9onk and n69wsp9p0f:
            right = '*' + n69wsp9onk + '/' + n69wsp9p0f
        elif n69wsp9onk:
            right = '*' + n69wsp9onk
        else:
            right = '/' + n69wsp9p0f
        n69wspa34p = 'classes' if not n69wsp9p0f else 'funcs'
        n69wspa2k3 = [v + right for v in n69wspa2z4]
        n69wsp9ov9 = [self.select(n69wspa34p, conds=[{'def_id': f}], targets=['def_id'], conn=conn, _skiplock=True) for f in n69wspa2k3]
        n69wsp9ov9 = await asyncio.gather(*n69wsp9ov9)
        n69wspa2xd = [i for i in range(len(n69wsp9ov9)) if len(n69wsp9ov9[i]) > 0]
        n69wspa2oi = [n69wspa2k3[i] for i in n69wspa2xd]
        return n69wspa2oi

    async def b69x8ynntq(self, conn, n69wspa337):
        n69wspa35s = False
        if not n69wspa337.find('/') > 0:
            return False
        else:
            n69wsp9oyc = await self.select('funcs', conds=[{'def_id': n69wspa337}], targets=['def_id'], conn=conn, _skiplock=True)
            if len(n69wsp9oyc) > 0:
                n69wspa35s = True
            return n69wspa35s

    async def b69x8ynnul(self, n69wspa38w, n69wsp9onk, n69wsp9p0f, n69wspa2z4, n69wspa381, current_space=0, conn=None, _skiplock=False):
        n69wsp9p0f = n69wsp9p0f.strip()
        n69wsp9onk = n69wsp9onk.strip()
        n69wspa38w = n69wspa38w.strip()
        if current_space > 0:
            n69wspa2z4 = []
        n69wspa2z4 = [v for v in n69wspa2z4 if x69xm5dtzx(v) == 'cond']
        assert x69xm5dtzx(n69wspa381) == 'folder'
        n69wspa2t0 = n69wspa381.replace('>', '.').strip()
        assert not '..' in n69wspa2t0

        async def b69wsp9mrq(conn):
            n69wspa306 = ''
            n69wspa35s = False
            n69wspa34s = ''
            n69wspa36e = False
            if n69wspa38w.lower() in ('<local>', '[local]'):
                n69wspa36e = True
                if not n69wsp9onk and (not n69wsp9p0f):
                    assert n69wspa2z4
                    n69wspa337 = n69wspa2z4[0].split('^')[0]
                    code, _ = await self.b69x8ynntv(n69wspa337, 'all', style='pure', tolerance=1, conn=conn, _skiplock=True)
                    return f"# code in module {n69wspa337.replace('^', '.').replace('/', '.')} (env_level=0):\n{code}"
                n69wspa2mo = await self.b69x8ynnt9(conn, n69wsp9onk, n69wsp9p0f, n69wspa2z4)
                if n69wspa2mo:
                    n69wspa306 = n69wspa2mo[-1]
                    n69wspa35s = True
                else:
                    return '# <NOT_FOUND>'
            else:
                n69wspa2j4 = x69xm5du08(n69wspa2t0, n69wspa38w)
                if n69wspa2j4 == '<UNDEFINED>':
                    return '# <NOT_FOUND>'
                n69wspa337 = x69xm5du01(n69wspa2j4) if '.' in n69wspa2j4 else n69wspa2j4
                n69wspa35s = await self.b69x8ynntq(conn, n69wspa337) if current_space == 0 else False
                if n69wspa35s:
                    if n69wsp9onk and n69wsp9p0f:
                        right = '*' + n69wsp9onk + '/' + n69wsp9p0f
                    elif n69wsp9onk:
                        right = '*' + n69wsp9onk
                    elif n69wsp9p0f:
                        right = '/' + n69wsp9p0f
                    else:
                        modcode, _ = await self.b69x8ynntv(n69wspa337, 'all', style='pure', tolerance=1, conn=conn, _skiplock=True)
                        return f"# code in module {n69wspa337.replace('^', '.').replace('/', '.')} (env_level=0):\n{modcode}"
                    n69wspa306 = n69wspa337 + '^1#_' + right
            if n69wspa35s:
                n69wspa2qq = {'func': 'funcs', 'class': 'classes'}
                n69wspa34p = n69wspa2qq[x69xm5dtzx(n69wspa306)]
                n69wsp9oyc = self.select(n69wspa34p, conds=[{'def_id': n69wspa306}], targets=['def_id'], conn=conn, _skiplock=True)
                n69wspa31f = self.select('funcs', conds=[{'def_id': n69wspa337}], targets=['imports_code'], conn=conn, _skiplock=True) if not n69wspa36e else aidle(default=pd.DataFrame(columns=['imports_code']))
                n69wsp9oyc, n69wspa31f = await asyncio.gather(n69wsp9oyc, n69wspa31f)
                if len(n69wspa31f) > 0:
                    if n69wspa31f['imports_code'].tolist()[0].strip():
                        _, _, n69wspa34s = b69wsp9mrs(b69wsp9mq1(n69wspa31f['imports_code'].tolist()[0]), expand=True)
                if len(n69wsp9oyc) == 0:
                    n69wspa37j = x69xm5dtzy(n69wsp9onk, n69wsp9p0f, n69wspa34s)
                    if n69wspa37j != '<NOT_FOUND>':
                        return f'# (env_level=0) This tool is imported into module {n69wspa38w}. Please divert to where it is imported from. Import code:\n' + n69wspa37j
                    return '# <NOT_FOUND>'
                code = await self.b69x8ynntv(n69wspa306, 'all', style='pure', tolerance=1, conn=conn, _skiplock=True)
                if n69wsp9onk and n69wsp9p0f:
                    n69wspa2uz = f'# (env_level=0) Code section of func {n69wsp9p0f} under class {n69wsp9onk} in module {n69wspa38w} :\n'
                elif n69wsp9p0f:
                    n69wspa2uz = f'# (env_level=0) Code section of func {n69wsp9p0f} in module {n69wspa38w} :\n'
                elif n69wsp9onk:
                    n69wspa2uz = f'# (env_level=0) Code section of class {n69wsp9onk} in module {n69wspa38w} :\n'
                code = n69wspa2uz + code[0]
                return code
            else:
                n69wspa2ne = ''
                n69wspa37e = sys.path.copy()
                n69wspa304 = 2
                try:
                    extpkgs = []
                    if current_space <= 1:
                        extpkgs = await self.select('misc', cond_sql='true', targets=['external_pkgs'], conn=conn, _skiplock=True)
                        extpkgs = extpkgs.loc[0, 'external_pkgs']
                        for p in extpkgs:
                            sys.path.insert(0, p)
                    n69wspa2lj = importlib.import_module(n69wspa38w)
                    try:
                        n69wspa304 = 1 if any([n69wspa2lj.__file__.startswith(ep + '/') for ep in extpkgs]) else 2
                    except:
                        traceback.print_exc()
                    n69wspa2iz = inspect.getsource(n69wspa2lj)
                except Exception as e:
                    return '# <NOT_FOUND>'
                finally:
                    sys.path = n69wspa37e
                if not n69wsp9onk and (not n69wsp9p0f):
                    return f'# (env_level={n69wspa304}) code in module {n69wspa38w}:\n{n69wspa2iz}'
                try:
                    n69wspa38x = n69wsp9onk if n69wsp9onk else n69wsp9p0f
                    n65d20cda3 = None
                    n69wspa33p = None if not (n69wsp9onk and n69wsp9p0f) else n69wsp9p0f
                    n69wspa2ix = None
                    n65d20cda3 = getattr(n69wspa2lj, n69wspa38x, None)
                    if not n65d20cda3:
                        return '# <NOT_FOUND>'
                    if not n69wspa33p:
                        n69wspa31w = 'class' if n69wsp9onk else 'func'
                        n69wspa2uz = f'# (env_level={n69wspa304}) Code section of {n69wspa31w} {n69wspa38x} in module {n69wspa38w} :\n'
                        n69wspa345 = n69wspa2uz + inspect.getsource(n65d20cda3)
                        return n69wspa345
                    n69wspa2ix = getattr(n65d20cda3, n69wspa33p, None)
                    if not n69wspa2ix:
                        return '# <NOT_FOUND>'
                    n69wspa2uz = f'# (env_level={n69wspa304}) Code section of func {n69wspa33p} under class {n69wspa38x} in module {n69wspa38w} :\n'
                    n69wspa345 = n69wspa2uz + inspect.getsource(n69wspa2ix)
                    return n69wspa345
                except Exception as e:
                    traceback.print_exc()
                    n69wspa2pm = b69wsp9mq1(n69wspa2iz, def_cutoff=True)
                    n69wspa2pm, _, n69wspa34s = b69wsp9mrs(n69wspa2pm, expand=True)
                    n69wspa38x = n69wsp9onk if n69wsp9onk else n69wsp9p0f
                    n69wspa2fk = ('ClassDef',) if n69wsp9onk else ('AsyncFunctionDef', 'FunctionDef')
                    n65d20cda3 = None
                    n69wspa33p = None if not (n69wsp9onk and n69wsp9p0f) else n69wsp9p0f
                    n69wspa37k = ('AsyncFunctionDef', 'FunctionDef')
                    n69wspa2ix = None
                    for n69wspa2mh in n69wspa2pm['body']:
                        if n69wspa2mh['ntype'] in n69wspa2fk:
                            if n69wspa2mh['name'] == n69wspa38x:
                                n65d20cda3 = n69wspa2mh
                    if not n65d20cda3:
                        n69wspa37j = x69xm5dtzy(n69wsp9onk, n69wsp9p0f, n69wspa34s)
                        if n69wspa37j != '<NOT_FOUND>':
                            return f'# (env_level={n69wspa304}) This tool is imported into module {n69wspa38w}. Please divert to where it is imported from. Import code:\n' + n69wspa37j
                        return '# <NOT_FOUND>'
                    if not n69wspa33p:
                        n69wspa31w = 'class' if n69wsp9onk else 'func'
                        n69wspa2uz = f'# (env_level={n69wspa304}) Code section of {n69wspa31w} {n69wspa38x} in module {n69wspa38w} :\n'
                        n69wspa345 = n69wspa2uz + n65d20cda3['code']
                        return n69wspa345
                    n69wspa328 = b69wsp9mq1(n65d20cda3['code'])
                    for n69wspa2mh in n69wspa328['body'][0]['body']:
                        if n69wspa2mh['ntype'] in n69wspa37k:
                            if n69wspa2mh['name'] == n69wspa33p:
                                n69wspa2ix = n69wspa2mh
                    if not n69wspa2ix:
                        return '# <NOT_FOUND>'
                    n69wspa2uz = f'# (env_level={n69wspa304}) Code section of func {n69wspa33p} under class {n69wspa38x} in module {n69wspa38w} :\n'
                    n69wspa345 = n69wspa2uz + n69wspa2ix['code']
                    return n69wspa345
        n69wsp9onl = await self._batch_read([b69wsp9mrq], conn=conn, _skiplock=_skiplock)
        n69wsp9onl = n69wsp9onl[0]
        return n69wsp9onl

    async def b69x8ynnte(self, n69wsp9onk, import_data, conn=None, _skiplock=False):
        pass

    async def b69x8ynnt1(self, modulename, import_data, conn=None, _skiplock=False):
        pass

    async def b69x8ynnv8(self, n65d20cda3, n69wsp9p0w, n69wspa317=None, style='both', tolerance=0, codeswaps={}, n69wspa39a=None, _skiplock=False, conn=None):
        assert x69xm5dtzx(n65d20cda3) == 'folder'
        assert n69wsp9p0w not in nesttypes
        n69wspa39a = n69wspa39a if n69wspa39a is not None else lambda: self.x69xm5dtzq

        async def b69wsp9mrq(conn):
            n69wspa2gl = n69wspa2hg = f"uid = '{n65d20cda3}'"
            n69wsp9oya = self.select('nodes', cond_sql=n69wspa2gl, conn=conn, _skiplock=True)
            n69wspa2mh = self.select('vars', cond_sql=n69wspa2hg, conn=conn, _skiplock=True)
            n69wsp9oya, n69wspa2mh = await asyncio.gather(n69wsp9oya, n69wspa2mh)
            return (n69wsp9oya, n69wspa2mh)
        n69wspa2q4 = ('_fAk84289epATh_/_fAke_FUnc',)
        if n69wspa317 and n69wspa39a():
            assert n69wspa317['data']['uid'] == n65d20cda3
            n69wspa317 = copy.deepcopy(n69wspa317)['data']
            n69wspa317['hid'] = '1.' + n69wspa317['uid']
            n69wspa317['branch'] = '_'
            n69wspa317['data_providers'] = []
            n69wspa317['pres'] = ['1.0']
            n69wspa317['nexts'] = []
            if n69wspa317['node_type'] in ('tool', 'conc'):
                n69wspa317['params_map'] = {p[0]: p[1] for p in n69wspa317['params_map']}
            n69wspa317 = {k: v for k, v in n69wspa317.items() if not k in ('handle_in', 'handle_outs', 'width', 'height', 'selected', 'positionAbsolute', 'dragging')}
            n69wspa2hn = {'uid': 'fakestart', 'hid': '1.0', 'branch': '_', 'node_type': 'start', 'pres': [], 'nexts': [n69wspa317['hid']], 'xpos': 0, 'ypos': 0, 'def_id': n69wspa317['def_id']}
            n69wsp9oya = pd.DataFrame([n69wspa2hn, n69wspa317])
            n69wspa2q4 = n69wspa317['def_id']
            n69wspa35f = [{'uid': 'f100', 'def_id': n69wspa317['def_id'], 'globals': [], 'nonlocals': [], 'imports_code': '', 'is_async': False, 'deco_expr': ''}]
            n69wsp9p3q = pd.DataFrame(n69wspa35f)
            n69wsp9p6q = pd.DataFrame(columns=list(n69wspa2dn.keys()))
            n69wsp9osv = pd.DataFrame(columns=list(n69wspa356.keys()))
            n69wsp9oya, _, _, n69wspa2mh, _ = b69wsp9mnk(n69wsp9oya, n69wsp9p3q, n69wsp9p6q, n69wsp9osv, n69wspa317['def_id'], ret_type='df', strict_order=True, n69wsp9ore=False, n69wsp9oy0=True, n69wspa2y2=False, recur_into_tools=True, n69wspa2f3=True)
            n69wsp9oya = n69wsp9oya[~(n69wsp9oya['node_type'] == 'start')].reset_index(drop=True)
            if 'toolcall' in n69wsp9oya.columns:
                n69wsp9oya['toolcall'] = n69wsp9oya.apply(lambda n69wspa2xo: x69xm5dtzz(n69wspa2xo['node_type'], n69wspa2xo['toolcall']), axis=1)
            if 'type' in n69wspa2mh.columns:
                n69wspa2mh['type'] = n69wspa2mh['type'].apply(lambda x: x.replace(DOT_REPL, '.') if isinstance(x, str) else x)
            if 'name' in n69wspa2mh.columns:
                n69wspa2mh['name'] = n69wspa2mh['name'].apply(lambda x: x.replace(DOT_REPL, '.') if isinstance(x, str) else x)
        else:
            n69wsp9oya, n69wspa2mh = (await self._batch_read([b69wsp9mrq], _skiplock=_skiplock, conn=conn))[0]
            if len(n69wsp9oya) > 0 and 'def_id' in n69wsp9oya.columns:
                n69wspa2q4 = n69wsp9oya['def_id'].tolist()[0]
        if len(n69wsp9oya) == 0:
            raise RuntimeError(f'[BUG] The backend may have failed to validate and save the workflow into the DB. This is a bug.')
        assert n69wsp9oya.loc[0, 'node_type'] == n69wsp9p0w or (n69wsp9oya.loc[0, 'node_type'] == 'code' and n69wsp9p0w == 'tool'), f"eh011：{n69wsp9oya.loc[0, 'node_type']} vs {n69wsp9p0w}, {n69wsp9oya.to_dict(orient='records')}"
        n69wsp9p3q = pd.DataFrame([{'uid': 'ffakerr', 'def_id': n69wspa2q4, 'globals': [], 'nonlocals': [], 'imports_code': '', 'is_async': False, 'deco_expr': '', 'xpos': 0, 'ypos': 0, 'doc': ''}])
        n69wsp9osv = pd.DataFrame([{'name': 'return', 'type': 'int', 'doc': '', 'def_id': n69wspa2q4, 'ctx': 'return', 'place': 0}])
        n69wsp9oya.loc[0, 'hid'] = '1.' + n65d20cda3
        n69wsp9oya.loc[0, 'branch'] = '_'
        n69wsp9oya.at[0, 'data_providers'] = []
        n69wsp9oya.at[0, 'pres'] = []
        n69wsp9oya.at[0, 'nexts'] = []
        n69wsp9oya.loc[0, 'def_id'] = n69wspa2q4
        if n65d20cda3 in codeswaps:
            n69wsp9oya.loc[0, 'code'] = codeswaps[n65d20cda3]
        n69wspa2r3 = {'by-branch': [{'uid': '1', 'func_id': n69wspa2q4, 'cond': '_', 'label1': SECTION_START_LABEL, 'label2': SECTION_END_LABEL, 'mode': 'embrace'}]}
        n69wsp9p6q = pd.DataFrame(columns=['bases', 'vars', 'deco_expr', 'def_id', 'uid', 'xpos', 'ypos', 'doc'])
        code = b69wsp9mnm(n69wsp9oya, n69wsp9p3q, n69wsp9p6q, n69wsp9osv, n69wspa2q4, n69wspa2in=False, n69wsp9ot1=n69wspa2mh, n69wspa2r3=n69wspa2r3, style=style, tolerance=tolerance)
        return (b6a0gjsmt7(code[0]), b6a0gjsmt7(code[1]))

    async def b69x8ynntv(self, base_id, choice, n69wspa2zf=None, n69wspa2wz=None, style='both', codeswaps={}, tolerance=0, n69wspa39a=None, conn=None, _skiplock=False):
        n69wspa39a = n69wspa39a if n69wspa39a is not None else lambda: self.x69xm5dtzq
        if x69xm5dtzx(base_id) == 'cond':
            assert not n69wspa2zf
            assert choice in ('funcs', 'classes')
            shellfid, hidbr = base_id.rsplit('^', 1)
            n69wsp9p6r = await self.b69x8ynnu5(shellfid, n69wspa2k7=hidbr, choices=['imports', choice], style=style, tolerance=tolerance, codeswaps=codeswaps, conn=conn, _skiplock=_skiplock)
            return n69wsp9p6r

        async def b69wsp9mrq(conn):
            n69wspa2gl = ''
            n69wspa37w = ''
            n69wspa2l8 = ''
            n69wspa31y = ''
            n69wspa2hg = ''
            n69wspa2in = False
            n69wspa2zj = base_id
            n69wspa387 = 0
            if x69xm5dtzx(base_id) in 'func':
                if choice in ('all', 'allbelow'):
                    n69wspa2gl = n69wspa2hg = f"def_id LIKE '{base_id}^%' OR def_id = '{base_id}'"
                    n69wspa37w = n69wspa2l8 = n69wspa31y = f"def_id LIKE '{base_id}^%' OR def_id = '{base_id}'"
                    if choice == 'all':
                        if base_id.count('/') == 1:
                            assert not '^' in base_id and (not '*' in base_id)
                        else:
                            n69wspa2in = True
                elif choice == 'funcs':
                    n69wspa2gl = n69wspa2hg = f"def_id LIKE '{base_id}^1#_/%'"
                    n69wspa37w = n69wspa2l8 = n69wspa31y = f"def_id LIKE '{base_id}^1#_/%' OR def_id = '{base_id}'"
                elif choice == 'classes':
                    n69wspa2gl = n69wspa2hg = f"def_id LIKE '{base_id}^1#_*%'"
                    n69wspa37w = n69wspa2l8 = n69wspa31y = f"def_id LIKE '{base_id}^1#_*%' OR def_id = '{base_id}'"
                else:
                    raise ValueError(base_id)
            elif x69xm5dtzx(base_id) in 'class':
                if choice == 'all':
                    n69wspa2gl = n69wspa2hg = n69wspa37w = n69wspa31y = f"def_id LIKE '{base_id}/%'"
                    n69wspa2l8 = f"def_id LIKE '{base_id}/%' OR def_id = '{base_id}'"
            else:
                raise ValueError(f'to_code base_id must point to either a class or a func, got a {x69xm5dtzx(base_id)}: {base_id}')
            n69wsp9oya = self.select('nodes', cond_sql=n69wspa2gl, conn=conn, _skiplock=True)
            n69wsp9p3q = self.select('funcs', cond_sql=n69wspa37w, conn=conn, _skiplock=True)
            n69wsp9p6q = self.select('classes', cond_sql=n69wspa2l8, conn=conn, _skiplock=True)
            n69wspa2mh = self.select('vars', cond_sql=n69wspa2hg, conn=conn, _skiplock=True)
            n69wsp9osv = self.select('params', cond_sql=n69wspa31y, conn=conn, _skiplock=True)
            n69wsp9oya, n69wsp9p3q, n69wsp9p6q, n69wsp9osv, n69wspa2mh = await asyncio.gather(n69wsp9oya, n69wsp9p3q, n69wsp9p6q, n69wsp9osv, n69wspa2mh)
            n69wsp9p51 = None
            if x69xm5dtzx(base_id) in 'class':
                assert '/' in base_id
                n69wspa2w8 = base_id[base_id.rfind('/') + 1:base_id.rfind('*')]
                n69wspa2w8 = n69wspa2w8[n69wspa2w8.find('^'):]
                n69wsp9p51 = n69wspa2w8[n69wspa2w8.rfind('#') + 1:]
                assert n69wspa2w8.count('^') == 1 and n69wspa2w8.count('#') == 1
                n69wspa2qn = base_id[:base_id.rfind('^')]
                n69wspa2zj = n69wspa2qn
                n69wspa2f6 = {'uid': 'fakeshell', 'def_id': n69wspa2qn}
                n69wspa2f6 = pd.DataFrame([n69wspa2f6])
                n69wsp9p3q = pd.concat([n69wsp9p3q, n69wspa2f6], ignore_index=True)
                if n69wspa2w8 == '^1#_':
                    pass
                else:
                    assert '.' in n69wspa2w8, n69wspa2w8
                    n69wspa366 = []
                    assert n69wspa2w8[0] == '^'
                    n69wspa2p6 = n69wspa2w8[1:].split('#')[0]
                    n69wspa387 = n69wspa2p6.count('.')
                    n69wspa34a = n69wspa2p6.split('.')
                    n69wspa2uh = []
                    for i in range(2, len(n69wspa34a)):
                        n69wspa2n6 = '.'.join(n69wspa34a[:i])
                        n69wspa366.append(n69wspa2n6)
                        n69wspa2nx = {'uid': n69wspa34a[i - 1], 'hid': n69wspa2n6, 'branch': '_', 'data_providers': [], 'node_type': 'while', 'pres': [], 'nexts': [], 'def_id': n69wspa2qn, 'expr': '1'}
                        n69wspa2rw = {'uid': n69wspa34a[i - 1] + '-endwhile', 'hid': n69wspa2n6 + '-endwhile', 'branch': '_', 'data_providers': [], 'node_type': 'endwhile', 'def_id': n69wspa2qn, 'source': n69wspa2n6}
                        n69wspa2uh.append(n69wspa2nx)
                        n69wspa2uh.append(n69wspa2rw)
                    n69wspa2n6 = '.'.join(n69wspa34a)
                    n69wspa366.append(n69wspa2n6)
                    n69wspa2vc = 'while'
                    if n69wsp9p51.lower() in ('true', 'false'):
                        n69wspa2vc = 'if'
                    elif n69wsp9p51.isdigit():
                        n69wspa2vc = 'match'
                    n69wspa2nm, _ = await self.b69x8ynnvo(n69wspa2vc, n69wspa2qn, '', '', 0, 0, save_new_node=False)
                    n69wspa2nm = n69wspa2nm[0]['data']
                    n69wspa2nm['uid'] = n69wspa34a[-1]
                    n69wspa2nm['hid'] = n69wspa2n6
                    n69wspa2nm['branch'] = '_'
                    n69wspa2nm = {k: v for k, v in n69wspa2nm.items() if not k in ('handle_in', 'handle_outs', 'width', 'height', 'selected', 'positionAbsolute', 'dragging', 'subWorkflowIds', 'tool_counts')}
                    if n69wspa2vc == 'match':
                        n69wspa2iq = int(n69wspa2w8.split('#')[-1]) + 1
                        n69wspa2hs = {c: {'expr': '', 'spawn_vars': []} for c in range(n69wspa2iq)}
                        n69wspa2nm['cases'] = n69wspa2hs
                    n69wspa2rw = {'uid': n69wspa2nm['uid'] + '-end' + n69wspa2vc, 'hid': n69wspa2nm['hid'] + '-end' + n69wspa2vc, 'branch': n69wspa2nm['branch'], 'data_providers': [], 'node_type': 'end' + n69wspa2vc, 'def_id': n69wspa2qn, 'source': n69wspa2n6}
                    n69wspa2uh.append(n69wspa2nm)
                    n69wspa2uh.append(n69wspa2rw)
                    n69wsp9oya = pd.concat([n69wsp9oya, pd.DataFrame(n69wspa2uh)], ignore_index=True)
            return (n69wsp9oya, n69wsp9p3q, n69wsp9p6q, n69wsp9osv, n69wspa2mh, n69wspa2in, n69wspa2zj, n69wspa387, n69wsp9p51)
        n69wspa2cs = None
        if x69xm5dtzx(base_id) == 'func' and choice in ('all', 'allbelow') and ((n69wspa2zf or {}).get('mode') in ('replace', 'insert', 'allbelow')) and (len((n69wspa2wz or {'nodes': []}).get('nodes')) > 0) and n69wspa39a():
            n69wspa2wz = copy.deepcopy(n69wspa2wz)
            n69wspa30p = []
            n69wsp9orl = []
            n69wspa34s = ''
            if n69wspa2zf['mode'] == 'allbelow':
                if n69wspa2zf['branch'] == '1#_':
                    n69wspa2v8 = await self.select('funcs', conds=[{'def_id': base_id}], targets=['globals', 'nonlocals', 'imports_code'], conn=conn, _skiplock=True)
                    n69wspa2v8 = n69wspa2v8.to_dict(orient='records')
                    if not n69wspa2v8:
                        pass
                    else:
                        n69wspa30p, n69wsp9orl, n69wspa34s = (n69wspa2v8[0]['globals'], n69wspa2v8[0]['nonlocals'], n69wspa2v8[0]['imports_code'])
            if n69wspa2zf['mode'] == 'replace':
                n69wspa2wr = await self.b69x8ynnv4(n69wspa2zf['section'][0], n69wspa2zf['section'][1], n69wspa2wz['nodes'], n69wspa2wz['edges'], dest='return', conn=conn, _skiplock=True)
            elif n69wspa2zf['mode'] == 'insert':
                n69wspa2wr = await self.b69x8ynnu0(n69wspa2zf['after'], n69wspa2wz['nodes'], n69wspa2wz['edges'], shell=n69wspa2zf['br_id'].split('^')[-1] if n69wspa2zf.get('br_id') else None, dest='return', conn=conn, _skiplock=True)
            elif n69wspa2zf['mode'] == 'allbelow':
                n69wspa2wr = await self.b69x8ynnui(n69wspa2zf['branch'], n69wspa2wz['nodes'], n69wspa2wz['edges'], dest='return', conn=conn, _skiplock=True)
            if n69wspa2wr['parent_hid'] == '1' and n69wspa2wr['parent_br'] == '_':
                n69wsp9oya, n69wsp9p3q, n69wsp9p6q, n69wsp9osv = (n69wspa2wr['nodes'], n69wspa2wr['funcs'], n69wspa2wr['classes'], n69wspa2wr['params'])
            else:
                n69wsp9oya, n69wsp9p3q, n69wsp9p6q, n69wsp9osv, _, _ = self.b69wspa0y2(base_id, '1', '_', n69wspa2wr, new_parent_class=None, renew_uids=False)
            if n69wspa2zf['mode'] == 'insert':
                n69wspa2zf['br_id'] = base_id + '^1#_'
            elif n69wspa2zf['mode'] == 'allbelow':
                n69wspa2zf['branch'] = '1#_'
            n69wspa35f = [{'uid': 'f100', 'def_id': n69wsp9oya['def_id'].tolist()[0], 'globals': n69wspa30p, 'nonlocals': n69wsp9orl, 'imports_code': n69wspa34s, 'is_async': False, 'deco_expr': ''}]
            if not n69wsp9oya['def_id'].tolist()[0] in n69wsp9p3q['def_id'].tolist():
                n69wsp9p3q = pd.concat([pd.DataFrame(n69wspa35f), n69wsp9p3q], ignore_index=True)
            if n69wspa2zf['mode'] == 'insert':
                n69wspa2cs = n69wsp9oya[(n69wsp9oya['def_id'] == base_id) & (n69wsp9oya['hid'].str[:2] == '1.') & ~n69wsp9oya['hid'].str[2:].str.contains('\\.')]
                n69wspa2cs = n69wspa2cs.to_dict(orient='records')[0]
            n69wsp9oya, n69wsp9p3q, n69wsp9p6q, n69wspa2mh, n69wsp9osv = b69wsp9mnk(n69wsp9oya, n69wsp9p3q, n69wsp9p6q, n69wsp9osv, base_id, ret_type='df', strict_order=True, n69wsp9ore=False, n69wsp9oy0=True, n69wspa2y2=False, recur_into_tools=True, n69wspa2f3=True, tolerance=tolerance)
            if 'toolcall' in n69wsp9oya.columns:
                n69wsp9oya['toolcall'] = n69wsp9oya.apply(lambda n69wspa2xo: x69xm5dtzz(n69wspa2xo['node_type'], n69wspa2xo['toolcall']), axis=1)
            if 'type' in n69wspa2mh.columns:
                n69wspa2mh['type'] = n69wspa2mh['type'].apply(lambda x: x.replace(DOT_REPL, '.') if isinstance(x, str) else x)
            if 'name' in n69wspa2mh.columns:
                n69wspa2mh['name'] = n69wspa2mh['name'].apply(lambda x: x.replace(DOT_REPL, '.') if isinstance(x, str) else x)
            n69wspa2in = base_id.count('/') == 1
            n69wspa2zj = base_id
            n69wspa387 = 0
            n69wsp9p51 = '_'
        else:
            n69wsp9otg = await self._batch_read([b69wsp9mrq], _skiplock=_skiplock, conn=conn)
            n69wsp9otg = n69wsp9otg[0]
            n69wsp9oya, n69wsp9p3q, n69wsp9p6q, n69wsp9osv, n69wspa2mh, n69wspa2in, n69wspa2zj, n69wspa387, n69wsp9p51 = n69wsp9otg
        n69wspa2r3 = {}
        if n69wspa2zf:
            if n69wspa2zf['mode'] == 'replace':
                n69wsp9p0q = n69wspa2zf['section'][0]
                n69wspa2ra = n69wspa2zf['section'][1]
                n69wspa30b = SECTION_START_LABEL
                n69wspa2ir = SECTION_END_LABEL
                if x69xm5dtzx(n69wsp9p0q) in ('class', 'func'):
                    assert n69wspa2ra == n69wsp9p0q
                    n69wspa2r3 = {'by-defnode': [{'def_id': n69wsp9p0q, 'label': n69wspa30b, 'mode': 'before'}, {'def_id': n69wspa2ra, 'label': n69wspa2ir, 'mode': 'after'}]}
                else:
                    assert x69xm5dtzx(n69wsp9p0q) == 'folder'
                    assert x69xm5dtzx(n69wspa2ra) == 'folder'
                    assert not n69wsp9p0q == '0' and (not n69wspa2ra == '0'), 'Section cannot include start node'
                    if '-end' in n69wsp9p0q:
                        n69wsp9p0q = n69wsp9p0q[:n69wsp9p0q.rfind('-end')]
                    if '-end' in n69wspa2ra:
                        n69wspa2ra = n69wspa2ra[:n69wspa2ra.rfind('-end')]
                    n69wspa2lb = n69wsp9p0q
                    n69wspa35r = n69wspa2ra
                    n69wspa2r3 = {'by-tasknode': [{'uid': n69wspa2lb, 'label': n69wspa30b, 'mode': 'before'}, {'uid': n69wspa35r, 'label': n69wspa2ir, 'mode': 'after'}]}
            elif n69wspa2zf['mode'] == 'append':
                n69wspa2y1 = INSERT_LABEL
                if x69xm5dtzx(n69wspa2zf['shell_id']) == 'cond':
                    n69wspa2j6, targ_br = n69wspa2zf['shell_id'].split('^')[-1].split('.')[-1].split('#')
                    assert '^' in n69wspa2zf['shell_id']
                    n69wspa2qn = n69wspa2zf['shell_id'][:n69wspa2zf['shell_id'].rfind('^')]
                    n69wspa2r3 = {'by-branch': [{'func_id': n69wspa2qn, 'uid': n69wspa2j6, 'cond': loadstr(targ_br), 'label': n69wspa2y1, 'mode': 'between'}]}
                elif x69xm5dtzx(n69wspa2zf['shell_id']) == 'class':
                    n69wspa2r3 = {'by-class': [{'class_id': n69wspa2zf['shell_id'], 'label': n69wspa2y1, 'mode': 'append'}]}
                else:
                    raise
            elif n69wspa2zf['mode'] == 'insert':
                n69wspa2y1 = INSERT_LABEL
                n69wspa2vx = 'after'
                n69wsp9oq2 = n69wspa2zf['after']
                if n69wsp9oq2 == '0':
                    n69wsp9oq2 = n69wspa2zf.get('target')
                    n69wspa2vx = 'before'
                if n69wsp9oq2 == '0' or not n69wsp9oq2:
                    n69wspa2tb = n69wspa2zf['br_id']
                    assert x69xm5dtzx(n69wspa2tb) == 'cond'
                    n69wspa2it, condpart = n69wspa2tb.rsplit('^', 1)
                    n69wsp9oz5, n69wspa2w8 = condpart.split('#')
                    n69wsp9oz5 = n69wsp9oz5.split('.')[-1]
                    n69wspa2w8 = loadstr(n69wspa2w8)
                    n69wspa2r3 = {'by-branch': [{'uid': n69wsp9oz5, 'func_id': n69wspa2it, 'cond': n69wspa2w8, 'label': n69wspa2y1, 'mode': 'append'}]}
                    if n69wspa2zf.get('label_roi'):
                        n69wspa2r3['by-branch'].append({'uid': n69wsp9oz5, 'func_id': n69wspa2it, 'cond': n69wspa2w8, 'label1': SECTION_START_LABEL, 'label2': SECTION_END_LABEL, 'mode': 'embrace'})
                elif x69xm5dtzx(n69wsp9oq2) in ('func', 'class'):
                    n69wspa2oz = n69wsp9oq2
                    n69wspa2r3 = {'by-defnode': [{'def_id': n69wspa2oz, 'label': n69wspa2y1, 'mode': 'after'}]}
                else:
                    assert x69xm5dtzx(n69wsp9oq2) == 'folder'
                    n69wspa2j6 = n69wsp9oq2
                    n69wspa2r3 = {'by-tasknode': [{'uid': n69wspa2j6, 'label': n69wspa2y1, 'mode': n69wspa2vx}]}
                    if n69wspa2zf.get('label_roi'):
                        n69wspa2ew = SECTION_START_LABEL
                        n69wspa2i7 = SECTION_END_LABEL
                        n69wspa2q8 = n69wsp9oya[n69wsp9oya['uid'] == n69wspa2j6].to_dict(orient='records')
                        if not n69wspa2cs:
                            assert len(n69wspa2q8) == 1, f"eh012：{n69wspa2j6},{n69wsp9oya['uid'].tolist()},{len(n69wspa2q8)}"
                        elif not n69wspa2q8:
                            n69wspa2q8 = [n69wspa2cs]
                        n69wspa2kd = n69wspa2q8[0]['hid']
                        n69wspa2ml = n69wspa2q8[0]['branch']
                        n69wspa2r0 = n69wspa2kd.split('.')[-2]
                        n69wspa2it = n69wspa2q8[0]['def_id']
                        n69wspa2r3['by-branch'] = [{'uid': n69wspa2r0, 'func_id': n69wspa2it, 'cond': loadstr(n69wspa2ml), 'label1': n69wspa2ew, 'label2': n69wspa2i7, 'mode': 'embrace'}]
            elif n69wspa2zf['mode'] == 'allbelow':
                n69wspa2j6, targ_br = n69wspa2zf['branch'].split('^')[-1].split('.')[-1].split('#')
                n69wspa2ew = SECTION_START_LABEL
                n69wspa2i7 = SECTION_END_LABEL
                n69wspa2r3 = {'by-branch': [{'uid': n69wspa2j6, 'func_id': n69wspa2zf['func_id'], 'cond': loadstr(targ_br), 'label1': n69wspa2ew, 'label2': n69wspa2i7, 'mode': 'embrace'}]}
        if len(n69wsp9oya) + len(n69wsp9p3q) + len(n69wsp9p6q) <= 1:
            return ('', '')
        if len(n69wsp9oya) == 0:
            n69wsp9oya = pd.DataFrame([{'uid': '0', 'hid': '1.0', 'branch': '_', 'node_type': 'start', 'def_id': trunk_to_func(base_id), 'pres': [], 'nexts': []}])
        if codeswaps:
            x69xm5du09(n69wsp9oya, codeswaps)
        n69wspa2xu = asyncio.get_running_loop()
        code = await n69wspa2xu.run_in_executor(None, lambda: b69wsp9mnm(n69wsp9oya, n69wsp9p3q, n69wsp9p6q, n69wsp9osv, n69wspa2zj, n69wspa2in=n69wspa2in, n69wsp9ot1=n69wspa2mh, n69wspa2r3=n69wspa2r3, style=style, tolerance=tolerance))
        if n69wspa387 > 0:

            def b69wspa0xl(n69wspa2ta):
                n69wspa2ls = b69wsp9mq1(n69wspa2ta, def_cutoff=True)
                for i in range(n69wspa387 + 1):
                    if n69wspa2ls['ntype'] == 'Match':
                        n69wspa2ls = n69wspa2ls['cases'][-1]
                    if n69wspa387 == i and n69wspa2ls['ntype'] == 'If' and (str(n69wsp9p51).lower() == 'false'):
                        n69wspa2ls = n69wspa2ls['orelse'][0]
                    else:
                        n69wspa2ls = n69wspa2ls['body'][0]
                return n69wspa2ls['code']
            n69wspa34d = ''
            n69wspa2ol = ''
            if style in ('both', 'pure'):
                n69wspa2ol = b69wspa0xl(code[0])
            if style in ('both', 'run'):
                n69wspa34d = b69wspa0xl(code[1])
            code = (n69wspa2ol, n69wspa34d)
        return (b6a0gjsmt7(code[0]), b6a0gjsmt7(code[1]))

    async def b69x8ynnu5(self, n69wspa2wq, n69wspa2k7='^1#_', choices=['imports', 'funcs', 'classes'], style='both', tolerance=0, codeswaps={}, add_start_uidcomments=False, conn=None, _skiplock=False):
        assert x69xm5dtzx(n69wspa2wq) == 'func'
        n69wspa2k7 = '^' + n69wspa2k7.lstrip('^')
        assert x69xm5dtzx(n69wspa2k7) == 'cond'
        n69wsp9p6r = self.select('funcs', conds=[{'def_id': n69wspa2wq}], targets=['globals', 'nonlocals', 'imports_code'], conn=conn, _skiplock=_skiplock) if 'imports' in choices else aidle(default=pd.DataFrame([{'def_id': n69wspa2wq, 'imports_code': '', 'globals': [], 'nonlocals': []}]))
        n69wspa2lb = aidle() if not add_start_uidcomments else self.select('nodes', conds=[{'def_id': n69wspa2wq, 'node_type': 'start'}], targets=['uid'], conn=conn, _skiplock=_skiplock)
        n69wspa2tb = n69wspa2wq + n69wspa2k7
        n69wspa2tb = self.b69x8ynnvl(n69wspa2tb, _skiplock=_skiplock, conn=conn)
        n69wspa2lb, n69wspa2tb = await asyncio.gather(n69wspa2lb, n69wspa2tb)
        if add_start_uidcomments:
            n69wspa2lb = n69wspa2lb['uid'].tolist()[0] if len(n69wspa2lb) > 0 else '<UNDEFINED>'
            if n69wspa2lb == '<UNDEFINED>':
                pass
        n69wspa353 = f"def_id LIKE '{n69wspa2tb}/%' AND\n                (SUBSTRING(def_id, LENGTH('{n69wspa2tb}/') + 1) NOT LIKE '%/%'\n                AND SUBSTRING(def_id, LENGTH('{n69wspa2tb}/') + 1) NOT LIKE '%#%'\n                AND SUBSTRING(def_id, LENGTH('{n69wspa2tb}/') + 1) NOT LIKE '%^%'\n                AND SUBSTRING(def_id, LENGTH('{n69wspa2tb}/') + 1) NOT LIKE '%*%')"
        n69wspa2i8 = f"def_id LIKE '{n69wspa2tb}*%' AND\n            (SUBSTRING(def_id, LENGTH('{n69wspa2tb}*') + 1) NOT LIKE '%/%'\n            AND SUBSTRING(def_id, LENGTH('{n69wspa2tb}*') + 1) NOT LIKE '%#%'\n            AND SUBSTRING(def_id, LENGTH('{n69wspa2tb}*') + 1) NOT LIKE '%^%'\n            AND SUBSTRING(def_id, LENGTH('{n69wspa2tb}*') + 1) NOT LIKE '%*%')"
        n69wspa2gf = self.select('funcs', cond_sql=n69wspa353, targets=['def_id'], conn=conn, _skiplock=_skiplock) if 'funcs' in choices else aidle(default=pd.DataFrame(columns=list(n69wspa2ts.keys())))
        n69wspa2dp = self.select('classes', cond_sql=n69wspa2i8, targets=['def_id', 'ypos'], conn=conn, _skiplock=_skiplock) if 'classes' in choices else aidle(default=pd.DataFrame(columns=list(n69wspa2dn.keys())))
        n69wsp9p6r, n69wspa2gf, n69wspa2dp = await asyncio.gather(n69wsp9p6r, n69wspa2gf, n69wspa2dp)
        n69wspa2gf = n69wspa2gf['def_id'].tolist()
        n69zpbli8f = n69wspa2dp['ypos'].tolist()
        n69wspa2dp = n69wspa2dp['def_id'].tolist()
        n69zpbli8f, n69wspa2dp = zip(*sorted(zip(n69zpbli8f, n69wspa2dp))) if n69wspa2dp else ([], [])
        n69wspa2vf = [self.b69x8ynntv(aid, 'all', style=style, tolerance=tolerance, codeswaps=codeswaps, conn=conn, _skiplock=_skiplock) for aid in n69wspa2gf]
        n69wspa2dh = [self.b69x8ynntv(aid, 'all', style=style, tolerance=tolerance, codeswaps=codeswaps, conn=conn, _skiplock=_skiplock) for aid in n69wspa2dp]
        n69wspa2vf = asyncio.gather(*n69wspa2vf)
        n69wspa2dh = asyncio.gather(*n69wspa2dh)
        n69wspa2vf, n69wspa2dh = await asyncio.gather(n69wspa2vf, n69wspa2dh)
        n69wspa2qt = [c[1] for c in n69wspa2vf]
        n69wspa31n = [c[1] for c in n69wspa2dh]
        n69wspa2qt = '\n'.join(n69wspa2qt)
        n69wspa31n = '\n'.join(n69wspa31n)
        n69wspa2il = [c[0] for c in n69wspa2vf]
        n69wspa2nw = [c[0] for c in n69wspa2dh]
        n69wspa2il = '\n'.join(n69wspa2il)
        n69wspa2nw = '\n'.join(n69wspa2nw)
        assert not len(n69wsp9p6r) == 0, f'Cannot find func {n69wspa2wq}'
        n69wsp9p6r = n69wsp9p6r.to_dict(orient='records')[0]
        if n69wspa2wq.count('/') == 1:
            if n69wsp9p6r.get('globals'):
                n69wsp9p6r['globals'] = []
            if n69wsp9p6r.get('nonlocals'):
                n69wsp9p6r['nonlocals'] = []
        n69wsp9p6r = (n69wsp9p6r.get('imports_code') or '') + '\n' + '\n'.join(['global ' + x.strip() for x in n69wsp9p6r.get('globals') or [] if x.strip()]) + '\n' + '\n'.join(['nonlocal ' + x.strip() for x in n69wsp9p6r.get('nonlocals') or [] if x.strip()])
        if add_start_uidcomments:
            n69wsp9or3 = [l + f' #{UID_COMMENT_LEFTLABEL}{n69wspa2lb}{UID_COMMENT_RIGHTLABEL}' for l in n69wsp9p6r.split('\n')]
            n69wsp9p6r = '\n'.join(n69wsp9or3)
            n69wsp9p6r = n69wsp9p6r + ('\n' + f"{VARSEND_FUNC}_safe('{n69wspa2lb}', {{}})")
        n69wspa2fz = n69wsp9p6r + '\n' + n69wspa2il + '\n' + n69wspa2nw if style in ('both', 'pure') else ''
        n69wspa38q = n69wsp9p6r + '\n' + n69wspa2qt + '\n' + n69wspa31n if style in ('both', 'run') else ''
        return (n69wspa2fz, n69wspa38q)

    async def b69x8ynntt(self, n69wspa2np, base_id, choice, n69wspa2zf, n69wspa2wz=None, style='run', tolerance=0, codeswaps={}, import_range='current', x69xm5dtzp=False, n69wspa39a=None, conn=None, _skiplock='auto'):
        n69wspa2np = n69wspa2np.strip()
        assert x69xm5dtzx(n69wspa2np) == 'folder', f'Project root must be a directory, got {n69wspa2np}'
        assert base_id.startswith(n69wspa2np + '>') or base_id.startswith(n69wspa2np + '/') or n69wspa2np == '', f"Project root '{n69wspa2np}' is not a parent folder of your flow '{base_id}'."
        n69wspa398 = base_id.split('^')[0]
        assert '/' in n69wspa398
        n69wspa398 = n69wspa398.split('/')[-1]
        n69wspa2lv = n69wspa2np.rstrip('>') + '/' + n69wspa398 if n69wspa2np else n69wspa398
        n69wspa2um = False
        if n69wspa2zf:
            if n69wspa2zf['mode'] == 'single':
                if n69wspa2zf['node_type'] not in nesttypes | {'start'}:
                    n69wspa2um = True
                elif n69wspa2zf['node_type'] == 'start':
                    n69wspa2zf = {'mode': 'shell_only', 'node_br': n69wspa2zf.get('node_br', '^1#_')}
                else:
                    n69wspa2zf = {'mode': 'replace', 'section': [n69wspa2zf['uid'], n69wspa2zf['uid']]}
        else:
            n69wspa2zf = {}
        n69wspa2qs = ['Annotated', 'Doc', 'Optional', 'Any']
        if _skiplock == 'auto':
            if x69xm5dtzx(base_id) == 'func' and choice in ('all', 'allbelow') and ((n69wspa2zf or {}).get('mode') in ('replace', 'insert', 'allbelow', 'single')) and (len((n69wspa2wz or {'nodes': []}).get('nodes')) > 0):
                _skiplock = True
            else:
                _skiplock = False

        async def b69wsp9mrq(conn):
            nonlocal n69wspa2qs
            if n69wspa2zf:
                if n69wspa2zf['mode'] == 'shell_only':
                    n69wspa2pu = self.b69x8ynnu5(base_id, n69wspa2k7=n69wspa2zf.get('node_br', '^1#_'), style=style, tolerance=tolerance, codeswaps=codeswaps, add_start_uidcomments=True, _skiplock=True, conn=conn)
                elif not n69wspa2um:
                    n69wspa2pu = self.b69x8ynntv(base_id, choice, n69wspa2zf=n69wspa2zf, n69wspa2wz=n69wspa2wz, style=style, tolerance=tolerance, codeswaps=codeswaps, n69wspa39a=n69wspa39a, _skiplock=True, conn=conn)
                else:
                    n69wspa317 = table_lambda_get(n69wspa2wz['nodes'], lambda x: x['data']['uid'] == n69wspa2zf['uid'])[0]
                    n69wspa2pu = self.b69x8ynnv8(n69wspa2zf['uid'], n69wspa2zf['node_type'], n69wspa317=n69wspa317, style=style, tolerance=tolerance, codeswaps=codeswaps, n69wspa39a=n69wspa39a, _skiplock=True, conn=conn)
            else:
                n69wspa2pu = self.b69x8ynntv(base_id, choice, n69wspa2zf=None, style=style, tolerance=tolerance, codeswaps=codeswaps, _skiplock=True, conn=conn)
            n69wspa2r3 = self.b69x8ynnw4(trunk_to_func(base_id), conn=conn, _skiplock=True) if import_range == 'allabove' else aidle()
            n69wspa2pu, n69wspa2r3 = await asyncio.gather(n69wspa2pu, n69wspa2r3)
            n69wspa2dc = n69wspa2pu[0] if style == 'pure' else n69wspa2pu[1]
            if import_range == 'current':
                pass
            else:
                n69wspa34s = n69wspa2r3['froms'] + '\n' + n69wspa2r3['imports']
                n69wspa30x = [l for l in n69wspa2dc.split('\n') if not (l.startswith('import ') or l.startswith('from '))]
                n69wspa2dc = '\n'.join(n69wspa30x)
                n69wspa2dc = n69wspa34s + '\n' + n69wspa2dc
            n69wspa339 = n69wspa2dc
            n69wspa2dc = extract_roi(n69wspa2dc, shift_after_varsender=n69wspa2zf.get('mode') in ('replace', 'single'))
            n69wspa395 = {n69wspa2lv: n69wspa2dc}
            n69wspa2ul = {}
            n69wspa2n3 = "\n            SELECT COUNT(def_id)>0 FROM funcs WHERE def_id = '{module_id}'\n            "

            async def b69x8ynnu9(n69wspa2ga, hier_module_id):
                nonlocal n69wspa2qs
                n69wspa375 = []
                n69wspa36u = n69wspa2ga.get('level', 1)
                if n69wspa36u > 1:
                    n69wspa2gt = hier_module_id.split('/')[0].split('>')
                    if n69wspa36u > len(n69wspa2gt):
                        raise ValueError(f"Import relative path beyond any known path. With entry path {'/'.join(n69wspa2gt)}, attempting to go to {n69wspa36u} levels higher.")
                    n69wspa396 = '>'.join(n69wspa2gt[:-n69wspa36u + 1]) if n69wspa36u != 1 else '>'.join(n69wspa2gt)
                else:
                    n69wspa396 = n69wspa2np
                if n69wspa2ga['ntype'] == 'ImportFrom':
                    n69wspa2my = n69wspa2ga['module']
                    if '.' in n69wspa2my:
                        n69wspa322, n69wspa38w = n69wspa2my.rsplit('.', 1)
                        n69wspa2u9 = (n69wspa396.rstrip('>') + '>' if n69wspa396 else '') + n69wspa322.replace('.', '>') + '/' + n69wspa38w
                        n69wspa375.append(n69wspa2u9)
                    else:
                        n69wspa2u9 = (n69wspa396.rstrip('>') + '/' if n69wspa396 else '') + n69wspa2my
                        n69wspa375.append(n69wspa2u9)
                    n69wspa322 = n69wspa2my.replace('.', '>')
                    n69wspa38w = n69wspa2ga['names'][0]['name']
                    n69wspa2pd = (n69wspa396.rstrip('>') + '>' if n69wspa396 else '') + n69wspa322 + '/' + n69wspa38w
                    n69wspa375.append(n69wspa2pd)
                elif n69wspa2ga['ntype'] == 'Import':
                    n69wspa38w = n69wspa2ga['names'][0]['name']
                    n69wspa2w1 = (n69wspa396.rstrip('>') + ('/' if not '.' in n69wspa38w else '>') if n69wspa396 else '') + x69xm5dtzw(n69wspa38w)
                    n69wspa375.append(n69wspa2w1)
                n69wspa38g = []
                for n69wspa38m in n69wspa375:
                    n69wspa2e9 = n69wspa2n3.format(module_id=n69wspa38m)
                    n69wsp9oq8 = conn.execute(text(n69wspa2e9))
                    n69wspa38g.append(n69wsp9oq8)
                n69wspa2d9 = await asyncio.gather(*n69wspa38g)
                n69wspa2d9 = [c.fetchall()[0][0] for c in n69wspa2d9]
                n69wspa2n4 = [c for c in n69wspa2d9 if c > 0]
                if len(n69wspa2n4) == 0:
                    for n69wsp9oul in n69wspa2ga['names']:
                        n69wspa309 = n69wsp9oul.get('asname') or n69wsp9oul['name']
                        if not '*' in n69wspa309:
                            n69wspa2qs.append(n69wspa309)
                    return None
                else:
                    if len(n69wspa2n4) > 1:
                        pass
                    n69wspa2ms = 0
                    for n69wspa2ms in range(len(n69wspa2d9)):
                        if n69wspa2d9[n69wspa2ms] > 0:
                            break
                    return n69wspa375[n69wspa2ms]

            async def b69x8ynnud(code, hier_module_id):
                nonlocal n69wspa395
                nonlocal n69wspa2ul
                _, _, n69wsp9oq3 = b69wsp9mrs(b69wsp9mq1(code))
                n69wspa2jm = b69wsp9mq1(n69wsp9oq3)['body']
                n69wspa33s = []
                for n69wspa2ga in n69wspa2jm:
                    for namedic in n69wspa2ga['names']:
                        n69wsp9p59 = copy.deepcopy(n69wspa2ga)
                        n69wsp9p59['names'] = [namedic]
                        n69wspa33s.append(n69wsp9p59)
                n69wspa2uo = [b69x8ynnu9(n69wspa2ga, hier_module_id) for n69wspa2ga in n69wspa33s]
                n69wspa2uo = await asyncio.gather(*n69wspa2uo)
                n69wspa2ul[hier_module_id] = [n69wspa2hh for n69wspa2hh in n69wspa2uo if not n69wspa2hh is None]
                n69wspa2uo = [n69wspa2hh for n69wspa2hh in n69wspa2uo if not n69wspa2hh is None and (not n69wspa2hh in n69wspa395)]
                n69wspa36c = []
                for n69wspa2hh in n69wspa2uo:
                    if not self.x69xm5dtzs.get(n69wspa2hh):
                        n69wspa2ze = self.b69x8ynntv(n69wspa2hh, choice='all', style='run', tolerance=tolerance, conn=conn, _skiplock=True)
                    else:

                        async def b69x8ynnsu(n69wspa2hh):
                            return (0, self.x69xm5dtzs[n69wspa2hh])
                        n69wspa2ze = b69x8ynnsu(n69wspa2hh)
                    n69wspa36c.append(n69wspa2ze)
                n69wspa36c = await asyncio.gather(*n69wspa36c)
                n69wspa36c = [c[1] for c in n69wspa36c]
                n69wspa2qw = []
                for i in range(len(n69wspa2uo)):
                    n69wspa395[n69wspa2uo[i]] = n69wspa36c[i]
                    n69wspa2qw.append(b69x8ynnud(n69wspa36c[i], n69wspa2uo[i]))
                await asyncio.gather(*n69wspa2qw)
            await b69x8ynnud(n69wspa339, n69wspa2lv)

            async def b69x8ynnua():
                try:
                    n69wspa2ky = await self.b69x8ynnvf(conn=conn, _skiplock=True)
                    return n69wspa2ky
                except Exception as e:
                    return []
            n69wspa2m5 = [to_func_id(base_id)] + list(n69wspa395.keys())[1:]
            n69wspa31u = self.b69x8ynnuc(n69wspa2m5, conn=conn, _skiplock=True)
            n69wspa2ky, n69wspa31u = await asyncio.gather(b69x8ynnua(), n69wspa31u)
            return (n69wspa395, n69wspa2ul, n69wspa2ky, n69wspa31u)
        n69wsp9otg = await self._batch_read([b69wsp9mrq], _skiplock=_skiplock, conn=conn)
        n69wsp9otg, n69wspa2ul, n69wspa2ky, n69wspa31u = n69wsp9otg[0]
        for x69xm5dtzo in n69wsp9otg:
            if x69xm5dtzo == n69wspa2lv:
                continue
            try:
                n69wsp9otg[x69xm5dtzo] = b69wsp9moj(n69wsp9otg[x69xm5dtzo])
            except Exception as e:
                traceback.print_exc()
        for n69wspa38w, modcode in n69wsp9otg.items():
            if n69wspa38w != n69wspa2lv:
                self.x69xm5dtzs[n69wspa38w] = modcode
        n69wspa35u = {}
        for k, v in n69wsp9otg.items():
            n69wspa2dd = k.replace('>', '/') + '.py'
            n69wsp9orm = PRECODE + v
            n69wspa35u[n69wspa2dd] = n69wsp9orm
        n69wspa2gp = n69wspa2lv.replace('>', '/') + '.py'
        n69wspa2dk = [r.replace('>', '.').replace('/', '.') for r in self.x69xm5dtzr]
        n69wspa38r = {}
        for k, v in n69wspa2ul.items():
            k = k.replace('>', '.').replace('/', '.')
            v = [name.replace('>', '.').replace('/', '.') for name in v]
            n69wspa38r[k] = v
        n69wspa2dk = [r.replace('>', '.').replace('/', '.') for r in n69wspa2dk if r.replace('>', '.').replace('/', '.') in n69wspa38r.keys()]
        if not n69wspa2lv.replace('>', '.').replace('/', '.') in n69wspa2dk:
            n69wspa2dk.append(n69wspa2lv.replace('>', '.').replace('/', '.'))

        def b69wspa0yl(n69wspa2x4):
            nonlocal n69wspa2dk
            n69wspa38l = []
            for k, v in n69wspa38r.items():
                if n69wspa2x4 in v:
                    if not k in n69wspa2dk:
                        n69wspa38l.append(k)
            n69wspa2dk = n69wspa2dk + n69wspa38l
            for supmod in n69wspa38l:
                b69wspa0yl(supmod)
        for n69wspa2x4 in n69wspa2dk.copy():
            b69wspa0yl(n69wspa2x4)
        n69wspa2f7 = None
        if x69xm5dtzp:
            n69wspa2sm = n69wspa35u[n69wspa2gp]
            n69wspa2to = b69wsp9mrv(n69wspa2sm)
            n69wspa2fl = f"    global {','.join(n69wspa2to)}" if n69wspa2to else '    '
            n69wspa2sm = f'async def __user_main__():\n{n69wspa2fl}\n' + '    try:\n' + n69wspa2sm.replace('\n', '\n        ') + '\n    finally:\n        globals().update(locals())\n'
            n69wspa35u[n69wspa2gp] = n69wspa2sm
            n69wspa2f7 = '__user_main__'
        n69wsp9onl = {'files': n69wspa35u, 'entry': n69wspa2gp, 'reloads': n69wspa2dk, 'builtins': n69wspa2qs, 'external_pkgs': n69wspa2ky, 'async_wrapper': n69wspa2f7, 'cnskey': base_id, 'untrack_vars': n69wspa31u}
        return n69wsp9onl

    async def b69x8ynnuf(self, n69wspa2it, conn=None, _skiplock=False):
        assert x69xm5dtzx(n69wspa2it) == 'func'

        async def b69wsp9mrq(conn):
            n69wspa2vq = await self.select('funcmeta', conds=[{'def_id': n69wspa2it}], conn=conn, _skiplock=True)
            n69wspa2vq = n69wspa2vq['untracked_vars'].tolist()
            if not n69wspa2vq:
                return []
            else:
                assert len(n69wspa2vq) == 1, n69wspa2vq
                assert isinstance(n69wspa2vq[0], list), n69wspa2vq
                return n69wspa2vq[0]
        n69wsp9otg = await self._batch_read([b69wsp9mrq], _skiplock=_skiplock, conn=conn)
        return n69wsp9otg[0]

    async def b69x8ynntm(self, n69wspa2it, n69wsp9p5l, conn=None, _skiplock=False):
        assert x69xm5dtzx(n69wspa2it) == 'func'
        assert isinstance(n69wsp9p5l, list)
        for v in n69wsp9p5l:
            assert v in ('<ALL>', '<UNTRACK>', '<TRACK>') or is_valid_varname(v) or all([is_valid_varname(a) for a in v.split('.')]), f'Invalid var name: {v}'
        if '<UNTRACK>' in n69wsp9p5l and '<TRACK>' in n69wsp9p5l:
            n69wsp9p5l = [v for v in n69wsp9p5l if v != '<TRACK>']
        if '<UNTRACK>' in n69wsp9p5l:
            n69wsp9p5l = ['<UNTRACK>'] + [v for v in n69wsp9p5l if v != '<UNTRACK>']
        elif '<TRACK>' in n69wsp9p5l:
            n69wsp9p5l = ['<TRACK>'] + [v for v in n69wsp9p5l if v != '<TRACK>']
        n69wspa2hy = pd.DataFrame([{'def_id': n69wspa2it, 'untracked_vars': n69wsp9p5l}])
        await self.upsert('funcmeta', n69wspa2hy, conn=conn, _skiplock=_skiplock)

    async def b69x8ynnuc(self, baseids, conn=None, _skiplock=False):

        async def b69wsp9mrq(conn):
            n69wspa2gf = [to_func_id(bid) for bid in baseids]
            n69wspa389 = [f"def_id LIKE '{n69wspa2jx}%'" for n69wspa2jx in n69wspa2gf]
            n69wspa32a = ' OR '.join(n69wspa389)
            n69wspa2o6 = await self.select('funcmeta', cond_sql=n69wspa32a, targets=['def_id', 'untracked_vars'], conn=conn, _skiplock=True)
            n69wspa31u = {}
            for r in n69wspa2o6.to_dict(orient='records'):
                if '<TRACK>' in r['untracked_vars']:
                    option = 'track'
                else:
                    option = 'untrack'
                n69wspa37f = {'option': option, 'vars': [v for v in r['untracked_vars'] if not v in ('<TRACK>', '<UNTRACK>')]}
                n69wspa31u[r['def_id']] = n69wspa37f
            return n69wspa31u
        n69wsp9otg = await self._batch_read([b69wsp9mrq], _skiplock=_skiplock, conn=conn)
        return n69wsp9otg[0]

    async def b69x8ynnw2(self, n69wspa2yk, n69wspa381, extpkgs, n69wspa2wc, conn, targcode=''):
        n69x75d5wx = '\n'.join([l.strip() for l in targcode.split('\n') if l.strip().startswith('import') or l.strip().startswith('from')])
        n69wspa371, _ = await self.b69x8ynnvd(n69wspa2yk, n69wspa381, 'both', extpkgs=extpkgs, n69wspa2z4=n69wspa2wc, recur_objs=True, n69x75d5wx=n69x75d5wx, conn=conn, _skiplock=True)
        n69wspa2xj = n69wspa371['classes']['def_id'].tolist()
        n69wspa2ka = n69wspa371['funcs']
        n69wspa2cc = n69wspa371['params']
        n69wspa2ka = n69wspa2ka[~n69wspa2ka['def_id'].str.contains('\\*')]
        n69wspa2ka = n69wspa2ka['def_id'].tolist()
        n69wspa2j1 = [(x.split('/')[-1], x) for x in n69wspa2ka]
        n69wspa2eh = [(x.split('*')[-1], x) for x in n69wspa2xj]
        n69wspa2es = n69wspa371['objs']
        n69wspa2tw = list(set(n69wspa2cc['def_id'].tolist()))
        n69wspa2ek = {}
        n69wspa2sb = 0
        n69wspa2jr = time.time()
        n69wspa2pf = {key: sub_df for key, sub_df in n69wspa2cc[n69wspa2cc['ctx'] == 'input'].groupby('def_id', sort=False)}
        n69wspa2ex = time.time()
        n69wspa2sb = n69wspa2ex - n69wspa2jr
        n69wspa2yq = 0
        for imppid in n69wspa2tw:
            n69wspa2ex = time.time()
            n69wspa2ln = n69wspa2pf.get(imppid)
            n69wspa2ln = [] if n69wspa2ln is None else n69wspa2ln.to_dict(orient='records')
            myexpr, myfmtd = b69wsp9mon(n69wspa2ln, resort=True)
            n69wspa302 = time.time()
            n69wspa2yq = n69wspa2yq + (n69wspa302 - n69wspa2ex)
            n69wspa2ek[imppid] = myfmtd
        n69wspa358 = await self.b69x8ynnv1(n69wspa2yk, n69wspa2wc=n69wspa2wc, conn=conn, _skiplock=True)
        n69wspa358 = pd.concat([n69wspa358, n69wspa2es], ignore_index=True)
        n69wspa358['type'] = n69wspa358['type'].apply(lambda x: replace_lastpart(x, '*', repfrom='.', repto=DOT_REPL) if isinstance(x, str) else x)
        n69wspa358['name'] = n69wspa358['name'].apply(lambda x: x.replace('.', DOT_REPL) if isinstance(x, str) else x)
        return (n69wspa2j1, n69wspa2eh, n69wspa2ek, n69wspa358)

    async def b69x8ynnuo(self, n69wsp9oya, conn=None, _skiplock=False):
        n69wspa2h8 = n69wsp9oya[n69wsp9oya['node_type'].isin(nesttypes)]
        n69wspa34c = pd.DataFrame(columns=list(n69wspa2ha.keys()))
        n69wsp9p2u = pd.DataFrame(columns=list(n69wspa2ts.keys()))
        n69wspa2xf = pd.DataFrame(columns=list(n69wspa2dn.keys()))
        n69wspa2d3 = pd.DataFrame(columns=list(n69wspa356.keys()))
        if len(n69wspa2h8) > 0:
            n69wspa2yx = n69wspa2h8['hid'].tolist()
            n69wspa354 = list(set(n69wspa2h8['def_id'].tolist()))
            assert len(n69wspa354) == 1, f'eh013：{n69wspa354}'
            n69wspa32r = n69wspa354[0]
            n69wspa2nn = await asyncio.gather(*[self.b69x8ynnt5(n69wspa32r, fnhid, conn=conn, _skiplock=_skiplock) for fnhid in n69wspa2yx])
            n69wspa37x = [x[0] for x in n69wspa2nn] + [n69wspa34c]
            n69wspa2nv = [x[1] for x in n69wspa2nn] + [n69wsp9p2u]
            n69wspa2hc = [x[2] for x in n69wspa2nn] + [n69wspa2xf]
            n69wspa2ia = [x[3] for x in n69wspa2nn] + [n69wspa2d3]
            n69wspa34c = pd.concat(n69wspa37x, ignore_index=True)
            n69wsp9p2u = pd.concat(n69wspa2nv, ignore_index=True)
            n69wspa2xf = pd.concat(n69wspa2hc, ignore_index=True)
            n69wspa2d3 = pd.concat(n69wspa2ia, ignore_index=True)
        return (n69wspa34c, n69wsp9p2u, n69wspa2xf, n69wspa2d3)

    async def b69x8ynnvs(self, n69wspa2tb, conn=None, _skiplock=False):
        n69wspa32s = f"\n        def_id LIKE '{n69wspa2tb}/%' OR def_id LIKE '{n69wspa2tb}*%'\n        "
        n69wspa34c = self.select('nodes', cond_sql=n69wspa32s, conn=conn, _skiplock=_skiplock)
        n69wsp9p2u = self.select('funcs', cond_sql=n69wspa32s, conn=conn, _skiplock=_skiplock)
        n69wspa2xf = self.select('classes', cond_sql=n69wspa32s, conn=conn, _skiplock=_skiplock)
        n69wspa2d3 = self.select('params', cond_sql=n69wspa32s, conn=conn, _skiplock=_skiplock)
        n69wspa34c, n69wsp9p2u, n69wspa2xf, n69wspa2d3 = await asyncio.gather(n69wspa34c, n69wsp9p2u, n69wspa2xf, n69wspa2d3)
        return (n69wspa34c, n69wsp9p2u, n69wspa2xf, n69wspa2d3)

    async def b69x8ynnv4(self, leftuid, rightuid, n69wsp9oya, n69wspa34y, dest='cache', conn=None, _skiplock=False):

        async def b69wsp9mrq(conn):
            n69wspa2zz = await self.b69x8ynntg(n69wsp9oya, n69wspa34y, n69wsp9ore=False, level='adlvf', section={'lefts': [leftuid], 'rights': [rightuid]}, _skiplock=True)
            assert not 'start' in n69wspa2zz['node_type'].tolist(), 'Cannot copy the start node.'
            if len(n69wspa2zz) == 0:
                raise RuntimeError(f'No selection detected with left={leftuid}, right={rightuid}')
            n69wspa2el = list(set(n69wspa2zz['def_id'].tolist()))
            assert len(n69wspa2el) == 1, f'eh014：{n69wspa2el}'
            n69wspa2el = n69wspa2el[0]
            n69wspa2gv = n69wspa2zz[n69wspa2zz['uid'] == leftuid]['hid'].tolist()[0]
            n69wspa2yl = n69wspa2zz[n69wspa2zz['uid'] == rightuid]['hid'].tolist()[0]
            assert not n69wspa2gv.endswith('.0') and (not n69wspa2yl.endswith('.0'))
            n69wspa2e1 = n69wspa2zz[n69wspa2zz['uid'] == leftuid]['branch'].tolist()[0]
            n69wspa2od = n69wspa2zz[n69wspa2zz['uid'] == rightuid]['branch'].tolist()[0]
            assert n69wspa2gv.count('.') == n69wspa2yl.count('.'), f'The two ends of the selection are not on the same level: {n69wspa2gv} vs {n69wspa2gv}'
            n69wsp9onr = n69wspa2gv.rsplit('.', 1)[0]
            assert n69wspa2yl.startswith(n69wsp9onr + '.'), f'The two ends of the selection are not under the same parent: {n69wspa2gv} vs {n69wspa2gv}'
            assert n69wspa2e1 == n69wspa2od, f'The two ends of the selection are not under the same branch: {n69wspa2e1} vs {n69wspa2od}'
            n69wspa34c, n69wsp9p2u, n69wspa2xf, n69wspa2d3 = await self.b69x8ynnuo(n69wspa2zz, conn=conn, _skiplock=True)
            n69wspa2pn = [c for c in n69wspa2zz.columns.tolist() if c in n69wspa2ha.keys()]
            n69wspa2g7 = n69wspa2zz[n69wspa2pn]
            n69wspa2fo = {'def_id': n69wspa2el, 'parent_hid': n69wsp9onr, 'left_hids': [n69wspa2gv], 'right_hids': [n69wspa2yl], 'parent_br': rectify_cond(n69wspa2e1), 'nodes': pd.concat([n69wspa2g7, n69wspa34c], ignore_index=True), 'funcs': n69wsp9p2u, 'classes': n69wspa2xf, 'params': n69wspa2d3}
            if dest == 'cache':
                self.n69wspa2fo = n69wspa2fo
            else:
                return n69wspa2fo
        n69wsp9onl = await self._batch_read([b69wsp9mrq], conn=conn, _skiplock=_skiplock)
        return n69wsp9onl[0]

    async def b69x8ynnu0(self, leftuid, n69wsp9oya, n69wspa34y, shell=None, dest='cache', conn=None, _skiplock=False):

        async def b69wsp9mrq(conn):
            if leftuid != '0':
                n69wspa2zz = await self.b69x8ynntg(n69wsp9oya, n69wspa34y, n69wsp9ore=False, level='adlvf', section={'lefts': [leftuid], 'rights': []}, _skiplock=True)
                assert not 'start' in n69wspa2zz['node_type'].tolist(), 'Cannot copy the start node.'
                n69wspa2z7 = n69wspa2zz[n69wspa2zz['uid'] == leftuid]['nexts'].tolist()[0]
                n69wspa2d5 = n69wspa2zz[n69wspa2zz['uid'] == leftuid]['hid'].tolist()[0]
                n69wspa2ip = loadstr(n69wspa2zz[n69wspa2zz['uid'] == leftuid]['branch'].tolist()[0])
                n69wsp9onr = n69wspa2d5.rsplit('.', 1)[0]
                n69wspa2zz = n69wspa2zz[(n69wspa2zz['uid'] != leftuid) & ~n69wspa2zz['hid'].str.contains(f'\\.{leftuid}\\.')]
            else:
                assert shell.count('#') == 1 and (not '^' in shell)
                n69wspa2p0 = await self.b69x8ynntg(n69wsp9oya, n69wspa34y, n69wsp9ore=False, level='adlvd', _skiplock=True)
                n69wsp9onr = shell.split('#')[0]
                n69wspa2ip = loadstr(shell.split('#')[1].strip())

                def b69wspa0y1(x):
                    x = loadstr(x)
                    n69wspa2wy = x == n69wspa2ip
                    return n69wspa2wy
                n69wspa2z7 = n69wspa2p0[n69wspa2p0['hid'].str.startswith(n69wsp9onr + '.') & ~n69wspa2p0['hid'].str[len(n69wsp9onr) + 1:].str.contains('\\.') & n69wspa2p0['branch'].apply(b69wspa0y1) & (n69wspa2p0['pres'].isna() | n69wspa2p0['pres'].apply(lambda x: x == [])) & ~n69wspa2p0['uid'].str.contains('-end')]['hid'].tolist()
                n69wspa2r3 = [h.split('.')[-1] for h in n69wspa2z7]
                if '0' in n69wspa2r3:
                    n69wspa2r3 = [l for l in n69wspa2r3 if l != '0']
                    n69wspa2hu = n69wspa2p0[n69wspa2p0['pres'].apply(lambda x: False if x is None else '1.0' in x)]
                    n69wspa2zm = n69wspa2hu['uid'].tolist()
                    n69wspa2yf = n69wspa2hu['hid'].tolist()
                    n69wspa2r3 = n69wspa2r3 + n69wspa2zm
                    n69wspa2z7 = [l for l in n69wspa2z7 if not l.endswith('.0')]
                    n69wspa2z7 = n69wspa2z7 + n69wspa2yf
                n69wspa2zz = await self.b69x8ynntg(n69wsp9oya, n69wspa34y, n69wsp9ore=False, level='adlvf', section={'lefts': n69wspa2r3, 'rights': []}, _skiplock=True)
            if len(n69wspa2zz) == 0:
                raise RuntimeError(f'No selection detected with left={leftuid}')
            n69wspa2el = list(set(n69wspa2zz['def_id'].tolist()))
            assert len(n69wspa2el) == 1, f'eh014：{n69wspa2el}'
            n69wspa2el = n69wspa2el[0]
            for n69wspa2gv in n69wspa2z7:
                assert not n69wspa2gv.endswith('.0')
                assert n69wspa2gv.startswith(n69wsp9onr + '.')
                assert n69wspa2gv[len(n69wsp9onr):].count('.') == 1
                assert not '-end' in n69wspa2gv
            n69wspa34q = n69wspa2zz[n69wspa2zz['hid'].isin(n69wspa2z7)]['branch'].tolist()
            assert len(set(n69wspa34q)) == 1, f'eh015：{set(n69wspa34q)}'
            n69wsp9p3g = n69wspa34q[0]
            assert n69wsp9p3g == n69wspa2ip
            n69wspa354 = n69wspa2zz[n69wspa2zz['hid'].str.startswith(n69wsp9onr + '.') & ~n69wspa2zz['hid'].str[len(n69wsp9onr) + 1:].str.contains('\\.') & (n69wspa2zz['branch'] == n69wspa2ip) & (n69wspa2zz['nexts'].isna() | n69wspa2zz['nexts'].apply(lambda x: x == []))]['hid'].tolist()
            n69wspa34c, n69wsp9p2u, n69wspa2xf, n69wspa2d3 = await self.b69x8ynnuo(n69wspa2zz, conn=conn, _skiplock=True)
            n69wspa2pn = [c for c in n69wspa2zz.columns.tolist() if c in n69wspa2ha.keys()]
            n69wspa2g7 = n69wspa2zz[n69wspa2pn]
            n69wspa2fo = {'def_id': n69wspa2el, 'parent_hid': n69wsp9onr, 'left_hids': n69wspa2z7, 'right_hids': n69wspa354, 'parent_br': rectify_cond(n69wsp9p3g), 'nodes': pd.concat([n69wspa2g7, n69wspa34c], ignore_index=True), 'funcs': n69wsp9p2u, 'classes': n69wspa2xf, 'params': n69wspa2d3}
            if dest == 'cache':
                self.n69wspa2fo = n69wspa2fo
            else:
                return n69wspa2fo
        n69wsp9onl = await self._batch_read([b69wsp9mrq], conn=conn, _skiplock=_skiplock)
        return n69wsp9onl[0]

    async def b69x8ynnui(self, shell, n69wsp9oya, n69wspa34y, dest='cache', conn=None, _skiplock=False):
        assert x69xm5dtzx(shell) == 'cond'

        async def b69wsp9mrq(conn):
            n69wspa2p0 = await self.b69x8ynntg(n69wsp9oya, n69wspa34y, n69wsp9ore=False, level='adlvd', _skiplock=True)
            n69wsp9onr = shell.split('#')[0]
            n69wspa2ip = loadstr(shell.split('#')[1].strip())

            def b69wspa0y1(x):
                x = loadstr(x)
                n69wspa2wy = x == n69wspa2ip
                return n69wspa2wy
            n69wspa2z7 = n69wspa2p0[n69wspa2p0['hid'].str.startswith(n69wsp9onr + '.') & ~n69wspa2p0['hid'].str[len(n69wsp9onr) + 1:].str.contains('\\.') & n69wspa2p0['branch'].apply(b69wspa0y1) & (n69wspa2p0['pres'].isna() | n69wspa2p0['pres'].apply(lambda x: x == [])) & ~n69wspa2p0['uid'].str.contains('-end')]['hid'].tolist()
            n69wspa2r3 = [h.split('.')[-1] for h in n69wspa2z7]
            if '0' in n69wspa2r3:
                n69wspa2r3 = [l for l in n69wspa2r3 if l != '0']
                n69wspa2hu = n69wspa2p0[n69wspa2p0['pres'].apply(lambda x: False if x is None else '1.0' in x)]
                n69wspa2zm = n69wspa2hu['uid'].tolist()
                n69wspa2yf = n69wspa2hu['hid'].tolist()
                n69wspa2r3 = n69wspa2r3 + n69wspa2zm
                n69wspa2z7 = [l for l in n69wspa2z7 if not l.endswith('.0')]
                n69wspa2z7 = n69wspa2z7 + n69wspa2yf
            n69wspa2zz = await self.b69x8ynntg(n69wsp9oya, n69wspa34y, n69wsp9ore=False, level='adlvf', section={'lefts': n69wspa2r3, 'rights': []}, _skiplock=True)
            n69wspa2el = list(set(n69wspa2zz['def_id'].tolist()))
            assert len(n69wspa2el) == 1, f'eh014：{n69wspa2el}'
            n69wspa2el = n69wspa2el[0]
            for n69wspa2gv in n69wspa2z7:
                assert not n69wspa2gv.endswith('.0')
                assert n69wspa2gv.startswith(n69wsp9onr + '.')
                assert n69wspa2gv[len(n69wsp9onr):].count('.') == 1
                assert not '-end' in n69wspa2gv
            n69wspa354 = n69wspa2zz[n69wspa2zz['hid'].str.startswith(n69wsp9onr + '.') & ~n69wspa2zz['hid'].str[len(n69wsp9onr) + 1:].str.contains('\\.') & (n69wspa2zz['branch'] == n69wspa2ip) & (n69wspa2zz['nexts'].isna() | n69wspa2zz['nexts'].apply(lambda x: x == []))]['hid'].tolist()
            n69wspa2nn = self.b69x8ynnuo(n69wspa2zz, conn=conn, _skiplock=True)
            n69wspa34h = self.b69x8ynnvs(n69wspa2el + '^' + n69wsp9onr + '#' + str(n69wspa2ip), conn=conn, _skiplock=True)
            n69wspa2nn, n69wspa34h = await asyncio.gather(n69wspa2nn, n69wspa34h)
            n69wspa34c, n69wsp9p2u, n69wspa2xf, n69wspa2d3 = n69wspa2nn
            innodes, infuncs, inclasses, inparams = n69wspa34h
            n69wspa2pn = [c for c in n69wspa2zz.columns.tolist() if c in n69wspa2ha.keys()]
            n69wspa2g7 = n69wspa2zz[n69wspa2pn]
            n69wspa2h6 = pd.concat([n69wspa2g7, n69wspa34c, innodes], ignore_index=True)
            n69wsp9p3q = pd.concat([n69wsp9p2u, infuncs], ignore_index=True)
            n69wsp9p6q = pd.concat([n69wspa2xf, inclasses], ignore_index=True)
            n69wsp9osv = pd.concat([n69wspa2d3, inparams], ignore_index=True)
            n69wspa2fo = {'def_id': n69wspa2el, 'parent_hid': n69wsp9onr, 'left_hids': n69wspa2z7, 'right_hids': n69wspa354, 'parent_br': n69wspa2ip, 'nodes': n69wspa2h6, 'funcs': n69wsp9p3q, 'classes': n69wsp9p6q, 'params': n69wsp9osv}
            if dest == 'cache':
                self.n69wspa2fo = n69wspa2fo
            else:
                return n69wspa2fo
            return
        n69wsp9onl = await self._batch_read([b69wsp9mrq], conn=conn, _skiplock=_skiplock)
        return n69wsp9onl[0]

    async def b69x8ynnts(self, n69wspa2wq, conn=None, _skiplock=False):

        async def b69wsp9mrq(conn):
            if x69xm5dtzx(n69wspa2wq) == 'class':
                n69wspa2xv = '*'
                n69wspa2sz = '/'
            elif x69xm5dtzx(n69wspa2wq) == 'func':
                n69wspa2xv = '/'
                n69wspa2sz = '^'
            else:
                raise f'def_id neither func nor class: {n69wspa2wq}'
            hier_defid, right = n69wspa2wq.rsplit('^', 1)
            n69wspa2cn, right = right.split('#', 1)
            n69wspa35y, right = right.split(n69wspa2xv, 1)
            n69wspa2op = None
            if '*' in n69wspa35y:
                n69wspa35y, n69wspa2op = n69wspa35y.split('*')
            assert x69xm5dtzx(hier_defid) == 'func'
            assert not any([x in n69wspa2cn for x in ('#', '/', '^', '*')]), n69wspa2cn
            if not n69wspa2cn.startswith('1.') and (not n69wspa2cn == '1'):
                assert not '.' in n69wspa2cn
                n69wspa2cn = await self.b69x8ynnvc(n69wspa2cn, _skiplock=True, conn=conn)
            assert not any([x in n69wspa35y for x in ('#', '/', '^', '*', '.')]), n69wspa35y
            n69wspa35y = loadstr(n69wspa35y)
            if n69wspa2op:
                assert not any([x in n69wspa2op for x in ('#', '/', '^', '*', '.')]), n69wspa2op
            n69wspa357 = f"\n            def_id = '{n69wspa2wq}' OR def_id LIKE '{n69wspa2wq + n69wspa2sz}%'\n            "
            n69wsp9oya = self.select('nodes', cond_sql=n69wspa357, conn=conn, _skiplock=True)
            n69wsp9p3q = self.select('funcs', cond_sql=n69wspa357, conn=conn, _skiplock=True)
            n69wsp9p6q = self.select('classes', cond_sql=n69wspa357, conn=conn, _skiplock=True)
            n69wsp9osv = self.select('params', cond_sql=n69wspa357, conn=conn, _skiplock=True)
            n69wsp9oya, n69wsp9p3q, n69wsp9p6q, n69wsp9osv = await asyncio.gather(n69wsp9oya, n69wsp9p3q, n69wsp9p6q, n69wsp9osv)
            self.n69wspa2fo = {'def_id': hier_defid, 'parent_hid': n69wspa2cn, 'left_hids': [], 'right_hids': [], 'parent_br': n69wspa35y, 'nodes': n69wsp9oya, 'funcs': n69wsp9p3q, 'classes': n69wsp9p6q, 'params': n69wsp9osv}
            if n69wspa2op:
                self.n69wspa2fo['parent_class'] = n69wspa2op
        await self._batch_read([b69wsp9mrq], conn=conn, _skiplock=_skiplock)

    def b69wspa0y2(self, new_def_id, new_parent_hid, new_parent_br, n69wspa2fo, new_parent_class=None, renew_uids=True):
        n69wspa2z0 = n69wspa2fo['nodes']['uid'].tolist()
        n69wspa2mn = [u for u in n69wspa2z0 if not '-end' in u]
        n69wspa2go = [u for u in n69wspa2z0 if '-end' in u]
        n69wsp9own = {n69wsp9oq0: gen_base36_id() if renew_uids else n69wsp9oq0 for n69wsp9oq0 in n69wspa2mn}
        for n69wspa2i0 in n69wspa2go:
            olduid, endfix = n69wspa2i0.split('-')
            if not olduid in n69wsp9own:
                continue
            n69wspa2ni = n69wsp9own[olduid] + '-' + endfix
            assert not n69wspa2i0 in n69wsp9own
            n69wsp9own[n69wspa2i0] = n69wspa2ni
        assert not '0' in n69wsp9own and (not '1' in n69wsp9own)
        n69wspa382 = n69wspa2fo['def_id'] + '^' + n69wspa2fo['parent_hid'] + '#' + str(n69wspa2fo['parent_br'])
        if n69wspa2fo.get('parent_class'):
            n69wspa382 = n69wspa382 + ('*' + n69wspa2fo['parent_class'])
        n69wspa36h = new_def_id + '^' + new_parent_hid + '#' + str(rectify_cond(new_parent_br))
        if new_parent_class:
            n69wspa36h = n69wspa36h + ('*' + new_parent_class)
        assert not any([u in n69wsp9own for u in n69wspa2fo['parent_hid'].split('.')])
        n69wsp9ozx = {}

        def b69wspa0yg(n69wspa2xo):
            n69wsp9p51 = n69wspa2xo['hid']
            n69wspa2el = n69wspa2xo['def_id']
            if n69wspa2el == n69wspa2fo['def_id']:
                assert n69wsp9p51.startswith(n69wspa2fo['parent_hid'] + '.')
                right = n69wsp9p51[len(n69wspa2fo['parent_hid']) + 1:].split('.')
                n69wspa2qk = [n69wsp9own[r] for r in right]
                n69wspa35e = '.'.join(n69wspa2qk)
                n69wsp9orn = new_parent_hid + '.' + n69wspa35e
                n69wsp9ozx[right[-1]] = n69wsp9orn
            else:
                n69wspa33t = n69wsp9p51.split('.')
                n69wspa2wq = [n69wsp9own[p] if not p in ('1', '0') else p for p in n69wspa33t]
                n69wsp9orn = '.'.join(n69wspa2wq)
                n69wsp9ozx[n69wspa33t[-1]] = n69wsp9orn
            return n69wsp9orn

        def b69wspa0xx(n69wspa2xo):
            if n69wspa2xo['def_id'] == n69wspa2fo['def_id']:
                if n69wspa2xo['hid'].startswith(new_parent_hid + '.'):
                    if not '.' in n69wspa2xo['hid'][len(new_parent_hid) + 1:]:
                        return new_parent_br
            return n69wspa2xo['branch']

        def b69wspa0yc(lst):
            if not lst:
                return lst
            n69wspa2n5 = [n69wsp9ozx.get(h.split('.')[-1], None) for h in lst]
            n69wspa2n5 = [n for n in n69wspa2n5 if n is not None]
            return n69wspa2n5

        def b69wspa0xk(h):
            if not h:
                return h
            return n69wsp9ozx.get(h.split('.')[-1])

        def b69wspa0y6(n69wspa2el):
            n69wspa2qv = [i for i in range(len(n69wspa2el)) if n69wspa2el[i] == '^']
            n69wspa2kk = [i for i in range(len(n69wspa2el)) if n69wspa2el[i] == '#']
            assert len(n69wspa2qv) == len(n69wspa2kk)
            n69wspa2ma = [n69wspa2el[n69wspa2qv[i] + 1:n69wspa2kk[i]] for i in range(len(n69wspa2qv))]
            n69wspa34b = [0] + n69wspa2kk
            n69wspa30g = n69wspa2qv + [len(n69wspa2el)]
            n69wspa2om = [n69wspa2el[n69wspa34b[i]:n69wspa30g[i] + 1] for i in range(len(n69wspa34b))]
            n69wspa2wq = []
            i = -1
            for i in range(len(n69wspa2ma)):
                n69wspa2wq.append(n69wspa2om[i])
                n69wspa2y7 = [n69wsp9own[olduid] if not olduid in ('0', '1', 'UNDEFINED') else olduid for olduid in n69wspa2ma[i].split('.')]
                n69wspa2wq.append('.'.join(n69wspa2y7))
            n69wspa2wq.append(n69wspa2om[i + 1])
            return ''.join(n69wspa2wq)

        def b69wspa0yi(n69wspa2gg, mode):
            if mode == 'sub':
                assert n69wspa2gg.startswith('.'), n69wspa2gg
                n69wspa30q = '_IMPOOSSSIBLE^UNDEFINED'
            elif mode == 'inner':
                assert n69wspa2gg.startswith('/') or n69wspa2gg.startswith('*'), n69wspa2gg
                n69wspa30q = ''
            n69wspa2eo = n69wspa30q + n69wspa2gg
            n69wspa2eo = b69wspa0y6(n69wspa2eo)
            n69wspa2v0 = n69wspa2eo[len(n69wspa30q):]
            return n69wspa2v0

        def b69wspa0xw(n69wspa2el):
            if not (n69wspa2el.startswith(n69wspa382 + '/') or n69wspa2el.startswith(n69wspa382 + '*')):
                if n69wspa2el == n69wspa2fo['def_id']:
                    return new_def_id
                else:
                    n69wspa2uc = n69wspa382.rsplit('#', 1)[0]
                    assert n69wspa2el.startswith(n69wspa2uc + '.')
                    n69wspa2v6 = n69wspa36h.rsplit('#', 1)[0]
                    n69wspa2jy = n69wspa2v6 + b69wspa0yi(n69wspa2el[len(n69wspa2uc):], 'sub')
                    return n69wspa2jy
            if x69xm5dtzx(n69wspa36h) == 'class':
                assert not n69wspa2el[len(n69wspa382):].startswith('*'), f'Cannot paste a class into another class'
            n69wspa2jy = n69wspa36h + b69wspa0yi(n69wspa2el[len(n69wspa382):], 'inner')
            return n69wspa2jy
        n69wsp9oz2 = n69wspa2fo['nodes'].copy()
        if len(n69wsp9oz2) > 0:
            n69wsp9oz2['uid'] = n69wsp9oz2['uid'].apply(lambda x: n69wsp9own[x])
            n69wsp9oz2['hid'] = n69wsp9oz2.apply(b69wspa0yg, axis=1)
            n69wsp9oz2['branch'] = n69wsp9oz2.apply(b69wspa0xx, axis=1)
            n69wsp9oz2['source'] = n69wsp9oz2['source'].apply(b69wspa0xk)
            n69wsp9oz2['data_providers'] = n69wsp9oz2['data_providers'].apply(b69wspa0yc)
            n69wsp9oz2['pres'] = n69wsp9oz2['pres'].apply(b69wspa0yc)
            n69wsp9oz2['nexts'] = n69wsp9oz2['nexts'].apply(b69wspa0yc)
        n69wsp9ox6 = n69wspa2fo['funcs'].copy()
        n69wsp9p0s = n69wspa2fo['classes'].copy()
        n69wspa2ws = n69wspa2fo['params'].copy()
        n69wsp9oz2['def_id'] = n69wsp9oz2['def_id'].apply(b69wspa0xw)
        n69wsp9ox6['def_id'] = n69wsp9ox6['def_id'].apply(b69wspa0xw)
        n69wsp9p0s['def_id'] = n69wsp9p0s['def_id'].apply(b69wspa0xw)
        n69wspa2ws['def_id'] = n69wspa2ws['def_id'].apply(b69wspa0xw)
        n69wsp9ox6['uid'] = n69wsp9ox6['uid'].apply(lambda x: idgen.generate('f'))
        n69wsp9p0s['uid'] = n69wsp9p0s['uid'].apply(lambda x: idgen.generate('c'))
        n69wspa364 = [n69wsp9ozx[lh.split('.')[-1]] for lh in n69wspa2fo['left_hids']]
        n69wspa2qk = [n69wsp9ozx.get(rh.split('.')[-1], '<UNDEFINED>') for rh in n69wspa2fo['right_hids'] if rh in n69wsp9ozx]
        return (n69wsp9oz2, n69wsp9ox6, n69wsp9p0s, n69wspa2ws, n69wspa364, n69wspa2qk)

    async def b69x8ynnt4(self, n69wspa2lp, n69wspa2nm=[], external_class_ids=[], n69wspa2da=None, cached=False, n69wspa381=None, sustainable=False, del_funcs=[], _skiplock=False, conn=None, tolerance=0):
        if not self.n69wspa2fo:
            raise ValueError('Nothing to paste.')

        async def b69x8ynnuq(conn):
            nonlocal n69wspa2lp
            n69wsp9onh = None
            if x69xm5dtzx(n69wspa2lp) == 'class':
                n69wspa2lp, n69wsp9onh = n69wspa2lp.rsplit('*', 1)
                assert n69wsp9onh.strip()
                assert x69xm5dtzx(n69wsp9onh) == 'folder'
            n69wspa2lp = await self.b69x8ynnvl(n69wspa2lp, _skiplock=True, conn=conn)
            n69wspa2yk = n69wspa2lp[:n69wspa2lp.rfind('^')]
            assert '/' in n69wspa2yk
            n69wspa2uy = n69wspa2lp[len(n69wspa2yk) + 1:n69wspa2lp.rfind('#')]
            n69wsp9omy = n69wspa2lp[n69wspa2lp.rfind('#') + 1:]
            n69wsp9omy = loadstr(n69wsp9omy)
            n69wsp9oz2, n69wsp9ox6, n69wsp9p0s, n69wspa2ws, n69wspa364, n69wspa2qk = self.b69wspa0y2(n69wspa2yk, n69wspa2uy, n69wsp9omy, self.n69wspa2fo, new_parent_class=n69wsp9onh)
            n69wsp9ox6 = pd.concat([n69wsp9ox6, pd.DataFrame([{'def_id': n69wspa2yk, 'code': '', 'globals': [], 'nonlocals': [], 'imports_code': '', 'xpos': 0, 'ypos': 0, 'doc': '', 'is_async': False, 'deco_expr': ''}])], ignore_index=True)
            n69wsp9oz2 = pd.concat([pd.DataFrame([{'hid': n69wspa2uy + '.0', 'uid': gen_base36_id(), 'branch': n69wsp9omy, 'data_providers': [], 'node_type': 'start', 'code': '', 'pres': [], 'nexts': n69wspa364, 'xpos': 0, 'ypos': 0, 'def_id': n69wspa2yk}]), n69wsp9oz2], ignore_index=True)
            if not n69wsp9onh:
                n69wspa2eu = True
                if len(n69wsp9ox6) == 1 and len(n69wsp9p0s) == 0:
                    if n69wsp9ox6['def_id'].tolist()[0] == n69wspa2yk:
                        n69wspa2hd = list(set(n69wsp9oz2['def_id'].tolist()))
                        if len(n69wspa2hd) == 1:
                            if n69wspa2hd[0] == n69wspa2yk:
                                n69wspa2eu = False
                n69wsp9onl = await self.b69x8ynnt6('', n69wspa2lp, n69wspa2nm=n69wspa2nm, external_class_ids=external_class_ids, n69wspa2da=n69wspa2da, cached=cached, n69wspa381=n69wspa381, sustainable=sustainable, pastedata=[n69wsp9oz2, n69wsp9ox6, n69wsp9p0s, n69wspa2ws], do_refresh=n69wspa2eu, _skiplock=True, conn=conn, tolerance=tolerance)
            else:
                n69wsp9onl = await self.b69x8ynntj('', n69wspa2lp + '*' + n69wsp9onh, n69wspa2nm=n69wspa2nm, external_class_ids=external_class_ids, del_funcs=del_funcs, n69wspa381=n69wspa381, cached=cached, pastedata=[n69wsp9oz2, n69wsp9ox6, n69wsp9p0s, n69wspa2ws], _skiplock=True, conn=conn, tolerance=tolerance)
            return n69wsp9onl
        n69wsp9onl = await self._batch_write([b69x8ynnuq], _skiplock=_skiplock, conn=conn)
        return n69wsp9onl[0]

    async def b69x8ynnt6(self, code, n69wspa2lp, n69wspa2nm=[], external_class_ids=[], n69wspa2da=None, cached=False, n69wspa381=None, sustainable=False, pastedata=[], do_refresh=True, n69znp79nl=True, _skiplock=False, conn=None, tolerance=0):
        assert x69xm5dtzx(n69wspa2lp) == 'cond', x69xm5dtzx(n69wspa2lp)
        assert '/' in n69wspa2lp
        assert n69wspa2da
        n69wsp9onl = None
        if n69wspa381 is None:
            n69wspa381 = ''
        n69wspa381 = n69wspa381.rsplit('/')[0]
        code = remove_common_indents(code)

        async def b69x8ynnvt(conn):
            nonlocal n69wsp9onl, n69wspa2da, code, n69wspa2lp
            n69wspa2lp = await self.b69x8ynnvl(n69wspa2lp, _skiplock=True, conn=conn)
            n69wspa2yk = n69wspa2lp[:n69wspa2lp.rfind('^')]
            assert '/' in n69wspa2yk
            n69wspa2uy = n69wspa2lp[len(n69wspa2yk) + 1:n69wspa2lp.rfind('#')]
            n69wsp9omy = n69wspa2lp[n69wspa2lp.rfind('#') + 1:]
            n69wsp9omy = loadstr(n69wsp9omy)
            n69wspa2he = None
            n69wspa2mz = None
            n69wspa34i = []
            n69wspa2wq = []
            n69wsp9ore = True
            n69wsp9oy0 = True
            n69wspa2y2 = True
            n69wspa2ca = True
            n69wspa2f3 = True
            n69wspa2da = copy.deepcopy(n69wspa2da)
            if n69wspa2da is None:
                raise
            elif n69wspa2da['mode'] == 'replace':
                n69wspa320 = n69wspa2da['section']
                if not '.' in n69wspa320[0]:
                    n69wspa320[0] = n69wspa2uy + '.' + n69wspa320[0]
                if not '.' in n69wspa320[1]:
                    n69wspa320[1] = n69wspa2uy + '.' + n69wspa320[1]
                assert n69wspa2uy == n69wspa320[0][:n69wspa320[0].rfind('.')]
                assert n69wspa320[0].count('.') == n69wspa320[1].count('.') == n69wspa2uy.count('.') + 1
                assert n69wspa320[0].startswith(n69wspa2uy + '.') and n69wspa320[1].startswith(n69wspa2uy + '.')
                n69wspa2he, n69wspa2mz = n69wspa320
                n69wspa2he = n69wspa2he.split('-end')[0]
                n69wspa2mz = n69wspa2mz.split('-end')[0]
            elif n69wspa2da['mode'] == 'allbelow':
                pass
            elif n69wspa2da['mode'] == 'insert':
                if not '.' in n69wspa2da['after']:
                    n69wspa2da['after'] = n69wspa2uy + '.' + n69wspa2da['after']
                n69wspa2he = n69wspa2da['after']
                n69wspa2he = n69wspa2he.split('-end')[0]
                assert n69wspa2uy == n69wspa2da['after'][:n69wspa2da['after'].rfind('.')]
            elif n69wspa2da['mode'] == 'refresh':
                assert not code.strip()
                assert n69wspa2uy == '1' and n69wsp9omy == '_', 'eh016'
                n69wsp9ore = n69wspa2da.get('repose', False)
                n69wsp9oy0 = False
                n69wspa2y2 = False
                n69wspa2ca = n69wspa2da.get('recur', True)
                n69wspa2f3 = n69wspa2da.get('revars', True)
            elif n69wspa2da['mode'] == 'tools':
                n69wspa37r = ['FunctionDef', 'AsyncFunctionDef', 'ClassDef']
                n69wspa37r = n69wspa37r + ['Import', 'ImportFrom', 'Nonlocal', 'Global']
                n69wspa2h9 = b69wsp9mq1(code, def_cutoff=True)
                n69wspa359 = [n for n in n69wspa2h9['body'] if n['ntype'] in n69wspa37r]
                if len(n69wspa359) < len(n69wspa2h9['body']):
                    n69wspa2xa = [n for n in n69wspa2h9['body'] if n['ntype'] not in n69wspa37r]
                    n69wspa2wh = {'ntype': 'Module', 'body': n69wspa2xa}
                    _, dirtycode = b65wsp9mrz(n69wspa2wh)
                n69wspa2h9['body'] = n69wspa359
                _, code = b65wsp9mrz(n69wspa2h9)
            n69wspa2z4 = self.b69x8ynnto(n69wspa2lp, n69wspa2nm=n69wspa2nm, external_class_ids=external_class_ids, inject_imports=True, return_visible_brs=True, _skiplock=True, conn=conn)
            n69wspa34t = self.b69x8ynnt3(n69wspa2yk, needvars=False, _skiplock=True, conn=conn)
            n69wspa2ow = f"\n            def_id = '{n69wspa2yk}'\n            "
            n69wspa31h = self.select('funcs', cond_sql=n69wspa2ow, targets=['def_id'], _skiplock=True, conn=conn)
            extpkgs = self.select('misc', cond_sql='true', targets=['external_pkgs'], conn=conn, _skiplock=True)
            n69wspa2z4, n69wspa34t, n69wspa31h, extpkgs = await asyncio.gather(n69wspa2z4, n69wspa34t, n69wspa31h, extpkgs)
            if len(n69wspa31h) == 0:
                raise ValueError(f'[404] Parent func no loger exists: {n69wspa2yk}')
            n69wsp9opz, n69wsp9oxq, n69wsp9p60, n69wspa2wc = n69wspa2z4
            n69wspa2gh, n69wspa354, n69wspa326, n69wspa2fn, _ = n69wspa34t
            extpkgs = extpkgs.loc[0, 'external_pkgs'] if len(extpkgs) > 0 else []
            if not do_refresh and pastedata:
                n69wspa2j1 = []
                n69wspa2eh = []
                n69wspa2ek = {}
                n69wspa358 = pd.DataFrame(columns=list(n69wspa2hw.keys()))
            else:
                n69wspa2j1, n69wspa2eh, n69wspa2ek, n69wspa358 = await self.b69x8ynnw2(n69wspa2yk, n69wspa381, extpkgs, n69wspa2wc, conn, targcode=code)
            n69wsp9opz = n69wsp9opz + n69wspa2j1
            n69wsp9oxq = n69wsp9oxq + n69wspa2eh
            n69wsp9p60 = {**n69wspa2ek, **n69wsp9p60}
            n69wspa2gh['toolcall'] = n69wspa2gh.apply(lambda n69wspa2xo: x69xm5du00(n69wspa2xo['node_type'], n69wspa2xo['toolcall']), axis=1)
            if 1:
                if n69wspa2uy == '1':
                    assert n69wsp9omy == '_'
                else:
                    assert n69wspa2uy in n69wspa2gh['hid'].tolist(), f'eh017{n69wspa2uy}'
                assert n69wspa2yk in n69wspa354['def_id'].tolist(), f'eh018:{n69wspa2yk}'
                if len(n69wspa2gh) == 0:
                    pass
                else:
                    n69wspa2hi = []
                    if n69wspa2da['mode'] in ('replace', 'insert'):
                        n69wspa2hi = []
                        n69wspa2nj = n69wspa2gh[n69wspa2gh['def_id'] == n69wspa2yk][['nexts', 'hid', 'branch']].to_dict(orient='records')
                        n69wsp9owh = {n['hid']: n for n in n69wspa2nj}
                        assert n69wspa2he in n69wsp9owh or n69wspa2he.endswith('.0'), f'The node from which inserts or replaces does not exist: {n69wspa2he}. Existing: {n69wsp9owh.keys()}'
                        if not n69wspa2he.endswith('.0'):
                            assert n69wsp9omy == n69wsp9owh[n69wspa2he]['branch'], (n69wsp9omy, n69wsp9owh[n69wspa2he])
                        if n69wspa2da['mode'] == 'replace':
                            assert n69wspa2mz in n69wsp9owh, (n69wsp9owh.keys(), n69wspa2mz)
                            assert not n69wspa2he.endswith('.0'), f'Cannot replace start node: {n69wspa2he}'
                            assert n69wsp9omy == n69wsp9owh[n69wspa2mz]['branch'], f"The selected section must be on the same branch in replace mode. Start hid: {n69wspa2he}, start branch: {n69wsp9omy}, end hid: {n69wspa2mz}, end branch: {n69wsp9owh[n69wspa2mz]['branch']}"
                            n69wspa2z5 = n69wspa2gh[(n69wspa2gh['hid'] == n69wspa2he) & (n69wspa2gh['def_id'] == n69wspa2yk)]
                            assert len(n69wspa2z5) == 1
                            n69wspa34i = copy.deepcopy(n69wspa2z5['pres'].tolist()[0])
                            if not n69wspa34i or n69wspa34i is np.nan:
                                n69wspa34i = []
                            n69wspa2lq = n69wspa2gh[(n69wspa2gh['hid'] == n69wspa2mz) & (n69wspa2gh['def_id'] == n69wspa2yk)]
                            assert len(n69wspa2lq) == 1
                            n69wspa2wq = copy.deepcopy(n69wspa2lq['nexts'].tolist()[0])
                            if not n69wspa2wq or n69wspa2wq is np.nan:
                                n69wspa2wq = []
                            if n69wspa2he != n69wspa2mz:
                                _, n69wspa2hi = b69wsp9mnd(n69wsp9owh, n69wsp9owh[n69wspa2he], 'nexts', endhid=n69wspa2mz, unend_behavior='raise')
                            else:
                                n69wspa2hi = {n69wspa2mz}
                            n69wspa2m9 = n69wspa2gh.loc[n69wspa2gh['hid'].str.startswith(tuple([r + '.' for r in n69wspa2hi] + [r + '-' for r in n69wspa2hi]))]
                            n69wspa2hi = n69wspa2hi.union(n69wspa2m9['hid'].tolist())
                        else:
                            n69wspa34i = [n69wspa2he]
                            if n69wspa2he.endswith('.0') and n69wspa2he != '1.0':
                                n69wspa34i = []

                                def b69wspa0y5(x):
                                    if isinstance(x, (list, tuple, set)):
                                        if len(x) == 0:
                                            return True
                                        return False
                                    return True
                                n69wspa2lq = n69wspa2gh[(n69wspa2gh['pres'].isna() | n69wspa2gh['pres'].apply(b69wspa0y5)) & (n69wspa2gh['def_id'] == n69wspa2yk) & (n69wspa2gh['branch'] == n69wsp9omy) & n69wspa2gh['hid'].str.startswith(n69wspa2uy + '.') & ~n69wspa2gh['hid'].str[len(n69wspa2uy) + 1:].str.contains('\\.', na=False) & ~n69wspa2gh['hid'].str.contains('-end')]
                                n69wspa2wq = n69wspa2lq['hid'].tolist()
                            else:
                                n69wspa2lq = n69wspa2gh[(n69wspa2gh['hid'] == n69wspa2he) & (n69wspa2gh['def_id'] == n69wspa2yk)]
                                assert len(n69wspa2lq) == 1
                                n69wspa2wq = copy.deepcopy(n69wspa2lq['nexts'].tolist()[0])
                            if not n69wspa2wq or n69wspa2wq is np.nan:
                                n69wspa2wq = []
                    elif n69wspa2da['mode'] == 'allbelow':
                        n69wspa2hi = n69wspa2gh[(n69wspa2gh['def_id'] == n69wspa2yk) & (n69wspa2gh['hid'].str.startswith(n69wspa2uy + '.') & ~n69wspa2gh['hid'].str[len(n69wspa2uy) + 1:].str.contains('\\.', na=False)) & (n69wspa2gh['branch'] == n69wsp9omy) & (n69wspa2gh['hid'].str.startswith(n69wspa2uy + '.') | n69wspa2gh['hid'].str.startswith(n69wspa2uy + '-'))]['hid'].tolist()
                        n69wspa2hi = [rp for rp in n69wspa2hi if rp.count('.') == n69wspa2uy.count('.') + 1]
                        n69wspa34i = []
                        n69wspa2wq = []
                        n69wspa2he = n69wspa2uy + '.0'
                    elif n69wspa2da['mode'] == 'refresh':
                        n69wspa2hi = []
            n69wspa2oe = []
            if not '*' in n69wspa2yk:
                n69wspa30u = None
            else:
                assert '/' in n69wspa2yk.split('*')[-1]
                n69wspa30u = [n69wspa2yk[:n69wspa2yk.rfind('/')], n69wspa2yk.split('*')[-1].split('/')[0]]
            n69wspa2wl = None
            n69wspa35e = None
            n69wspa341 = None
            if not pastedata:
                n69wspa2xu = asyncio.get_running_loop()
                n69wspa32w, n69wspa2ei, n69wspa35a, myvarsdf, n69wspa34y = await n69wspa2xu.run_in_executor(None, lambda: b69wsp9mpo(code, n69wspa2yk, n69wspa2uy=n69wspa2uy, n69wsp9omy=n69wsp9omy, n69wspa2oe=n69wspa2oe, base_funcs=n69wsp9opz, base_classes=n69wsp9oxq, n69wsp9p19=n69wspa30u, base_extra_info={'func_params': n69wsp9p60}, relocate_asks=False, ret_type='df', base_xpos=0, base_ypos=0, n69wsp9osu=NODESPACINGX, n69wsp9ozl=NODESPACINGY, strict_order=True, skip_outershell=False, needvars=False, n69wsp9p3l=n69wspa358))
                if 'toolcall' in n69wspa32w.columns:
                    n69wspa32w['toolcall'] = n69wspa32w.apply(lambda n69wspa2xo: x69xm5dtzz(n69wspa2xo['node_type'], n69wspa2xo['toolcall']), axis=1)
            else:
                assert not code.strip() or code.strip() in ('pass', EMPTY_PASSER)
                n69wspa32w, n69wspa2ei, n69wspa35a, n69wspa34y = pastedata
            if sustainable and n69wspa2da['mode'] in ('allbelow', 'replace'):
                n69wspa35h = n69wspa32w[(n69wspa32w['def_id'] == n69wspa2yk) & (n69wspa32w['node_type'] != 'start')]
                if len(n69wspa35h) == 0:
                    n69wspa32n = gen_base36_id()
                    n69wspa2nx = pd.DataFrame([{'uid': n69wspa32n, 'hid': n69wspa2uy + '.' + n69wspa32n, 'node_type': 'pass', 'branch': n69wsp9omy, 'def_id': n69wspa2yk, 'pres': [], 'nexts': [], 'code': 'pass'}])
                    n69wspa32w = pd.concat([n69wspa32w, n69wspa2nx], ignore_index=True)
                n69wspa363 = n69wspa32w[(n69wspa32w['def_id'] == n69wspa2yk) & (n69wspa32w['node_type'] == 'start')]
                n69wspa32v = [] if len(n69wspa363) == 0 else [n69wspa363['hid'].tolist()[0]]
                n69wspa364 = n69wspa32w[(n69wspa32w['def_id'] == n69wspa2yk) & (n69wspa32w['node_type'] != 'start') & n69wspa32w['hid'].str.startswith(n69wspa2uy + '.') & ~n69wspa32w['hid'].str[len(n69wspa2uy) + 1:].str.contains('\\.', na=False) & (n69wspa32w['branch'] == n69wsp9omy) & (n69wspa32w['pres'].isna() | n69wspa32w['pres'].apply(lambda x: x == [] or x == n69wspa32v))]
                n69wspa2qk = n69wspa32w[(n69wspa32w['def_id'] == n69wspa2yk) & (n69wspa32w['node_type'] != 'start') & n69wspa32w['hid'].str.startswith(n69wspa2uy + '.') & ~n69wspa32w['hid'].str[len(n69wspa2uy) + 1:].str.contains('\\.', na=False) & (n69wspa32w['branch'] == n69wsp9omy) & (n69wspa32w['nexts'].isna() | n69wspa32w['nexts'].apply(lambda x: x == []))]
                if len(n69wspa364) > 0:
                    n69wspa2wl = n69wspa364['uid'].tolist()[0]
                if len(n69wspa2qk) > 0:
                    n69wspa35e = n69wspa2qk['uid'].tolist()[0]
            n69wsp9owv = n69wspa2ei[n69wspa2ei['def_id'] == n69wspa2yk].to_dict(orient='records')[0]
            n69wspa331 = aidle
            n69wsp9oye = None
            if n69wsp9owv.get('globals') or n69wsp9owv.get('nonlocals') or n69wsp9owv.get('imports_code'):
                n69wspa2zq = n69wspa354[n69wspa354['def_id'] == n69wspa2yk][['uid', 'def_id', 'nonlocals', 'globals', 'imports_code']]
                n69wspa2zq = n69wspa2zq.to_dict(orient='records')[0]
                if '*' in n69wspa2yk or '/' in n69wspa2yk:
                    n69wspa2xk = max([ii for ii in [n69wspa2yk.rfind('/'), n69wspa2yk.rfind('*')] if ii != -1])
                    n69wsp9oye = self.b69wspa0yj(n69wspa2yk[:n69wspa2xk]) + ':funcs'
                n69wspa2w0 = copy.deepcopy(n69wspa2zq)
                n69wspa2x5 = n69wsp9owv['imports_code'] + '\n' + n69wspa2w0['imports_code']
                _, _, n69wspa2x5 = b69wsp9mrs(b69wsp9mq1(n69wspa2x5), expand=True)
                n69wspa2x5 = '\n'.join(list(set(n69wspa2x5.split('\n'))))
                n69wspa2w0['imports_code'] = n69wspa2x5
                for ng in n69wsp9owv['globals']:
                    if not ng in n69wspa2w0['globals']:
                        n69wspa2w0['globals'].append(ng)
                for nn in n69wsp9owv['nonlocals']:
                    if not nn in n69wspa2w0['nonlocals']:
                        n69wspa2w0['nonlocals'].append(nn)
                n69wspa2gz = n69wspa354[n69wspa354['def_id'] == n69wspa2yk].index.tolist()
                assert len(n69wspa2gz) == 1, n69wspa354.to_dict(orient='records')
                n69wspa2gz = n69wspa2gz[0]
                n69wspa354.loc[n69wspa2gz, 'imports_code'] = n69wspa2w0['imports_code']
                n69wspa354.at[n69wspa2gz, 'globals'] = n69wspa2w0['globals']
                n69wspa354.at[n69wspa2gz, 'nonlocals'] = n69wspa2w0['nonlocals']
            n69wspa2ei = n69wspa2ei[n69wspa2ei['def_id'] != n69wspa2yk]
            n69wspa38s = n69wspa35a[n69wspa35a['def_id'].str.startswith(n69wspa2lp + '*') & ~n69wspa35a['def_id'].str[len(n69wspa2lp) + 1:].str.contains('\\*') & ~n69wspa35a['def_id'].str[len(n69wspa2lp) + 1:].str.contains('/')]
            n69wspa2po = n69wspa2ei[n69wspa2ei['def_id'].str.startswith(n69wspa2lp + '/') & ~n69wspa2ei['def_id'].str[len(n69wspa2lp) + 1:].str.contains('\\*') & ~n69wspa2ei['def_id'].str[len(n69wspa2lp) + 1:].str.contains('/')]
            n69wspa38k = len(n69wspa38s)
            n69wspa2m2 = len(n69wspa2po)

            def b69wspa0xn(maindf, targdf, n69wspa34i, n69wspa2wq, dotcount, n69wspa2hi):
                nonlocal n69wspa2wl, n69wspa35e
                assert targdf.loc[0, 'node_type'] == 'start', targdf.loc[0, 'node_type']
                assert targdf.loc[0, 'def_id'] == n69wspa2yk, (targdf.loc[0, 'def_id'], n69wspa2yk, targdf.to_dict(orient='records'))
                if len(targdf) <= 1:
                    return maindf
                n69wspa315 = targdf[targdf['def_id'] == n69wspa2yk]
                if len(n69wspa315) <= 1:
                    return pd.concat([maindf, targdf[1:]], ignore_index=True)
                targdf.at[1, 'pres'] = n69wspa34i
                for n69wspa380 in n69wspa34i:
                    assert n69wspa380.count('.') == dotcount
                    assert n69wspa380.startswith(n69wspa2uy + '.')
                    n69wspa2of = maindf[(maindf['hid'] == n69wspa380) & (maindf['def_id'] == n69wspa2yk)].index.tolist()
                    if len(n69wspa2of) == 0:
                        continue
                    else:
                        if len(n69wspa2of) > 1:
                            pass
                        n69wspa2of = n69wspa2of[0]
                        if maindf.loc[n69wspa2of, 'nexts'] is None or maindf.loc[n69wspa2of, 'nexts'] is np.nan:
                            maindf.at[n69wspa2of, 'nexts'] = []
                        else:
                            maindf.at[n69wspa2of, 'nexts'] = [n for n in maindf.loc[n69wspa2of, 'nexts'] if not n in n69wspa2hi]
                        maindf.loc[n69wspa2of, 'nexts'].append(targdf.loc[1, 'hid'])
                        n69wspa2wl = targdf.loc[1, 'hid'].split('.')[-1]
                n69wspa2wk = []
                for i in range(len(targdf)):
                    if targdf.loc[i, 'hid'].count('.') == dotcount and targdf.loc[i, 'def_id'] == n69wspa2yk:
                        if targdf.loc[i, 'nexts'] is np.nan or not targdf.loc[i, 'nexts']:
                            n69wspa2wk.append(targdf.loc[i, 'hid'])
                            targdf.at[i, 'nexts'] = n69wspa2wq
                            n69wspa35e = targdf.loc[i, 'hid'].split('.')[-1]
                for n69wspa380 in n69wspa2wq:
                    assert n69wspa380.count('.') == dotcount, n69wspa2wq
                    assert n69wspa380.startswith(n69wspa2uy + '.')
                    n69wspa2of = maindf[(maindf['hid'] == n69wspa380) & (maindf['def_id'] == n69wspa2yk)].index.tolist()
                    if len(n69wspa2of) == 0:
                        continue
                    else:
                        if len(n69wspa2of) > 1:
                            pass
                        n69wspa2of = n69wspa2of[0]
                        if maindf.loc[n69wspa2of, 'pres'] is None or maindf.loc[n69wspa2of, 'pres'] is np.nan:
                            maindf.at[n69wspa2of, 'pres'] = [n for n in maindf.loc[n69wspa2of, 'pres'] if not n in n69wspa2hi]
                        else:
                            maindf.at[n69wspa2of, 'pres'] = []
                        for tr in n69wspa2wk:
                            maindf.loc[n69wspa2of, 'pres'].append(tr)
                n69wspa2dy = pd.concat([maindf, targdf[1:]], ignore_index=True)
                n69wspa2dy = n69wspa2dy.replace(np.nan, None)
                return n69wspa2dy
            if 1:
                n69wspa2ly = n69wspa2gh
                n69wspa316 = n69wspa35a['def_id'].tolist()
                n69wspa33k = n69wspa2ei['def_id'].tolist()
                n69wspa33k = [rf for rf in n69wspa33k if rf != n69wspa2yk]
                if n69wspa2da:
                    if n69wspa2da['mode'] == 'tools':
                        if n69wspa2da.get('dels'):
                            if n69wspa2da['scope'] == 'classes':
                                n69wspa316 = n69wspa316 + n69wspa2da['dels']
                            elif n69wspa2da['scope'] == 'funcs':
                                n69wspa33k = n69wspa33k + n69wspa2da['dels']
                n69wspa2i6 = n69wspa2ly['hid'].str.startswith(n69wspa2uy)
                n69wspa2ly.loc[n69wspa2i6] = n69wspa2ly.loc[n69wspa2i6].apply(self.b69wspa0xi, axis=1, args=(n69wspa2hi, ['pres', 'nexts', 'data_providers']))
                for rhid in n69wspa2hi:
                    n69wspa2ly = n69wspa2ly[~n69wspa2ly['def_id'].str.startswith(n69wspa2yk + '^' + rhid)]
                    n69wspa2ly = n69wspa2ly[~((n69wspa2ly['def_id'] == n69wspa2yk) & (n69wspa2ly['hid'] == rhid))]
                    n69wspa354 = n69wspa354[~n69wspa354['def_id'].str.startswith(n69wspa2yk + '^' + rhid)]
                    n69wspa2fn = n69wspa2fn[~n69wspa2fn['def_id'].str.startswith(n69wspa2yk + '^' + rhid)]
                    n69wspa326 = n69wspa326[~n69wspa326['def_id'].str.startswith(n69wspa2yk + '^' + rhid)]
                for rf in n69wspa33k:
                    n69wspa2ly = n69wspa2ly[~(n69wspa2ly['def_id'].str.startswith(rf + '^') | (n69wspa2ly['def_id'] == rf))]
                    n69wspa354 = n69wspa354[~((n69wspa354['def_id'] == rf) | n69wspa354['def_id'].str.startswith(rf + '^'))]
                    n69wspa2fn = n69wspa2fn[~((n69wspa2fn['def_id'] == rf) | n69wspa2fn['def_id'].str.startswith(rf + '^'))]
                    n69wspa326 = n69wspa326[~n69wspa326['def_id'].str.startswith(rf + '^')]
                for rc in n69wspa316:
                    n69wspa326 = n69wspa326[~(n69wspa326['def_id'] == rc)]
                if n69wspa2da['mode'] == 'tools':
                    n69wsp9on0 = len(n69wspa32w)
                    n69wspa35d = n69wspa32w[~(n69wspa32w['def_id'] == n69wspa2yk)]
                    n69wsp9oz4 = len(n69wspa35d)
                    if n69wsp9oz4 - n69wsp9on0 < -1:
                        pass
                    n69wspa2rm = pd.concat([n69wspa2ly, n69wspa35d], ignore_index=True)
                    if n69wspa2da['scope'] == 'funcs':
                        n69wspa33u = n69wspa2ei['def_id'].tolist()
                        n69wspa2w5 = '/'
                    elif n69wspa2da['scope'] == 'classes':
                        n69wspa33u = n69wspa35a['def_id'].tolist()
                        n69wspa2w5 = '*'
                    for tid in n69wspa33u:
                        if tid.startswith(n69wspa2lp + n69wspa2w5):
                            if not any([fix in tid[len(n69wspa2lp) + 1:] for fix in ('*', '/', '.', '^', '#')]):
                                n69wspa341 = tid
                                break
                elif n69wspa2da['mode'] not in 'refresh':
                    n69wspa2rm = b69wspa0xn(n69wspa2ly, n69wspa32w, n69wspa34i, n69wspa2wq, n69wspa2he.count('.'), n69wspa2hi)
                else:
                    n69wspa2rm = n69wspa2ly
                n69wspa2tl = pd.concat([n69wspa354, n69wspa2ei], ignore_index=True)
                n65d20cda3 = pd.concat([n69wspa326, n69wspa35a], ignore_index=True)
                n69wspa2m8 = pd.concat([n69wspa2fn, n69wspa34y], ignore_index=True)
                n69wspa33a = set(n69wspa32w['def_id'].tolist()) if not pastedata else set()
                n69wspa2lx = {}
                for n69wspa2hh in n69wspa33a:
                    n69wspa2lx[n69wspa2hh] = n69wspa32w[n69wspa32w['def_id'] == n69wspa2hh]['hid'].tolist()
                n69wspa31j = n69wspa2rm[['uid', 'code']].copy()
                if not do_refresh:
                    assert pastedata, f'eh019'
                    n69wspa2zr = pd.DataFrame(columns=list(n69wspa2hw.keys()))
                    n69wspa2on = n69wspa2rm[n69wspa2rm['def_id'] == n69wspa2yk]
                    n69wspa365 = n69wspa2rm[n69wspa2rm['def_id'] != n69wspa2yk]
                    n69wspa2on = n69wspa2on.reset_index(drop=True)
                    n69wspa2on = b69wsp9mpz(n69wspa2on, n69wspa2yk, n69wsp9osu=NODESPACINGX, n69wsp9ozl=NODESPACINGY, tolerance=tolerance)
                    n69wspa2rm = pd.concat([n69wspa2on, n69wspa365], ignore_index=True)
                else:
                    n69wspa2xu = asyncio.get_running_loop()
                    n69wspa2rm, n69wspa2tl, n65d20cda3, n69wspa2zr, n69wspa2m8 = await n69wspa2xu.run_in_executor(None, lambda: b69wsp9mnk(n69wspa2rm, n69wspa2tl, n65d20cda3, n69wspa2m8, n69wspa2yk, ret_type='df', n69wsp9osu=NODESPACINGX, n69wsp9ozl=NODESPACINGY, base_funcs=n69wsp9opz, base_classes=n69wsp9oxq, n69wsp9p60=n69wsp9p60, strict_order=True, n69wsp9ore=n69wsp9ore, n69wsp9oy0=n69wsp9oy0, n69wspa2y2=n69wspa2y2, n69wspa2lx=n69wspa2lx, recur_into_tools=n69wspa2ca, n69wspa2f3=n69wspa2f3, n69wsp9p3l=n69wspa358, tolerance=tolerance, n69znp79nl=n69znp79nl))
                n69wspa2y7 = n69wspa32w['uid'].tolist()
                n69wspa31j = n69wspa31j[~n69wspa31j['uid'].isin(n69wspa2y7)]
                n69wspa34e = n69wspa31j.set_index('uid')['code']
                n69wspa2rm['code'] = n69wspa2rm['uid'].map(n69wspa34e).fillna(n69wspa2rm['code'])
                n69wspa2rm = n69wspa2rm.replace(np.nan, None)
                n69wspa2tl = n69wspa2tl.replace(np.nan, None)
                n65d20cda3 = n65d20cda3.replace(np.nan, None)
                n69wspa2m8 = n69wspa2m8.replace(np.nan, None)
                n69wspa2zr = n69wspa2zr.replace(np.nan, None)
                if 'toolcall' in n69wspa2rm.columns:
                    n69wspa2rm['toolcall'] = n69wspa2rm.apply(lambda n69wspa2xo: x69xm5dtzz(n69wspa2xo['node_type'], n69wspa2xo['toolcall']), axis=1)
                if 'type' in n69wspa2zr.columns:
                    n69wspa2zr['type'] = n69wspa2zr['type'].apply(lambda x: x.replace(DOT_REPL, '.') if isinstance(x, str) else x)
                if 'name' in n69wspa2zr.columns:
                    n69wspa2zr['name'] = n69wspa2zr['name'].apply(lambda x: x.replace(DOT_REPL, '.') if isinstance(x, str) else x)
                if n69wspa2da['mode'] == 'tools':
                    n69wspa31g = {'newtoolid': n69wspa341}
                else:
                    n69wspa31g = {'newleft': n69wspa2wl, 'newright': n69wspa35e}

                async def b69x8ynnu4(conn):
                    if n69wspa2da['mode'] != 'refresh':
                        if not do_refresh:
                            n69wsp9ot1 = None
                        await self.b69x8ynnvm(n69wspa2rm, n69wspa2tl, n69wsp9oni=n65d20cda3, n69wsp9ot1=n69wspa2zr, n69wsp9oxa=n69wspa2m8, _skiplock=True, conn=conn)
                    else:
                        n69wsp9oye = set(n69wspa2rm['def_id'].tolist() + n69wspa2tl['def_id'].tolist()) if n69wspa2f3 else set()
                        n69wspa2xq = self.delete('vars', [{'def_id': did} for did in n69wsp9oye], _skiplock=True, conn=conn) if n69wspa2f3 else aidle()
                        n69wspa38i = self.delete('params', [{'def_id': did} for did in n69wsp9oye], _skiplock=True, conn=conn) if n69wspa2f3 else aidle()
                        await asyncio.gather(n69wspa2xq, n69wspa38i)
                        n69wspa2gd = self.upsert('vars', n69wspa2zr, _skiplock=True, conn=conn) if n69wspa2f3 else aidle()
                        n69wspa327 = self.upsert('params', n69wspa2m8, _skiplock=True, conn=conn) if n69wspa2f3 else aidle()
                        n69wspa2f4 = aidle()
                        if n69wsp9ore:
                            n69wspa2f4 = self.upsert('nodes', n69wspa2rm[['uid', 'xpos', 'ypos']], _skiplock=True, conn=conn)
                        await asyncio.gather(n69wspa2gd, n69wspa327, n69wspa2f4)
                await self._batch_write([b69x8ynnu4], _skiplock=True, conn=conn)
                n69wspa2rm = n69wspa2rm.replace(np.nan, None)
                n69wsp9onl = n69wspa2rm
            if cached:
                if n69wspa2da['mode'] == 'tools':
                    n69wspa2ok = n69wspa2da['scope']
                    n69wspa2do = n69wspa2lp
                else:
                    n69wspa2ok = 'dag'
                    n69wspa2do = n69wspa2yk
                n69wspa36f = await self.b69x8ynnvg(n69wspa2do, n69wspa2ok, debugtag=f'(dc01)', existing_data={'nodes': n69wspa2rm, 'funcs': n69wspa2tl, 'params': n69wspa2m8, 'classes': n65d20cda3}, conn=conn, _skiplock=True)
                n69wsp9onl = (n69wsp9onl, n69wspa36f['data']['app'], {**n69wspa31g, 'alsoreload': n69wsp9oye, 'numclasss': n69wspa38k, 'numfuncs': n69wspa2m2})
        await self._batch_write([b69x8ynnvt], _skiplock=_skiplock, conn=conn)
        return n69wsp9onl

    async def b69x8ynntj(self, code, base_class_id, n69wspa2nm=[], external_class_ids=[], del_funcs=[], n69wspa381=None, cached=False, pastedata={}, n69znp79nl=True, _skiplock=False, conn=None, tolerance=1):
        assert x69xm5dtzx(base_class_id) == 'class'
        if n69wspa381 is None:
            n69wspa381 = ''
        n69wspa381 = n69wspa381.rsplit('/')[0]
        n69wsp9onl = None

        async def b69x8ynnvt(conn):
            nonlocal n69wsp9onl, code, base_class_id
            n69wspa2lm = base_class_id.rfind('*')
            n69wspa2lp = base_class_id[:n69wspa2lm]
            assert x69xm5dtzx(n69wspa2lp) == 'cond'
            n69wspa2yk = n69wspa2lp[:n69wspa2lp.rfind('^')]
            assert '/' in n69wspa2yk
            n69wspa2uy = n69wspa2lp[len(n69wspa2yk) + 1:n69wspa2lp.rfind('#')]
            n69wsp9omy = n69wspa2lp[n69wspa2lp.rfind('#') + 1:]
            n69wsp9omy = loadstr(n69wsp9omy)
            n69wspa313 = base_class_id.split('*')[-1]
            n69wspa2z4 = self.b69x8ynnto(n69wspa2lp, n69wspa2nm=n69wspa2nm, external_class_ids=external_class_ids, inject_imports=True, return_visible_brs=True, _skiplock=True, conn=conn)
            n69wspa2ow = f"\n            def_id = '{base_class_id}'\n            "
            n69wspa31h = self.select('classes', cond_sql=n69wspa2ow, targets=['def_id'], _skiplock=True, conn=conn)
            n69wspa34t = self.b69x8ynnt3(n69wspa2yk, needvars=False, _skiplock=True, conn=conn)
            extpkgs = self.select('misc', cond_sql='true', targets=['external_pkgs'], conn=conn, _skiplock=True)
            n69wspa2z4, n69wspa34t, n69wspa31h, extpkgs = await asyncio.gather(n69wspa2z4, n69wspa34t, n69wspa31h, extpkgs)
            if len(n69wspa31h) == 0:
                raise ValueError(f'[404] Parent class no loger exists: {base_class_id}')
            n69wsp9opz, n69wsp9oxq, n69wsp9p60, n69wspa2wc = n69wspa2z4
            n69wspa2gh, n69wspa354, n69wspa326, n69wspa2fn, _ = n69wspa34t
            extpkgs = extpkgs.loc[0, 'external_pkgs'] if len(extpkgs) > 0 else []
            n69wspa2j1, n69wspa2eh, n69wspa2ek, n69wspa358 = await self.b69x8ynnw2(n69wspa2yk, n69wspa381, extpkgs, n69wspa2wc, conn, targcode=code)
            n69wsp9opz = n69wsp9opz + n69wspa2j1
            n69wsp9oxq = n69wsp9oxq + n69wspa2eh
            n69wsp9p60 = {**n69wspa2ek, **n69wsp9p60}
            n69wspa2gh['toolcall'] = n69wspa2gh.apply(lambda n69wspa2xo: x69xm5du00(n69wspa2xo['node_type'], n69wspa2xo['toolcall']), axis=1)
            n69wspa37r = ['FunctionDef', 'AsyncFunctionDef']
            n69wspa2zl = ['Import', 'ImportFrom', 'Nonlocal', 'Global']
            n69wspa2h9 = b69wsp9mq1(code, def_cutoff=True)
            n69wspa359 = [n for n in n69wspa2h9['body'] if n['ntype'] in n69wspa37r]
            n69wspa35z = [n for n in n69wspa2h9['body'] if n['ntype'] in n69wspa2zl]
            if len(n69wspa359) < len(n69wspa2h9['body']):
                n69wspa2xa = [n for n in n69wspa2h9['body'] if n['ntype'] not in n69wspa37r]
                n69wspa2wh = {'ntype': 'Module', 'body': n69wspa2xa}
                _, dirtycode = b65wsp9mrz(n69wspa2wh)
            n69wspa2yy = {'ntype': 'Module', 'body': [n69wspa35z, {'ntype': 'ClassDef', 'name': n69wspa313, 'bases': [], 'keywords': [], 'body': n69wspa359, 'decorator_list': []}]}
            _, code = b65wsp9mrz(n69wspa2yy)
            if not pastedata:
                n69wspa32w, n69wspa2ei, n69wspa35a, myvarsdf, n69wspa34y = b69wsp9mpo(code, n69wspa2yk, n69wspa2uy=n69wspa2uy, n69wsp9omy=n69wsp9omy, base_funcs=n69wsp9opz, base_classes=n69wsp9oxq, n69wsp9p19=None, base_extra_info={'func_params': n69wsp9p60}, relocate_asks=False, ret_type='df', base_xpos=0, base_ypos=0, n69wsp9osu=NODESPACINGX, n69wsp9ozl=NODESPACINGY, strict_order=True, skip_outershell=False, needvars=False, n69wsp9p3l=n69wspa358)
                if 'toolcall' in n69wspa32w.columns:
                    n69wspa32w['toolcall'] = n69wspa32w.apply(lambda n69wspa2xo: x69xm5dtzz(n69wspa2xo['node_type'], n69wspa2xo['toolcall']), axis=1)
            else:
                assert not code.strip() or code.strip() in ('pass', EMPTY_PASSER), code
                n69wspa32w, n69wspa2ei, n69wspa35a, n69wspa34y = pastedata
            n69wsp9owv = n69wspa2ei[n69wspa2ei['def_id'] == n69wspa2yk].to_dict(orient='records')[0]
            n69wsp9oye = None
            if n69wsp9owv.get('globals') or n69wsp9owv.get('nonlocals') or n69wsp9owv.get('imports_code'):
                n69wspa2zq = n69wspa354[n69wspa354['def_id'] == n69wspa2yk][['uid', 'def_id', 'nonlocals', 'globals', 'imports_code']]
                n69wspa2zq = n69wspa2zq.to_dict(orient='records')[0]
                if '*' in n69wspa2yk or '/' in n69wspa2yk:
                    n69wspa2xk = max([ii for ii in [n69wspa2yk.rfind('/'), n69wspa2yk.rfind('*')] if ii != -1])
                    n69wsp9oye = self.b69wspa0yj(n69wspa2yk[:n69wspa2xk]) + ':funcs'
                n69wspa2w0 = copy.deepcopy(n69wspa2zq)
                n69wspa2ve = n69wsp9owv['imports_code'].split('\n')
                for nimp in n69wspa2ve:
                    if not nimp + '\n' in n69wspa2w0['imports_code']:
                        n69wspa2w0['imports_code'] = n69wspa2w0['imports_code'] + (nimp + '\n' if n69wspa2w0['imports_code'].endswith('\n') else '\n' + nimp + '\n')
                for ng in n69wsp9owv['globals']:
                    if not ng in n69wspa2w0['globals']:
                        n69wspa2w0['globals'].append(ng)
                for nn in n69wsp9owv['nonlocals']:
                    if not nn in n69wspa2w0['nonlocals']:
                        n69wspa2w0['nonlocals'].append(nn)
                n69wspa2gz = n69wspa354[n69wspa354['def_id'] == n69wspa2yk].index.tolist()
                assert len(n69wspa2gz) == 1, n69wspa354.to_dict(orient='records')
                n69wspa2gz = n69wspa2gz[0]
                n69wspa354.loc[n69wspa2gz, 'imports_code'] = n69wspa2w0['imports_code']
                n69wspa354.at[n69wspa2gz, 'globals'] = n69wspa2w0['globals']
                n69wspa354.at[n69wspa2gz, 'nonlocals'] = n69wspa2w0['nonlocals']
            n69wspa2ei = n69wspa2ei[n69wspa2ei['def_id'] != n69wspa2yk]
            n69wspa35a = n69wspa35a[n69wspa35a['def_id'] != base_class_id]
            n69wspa316 = n69wspa35a['def_id'].tolist()
            n69wspa33k = n69wspa2ei['def_id'].tolist()
            if del_funcs:
                n69wspa33k = n69wspa33k + del_funcs
            for rf in n69wspa33k:
                n69wspa2gh = n69wspa2gh[~(n69wspa2gh['def_id'].str.startswith(rf + '^') | (n69wspa2gh['def_id'] == rf))]
                n69wspa354 = n69wspa354[~((n69wspa354['def_id'] == rf) | n69wspa354['def_id'].str.startswith(rf + '^'))]
                n69wspa2fn = n69wspa2fn[~((n69wspa2fn['def_id'] == rf) | n69wspa2fn['def_id'].str.startswith(rf + '^'))]
                n69wspa326 = n69wspa326[~n69wspa326['def_id'].str.startswith(rf + '^')]
            for rc in n69wspa316:
                n69wspa326 = n69wspa326[~(n69wspa326['def_id'] == rc)]
            n69wsp9on0 = len(n69wspa32w)
            n69wspa35d = n69wspa32w[~(n69wspa32w['def_id'] == n69wspa2yk)]
            n69wsp9oz4 = len(n69wspa35d)
            if n69wsp9oz4 - n69wsp9on0 < -1:
                pass
            n69wspa2rm = pd.concat([n69wspa2gh, n69wspa35d], ignore_index=True)
            n69wspa2tl = pd.concat([n69wspa354, n69wspa2ei], ignore_index=True)
            n65d20cda3 = pd.concat([n69wspa326, n69wspa35a], ignore_index=True)
            n69wspa2m8 = pd.concat([n69wspa2fn, n69wspa34y], ignore_index=True)
            n69wspa341 = None
            n69wspa33u = n69wspa2ei['def_id'].tolist()
            n69wspa2w5 = '/'
            for tid in n69wspa33u:
                if tid.startswith(n69wspa2lp + n69wspa2w5):
                    if not any([fix in tid[len(n69wspa2lp) + 1:] for fix in ('*', '/', '.', '^', '#')]):
                        n69wspa341 = tid
                        break
            n69wspa33a = set(n69wspa32w['def_id'].tolist()) if not pastedata else set()
            n69wspa2lx = {}
            for n69wspa2hh in n69wspa33a:
                n69wspa2lx[n69wspa2hh] = n69wspa32w[n69wspa32w['def_id'] == n69wspa2hh]['hid'].tolist()
            n69wspa31j = n69wspa2rm[['uid', 'code']].copy()
            n69wspa2rm, n69wspa2tl, n65d20cda3, n69wspa2zr, n69wspa2m8 = b69wsp9mnk(n69wspa2rm, n69wspa2tl, n65d20cda3, n69wspa2m8, n69wspa2yk, ret_type='df', n69wsp9osu=NODESPACINGX, n69wsp9ozl=NODESPACINGY, base_funcs=n69wsp9opz, base_classes=n69wsp9oxq, n69wsp9p60=n69wsp9p60, strict_order=True, n69wsp9ore=True, n69wsp9oy0=True, n69wspa2y2=True, n69wspa2lx=n69wspa2lx, recur_into_tools=True, n69wspa2f3=True, n69wsp9p3l=n69wspa358, tolerance=tolerance, n69znp79nl=n69znp79nl)
            n69wspa2rm = n69wspa2rm.replace(np.nan, None)
            n69wspa2tl = n69wspa2tl.replace(np.nan, None)
            n65d20cda3 = n65d20cda3.replace(np.nan, None)
            n69wspa2m8 = n69wspa2m8.replace(np.nan, None)
            n69wspa2zr = n69wspa2zr.replace(np.nan, None)
            n69wspa2y7 = n69wspa32w['uid'].tolist()
            n69wspa31j = n69wspa31j[~n69wspa31j['uid'].isin(n69wspa2y7)]
            n69wspa34e = n69wspa31j.set_index('uid')['code']
            n69wspa2rm['code'] = n69wspa2rm['uid'].map(n69wspa34e).fillna(n69wspa2rm['code'])
            if 'toolcall' in n69wspa2rm.columns:
                n69wspa2rm['toolcall'] = n69wspa2rm.apply(lambda n69wspa2xo: x69xm5dtzz(n69wspa2xo['node_type'], n69wspa2xo['toolcall']), axis=1)
            if 'type' in n69wspa2zr.columns:
                n69wspa2zr['type'] = n69wspa2zr['type'].apply(lambda x: x.replace(DOT_REPL, '.') if isinstance(x, str) else x)
            if 'name' in n69wspa2zr.columns:
                n69wspa2zr['name'] = n69wspa2zr['name'].apply(lambda x: x.replace(DOT_REPL, '.') if isinstance(x, str) else x)
            await self.b69x8ynnvm(n69wspa2rm, n69wspa2tl, n69wsp9oni=n65d20cda3, n69wsp9ot1=n69wspa2zr, n69wsp9oxa=n69wspa2m8, _skiplock=True, conn=conn)
            n69wsp9onl = n69wspa2rm
            if cached:
                n69wspa36f = await self.b69x8ynnvg(base_class_id, 'funcs', debugtag='(dc02)', existing_data={'nodes': n69wspa2rm, 'funcs': n69wspa2tl, 'params': n69wspa2m8, 'classes': n65d20cda3}, conn=conn, _skiplock=True)
                n69wsp9onl = (n69wsp9onl, n69wspa36f['data']['app'], {'alsoreload': n69wsp9oye, 'newtoolid': n69wspa341})
        await self._batch_write([b69x8ynnvt], _skiplock=_skiplock, conn=conn)
        return n69wsp9onl

    def b69wspa0yb(self, n69wspa38m):
        n69wspa2h0 = n69wspa38m.replace('>', '\\>').replace('/', '\\/')
        n69wspa357 = f'@module_id:{{{n69wspa2h0}}} @undoed:{{T}}'
        n69wspa2vo = self.n69wspa36f.execute_command('FT.SEARCH', 'idx:undo', n69wspa357, 'VERBATIM', 'LIMIT', '0', f'{MAX_REDOS}', 'RETURN', '0')
        n69wspa2ea = self.n69wspa36f.execute_command('FT.SEARCH', 'idx:undo', f'@module_id:{{{n69wspa2h0}}}', 'SORTBY', 'save_id', 'DESC', 'VERBATIM', 'LIMIT', f'{MAX_REDOS}', '10000', 'RETURN', '0')
        n69wspa34l = n69wspa2vo[1:]
        n69wspa2qo = n69wspa2ea[1:]
        n69wspa2ou = list(set(n69wspa34l + n69wspa2qo))
        n69wspa2tp = self.n69wspa36f.pipeline()
        [n69wspa2tp.delete(u) for u in n69wspa2ou]
        n69wspa2tp.execute()

    @n69wspa2rv.debounce([1, 2, 'n69wsp9ore', 'n69wspa2f3', 'timestamp'], [dict], compare_func=b69wspa0xp, skipper=lambda *args, **kwargs: kwargs.get('level') != 'adlvt', intercepted_kws=['timestamp'])
    async def b69x8ynntg(self, n69wsp9oya, n69wspa34y, n69wspa2wq=None, n69wsp9ore=False, n69wsp9osu=NODESPACINGX, _skiplock=False, conn=None, rectify_caseids=False, level='adlvt', n69wspa2f3=True, revars_failure_behavior='warning', cached=False, n69wspa2sf=False, n69wspa381=None, section={'lefts': [], 'rights': []}):
        if level == 'adlvv' or level == 'adlvd' or level == 'adlvl' or (level == 'adlvx') or (level == 'adlvf'):
            n69wspa2f3 = False
        if level == 'adlvv' or level == 'adlvl':
            assert n69wsp9ore
        if level == 'adlvx':
            assert not n69wsp9ore
        if not n69wsp9oya:
            n69wsp9oyq = gen_base36_id()
            assert n69wspa2wq
            n69wsp9oya = [{'type': 'plain', 'position': {'x': 0, 'y': 0}, 'id': n69wsp9oyq, 'data': {'uid': n69wsp9oyq, 'hid': '1.0', 'branch': '_', 'data_providers': [], 'node_type': 'start', 'pres': [], 'nexts': [], 'xpos': 0, 'ypos': 0, 'def_id': n69wspa2wq}}]
        if n69wspa2sf and (not cached):
            n69wspa2sf = False
        n69wsp9oy3 = {n['id']: {**n['data'], 'type': n['type'], 'xpos': n['position']['x'], 'ypos': n['position']['y']} for n in n69wsp9oya}
        n69wspa2un = []
        for n69wspa2ww in n69wspa34y:
            if not '#' in n69wspa2ww['sourceHandle']:
                n69wspa2un.append(n69wspa2ww)
                continue
            assert n69wspa2ww['sourceHandle'][0] == '#' and '@' in n69wspa2ww['sourceHandle']
            n69wspa38a = n69wspa2ww['source']
            if not n69wspa38a in n69wsp9oy3:
                continue
            n69wsp9p3g = n69wspa2ww['sourceHandle'][1:n69wspa2ww['sourceHandle'].find('@')]
            assert not n69wsp9p3g in ('true', 'false'), 'eh020'
            if n69wsp9oy3[n69wspa38a]['node_type'] in ('match', 'excepts'):
                n69wspa37b = [str(k) for k in list(n69wsp9oy3[n69wspa38a]['cases'].keys())]
            elif n69wsp9oy3[n69wspa38a]['node_type'] in 'if':
                n69wspa37b = ['True', 'False']
            else:
                n69wspa37b = ['_']
            if not n69wsp9p3g in n69wspa37b:
                continue
            n69wspa2un.append(n69wspa2ww)
        n69wspa2km = pd.DataFrame(n69wspa2un) if len(n69wspa2un) > 0 else pd.DataFrame(columns=['id', 'source', 'sourceHandle', 'target', 'targetHandle'])

        def b69wspa0xs(n69wspa2ww):
            assert n69wspa2ww.count('#') == 1 and n69wspa2ww.count('@') == 1
            n69wspa2s3 = n69wspa2ww[n69wspa2ww.find('#') + 1:n69wspa2ww.find('@')]
            assert n69wspa2s3.isdigit()
            return int(n69wspa2s3)
        n69wspa38y = {}
        if rectify_caseids:
            for k, n69wspa2mh in n69wsp9oy3.items():
                if n69wspa2mh['node_type'] in ('excepts', 'match'):
                    n69wsp9ot8 = {}
                    n69wspa2oj = {}
                    n69wspa2up = {}
                    n69wsp9osi = 0
                    for oldc, caseinfo in n69wspa2mh['cases'].items():
                        assert oldc.isdigit()
                        n69wsp9ot8[n69wsp9osi] = caseinfo
                        n69wspa2oj[oldc] = n69wsp9osi
                        n69wspa2up[n69wsp9osi] = oldc
                        n69wsp9osi = n69wsp9osi + 1
                    n69wspa2mh['cases'] = n69wsp9ot8
                    n69wspa37s = {}
                    n69wspa2rf = {}
                    n69wspa2x6 = {}
                    for newc in n69wsp9ot8.keys():
                        n69wspa2rp = n69wspa2mh['handle_outs'][n69wspa2up[newc]]
                        n69wspa2vz = f'#{rectify_cond(newc)}@source'
                        n69wspa2x6[n69wspa2rp] = n69wspa2vz
                        n69wspa37s[newc] = n69wspa2vz
                        n69wspa2rf[newc] = {'funcs': f"{n69wspa2mh['def_id']}^{n69wspa2mh['uid']}#{newc}:funcs", 'classes': f"{n69wspa2mh['def_id']}^{n69wspa2mh['uid']}#{newc}:classes"}
                    n69wspa2mh['handle_outs'] = n69wspa37s
                    n69wspa2mh['subWorkflowIds'] = n69wspa2rf
                    n69wspa2km['source'] = n69wspa2km['source'].apply(lambda x: n69wspa2x6.get(x, x))
                    n69wspa38y[n69wspa2mh['uid']] = n69wspa2oj
                else:
                    n69wspa2mh['cases'] = {rectify_cond(k): v for k, v in n69wspa2mh['cases'].items()}
        n69wspa2pt = set(n69wsp9oy3.keys())
        if 1:
            pass
        n69wsp9oo8 = []
        n69wsp9p0q = table_unique_get(list(n69wsp9oy3.values()), {'node_type': 'start'})
        n69wsp9p0q['hid'] = '1.0'
        n69wsp9p0q['branch'] = '_'
        n69wsp9p0q['pres'] = []
        n69wsp9p0q['nexts'] = []
        n69wspa2yk = n69wsp9p0q['def_id']
        n69wspa2ns = n69wspa2km[n69wspa2km['source'] == n69wsp9p0q['uid']]
        n69wspa2x8 = n69wspa2ns['target'].tolist()
        n69wspa2ll = [n69wsp9oy3[nid] for nid in n69wspa2x8]
        n69wspa346 = set()
        n69wspa32o = n69wsp9osu
        n69wspa2t2 = level == 'adlvx'
        n69wspa2y9 = set()

        def b69wspa0y9(n69wspa2mh, n69wspa2w9=0, n69wsp9oz5='-1', insection=False):
            assert n69wspa2mh['def_id'] == n69wspa2yk, (n69wspa2mh['def_id'], n69wspa2yk)
            nonlocal n69wsp9oo8, n69wsp9oy3, n69wspa346
            n65d20cda3 = n69wspa2mh['uid']
            assert '.' in n69wspa2mh['hid']
            if n65d20cda3 in n69wspa2pt and n69wspa2t2:
                if n69wspa2w9 != 0:
                    n69wspa346.add(n65d20cda3)
                    n69wspa2mh['xpos'] = n69wspa2mh['xpos'] + n69wsp9osu * n69wspa2w9
            if not n65d20cda3 in n69wspa2pt:
                return n69wspa2w9
            if insection:
                n69wspa2y9.add(n65d20cda3)
            n69wsp9oo8.append(n69wspa2mh)
            n69wspa2pt.remove(n65d20cda3)
            n69wsp9oup = n69wspa2mh
            n69wspa2n8 = []
            n69wspa2fg = n69wsp9oup['xpos']
            if n69wsp9oup['type'] != 'switch':
                n69wspa2n8 = n69wspa2km[n69wspa2km['source'] == n69wsp9oup['uid']].to_dict(orient='records')
            else:
                n69wspa2ld = n69wspa2km[n69wspa2km['source'] == n69wsp9oup['uid']].to_dict(orient='records')
                for n69wspa2ww in n69wspa2ld:
                    assert n69wspa2ww['sourceHandle'].count('#') == 1 and n69wspa2ww['sourceHandle'].count('@') == 1
                    n69wspa2s3 = n69wspa2ww['sourceHandle'][n69wspa2ww['sourceHandle'].find('#') + 1:n69wspa2ww['sourceHandle'].find('@')]
                    n69wspa2s3 = rectify_cond(n69wspa2s3)
                    n69wspa2yg = n69wspa2ww['target']
                    if '-end' in n69wspa2yg and n69wspa2yg in n69wspa2pt:
                        assert n69wspa2yg[:n69wspa2yg.find('-end')] == n65d20cda3, (n69wspa2yg, n65d20cda3)
                        continue
                    n69wspa2n7 = copy.deepcopy(n69wsp9oy3[n69wspa2yg])
                    assert n69wspa2yg == n69wspa2n7['uid']
                    n69wspa2n7['pres'] = []
                    n69wspa2n7['hid'] = n69wsp9oup['hid'] + '.' + n69wspa2yg
                    n69wspa2n7['branch'] = n69wspa2s3
                    _ = b69wspa0y9(n69wspa2n7, n69wspa2w9=n69wspa2w9, n69wsp9oz5=n69wsp9oup['uid'], insection=insection or n69wspa2n7['uid'] in section['lefts'])
                n69wspa318 = n69wspa2w9
                n69wspa31r = False
                n69wspa2ff = table_lambda_get(n69wsp9oo8, lambda x: x['hid'].startswith(n69wsp9oup['hid'] + '.'))
                n69wspa36k = [c['xpos'] for c in n69wspa2ff]
                n69wspa2m4 = max(n69wspa36k) if n69wspa36k else n69wsp9oup['xpos']
                n69wspa346.add(n69wsp9oup['uid'])
                n69wspa31r = n69wspa2t2
                n69wspa32g = n69wsp9oup['uid'] + '-end' + n69wsp9oup['node_type']
                if insection:
                    n69wspa2y9.add(n69wspa32g)
                if n69wspa32g in n69wspa2pt:
                    n69wspa335 = n69wsp9oy3[n69wspa32g]
                    n69wspa335['source'] = n69wsp9oup['hid']
                    n69wspa335['branch'] = n69wsp9oup['branch']
                    n69wspa335['hid'] = n69wsp9oup['hid'] + '-end' + n69wsp9oup['node_type']
                    n69wsp9oo8.append(n69wspa335)
                    n69wspa2pt.remove(n69wspa32g)
                    n69wspa2i0 = n69wspa32g
                    if n69wspa31r:
                        n69wspa318 = (n69wspa2m4 - n69wspa335['xpos']) / n69wsp9osu + 1
                        n69wspa346.add(n69wspa335['uid'])
                        n69wspa335['xpos'] = n69wspa335['xpos'] + n69wsp9osu * n69wspa318
                elif not n69wspa32g in n69wsp9oy3:
                    n69wspa2dl = [n69wspa5l8.get('xpos', 0) for n69wspa5l8 in n69wsp9oy3.values() if n69wspa5l8.get('hid', '').startswith(n69wsp9oup['hid'] + '.')] + [n69wsp9oup['xpos']]
                    n69wspa2mr = max(n69wspa2dl) + n69wsp9osu
                    n69wspa335 = self.b69wspa0xu(n69wspa2yk, n69wsp9oup['uid'], n69wsp9oup['node_type'], n69wspa2mr, n69wsp9oup['ypos'])
                    n69wspa335 = {**n69wspa335['data'], 'type': n69wspa335['type'], 'xpos': n69wspa335['position']['x'], 'ypos': n69wspa335['position']['y'], 'hid': n69wsp9oup['hid'] + '-end' + n69wsp9oup['node_type'], 'branch': n69wsp9oup['branch']}
                    n69wsp9oo8.append(n69wspa335)
                    n69wsp9oy3[n69wspa32g] = n69wspa335
                    n69wspa2i0 = n69wspa32g
                    assert n69wspa335['uid'] == n69wspa32g
                    if n69wspa31r:
                        n69wspa346.add(n69wspa335['uid'])
                        n69wspa318 = (n69wspa2m4 - n69wspa2mr) / n69wsp9osu + 1
                        n69wspa335['xpos'] = n69wspa335['xpos'] + n69wsp9osu * n69wspa318
                else:
                    n69wspa2i0 = n69wspa32g
                    if n69wspa31r:
                        n69wspa335 = table_unique_get(n69wsp9oo8, {'uid': n69wspa32g}, return_pointer=True)
                        n69wspa346.add(n69wspa2i0)
                        n69wspa318 = (n69wspa2m4 - n69wspa335['xpos']) / n69wsp9osu + 1
                        n69wspa335['xpos'] = n69wspa335['xpos'] + n69wsp9osu * n69wspa318
                n69wspa2n8 = n69wspa2km[n69wspa2km['source'] == n69wspa2i0].to_dict(orient='records')
                n69wspa2w9 = n69wspa318
            n69wspa2y8 = n69wspa2mh['hid'][:n69wspa2mh['hid'].rfind('.')]
            n69wspa2cu = []
            for e in n69wspa2n8:
                assert e['targetHandle'] == '@target', e
                assert e['sourceHandle'] == '@source', e
                if not '-end' in e['target']:
                    n69wspa2cu.append(e['target'])
                elif e['target'] in n69wspa2pt:
                    n69wspa2gq = copy.deepcopy(n69wsp9oy3[e['target']])
                    if n69wspa2gq['node_type'][3:] != (n69wsp9oy3[n69wspa2y8.split('.')[-1]]['node_type'] if n69wspa2y8.split('.')[-1] != '1' else 'module'):
                        continue
                    n69wspa2gq['hid'] = n69wspa2y8 + '-' + n69wspa2gq['node_type']
                    n69wspa2gq['source'] = n69wspa2y8
                    try:
                        n69wspa2gq['branch'] = table_unique_get(n69wsp9oo8, {'hid': n69wspa2y8})['branch']
                        n69wsp9oo8.append(n69wspa2gq)
                        n69wspa2pt.remove(e['target'])
                    except:
                        pass
            n69wspa2pi = []
            n69wspa2mg = []
            for auid in n69wspa2cu:
                if n69wspa2t2:
                    n69wspa2gh = n69wspa2km[n69wspa2km['target'] == auid].to_dict(orient='records')
                    n69wspa344 = [ed['source'] for ed in n69wspa2gh]
                    if any([ts in n69wspa2pt for ts in n69wspa344]):
                        continue
                    n69wspa2mg = n69wspa2mg + n69wspa344
                n69wspa34n = copy.deepcopy(n69wsp9oy3[auid])
                n69wspa2pi.append(n69wspa34n)
            n69wsp9oup['nexts'] = []
            if not n69wspa2pi:
                return n69wspa2w9
            if n69wspa2t2:
                n69wspa2mg = list(set(n69wspa2mg))
                n69wsp9oye = [n['xpos'] for n in n69wspa2pi]
                n69wspa2dj = [srcn['xpos'] for srcn in table_lambda_get(n69wsp9oo8, lambda x: x['uid'] in n69wspa2mg)]
                assert n69wspa2dj
                n69wspa2qd = max(n69wspa2dj)
                n69wspa2os = min(n69wsp9oye) if n69wsp9oye else n69wspa2qd + n69wsp9osu
                n69wspa2uv = (n69wspa2qd - n69wspa2os) / n69wsp9osu + 1
            else:
                n69wspa2uv = 0
            for n69wsp9p34, nn in enumerate(n69wspa2pi):
                n69wspa2ti = insection
                if nn['uid'] in section['lefts']:
                    n69wspa2ti = True
                if n65d20cda3 in section['rights']:
                    n69wspa2ti = False
                nn['hid'] = n69wspa2y8 + '.' + nn['uid']
                n69wsp9oup['nexts'].append(nn['hid'])
                if not nn.get('pres'):
                    nn['pres'] = []
                if not n69wsp9oup['hid'] in nn['pres']:
                    nn['pres'].append(n69wsp9oup['hid'])
                nn['branch'] = n69wsp9oup['branch']
                b69wspa0y9(nn, n69wspa2w9=n69wspa2uv, n69wsp9oz5=n69wsp9oz5, insection=n69wspa2ti)
            return n69wspa2w9
        n69wspa2gh = n69wsp9p0q
        for i in range(999):
            b69wspa0y9(n69wspa2gh, n69wspa2w9=0, insection=n69wspa2gh['uid'] in section['lefts'])
            if not [rid for rid in n69wspa2pt if not '-end' in rid]:
                n69wspa2p5 = [rid for rid in n69wspa2pt if '-end' in rid]
                if n69wspa2p5:
                    pass
                break
            n69wspa388 = None
            for rid in n69wspa2pt:
                if '-end' in rid:
                    continue
                if len(n69wspa2km[n69wspa2km['target'] == rid]) == 0:
                    n69wspa388 = rid
            assert n69wspa388 is not None, f'Possibly cyclic pattern: {n69wspa2pt}.'
            n69wspa2gh = copy.deepcopy(n69wsp9oy3[n69wspa388])
            n69wspa2gh['pres'] = []
            n69wspa2gh['branch'] = '_'
            n69wspa2gh['hid'] = '1.' + n69wspa2gh['uid']
        if i > 998:
            pass
        for n69wspa2mh in n69wsp9oo8:
            if n69wspa2mh['node_type'] in ('tool', 'tool_conc'):
                n69wspa2mh['params_map'] = {p[0]: p[1] for p in n69wspa2mh['params_map']}
        n69wspa2uq = pd.DataFrame(n69wsp9oo8)
        n69wspa2uq = n69wspa2uq.replace(np.nan, None) if len(n69wspa2uq) > 0 else pd.DataFrame(columns=list(n69wspa2ha.keys()))
        if n69wsp9ore and (not n69wspa2f3):
            n69wspa2yv = n69wspa2uq[['uid', 'code']].copy() if 'code' in n69wspa2uq.columns else None
            n69wspa2xu = asyncio.get_running_loop()
            n69wspa2uq = await n69wspa2xu.run_in_executor(None, lambda: b69wsp9mpz(n69wsp9oo8, n69wspa2yk, n69wsp9osu=NODESPACINGX, n69wsp9ozl=NODESPACINGY, tolerance=2 if level == 'adlvl' else 0))
            if level == 'adlvl' and 'code' in n69wspa2uq.columns:
                n69wspa2uq = n69wspa2uq[[c for c in n69wspa2uq.columns if c != 'code']]
                n69wspa2uq = pd.merge(n69wspa2uq, n69wspa2yv, on='uid', how='left')
        if level == 'adlvf':
            n69wspa2vu = n69wspa2uq[n69wspa2uq['uid'].isin(n69wspa2y9)]
            return n69wspa2vu
        if level == 'adlvd':
            return n69wspa2uq
        if level == 'adlvv' or level == 'adlvl':
            if not n69wsp9oo8:
                return {'nodes': [], 'edges': []}
            n69wsp9onl = await self.b69x8ynnta(n69wspa2uq, 'dag', count_previews=True, conn=conn, _skiplock=_skiplock)
            return n69wsp9onl
        if level == 'adlvx':
            n69wspa2g6 = n69wspa2uq[['uid', 'xpos', 'ypos']].to_dict(orient='records')
            n69wspa2g6 = {r['uid']: {'x': r['xpos'], 'y': r['ypos']} for r in n69wspa2g6}
            n69wspa2je = copy.deepcopy(n69wsp9oya)
            for n in n69wspa2je:
                if n['id'] in n69wspa2g6:
                    n['position'] = n69wspa2g6[n['id']]
            return {'nodes': n69wspa2je, 'edges': n69wspa34y}
        n69wspa2i4 = {n['uid']: n for n in n69wsp9oo8}
        n69wspa34v = None
        n69wspa36f = {}
        n69wspa2ep = {}
        n69wspa36o = {}

        async def b69x8ynnu4(conn):
            nonlocal n69wspa36f, n69wspa2ep
            n69wspa33n = f"def_id = '{n69wspa2yk}'"
            n69wspa357 = f"\n            def_id LIKE '{n69wspa2yk}^%' AND NOT def_id LIKE '{n69wspa2yk}^1#_/%' AND NOT def_id LIKE '{n69wspa2yk}^1#_*%' \n            "
            n69wspa2ow = f"\n            def_id = '{n69wspa2yk}'\n            "
            n69wsp9p2u = self.select('funcs', cond_sql=n69wspa357, targets=list(n69wspa2ts.keys()) if n69wspa2sf else ['uid', 'def_id'], _skiplock=True, conn=conn)
            n69wspa2xf = self.select('classes', cond_sql=n69wspa357, targets=list(n69wspa2dn.keys()) if n69wspa2sf else ['uid', 'def_id'], _skiplock=True, conn=conn)
            n69wspa2y6 = self.select('nodes', cond_sql=n69wspa33n, targets=list(n69wspa2ha.keys()) if n69wspa2sf else ['uid'], _skiplock=True, conn=conn)
            n69wspa2d3 = self.select('params', cond_sql=n69wspa357, targets=list(n69wspa356.keys()), _skiplock=True, conn=conn) if n69wspa2sf else aidle()
            n69wspa34c = self.select('nodes', cond_sql=n69wspa357, targets=list(n69wspa2ha.keys()), _skiplock=True, conn=conn) if n69wspa2sf else aidle()
            n69wspa31h = self.select('funcs', cond_sql=n69wspa2ow, targets=['def_id'], _skiplock=True, conn=conn)
            n69wsp9p2u, n69wspa2xf, n69wspa2y6, n69wspa2d3, n69wspa34c, n69wspa31h = await asyncio.gather(n69wsp9p2u, n69wspa2xf, n69wspa2y6, n69wspa2d3, n69wspa34c, n69wspa31h)
            if len(n69wspa31h) == 0:
                raise ValueError(f'[404] Parent func no loger exists: {n69wspa2yk}')
            n69wspa2u4 = n69wspa2y6['uid'].tolist()
            n69wspa2u4 = [nid for nid in n69wspa2u4 if not nid in n69wspa2i4]

            def b69wspa0ya(n69wspa2xo):
                assert n69wspa2xo['def_id'].startswith(n69wspa2yk + '^'), (n69wspa2xo['def_id'], n69wspa2yk)
                n69wspa2gg = n69wspa2xo['def_id'][len(n69wspa2yk) + 1:]
                n69wspa2sl = n69wspa2gg[:n69wspa2gg.find('#')]
                assert '/' in n69wspa2gg[n69wspa2gg.find('#') + 1:] or '*' in n69wspa2gg[n69wspa2gg.find('#') + 1:]
                n69wspa2xm = n69wspa2gg[n69wspa2gg.find('#') + 1:min(n69wspa2gg.find('*') if '*' in n69wspa2gg[n69wspa2gg.find('#') + 1:] else 9999, n69wspa2gg.find('/') if '/' in n69wspa2gg[n69wspa2gg.find('#') + 1:] else 9999)]
                n69wsp9p2p = n69wspa2gg[min(n69wspa2gg.find('*') if '*' in n69wspa2gg[n69wspa2gg.find('#') + 1:] else 9999, n69wspa2gg.find('/') if '/' in n69wspa2gg[n69wspa2gg.find('#') + 1:] else 9999):]
                n65d20cda3 = n69wspa2sl[n69wspa2sl.rfind('.') + 1:]
                if not n65d20cda3 in n69wspa2i4:
                    return '<DEL>:' + n69wspa2xo['def_id']
                if n69wspa2i4[n65d20cda3]['hid'].count('.') > 1:
                    if n69wspa2i4[n65d20cda3]['node_type'] in ('excepts', 'match'):
                        if rectify_caseids:
                            if n69wspa2xm in n69wspa38y[n65d20cda3]:
                                n69wspa2s3 = n69wspa38y[n65d20cda3][n69wspa2xm]
                            else:
                                return '<DEL>:' + n69wspa2xo['def_id']
                        elif n69wspa2xm in n69wspa2i4[n65d20cda3]['cases']:
                            n69wspa2s3 = n69wspa2xm
                        else:
                            return '<DEL>:' + n69wspa2xo['def_id']
                    else:
                        n69wspa2s3 = n69wspa2xm
                else:
                    n69wspa2s3 = n69wspa2xm
                if n69wspa2s3 is None:
                    return '<DEL>:' + n69wspa2xo['def_id']
                n69wsp9orn = n69wspa2i4[n65d20cda3]['hid']
                n69wspa2s3 = rectify_cond(n69wspa2s3)
                n69wspa2mt = n69wspa2yk + '^' + n69wsp9orn + '#' + str(n69wspa2s3) + n69wsp9p2p
                return n69wspa2mt
            if len(n69wsp9p2u) > 0:
                n69wsp9p2u['new_def_id'] = n69wsp9p2u.apply(lambda n69wspa2xo: b69wspa0ya(n69wspa2xo), axis=1)
            else:
                n69wsp9p2u['new_def_id'] = []
            if len(n69wspa2xf) > 0:
                n69wspa2xf['new_def_id'] = n69wspa2xf.apply(lambda n69wspa2xo: b69wspa0ya(n69wspa2xo), axis=1)
            else:
                n69wspa2xf['new_def_id'] = []
            n69wsp9p2u = n69wsp9p2u[n69wsp9p2u['def_id'] != n69wsp9p2u['new_def_id']]
            n69wspa2xf = n69wspa2xf[n69wspa2xf['def_id'] != n69wspa2xf['new_def_id']]
            n69wsp9p2u = n69wsp9p2u[['uid', 'new_def_id']]
            n69wspa2xf = n69wspa2xf[['uid', 'new_def_id']]
            n69wsp9p2u = n69wsp9p2u.rename(columns={'new_def_id': 'def_id'})
            n69wspa2xf = n69wspa2xf.rename(columns={'new_def_id': 'def_id'})
            n69wspa2wp = n69wsp9p2u[~n69wsp9p2u['def_id'].str.startswith('<DEL>:')] if len(n69wsp9p2u) > 0 else pd.DataFrame(columns=['uid', 'def_id'])
            n69wspa2yu = n69wsp9p2u[n69wsp9p2u['def_id'].str.startswith('<DEL>:')] if len(n69wsp9p2u) > 0 else pd.DataFrame(columns=['uid', 'def_id'])
            n69wspa2yc = n69wspa2xf[~n69wspa2xf['def_id'].str.startswith('<DEL>:')] if len(n69wspa2xf) > 0 else pd.DataFrame(columns=['uid', 'def_id'])
            n69wspa2uw = n69wspa2xf[n69wspa2xf['def_id'].str.startswith('<DEL>:')] if len(n69wspa2xf) > 0 else pd.DataFrame(columns=['uid', 'def_id'])
            n69wspa2yu['active'] = False
            n69wspa2uw['active'] = False
            n69wspa2yu = n69wspa2yu[['uid', 'def_id', 'active']]
            n69wspa2uw = n69wspa2uw[['uid', 'def_id', 'active']]
            if len(n69wspa2yu) > 0:
                n69wspa2yu['def_id'] = n69wspa2yu.apply(lambda n69wspa2xo: n69wspa2xo['def_id'][6:], axis=1)
            if len(n69wspa2uw) > 0:
                n69wspa2uw['def_id'] = n69wspa2uw.apply(lambda n69wspa2xo: n69wspa2xo['def_id'][6:], axis=1)
            n69wsp9onf = n69wspa2uq.columns.intersection(list(n69wspa2ha.keys()), sort=False)
            nonlocal n69wspa34v
            n69wspa2q1 = self.upsert('funcs', n69wspa2wp, _skiplock=True, conn=conn)
            n69wspa2nc = [self.b69x8ynnuu(did, conn=conn, _skiplock=True) for did in n69wspa2yu['def_id'].tolist()]
            n69wspa33x = asyncio.gather(*n69wspa2nc)
            n69wspa2ig = self.upsert('classes', n69wspa2yc, _skiplock=True, conn=conn)
            n69wspa30i = [self.b69x8ynnuu(did, conn=conn, _skiplock=True) for did in n69wspa2uw['def_id'].tolist()]
            n69wspa2r8 = asyncio.gather(*n69wspa30i)
            if n69wspa2u4:
                n69wspa2hm = self.delete('nodes', conds=[{'uid': u} for u in n69wspa2u4], _skiplock=True, conn=conn)
            else:
                n69wspa2hm = aidle()
            await asyncio.gather(n69wspa2q1, n69wspa33x, n69wspa2ig, n69wspa2r8, n69wspa2hm)
            n69wspa355 = n69wspa2uq
            await self.upsert('nodes', n69wspa2uq, _skiplock=True, conn=conn)
            if n69wspa2f3:
                try:
                    n69wspa34v = await self.b69x8ynnt6('', n69wspa2yk + '^1#_', n69wspa2da={'mode': 'refresh', 'recur': False, 'repose': n69wsp9ore, 'revars': True}, n69wspa381=n69wspa381, _skiplock=True, conn=conn, tolerance=2)
                except Exception as e:
                    n69wspa2ui = {'debug': logger.debug, 'info': logger.info, 'warning': logger.warning, 'error': logger.error}
                    if revars_failure_behavior in n69wspa2ui:
                        n69wspa2ui[revars_failure_behavior](f'eh027：{e}')
                        traceback.print_exc()
                    else:
                        raise RuntimeError(e)
                    n69wspa34v = n69wspa2uq
            else:
                n69wspa34v = n69wspa2uq
            n69wspa34v = n69wspa34v.replace(np.nan, None)
        gatherables = [b69x8ynnu4]
        await self._batch_write(gatherables, _skiplock=_skiplock, conn=conn)
        assert n69wspa34v is not None
        if 'rerender' in level:
            n69wspa34v = await self.b69x8ynnvx(n69wspa2yk, 'dag', count_previews=True, conn=conn, _skiplock=_skiplock)
        else:
            n69wspa34v = n69wspa34v.replace(np.nan, None)
            n69wspa34v = await self.b69x8ynnta(n69wspa34v, 'dag', count_previews=True, conn=conn, _skiplock=_skiplock)
        if cached:
            await self.b69x8ynnvg(n69wspa2yk, 'dag', debugtag='(dc03)', conn=conn, _skiplock=_skiplock)
        return n69wspa34v

    async def b69x8ynnvk(self, n65d20cda3, rela, todel=None, conn=None, _skiplock=False):
        n69wspa2yo = []
        for k, v in rela.items():
            assert k.count('#') == 1 and k.count('@') == 1
            assert v.count('#') == 1 and v.count('@') == 1
            assert k.find('#') < k.find('@')
            assert v.find('#') < v.find('@')
            n69wspa2mx = k[k.find('#') + 1:k.find('@')]
            n69wspa2vb = v[v.find('#') + 1:v.find('@')]
            assert n69wspa2mx.isdigit(), n69wspa2mx
            assert n69wspa2vb.isdigit(), n69wspa2vb
            n69wspa2yo.append((n69wspa2mx, n69wspa2vb))
        if todel is None:
            n69wspa2yo.reverse()
        if len(n69wspa2yo) > 0:
            assert all([n69wspa2yo[i][0] == n69wspa2yo[i + 1][1] for i in range(len(n69wspa2yo) - 1)]), f"eh021-1{('删减' if todel is not None else '新增')}eh021-2{n69wspa2yo}"
        n69wspa2jg = None
        if todel is not None:
            if not todel.isdigit():
                assert todel.count('#') == 1 and todel.count('@') == 1
                assert todel.find('#') < todel.find('@')
                n69wspa2jg = todel[todel.find('#') + 1:todel.find('@')]
            else:
                n69wspa2jg = todel
            assert n69wspa2jg.isdigit(), n69wspa2jg
            if len(n69wspa2yo) > 0:
                assert n69wspa2jg == n69wspa2yo[0][1], f'eh023{n69wspa2jg}，{n69wspa2yo}'

        async def b69x8ynnu4(conn):
            n69wspa2ue = await self.select('nodes', [{'uid': n65d20cda3}], targets=['uid', 'def_id', 'hid'], _skiplock=True, conn=conn)
            assert len(n69wspa2ue) <= 1, n69wspa2ue
            if len(n69wspa2ue) == 0:
                return
            n69wspa2kd = n69wspa2ue.loc[0, 'hid']
            n69wspa2ob = n69wspa2ue.loc[0, 'def_id']
            n69wspa38z = f'{n69wspa2ob}^{n69wspa2kd}'
            if n69wspa2jg is not None:
                n69wspa37u = n69wspa38z + '#' + n69wspa2jg
                n69wspa2cr = f"\n                def_id LIKE '{n69wspa37u}/%' OR def_id LIKE '{n69wspa37u}*%'\n                "
                n69wspa2q9 = self.delete('funcs', cond_sql=n69wspa2cr, _skiplock=True, conn=conn)
                n69wspa321 = self.delete('classes', cond_sql=n69wspa2cr, _skiplock=True, conn=conn)
                await asyncio.gather(n69wspa2q9, n69wspa321)
            n69wspa2m3 = "\n            UPDATE funcs\n            SET def_id = CONCAT('{new_branch_id}',SUBSTRING(def_id,LENGTH('{old_branch_id}')+1))\n            WHERE def_id LIKE '{old_branch_id}*%' OR def_id LIKE '{old_branch_id}/%'\n            "
            n69wspa2t3 = "\n            UPDATE classes\n            SET def_id = CONCAT('{new_branch_id}',SUBSTRING(def_id,LENGTH('{old_branch_id}')+1))\n            WHERE def_id LIKE '{old_branch_id}*%' OR def_id LIKE '{old_branch_id}/%'\n            "
            for revise in n69wspa2yo:
                n69wspa2v9 = n69wspa38z + '#' + revise[0]
                n69wspa34r = n69wspa38z + '#' + revise[1]
                n69wspa2zu = text(n69wspa2m3.format(old_branch_id=n69wspa2v9, new_branch_id=n69wspa34r))
                n69wspa2p2 = text(n69wspa2t3.format(old_branch_id=n69wspa2v9, new_branch_id=n69wspa34r))
                n69wspa2dx = conn.execute(n69wspa2zu)
                n69wspa348 = conn.execute(n69wspa2p2)
                await asyncio.gather(n69wspa2dx, n69wspa348)
        await self._batch_write([b69x8ynnu4], _skiplock=_skiplock, conn=conn)

    async def b69x8ynnvl(self, n69wspa2tb, _skiplock=False, conn=None):
        assert x69xm5dtzx(n69wspa2tb) == 'cond', n69wspa2tb
        assert '^' in n69wspa2tb and '#' in n69wspa2tb
        n69wspa2ox = n69wspa2tb.rfind('#')
        n69wspa2eq = n69wspa2tb.rfind('^') + 1
        assert n69wspa2ox > n69wspa2eq
        n65d20cda3 = n69wspa2tb[n69wspa2eq:n69wspa2ox]
        if '.' in n65d20cda3:
            return n69wspa2tb
        if n65d20cda3 == '1':
            return n69wspa2tb
        n69wsp9p51, _ = await self.b69x8ynnvc(n65d20cda3, _skiplock=_skiplock, conn=conn)
        n69wspa37n = n69wspa2tb[:n69wspa2eq] + n69wsp9p51 + n69wspa2tb[n69wspa2ox:]
        return n69wspa37n

    def b69wspa0yj(self, n69wspa2tb):
        assert x69xm5dtzx(n69wspa2tb) in ('class', 'cond', 'folder'), n69wspa2tb
        if x69xm5dtzx(n69wspa2tb) in ('folder', 'class'):
            return n69wspa2tb
        assert '^' in n69wspa2tb and '#' in n69wspa2tb
        n69wspa2ox = n69wspa2tb.rfind('#')
        n69wspa2eq = n69wspa2tb.rfind('^') + 1
        assert n69wspa2ox > n69wspa2eq
        n69wsp9p51 = n69wspa2tb[n69wspa2eq:n69wspa2ox]
        n65d20cda3 = n69wsp9p51.split('.')[-1]
        n69wspa37n = n69wspa2tb[:n69wspa2eq] + n65d20cda3 + n69wspa2tb[n69wspa2ox:]
        return n69wspa37n

    @n69wspa2jh.debounce([1, 2, 'timestamp'], [list], compare_func=b69wspa0xo, intercepted_kws=['timestamp'])
    async def b69x8ynnvr(self, n69wspa2qg, app_funcs, n69wsp9ore=False, n69wsp9ozl=NODESPACINGY, cached=False, n69wspa2sf=False, _skiplock=False, conn=None):
        if n69wspa2sf and (not cached):
            n69wspa2sf = False
        assert x69xm5dtzx(n69wspa2qg) in ('cond', 'class', 'folder')
        n69wspa30v = n69wspa2qg
        if x69xm5dtzx(n69wspa2qg) == 'cond':
            n69wspa2qg = await self.b69x8ynnvl(n69wspa2qg, _skiplock=_skiplock, conn=conn)
        n69wspa2p7 = copy.deepcopy(app_funcs)
        n69wspa32y = []
        for b69wsp9mos in app_funcs:
            if b69wsp9mos['data']['def_id'].split('/')[-1] in n69wspa32y:
                raise ValueError(f"Repeating func name: {b69wsp9mos['data']['def_id'].split('/')[-1]}")
            n69wspa32y.append(b69wsp9mos['data']['def_id'].split('/')[-1])
            assert is_valid_varname(b69wsp9mos['data']['def_id'].split('/')[-1]), f"Invalid func name: {b69wsp9mos['data']['def_id'].split('/')[-1]}"
            assert 'inputs' in b69wsp9mos['data'], b69wsp9mos['data']
            if x69xm5dtzx(n69wspa30v) == 'folder':
                if b69wsp9mos['data']['inputs']:
                    b69wsp9mos['data']['inputs'] = []
                if b69wsp9mos['data'].get('is_async'):
                    b69wsp9mos['data']['is_async'] = False
            b69wsp9mos['data']['inputs'] = [inp for inp in b69wsp9mos['data']['inputs'] if inp['name'].strip()]
            for inp in b69wsp9mos['data']['inputs']:
                assert is_valid_varname(inp['name'].strip().lstrip('*')), f"illegal variable name: {inp['name']}"
                if x69xm5dtzx(n69wspa2qg) == 'class':
                    if inp['name'] == 'self':
                        inp['type'] = n69wspa2qg.split('*')[-1]
                    elif inp['name'] == 'cls':
                        inp['type'] = f"Type[{n69wspa2qg.split('*')[-1]}]"
            b69wsp9mos['subWorkflowIds'] = {'_': {'funcs': b69wsp9mos['data']['def_id'] + '^1#_:funcs', 'classes': b69wsp9mos['data']['def_id'] + '^1#_:classes', 'dag': b69wsp9mos['data']['def_id'] + ':dag'}}
        n69wspa2oy = copy.deepcopy(app_funcs)
        n69wspa2d8 = []
        n69wsp9osv = []
        for n69wspa317 in app_funcs:
            assert n69wspa317['type'] == 'switch'
            assert x69xm5dtzx(n69wspa317['data']['def_id']) == 'func'
            assert n69wspa317['data']['def_id'].startswith(n69wspa2qg + '/'), (n69wspa317['data']['def_id'], n69wspa2qg)
            assert not '/' in n69wspa317['data']['def_id'][len(n69wspa2qg) + 1:]
            func = n69wspa317['data']
            assert 'uid' in func
            del func['subWorkflowIds']
            assert func['node_type'] == 'func'
            del func['node_type']
            if func.get('imports_code'):
                try:
                    b69wsp9mo8(func['imports_code'], ['Import', 'ImportFrom'])
                except Exception as e:
                    raise ValueError(f'[404] Error in imports: {e}')
            func['xpos'] = n69wspa317['position']['x']
            func['ypos'] = n69wspa317['position']['y']
            n69wspa2d8.append(func)
            n69wsp9ozv = copy.deepcopy(func['inputs'])
            n69wsp9onl = copy.deepcopy(func['return'])
            if not n69wsp9onl.get('name') in (None, 'return'):
                pass
            n69wsp9onl['name'] = 'return'
            del func['inputs'], func['return']
            for ii, inp in enumerate(n69wsp9ozv):
                if not inp['name'].strip():
                    continue
                inp['def_id'] = func['def_id']
                inp['ctx'] = 'input'
                inp['place'] = ii
                n69wsp9osv.append(inp)
            n69wsp9onl['def_id'] = func['def_id']
            n69wsp9onl['ctx'] = 'return'
            n69wsp9onl['place'] = 0
            n69wsp9osv.append(n69wsp9onl)
        n69wsp9osv = pd.DataFrame(n69wsp9osv) if n69wsp9osv else pd.DataFrame(columns=['name', 'type', 'doc', 'def_id', 'ctx', 'default', 'place'])
        n69wsp9omb = pd.DataFrame(n69wspa2d8) if n69wspa2d8 else pd.DataFrame(columns=['uid', 'def_id', 'globals', 'nonlocals'])
        n69wsp9omb['nonlocals'] = n69wsp9omb['nonlocals'].apply(lambda n69wsp9oxo: [x.strip() for x in n69wsp9oxo.strip().split(',') if x.strip()])
        n69wsp9omb['globals'] = n69wsp9omb['globals'].apply(lambda n69wsp9oxo: [x.strip() for x in n69wsp9oxo.strip().split(',') if x.strip()])
        n69wspa36f = {}
        n69wspa2ep = {}
        n69wspa36o = {}

        async def b69x8ynnu4(conn):
            nonlocal n69wspa36f, n69wspa2ep, n69wspa36o
            n69wspa357 = f"def_id LIKE '{n69wspa2qg}/%' AND\n                    (SUBSTRING(def_id, LENGTH('{n69wspa2qg}/') + 1) NOT LIKE '%/%'\n                    AND SUBSTRING(def_id, LENGTH('{n69wspa2qg}/') + 1) NOT LIKE '%#%'\n                    AND SUBSTRING(def_id, LENGTH('{n69wspa2qg}/') + 1) NOT LIKE '%^%'\n                    AND SUBSTRING(def_id, LENGTH('{n69wspa2qg}/') + 1) NOT LIKE '%*%')\n            "
            n69wspa2p1 = f"def_id LIKE '{n69wspa2qg}/%'"
            n69wspa34j = self.select('funcs', cond_sql=n69wspa357, targets=['uid', 'def_id'], _skiplock=True, conn=conn)
            n69wspa329 = self.select('funcs', cond_sql=n69wspa2p1, _skiplock=True, conn=conn) if n69wspa2sf else aidle()
            n69wspa2qh = self.select('params', cond_sql=n69wspa2p1, _skiplock=True, conn=conn) if n69wspa2sf else aidle()
            n69wspa2no = self.select('nodes', cond_sql=n69wspa2p1, _skiplock=True, conn=conn) if n69wspa2sf else aidle()
            n69wspa2vt = self.select('classes', cond_sql=n69wspa2p1, _skiplock=True, conn=conn) if n69wspa2sf else aidle()
            if x69xm5dtzx(n69wspa2qg) == 'class':
                n69wspa2xc = f"def_id = '{n69wspa2qg}'"
                n69wspa31h = self.select('classes', cond_sql=n69wspa2xc, targets=['def_id'], _skiplock=True, conn=conn)
            else:
                assert x69xm5dtzx(n69wspa2qg) in ('cond', 'folder')
                if x69xm5dtzx(n69wspa2qg) == 'folder':
                    n69wspa31h = aidle(default=[1])
                else:
                    assert '^' in n69wspa2qg
                    n69wspa2nb = n69wspa2qg.rsplit('^', 1)[0]
                    n69wspa2xc = f"def_id = '{n69wspa2nb}'"
                    n69wspa31h = self.select('funcs', cond_sql=n69wspa2xc, targets=['def_id'], _skiplock=True, conn=conn)
            n69wspa34j, n69wspa329, n69wspa2qh, n69wspa2no, n69wspa2vt, n69wspa31h = await asyncio.gather(n69wspa34j, n69wspa329, n69wspa2qh, n69wspa2no, n69wspa2vt, n69wspa31h)
            if len(n69wspa31h) == 0:
                raise ValueError(f'[404] Parent element no longer exists. Cannot edit funcs.')
            if cached:
                self.b69wspa0yb(to_module_id(n69wspa30v))
                n69wspa38m = to_module_id(n69wspa30v)
                n69wspa36f = {'module_id': n69wspa38m, 'scope_type': 'funcs', 'def_id': n69wspa30v, 'undoed': 'F', 'data': {}, 'focus_scope': None}
                if n69wspa2sf:
                    n69wspa31i = copy.deepcopy(n69wspa36f)
                    n69wspa2mb = idgen.generate(return_type='int')
                    n69wspa31i['save_id'] = n69wspa2mb
                    n69wspa31i['data']['db'] = {'nodes': n69wspa2no.to_dict(orient='records'), 'funcs': n69wspa329.to_dict(orient='records'), 'classes': n69wspa2vt.to_dict(orient='records'), 'params': n69wspa2qh.to_dict(orient='records')}
                    n69wspa2zv = await self.b69x8ynnu2(n69wspa2qg, conn=conn, _skiplock=True)
                    n69wspa2zv = await self.b69x8ynnta(n69wspa2zv, 'funcs', count_previews=True, conn=conn, _skiplock=True)
                    n69wspa31i['data']['app'] = n69wspa2zv
                    self.n69wspa36f.json().set(f"undo:{n69wspa31i['save_id']}", '$', n69wspa31i)
                n69wspa2jn = idgen.generate(return_type='int')
                n69wspa36f['save_id'] = n69wspa2jn
            n69wspa2nu = n69wsp9omb[n69wsp9omb['uid'].isin(n69wspa34j['uid'])]
            n69wspa2xr = n69wsp9omb[~n69wsp9omb['uid'].isin(n69wspa34j['uid'])]
            n69wspa2qz = n69wspa34j[['uid', 'def_id']]
            n69wspa2qz = n69wspa2qz.rename(columns={'def_id': 'old_def_id'})
            n69wspa36v = pd.merge(n69wspa2nu, n69wspa2qz, on='uid', how='left')
            n69wspa36v = n69wspa36v[~(n69wspa36v['def_id'] == n69wspa36v['old_def_id'])]
            n69wsp9oqx = n69wspa2qz[~n69wspa2qz['uid'].isin(n69wsp9omb['uid'])]
            n69wspa2m3 = "\n            \n            UPDATE funcs\n            SET def_id = CONCAT('{new_def_id}',SUBSTRING(def_id,LENGTH('{old_def_id}')+1))\n            WHERE def_id LIKE '{old_def_id}^%'\n            "
            n69wspa2t3 = "\n            UPDATE classes\n            SET def_id = CONCAT('{new_def_id}',SUBSTRING(def_id,LENGTH('{old_def_id}')+1))\n            WHERE def_id LIKE '{old_def_id}^%'\n            "
            n69wspa2mu = "\n            UPDATE nodes \n            SET def_id = CONCAT('{new_def_id}',SUBSTRING(def_id,LENGTH('{old_def_id}')+1))\n            WHERE def_id LIKE '{old_def_id}^%'\n            "
            n69wspa2dx = self.upsert('funcs', n69wsp9omb, conn=conn, _skiplock=True)
            n69wspa2qm = [n69wspa2dx]
            for ri in n69wspa36v.index.tolist():
                n69wspa2u1 = conn.execute(text(n69wspa2m3.format(new_def_id=n69wspa36v.loc[ri, 'def_id'], old_def_id=n69wspa36v.loc[ri, 'old_def_id'])))
                n69wspa2o2 = conn.execute(text(n69wspa2t3.format(new_def_id=n69wspa36v.loc[ri, 'def_id'], old_def_id=n69wspa36v.loc[ri, 'old_def_id'])))
                n69wspa2sv = conn.execute(text(n69wspa2mu.format(new_def_id=n69wspa36v.loc[ri, 'def_id'], old_def_id=n69wspa36v.loc[ri, 'old_def_id'])))
                n69wspa2qm = n69wspa2qm + [n69wspa2u1, n69wspa2o2, n69wspa2sv]
            n69wspa2px = [self.b69x8ynnuu(adel, conn=conn, _skiplock=True) for adel in n69wsp9oqx['old_def_id']]
            n69wspa2sd = self.delete('params', [{'def_id': did} for did in n69wspa2qz['old_def_id']], conn=conn, _skiplock=True) if len(n69wspa2qz) > 0 else aidle()
            await asyncio.gather(*n69wspa2qm, *n69wspa2px, n69wspa2sd)
            n69wspa34y = []
            for anew in n69wspa2xr['def_id']:
                b69wsp9mo6 = {'uid': gen_base36_id(), 'hid': '1.0', 'branch': '_', 'data_providers': [], 'node_type': 'start', 'pres': [], 'nexts': [], 'xpos': 0, 'ypos': 0, 'def_id': anew}
                n69wspa34y.append(b69wsp9mo6)
            n69wspa34y = pd.DataFrame(n69wspa34y)
            n69wspa2qx = self.upsert('nodes', n69wspa34y, conn=conn, _skiplock=True) if len(n69wspa34y) > 0 else aidle()
            n69wspa2cd = self.upsert('params', n69wsp9osv, conn=conn, _skiplock=True)
            await asyncio.gather(n69wspa2qx, n69wspa2cd)
            if cached:
                await self.b69x8ynnvg(n69wspa30v, 'funcs', debugtag='(dc04)', conn=conn, _skiplock=True)
        await self._batch_write([b69x8ynnu4], _skiplock=_skiplock, conn=conn)
        return n69wspa2oy

    async def b69x8ynnuz(self, n69wspa2qg, src_x, src_y, n69wsp9osu=0, n69wsp9ozl=NODESPACINGY):
        n65d20cda3 = idgen.generate('f')
        assert x69xm5dtzx(n69wspa2qg) in ('cond', 'class', 'folder')
        if x69xm5dtzx(n69wspa2qg) == 'cond':
            n69wspa2qg = await self.b69x8ynnvl(n69wspa2qg)
        n69wspa30z = 'your_func_name'
        n69wspa2cw = {'id': n65d20cda3, 'type': 'switch', 'position': {'x': (src_x or 0) + n69wsp9osu, 'y': (src_y or 0) + n69wsp9ozl}, 'data': {'uid': n65d20cda3, 'node_type': 'func', 'def_id': n69wspa2qg + '/' + n69wspa30z, 'globals': '', 'nonlocals': '', 'imports_code': '', 'is_async': 0, 'deco_expr': '', 'inputs': [] if not '*' in n69wspa2qg else [{'name': 'self', 'ctx': 'intput', 'place': 0}], 'return': {'name': 'result', 'type': '', 'doc': '', 'default': None}, 'tool_counts': {'_': {'nodes': 0, 'funcs': 0, 'classes': 0}}, 'subWorkflowIds': {'_': {'funcs': f'{n69wspa2qg}/{n69wspa30z}^1#_:funcs', 'classes': f'{n69wspa2qg}/{n69wspa30z}^1#_:classes', 'dag': f'{n69wspa2qg}/{n69wspa30z}:dag'}}}}
        return n69wspa2cw

    @n69wspa2pp.debounce([1, 2, 'timestamp'], [list], compare_func=n69wspa2s9, intercepted_kws=['timestamp'])
    async def b69x8ynntk(self, n69wspa2qg, app_classes, n69wsp9ore=False, n69wsp9ozl=NODESPACINGY, cached=False, n69wspa2sf=False, _skiplock=False, conn=None):
        if n69wspa2sf and (not cached):
            n69wspa2sf = False
        assert x69xm5dtzx(n69wspa2qg) in 'cond'
        n69wspa30v = n69wspa2qg
        n69wspa372 = copy.deepcopy(app_classes)
        n69wspa2qg = await self.b69x8ynnvl(n69wspa2qg, _skiplock=_skiplock, conn=conn)
        n69wspa319 = []
        n69wspa32y = []
        for n69wsp9p51 in app_classes:
            assert n69wsp9p51['type'] == 'switch'
            assert x69xm5dtzx(n69wsp9p51['data']['def_id']) == 'class'
            assert n69wsp9p51['data']['def_id'].startswith(n69wspa2qg + '*'), (n69wsp9p51['data']['def_id'], n69wspa2qg)
            assert not '/' in n69wsp9p51['data']['def_id'][len(n69wspa2qg) + 1:]
            assert is_valid_varname(n69wsp9p51['data']['def_id'].split('*')[-1]), f"invalid class name: {n69wsp9p51['data']['def_id'].split('*')[-1]}"
            if n69wsp9p51['data']['def_id'].split('*')[-1] in n69wspa32y:
                raise ValueError(f"Repeating class name: {n69wsp9p51['data']['def_id'].split('*')[-1]}")
            n69wspa32y.append(n69wsp9p51['data']['def_id'].split('*')[-1])
            n69wsp9p51['data']['vars'] = [inp for inp in n69wsp9p51['data']['vars'] if inp['name'].strip()]
            n69wsp9p51['subWorkflowIds'] = {'_': {'funcs': n69wsp9p51['data']['def_id'] + ':funcs'}}
            n69wsp9orc = copy.deepcopy(n69wsp9p51['data'])
            assert 'uid' in n69wsp9orc
            del n69wsp9orc['subWorkflowIds']
            n69wsp9orc['xpos'] = n69wsp9p51['position']['x']
            n69wsp9orc['ypos'] = n69wsp9p51['position']['y']
            if 'node_type' in n69wsp9orc:
                assert n69wsp9orc['node_type'] == 'class'
                del n69wsp9orc['node_type']
            n69wspa319.append(n69wsp9orc)
        n69wsp9oni = pd.DataFrame(n69wspa319) if n69wspa319 else pd.DataFrame(columns=['uid', 'def_id', 'bases'])
        n69wsp9oni['bases'] = n69wsp9oni['bases'].apply(lambda n69wsp9oxo: [x.strip() for x in n69wsp9oxo.strip().split(',') if x.strip()])
        n69wspa36f = {}
        n69wspa2ep = {}
        n69wspa36o = {}

        async def b69x8ynnu4(conn):
            nonlocal n69wspa36f, n69wspa2ep, n69wspa36o
            n69wspa357 = f"def_id LIKE '{n69wspa2qg}*%' AND\n                (SUBSTRING(def_id, LENGTH('{n69wspa2qg}*') + 1) NOT LIKE '%/%'\n                AND SUBSTRING(def_id, LENGTH('{n69wspa2qg}*') + 1) NOT LIKE '%#%'\n                AND SUBSTRING(def_id, LENGTH('{n69wspa2qg}*') + 1) NOT LIKE '%^%'\n                AND SUBSTRING(def_id, LENGTH('{n69wspa2qg}*') + 1) NOT LIKE '%*%')"
            n69wspa2i2 = self.select('classes', cond_sql=n69wspa357, targets=['uid', 'def_id'], _skiplock=True, conn=conn)
            n69wspa2p1 = f"def_id LIKE '{n69wspa2qg}*%'"
            n69wspa329 = self.select('funcs', cond_sql=n69wspa2p1, _skiplock=True, conn=conn) if n69wspa2sf else aidle()
            n69wspa2qh = self.select('params', cond_sql=n69wspa2p1, _skiplock=True, conn=conn) if n69wspa2sf else aidle()
            n69wspa2no = self.select('nodes', cond_sql=n69wspa2p1, _skiplock=True, conn=conn) if n69wspa2sf else aidle()
            n69wspa2vt = self.select('classes', cond_sql=n69wspa2p1, _skiplock=True, conn=conn) if n69wspa2sf else aidle()
            assert '^' in n69wspa2qg
            n69wspa36j = n69wspa2qg.rsplit('^', 1)[0]
            n69wspa2ow = f"\n            def_id = '{n69wspa36j}'\n            "
            n69wspa31h = self.select('funcs', cond_sql=n69wspa2ow, targets=['def_id'], _skiplock=True, conn=conn)
            n69wspa2i2, n69wspa329, n69wspa2qh, n69wspa2no, n69wspa2vt, n69wspa31h = await asyncio.gather(n69wspa2i2, n69wspa329, n69wspa2qh, n69wspa2no, n69wspa2vt, n69wspa31h)
            if len(n69wspa31h) == 0:
                raise ValueError(f'[404] Parent func no loger exists: {n69wspa36j}')
            if cached:
                self.b69wspa0yb(to_module_id(n69wspa30v))
                assert '/' in n69wspa30v
                n69wspa36f = {'module_id': to_module_id(n69wspa30v), 'scope_type': 'classes', 'def_id': n69wspa30v, 'undoed': 'F', 'data': {}, 'focus_scope': None}
                if n69wspa2sf:
                    n69wspa31i = copy.deepcopy(n69wspa36f)
                    n69wspa31i['save_id'] = idgen.generate(return_type='int')
                    n69wspa31i['data']['db'] = {'nodes': n69wspa2no.to_dict(orient='records'), 'funcs': n69wspa329.to_dict(orient='records'), 'classes': n69wspa2vt.to_dict(orient='records'), 'params': n69wspa2qh.to_dict(orient='records')}
                    n69wspa2vs = await self.b69x8ynntl(n69wspa2qg, conn=conn, _skiplock=True)
                    n69wspa2vs = await self.b69x8ynnta(n69wspa2vs, 'classes', count_previews=True, conn=conn, _skiplock=True)
                    n69wspa31i['data']['app'] = n69wspa2vs
                    self.n69wspa36f.json().set(f"undo:{n69wspa31i['save_id']}", '$', n69wspa31i)
                n69wspa2jn = idgen.generate(return_type='int')
                n69wspa36f['save_id'] = n69wspa2jn
            n69wspa2nu = n69wsp9oni[n69wsp9oni['uid'].isin(n69wspa2i2['uid'])]
            n69wspa2xr = n69wsp9oni[~n69wsp9oni['uid'].isin(n69wspa2i2['uid'])]
            n69wspa2qz = n69wspa2i2[['uid', 'def_id']]
            n69wspa2qz = n69wspa2qz.rename(columns={'def_id': 'old_def_id'})
            n69wspa36v = pd.merge(n69wspa2nu, n69wspa2qz, on='uid', how='left')
            n69wspa36v = n69wspa36v[~(n69wspa36v['def_id'] == n69wspa36v['old_def_id'])]
            n69wsp9oqx = n69wspa2qz[~n69wspa2qz['uid'].isin(n69wsp9oni['uid'])]
            n69wspa2m3 = "\n            \n            UPDATE funcs\n            SET def_id = CONCAT('{new_def_id}',SUBSTRING(def_id,LENGTH('{old_def_id}')+1))\n            WHERE def_id LIKE '{old_def_id}/%'\n            "
            n69wspa2t3 = "\n            UPDATE classes\n            SET def_id = CONCAT('{new_def_id}',SUBSTRING(def_id,LENGTH('{old_def_id}')+1))\n            WHERE def_id LIKE '{old_def_id}/%'\n            "
            n69wspa2mu = "\n            UPDATE nodes \n            SET def_id = CONCAT('{new_def_id}',SUBSTRING(def_id,LENGTH('{old_def_id}')+1))\n            WHERE def_id LIKE '{old_def_id}/%'\n            "
            n69wspa323 = "\n            UPDATE params\n            SET type = '{new_cls_name}'\n            WHERE ctx = 'input' AND name = 'self' AND def_id LIKE '{new_def_id}/%' AND place = 0\n            AND NOT SUBSTRING(def_id,LENGTH('{new_def_id}')+2) LIKE '%^%'\n            "
            n69wspa2er = "\n            UPDATE params\n            SET type = 'Type[{new_cls_name}]'\n            WHERE ctx = 'input' AND name = 'cls' AND def_id LIKE '{new_def_id}/%' AND place = 0\n            AND NOT SUBSTRING(def_id,LENGTH('{new_def_id}')+2) LIKE '%^%'\n            "
            n69wspa2rc = self.upsert('classes', n69wsp9oni, conn=conn, _skiplock=True)
            n69wspa2qm = [n69wspa2rc]
            n69wspa2eb = []
            for ri in n69wspa36v.index.tolist():
                n69wspa2u1 = conn.execute(text(n69wspa2m3.format(new_def_id=n69wspa36v.loc[ri, 'def_id'], old_def_id=n69wspa36v.loc[ri, 'old_def_id'])))
                n69wspa2o2 = conn.execute(text(n69wspa2t3.format(new_def_id=n69wspa36v.loc[ri, 'def_id'], old_def_id=n69wspa36v.loc[ri, 'old_def_id'])))
                n69wspa2sv = conn.execute(text(n69wspa2mu.format(new_def_id=n69wspa36v.loc[ri, 'def_id'], old_def_id=n69wspa36v.loc[ri, 'old_def_id'])))
                n69wspa2qm = n69wspa2qm + [n69wspa2u1, n69wspa2o2, n69wspa2sv]
                n69wspa2mv = conn.execute(text(n69wspa323.format(new_cls_name=n69wspa36v.loc[ri, 'def_id'].split('*')[-1], new_def_id=n69wspa36v.loc[ri, 'def_id'])))
                n69wspa2yi = conn.execute(text(n69wspa2er.format(new_cls_name=n69wspa36v.loc[ri, 'def_id'].split('*')[-1], new_def_id=n69wspa36v.loc[ri, 'def_id'])))
                n69wspa2eb = n69wspa2eb + [n69wspa2mv, n69wspa2yi]
            n69wspa2px = [self.b69x8ynnuu(adel, conn=conn, _skiplock=True) for adel in n69wsp9oqx['old_def_id']]
            await asyncio.gather(*n69wspa2qm, *n69wspa2px)
            await asyncio.gather(*n69wspa2eb)
            if cached:
                await self.b69x8ynnvg(n69wspa30v, 'classes', debugtag='(dc05)', conn=conn, _skiplock=True)
        await self._batch_write([b69x8ynnu4], _skiplock=_skiplock, conn=conn)
        return app_classes

    async def b69x8ynnvy(self, n69wspa2qg, src_x, src_y, n69wsp9osu=0, n69wsp9ozl=NODESPACINGY):
        n65d20cda3 = idgen.generate('c')
        assert x69xm5dtzx(n69wspa2qg) in 'cond'
        n69wspa2qg = await self.b69x8ynnvl(n69wspa2qg)
        n69wspa30z = 'YourClassName'
        n69wspa2cw = {'id': n65d20cda3, 'type': 'switch', 'position': {'x': (src_x or 0) + n69wsp9osu, 'y': (src_y or 0) + n69wsp9ozl}, 'data': {'uid': n65d20cda3, 'node_type': 'class', 'def_id': n69wspa2qg + '*' + n69wspa30z, 'vars': [], 'deco_expr': '', 'subWorkflowIds': {'_': {'funcs': f'{n69wspa2qg}/{n69wspa30z}:funcs'}}, 'tool_counts': {'_': {'funcs': 0, 'classes': 0}}, 'bases': ''}}
        return n69wspa2cw

    async def b69x8ynntb(self, n69wspa2qg, n69wspa2jq, debugtag='', existing_data=None, conn=None, _skiplock=False):
        n69wspa30v = n69wspa2qg
        if n69wspa2jq == 'dag':
            assert x69xm5dtzx(n69wspa2qg) in ('func', 'folder'), n69wspa2qg
        elif n69wspa2jq == 'funcs':
            assert x69xm5dtzx(n69wspa2qg) in ('cond', 'class', 'folder'), n69wspa2qg
        elif n69wspa2jq == 'classes':
            assert x69xm5dtzx(n69wspa2qg) in 'cond', n69wspa2qg

        async def b69x8ynnsr(conn):

            def b69wspa0xv(existing_data, n69wspa2qn, separator):
                n69wsp9p2u, n69wspa2xf, n69wspa34c, n69wspa2d3 = (existing_data['funcs'], existing_data['classes'], existing_data['nodes'], existing_data['params'])
                n69wsp9p2u = n69wsp9p2u[(n69wsp9p2u['def_id'] == n69wspa2qn) | n69wsp9p2u['def_id'].str.startswith(n69wspa2qn + separator)]
                n69wspa2xf = n69wspa2xf[n69wspa2xf['def_id'].str.startswith(n69wspa2qn + separator)]
                n69wspa2d3 = n69wspa2d3[(n69wspa2d3['def_id'] == n69wspa2qn) | n69wspa2d3['def_id'].str.startswith(n69wspa2qn + separator)]
                n69wspa34c = n69wspa34c[(n69wspa34c['def_id'] == n69wspa2qn) | n69wspa34c['def_id'].str.startswith(n69wspa2qn + separator)]
                return (n69wspa34c, n69wsp9p2u, n69wspa2xf, n69wspa2d3)

            def b69wspa0yo(n69wsp9p2u, n69wspa2d3, n69wspa31m):
                n69wspa34x = copy.deepcopy(n69wsp9p2u)
                n69wspa34x = n69wspa34x[n69wspa34x['def_id'].str.startswith(n69wspa31m + '/') & ~n69wspa34x['def_id'].str[len(n69wspa31m) + 1:].str.contains('/') & ~n69wspa34x['def_id'].str[len(n69wspa31m) + 1:].str.contains('\\*') & ~n69wspa34x['def_id'].str[len(n69wspa31m) + 1:].str.contains('#') & ~n69wspa34x['def_id'].str[len(n69wspa31m) + 1:].str.contains('\\^')]
                n69wspa2qe = n69wspa2d3[n69wspa2d3['def_id'].str.startswith(n69wspa31m + '/') & ~n69wspa2d3['def_id'].str[len(n69wspa31m) + 1:].str.contains('/') & ~n69wspa2d3['def_id'].str[len(n69wspa31m) + 1:].str.contains('\\*') & ~n69wspa2d3['def_id'].str[len(n69wspa31m) + 1:].str.contains('#') & ~n69wspa2d3['def_id'].str[len(n69wspa31m) + 1:].str.contains('\\^')]
                n69wsp9ozv = n69wspa2qe[n69wspa2qe['ctx'] == 'input']
                n69wspa31a = n69wspa2qe[n69wspa2qe['ctx'] == 'return']

                def b69wspa0y4(n69wspa2u7, new_col_name):
                    n69wspa2xy = 'def_id'
                    if len(n69wspa2u7) == 0:
                        return pd.DataFrame({'def_id': [], new_col_name: []})
                    n69wspa2j2 = [n69wspa2id for n69wspa2id in n69wspa2u7.columns if n69wspa2id != n69wspa2xy]
                    n69wsp9oq8 = n69wspa2u7.groupby(n69wspa2xy).apply(lambda x: x[n69wspa2j2].to_dict('records') if new_col_name == 'inputs' else x[n69wspa2j2].iloc[0].to_dict()).reset_index(name=new_col_name)
                    return n69wsp9oq8
                n69wspa2sa = b69wspa0y4(n69wsp9ozv, 'inputs')
                n69wspa37c = b69wspa0y4(n69wspa31a, 'return')
                n69wspa2m1 = set(n69wsp9p2u['def_id'].tolist()) - set(n69wspa2sa['def_id'].tolist())
                n69wspa2qi = set(n69wsp9p2u['def_id'].tolist()) - set(n69wspa37c['def_id'].tolist())
                n69wspa2ur = [{'def_id': n69wspa2el, 'inputs': []} for n69wspa2el in n69wspa2m1]
                n69wspa2j9 = [{'def_id': n69wspa2el, 'return': {'name': 'return'}} for n69wspa2el in n69wspa2qi]
                n69wspa2ur = pd.DataFrame(n69wspa2ur)
                n69wspa2j9 = pd.DataFrame(n69wspa2j9)
                n69wspa2sa = pd.concat([n69wspa2sa, n69wspa2ur], ignore_index=True)
                n69wspa37c = pd.concat([n69wspa37c, n69wspa2j9], ignore_index=True)
                n69wspa34x = pd.merge(n69wspa34x, n69wspa2sa, on='def_id', how='left')
                n69wspa34x = pd.merge(n69wspa34x, n69wspa37c, on='def_id', how='left')
                return n69wspa34x
            nonlocal n69wspa2qg
            n69wspa31m = await self.b69x8ynnvl(n69wspa2qg, _skiplock=True, conn=conn) if x69xm5dtzx(n69wspa2qg) == 'cond' else n69wspa2qg
            if x69xm5dtzx(n69wspa2qg) in ('func', 'cond', 'class'):
                n69wspa2pz = n69wspa30v + ':' + n69wspa2jq
                if x69xm5dtzx(n69wspa2qg) in ('cond', 'class'):
                    n69wspa2qn = n69wspa2qg[:n69wspa2qg.rfind('^')]
                else:
                    n69wspa2qn = n69wspa2qg
                n69wspa36f = {'module_id': to_module_id(n69wspa2qg), 'save_id': idgen.generate(return_type='int'), 'scope_type': 'all', 'def_id': n69wspa2qn, 'undoed': 'F', 'data': {}, 'focus_scope': n69wspa2pz}
                if not existing_data:
                    n69wspa357 = f"def_id LIKE '{n69wspa2qn}^%' OR def_id = '{n69wspa2qn}'"
                    n69wsp9p2u = self.select('funcs', cond_sql=n69wspa357, _skiplock=True, conn=conn)
                    n69wspa2xf = self.select('classes', cond_sql=n69wspa357, _skiplock=True, conn=conn)
                    n69wspa2d3 = self.select('params', cond_sql=n69wspa357, _skiplock=True, conn=conn)
                    n69wspa34c = self.select('nodes', cond_sql=n69wspa357, _skiplock=True, conn=conn)
                    n69wsp9p2u, n69wspa2xf, n69wspa34c, n69wspa2d3 = await asyncio.gather(n69wsp9p2u, n69wspa2xf, n69wspa34c, n69wspa2d3)
                else:
                    n69wspa34c, n69wsp9p2u, n69wspa2xf, n69wspa2d3 = b69wspa0xv(existing_data, n69wspa2qn, '^')
                    assert n69wspa2qn in n69wsp9p2u['def_id'].tolist(), f"eh024 {n69wspa2qn}。有的funcs：{n69wsp9p2u['def_id'].tolist()}"
                assert n69wspa2pz.count(':') == 1
                if n69wspa2jq == 'dag':
                    n69wspa37v = n69wspa34c[n69wspa34c['def_id'] == n69wspa2qn]
                    if not len(n69wspa37v):
                        raise ValueError(f'ehdl001func id: {n69wspa30v}')
                    n69wspa2mh = await self.b69x8ynnta(n69wspa37v, 'dag', count_previews=True, conn=conn, _skiplock=True)
                elif n69wspa2jq == 'funcs':
                    n69wspa34x = b69wspa0yo(n69wsp9p2u, n69wspa2d3, n69wspa31m)
                    n69wspa2mh = await self.b69x8ynnta(n69wspa34x, 'funcs', count_previews=True, conn=conn, _skiplock=True)
                elif n69wspa2jq == 'classes':
                    n69wspa37v = n69wspa2xf[n69wspa2xf['def_id'].str.startswith(n69wspa2qg + '*') & ~n69wspa2xf['def_id'].str[len(n69wspa2qg) + 1:].str.contains('/') & ~n69wspa2xf['def_id'].str[len(n69wspa2qg) + 1:].str.contains('\\*') & ~n69wspa2xf['def_id'].str[len(n69wspa2qg) + 1:].str.contains('#') & ~n69wspa2xf['def_id'].str[len(n69wspa2qg) + 1:].str.contains('\\^')]
                    n69wspa2mh = await self.b69x8ynnta(n69wspa37v, 'classes', count_previews=True, conn=conn, _skiplock=True)
                n69wspa36f['data']['db'] = {'nodes': n69wspa34c.to_dict(orient='records'), 'funcs': n69wsp9p2u.to_dict(orient='records'), 'classes': n69wspa2xf.to_dict(orient='records'), 'params': n69wspa2d3.to_dict(orient='records')}
                assert n69wspa2mh is not None
                n69wspa36f['data']['app'] = n69wspa2mh
                return n69wspa36f
            elif x69xm5dtzx(n69wspa2qg) == 'folder':
                print(inspect.stack())
                n69wspa36f = {'module_id': n69wspa2qg, 'save_id': idgen.generate(return_type='int'), 'scope_type': 'funcs', 'def_id': n69wspa2qg, 'undoed': 'F', 'data': {}, 'focus_scope': None}
                if not existing_data:
                    n69wspa2p1 = f"def_id LIKE '{n69wspa2qg}/%'"
                    n69wspa2n0 = self.select('funcs', cond_sql=n69wspa2p1, _skiplock=True, conn=conn)
                    n69wspa2jt = self.select('params', cond_sql=n69wspa2p1, _skiplock=True, conn=conn)
                    n69wsp9omx = self.select('nodes', cond_sql=n69wspa2p1, _skiplock=True, conn=conn)
                    n69wspa2uj = self.select('classes', cond_sql=n69wspa2p1, _skiplock=True, conn=conn)
                    n69wspa2n0, n69wspa2jt, n69wsp9omx, n69wspa2uj = await asyncio.gather(n69wspa2n0, n69wspa2jt, n69wsp9omx, n69wspa2uj)
                else:
                    n69wsp9omx, n69wspa2n0, n69wspa2uj, n69wspa2jt = b69wspa0xv(existing_data, n69wspa2qg, '/')
                if n69wspa2jq == 'funcs':
                    n69wspa34x = b69wspa0yo(n69wspa2n0, n69wspa2jt, n69wspa31m)
                    n69wspa2mh = await self.b69x8ynnta(n69wspa34x, 'funcs', count_previews=True, conn=conn, _skiplock=True)
                else:
                    raise
                n69wspa36f['data']['db'] = {'nodes': n69wsp9omx.to_dict(orient='records'), 'funcs': n69wspa2n0.to_dict(orient='records'), 'classes': n69wspa2uj.to_dict(orient='records'), 'params': n69wspa2jt.to_dict(orient='records')}
                n69wspa36f['data']['app'] = n69wspa2mh
                return n69wspa36f
            else:
                raise ValueError(n69wspa2qg)
        n69wsp9onl = await self._batch_read([b69x8ynnsr], _skiplock=_skiplock, conn=conn)
        return n69wsp9onl[0]

    def b69wspa0xr(self, n69wspa38m):
        self.x69xm5dtzr.add(n69wspa38m)
        if n69wspa38m in self.x69xm5dtzs:
            del self.x69xm5dtzs[n69wspa38m]

    async def b69x8ynnvg(self, n69wspa2qg, n69wspa2jq, debugtag='(dc06)', existing_data=None, del_undoed=True, conn=None, _skiplock=False):
        if del_undoed:
            self.b69wspa0yb(to_module_id(n69wspa2qg))
        n69wspa36f = await self.b69x8ynntb(n69wspa2qg, n69wspa2jq, debugtag=debugtag, existing_data=existing_data, conn=conn, _skiplock=_skiplock)
        self.n69wspa36f.json().set(f"undo:{n69wspa36f['save_id']}", '$', n69wspa36f)
        await self.b69x8ynntp(n69wspa36f['data']['db']['funcs'], conn=conn, _skiplock=_skiplock)
        n69wspa2vv = n69wspa36f['def_id'].split('^')[0]
        self.b69wspa0xr(n69wspa2vv)
        return n69wspa36f

    async def b69x8ynnst(self, n69wspa2qg, n69wspa2jq, conn=None, _skiplock=False):
        if x69xm5dtzx(n69wspa2qg.split(':')[0]) == 'folder':
            return
        n69wspa31i = await self.b69x8ynntb(n69wspa2qg, n69wspa2jq, debugtag='(dc07)', conn=conn, _skiplock=_skiplock)
        self.n69wspa31i = n69wspa31i

    async def b69x8ynnun(self, n69wspa2qg):
        if x69xm5dtzx(n69wspa2qg.split(':')[0]) == 'folder':
            return
        try:
            if n69wspa2qg.split(':')[0].startswith(self.n69wspa31i.get('def_id', 'WHATS!THE!FUCK') + '^') or n69wspa2qg.split(':')[0] == self.n69wspa31i.get('def_id', 'WHATS!THE!FUCK'):
                self.n69wspa36f.json().set(f"undo:{self.n69wspa31i['save_id']}", '$', self.n69wspa31i)
            else:
                pass
        except Exception as e:
            traceback.print_exc()
        self.n69wspa31i = {}

    async def b69x8ynnv2(self, n69wspa2qg, n69wspa2jq, conn=None, _skiplock=False):
        n69wspa30v = n69wspa2qg
        if n69wspa2jq == 'dag':
            assert x69xm5dtzx(n69wspa2qg) in ('func', 'folder')
        elif n69wspa2jq == 'funcs':
            assert x69xm5dtzx(n69wspa2qg) in ('cond', 'class', 'folder')
        elif n69wspa2jq == 'classes':
            assert x69xm5dtzx(n69wspa2qg) in 'cond'

        async def b69x8ynnsr(conn):
            nonlocal n69wspa2qg
            if x69xm5dtzx(n69wspa2qg) == 'cond':
                n69wspa2qg = await self.b69x8ynnvl(n69wspa2qg, conn=conn, _skiplock=True)
            n69wspa36f = {'module_id': to_module_id(n69wspa2qg), 'save_id': idgen.generate(return_type='int'), 'scope_type': 'all' if x69xm5dtzx(n69wspa2qg) != 'class' else 'funcs', 'def_id': n69wspa30v, 'undoed': 'F', 'data': {}, 'focus_scope': None if x69xm5dtzx(n69wspa2qg) == 'class' else n69wspa30v + ':' + n69wspa2jq}
            if n69wspa2jq == 'dag':
                n69wspa33n = f"def_id = '{n69wspa2qg}'"
                n69wspa357 = f"\n                def_id LIKE '{n69wspa2qg}^%' AND NOT def_id LIKE '{n69wspa2qg}^1#_/%' AND NOT def_id LIKE '{n69wspa2qg}^1#_*%' \n                "
                n69wsp9p2u = self.select('funcs', cond_sql=n69wspa357, targets=list(n69wspa2ts.keys()), _skiplock=True, conn=conn)
                n69wspa2xf = self.select('classes', cond_sql=n69wspa357, targets=list(n69wspa2dn.keys()), _skiplock=True, conn=conn)
                n69wspa2y6 = self.select('nodes', cond_sql=n69wspa33n, targets=list(n69wspa2ha.keys()), _skiplock=True, conn=conn)
                n69wspa2d3 = self.select('params', cond_sql=n69wspa357, targets=list(n69wspa356.keys()), _skiplock=True, conn=conn)
                n69wspa34c = self.select('nodes', cond_sql=n69wspa357, targets=list(n69wspa2ha.keys()), _skiplock=True, conn=conn)
                n69wsp9p2u, n69wspa2xf, n69wspa2y6, n69wspa2d3, n69wspa34c = await asyncio.gather(n69wsp9p2u, n69wspa2xf, n69wspa2y6, n69wspa2d3, n69wspa34c)
                n69wspa31i = copy.deepcopy(n69wspa36f)
                n69wspa31i['data']['db'] = {'nodes': n69wspa2y6.to_dict(orient='records') + n69wspa34c.to_dict(orient='records'), 'funcs': n69wsp9p2u.to_dict(orient='records'), 'classes': n69wspa2xf.to_dict(orient='records'), 'params': n69wspa2d3.to_dict(orient='records')}
                n69wspa31i['data']['app'] = await self.b69x8ynnta(n69wspa2y6, 'dag', count_previews=True, conn=conn, _skiplock=True)
            elif n69wspa2jq == 'funcs':
                n69wspa2p1 = f"def_id LIKE '{n69wspa2qg}/%'"
                n69wspa329 = self.select('funcs', cond_sql=n69wspa2p1, _skiplock=True, conn=conn)
                n69wspa2qh = self.select('params', cond_sql=n69wspa2p1, _skiplock=True, conn=conn)
                n69wspa2no = self.select('nodes', cond_sql=n69wspa2p1, _skiplock=True, conn=conn)
                n69wspa2vt = self.select('classes', cond_sql=n69wspa2p1, _skiplock=True, conn=conn)

                async def b69x8ynnve():
                    n69wspa2zv = await self.b69x8ynnu2(n69wspa2qg, conn=conn, _skiplock=True)
                    n69wspa2zv = await self.b69x8ynnta(n69wspa2zv, 'funcs', count_previews=True, conn=conn, _skiplock=True)
                    return n69wspa2zv
                n69wspa2zv, n69wspa329, n69wspa2qh, n69wspa2no, n69wspa2vt = await asyncio.gather(b69x8ynnve(), n69wspa329, n69wspa2qh, n69wspa2no, n69wspa2vt)
                n69wspa31i = copy.deepcopy(n69wspa36f)
                n69wspa31i['data']['db'] = {'nodes': n69wspa2no.to_dict(orient='records'), 'funcs': n69wspa329.to_dict(orient='records'), 'classes': n69wspa2vt.to_dict(orient='records'), 'params': n69wspa2qh.to_dict(orient='records')}
                assert n69wspa2zv is not None
                n69wspa31i['data']['app'] = n69wspa2zv
            elif n69wspa2jq == 'classes':
                n69wspa2p1 = f"def_id LIKE '{n69wspa2qg}*%'"
                n69wspa329 = self.select('funcs', cond_sql=n69wspa2p1, _skiplock=True, conn=conn)
                n69wspa2qh = self.select('params', cond_sql=n69wspa2p1, _skiplock=True, conn=conn)
                n69wspa2no = self.select('nodes', cond_sql=n69wspa2p1, _skiplock=True, conn=conn)
                n69wspa2vt = self.select('classes', cond_sql=n69wspa2p1, _skiplock=True, conn=conn)

                async def b69x8ynntn():
                    n69wspa2vs = await self.b69x8ynntl(n69wspa2qg, conn=conn, _skiplock=True)
                    n69wspa2vs = await self.b69x8ynnta(n69wspa2vs, 'classes', count_previews=True, conn=conn, _skiplock=True)
                    return n69wspa2vs
                n69wspa2vs, n69wspa329, n69wspa2qh, n69wspa2no, n69wspa2vt = await asyncio.gather(b69x8ynntn(), n69wspa329, n69wspa2qh, n69wspa2no, n69wspa2vt)
                n69wspa31i = copy.deepcopy(n69wspa36f)
                n69wspa31i['data']['db'] = {'nodes': n69wspa2no.to_dict(orient='records'), 'funcs': n69wspa329.to_dict(orient='records'), 'classes': n69wspa2vt.to_dict(orient='records'), 'params': n69wspa2qh.to_dict(orient='records')}
                assert not n69wspa2vs is None
                n69wspa31i['data']['app'] = n69wspa2vs
            self.n69wspa31i = n69wspa31i
        await self._batch_read([b69x8ynnsr], _skiplock=_skiplock, conn=conn)

    async def b69x8ynntx(self, n69wspa38m, target, conn=None, _skiplock=False):
        assert x69xm5dtzx(n69wspa38m) == 'func'
        assert not '^' in n69wspa38m and n69wspa38m.count('/') == 1
        assert target in ('desc', 'imports')
        n69wspa35g = target
        target = {'desc': 'doc', 'imports': 'imports_code'}[target]
        n69wspa357 = f"def_id = '{n69wspa38m}'"
        n69wspa2kr = await self.select('funcs', cond_sql=n69wspa357, targets=['def_id', target], _skiplock=_skiplock, conn=conn)
        if len(n69wspa2kr) == 0:
            raise ValueError(f'module {n69wspa38m} does not exist')
        if len(n69wspa2kr) > 1:
            pass
        return n69wspa2kr.to_dict(orient='records')[0][target]

    async def b69x8ynnvn(self, n69wspa38m, target, n69wsp9p72, conn=None, _skiplock=False):
        assert x69xm5dtzx(n69wspa38m) == 'func'
        assert not '^' in n69wspa38m and n69wspa38m.count('/') == 1
        assert target in ('desc', 'imports')
        if target == 'imports':
            b69wsp9mo8(n69wsp9p72, ['Import', 'ImportFrom'])
        target = {'desc': 'doc', 'imports': 'imports_code'}[target]
        n69wspa32l = pd.DataFrame([{'def_id': n69wspa38m, target: n69wsp9p72}])

        async def b69x8ynnu4(conn):
            n69wspa2dx = self.upsert('funcs', n69wspa32l, skipna=True, force_update=4, primekeys=['def_id'], _skiplock=True, conn=conn)
            n69wspa36i = self.b69x8ynntp(n69wspa32l, conn=conn, _skiplock=True) if target == 'imports_code' else aidle()
            await asyncio.gather(n69wspa2dx, n69wspa36i)
            await self.b69x8ynnvg(n69wspa38m, 'dag', debugtag='(dc08)', conn=conn, _skiplock=True)
        await self._batch_write([b69x8ynnu4], _skiplock=_skiplock, conn=conn)

    async def b69x8ynnvj(self, n69wspa38m, rootfilter='', conn=None, _skiplock=False):
        n69wspa38m = n69wspa38m.strip().strip('>').strip('/').strip()
        assert '/' in n69wspa38m
        n69wspa33t = n69wspa38m.split('/')
        n69wspa38m = '>'.join(n69wspa33t[:-1]) + '/' + n69wspa33t[-1]
        assert x69xm5dtzx(n69wspa38m) == 'func', f"invalid module id: {n69wspa38m}. Module_id must contain EXACTLY one '/', and nested folders are connected with '>', like 'path>to>folder/module_name'."
        assert not '^' in n69wspa38m and n69wspa38m.count('/') == 1, f"module_id must contain EXACTLY one '/', and nested folders are connected with '>', like 'path>to>folder/module_name'."
        n69wspa322, n69wspa38w = n69wspa38m.split('/')
        assert n69wspa322 == n69wspa322.strip(), 'Heading and trailing spaces are not allowed'
        assert n69wspa38w == n69wspa38w.strip(), 'Heading and trailing spaces are not allowed'
        assert x69xm5dtzx(n69wspa322) == 'folder' and (not ' ' in n69wspa322), f'invalid module id (invalid dir): {n69wspa38m}'
        assert x69xm5dtzx(n69wspa38w) == 'folder' and (not ' ' in n69wspa38w) and (not n69wspa38w[0].isdigit()), f'invalid module id (invalid module name): {n69wspa38m}'
        assert not n69wspa38m.strip().startswith('/'), f"Module id must have a directory name before the '/'"
        assert is_valid_varname(n69wspa38w), f'invalid module id (invalid module name): {n69wspa38m}'
        n65d20cda3 = idgen.generate('f')
        n69wspa38n = []

        async def b69x8ynnuq(conn):
            nonlocal n69wspa38n
            n69wspa30y = f"\n            def_id = '{n69wspa38m}' OR def_id LIKE '{n69wspa38m.replace('/', '>')}>%' OR def_id LIKE '{n69wspa38m.replace('/', '>')}/%'\n            "
            n69wspa2zb = await self.select('funcs', cond_sql=n69wspa30y, targets=['def_id'], conn=conn, _skiplock=True)
            if len(n69wspa2zb) > 0:
                raise ValueError(f'The module {n69wspa38m} already exists.')
            n69wspa32l = pd.DataFrame([{'uid': n65d20cda3, 'def_id': n69wspa38m, 'imports_code': '', 'is_async': 0, 'xpos': 0, 'ypos': 0, 'doc': ''}])
            n69wspa2yh = gen_base36_id()
            n69wspa2sy = pd.DataFrame([{'uid': n69wspa2yh, 'hid': '1.0', 'branch': '_', 'data_providers': [], 'node_type': 'start', 'pres': [], 'nexts': [], 'xpos': 0, 'ypos': 0, 'def_id': n69wspa38m}])
            n69wspa31p = self.upsert('funcs', n69wspa32l, _skiplock=True, conn=conn)
            n69wspa2xw = self.upsert('nodes', n69wspa2sy, _skiplock=True, conn=conn)
            await asyncio.gather(n69wspa31p, n69wspa2xw)
        await self._batch_write([b69x8ynnuq], conn=conn, _skiplock=_skiplock)

    async def b69x8ynnt0(self, n69wspa38m, rootfilter='', conn=None, _skiplock=False):
        assert x69xm5dtzx(n69wspa38m) == 'func'
        assert not '^' in n69wspa38m and n69wspa38m.count('/') == 1
        n69wspa38n = []

        async def b69x8ynnuq(conn):
            nonlocal n69wspa38n
            await self.b69x8ynnuu(n69wspa38m, conn=conn, _skiplock=True)
        await self._batch_write([b69x8ynnuq], conn=conn, _skiplock=_skiplock)

    async def b69x8ynnw3(self, old_module_id, n69wspa2o4, rootfilter='', conn=None, _skiplock=False):
        n69wspa2o4 = n69wspa2o4.strip().strip('>').strip('/').strip()
        assert '/' in n69wspa2o4
        n69wspa33t = n69wspa2o4.split('/')
        n69wspa2o4 = '>'.join(n69wspa33t[:-1]) + '/' + n69wspa33t[-1]
        assert x69xm5dtzx(old_module_id) == 'func', old_module_id
        assert not '^' in old_module_id and old_module_id.count('/') == 1
        assert x69xm5dtzx(n69wspa2o4) == 'func', f'invalid module id: {n69wspa2o4}'
        assert n69wspa2o4.count('/') == 1, f'invalid module id {n69wspa2o4}, must contain one "/"'
        assert not n69wspa2o4.strip().startswith('/'), f"Module id must have a directory name before the '/'"
        n69wspa322, n69wspa38w = n69wspa2o4.split('/')
        assert n69wspa322 == n69wspa322.strip(), 'Heading and trailing spaces are not allowed'
        assert n69wspa38w == n69wspa38w.strip(), 'Heading and trailing spaces are not allowed'
        assert x69xm5dtzx(n69wspa322) == 'folder' and (not ' ' in n69wspa322), f'invalid module id (invalid dir): {n69wspa2o4}'
        assert x69xm5dtzx(n69wspa38w) == 'folder' and (not ' ' in n69wspa38w) and (not n69wspa38w[0].isdigit()), f'invalid module id (invalid module name): {n69wspa2o4}'
        assert is_valid_varname(n69wspa38w), f'invalid module id (invalid module name): {n69wspa2o4}'
        n69wspa38n = []

        async def b69x8ynnuq(conn):
            nonlocal n69wspa38n
            n69wspa30y = f"\n            def_id = '{n69wspa2o4}' OR def_id LIKE '{n69wspa2o4.replace('/', '>')}>%' OR def_id LIKE '{n69wspa2o4.replace('/', '>')}/%'\n            "
            n69wspa2zb = await self.select('funcs', cond_sql=n69wspa30y, targets=['def_id'], conn=conn, _skiplock=True)
            if len(n69wspa2zb) > 0:
                raise ValueError(f'The module {n69wspa2o4} already exists.')
            n69wspa35t = []
            for tablename in ['funcs', 'classes', 'nodes']:
                n69wspa2wm = f"\n                UPDATE {tablename}\n                SET def_id = CONCAT('{n69wspa2o4}', SUBSTRING(def_id, LENGTH('{old_module_id}') + 1))\n                WHERE def_id LIKE '{old_module_id}^%' OR def_id = '{old_module_id}'\n                "
                n69wspa2zw = conn.execute(text(n69wspa2wm))
                n69wspa35t.append(n69wspa2zw)
            await asyncio.gather(n69wspa35t[0], n69wspa35t[1])
            await n69wspa35t[2]
        await self._batch_write([b69x8ynnuq], conn=conn, _skiplock=_skiplock)

    async def b69x8ynnu6(self, n69wspa38m, xx, conn=None, _skiplock=False):
        n69wspa2mh = None
        n69wspa2ua = None
        n69wsp9oye = None
        n69yxx8iwg = None
        n69wspa38m = to_module_id(n69wspa38m)

        async def b69x8ynnu4(conn):
            n69wspa2v1 = time.time()
            nonlocal n69wspa2mh, n69wspa2ua, n69wsp9oye, n69yxx8iwg
            n69wspa35c = None
            n69wspa35i = None
            n69wspa2pc = True
            n69wspa2h0 = n69wspa38m.replace('>', '\\>').replace('/', '\\/')
            n69wspa340 = time.time()
            if xx == 'un':
                n69wspa357 = f'@module_id:{{{n69wspa2h0}}} @undoed:{{F}}'
                n69wspa2xr = self.n69wspa36f.execute_command('FT.SEARCH', f'idx:undo', n69wspa357, 'SORTBY', 'save_id', 'DESC', 'VERBATIM', 'LIMIT', '0', '2', 'RETURN', '5', 'def_id', 'save_id', 'scope_type', '$.data', '$.focus_scope')
                if len(n69wspa2xr) < 5:
                    n69wspa2ua = '<EMPTY>'
                    n69wspa2mh = None
                    return

                def b69wspa0y3():
                    self.n69wspa36f.json().set(n69wspa2xr[1], '$.undoed', 'T')
            elif xx == 're':

                async def b69x8ynnuk():
                    n69wspa37y = f'@module_id:{{{n69wspa2h0}}} @undoed:{{F}}'
                    n69wspa2d6 = self.n69wspa36f.execute_command('FT.SEARCH', f'idx:undo', n69wspa37y, 'SORTBY', 'save_id', 'DESC', 'VERBATIM', 'LIMIT', '0', '1', 'RETURN', '5', 'def_id', 'save_id', 'scope_type', '$.data', '$.focus_scope')
                    return n69wspa2d6

                async def b69x8ynnub():
                    n69wspa2gk = f'@module_id:{{{n69wspa2h0}}} @undoed:{{T}}'
                    n69wspa2fc = self.n69wspa36f.execute_command('FT.SEARCH', f'idx:undo', n69wspa2gk, 'SORTBY', 'save_id', 'ASC', 'VERBATIM', 'LIMIT', '0', '1', 'RETURN', '5', 'def_id', 'save_id', 'scope_type', '$.data', '$.focus_scope')
                    return n69wspa2fc
                n69wspa2d6, n69wspa2fc = await asyncio.gather(b69x8ynnuk(), b69x8ynnub())
                if len(n69wspa2d6) != 3:
                    n69wspa2ua = '<EMPTY>'
                    n69wspa2mh = None
                    return
                if len(n69wspa2fc) != 3:
                    n69wspa2ua = '<EMPTY>'
                    n69wspa2mh = None
                    return
                n69wspa2xr = [None] + n69wspa2d6[1:] + n69wspa2fc[1:]

                def b69wspa0y3():
                    self.n69wspa36f.json().set(n69wspa2fc[1], '$.undoed', 'F')
            n69wspa2ug = time.time()
            if n69wspa2xr[2][3] == n69wspa2xr[4][3] and n69wspa2xr[2][5] == n69wspa2xr[4][5]:
                n69wspa35c = json.loads(n69wspa2xr[2][7])
            else:
                n69wspa2pc = False
            n69wspa2wq = n69wspa2xr[4][3]
            if not n69wspa2xr[4][9]:
                n69wspa2ua = n69wspa2xr[4][3] + ':' + (n69wspa2xr[4][5] if n69wspa2xr[4][5] != 'all' else 'dag')
            else:
                assert n69wspa2xr[4][5] == 'all'
                n69wspa2ua = n69wspa2xr[4][9]
            if n69wspa2xr[4][5] == 'all':
                if '*' in n69wspa2wq or '/' in n69wspa2wq:
                    n69wspa2xk = max([ii for ii in [n69wspa2wq.rfind('/'), n69wspa2wq.rfind('*')] if ii != -1])
                    n69wsp9oye = n69wspa2wq[:n69wspa2xk] + ':funcs'
            n69wspa2jq = n69wspa2xr[4][5]
            n69wspa35i = json.loads(n69wspa2xr[4][7])
            n69wspa2mh = n69wspa35i['app']
            assert n69wspa2mh is not None

            def b69wspa0yk(lod1, lod2, primekey, scopetype):

                def b69wspa0yn(n1, n2):
                    if n1 == n2:
                        return True
                    n69wspa2rb = []
                    if n69wspa2jq in ('funcs', 'classes', 'params'):
                        n69wspa2rb = ['code']
                    elif n69wspa2jq in 'nodes':
                        n69wspa2rb = ['data_providers']
                        if n1.get('node_type') != 'code' and n2.get('node_type') != 'code':
                            n69wspa2rb.append('code')
                    for k, v in n1.items():
                        if k in n69wspa2rb:
                            continue
                        if n2.get(k) != v:
                            return False
                    return True
                n69wspa33y = []
                n69wspa2ec = set()
                n69wsp9oqx = set()
                n69wspa38e = {l[primekey]: l for l in lod1}
                n69wspa385 = set(n69wspa38e.keys())
                n69wspa32b = {l[primekey]: l for l in lod2}
                for k2 in n69wspa32b:
                    if not k2 in n69wspa38e:
                        n69wspa33y.append(n69wspa32b[k2])
                        n69wspa2ec.add(k2)
                    else:
                        if b69wspa0yn(n69wspa38e[k2], n69wspa32b[k2]):
                            pass
                        else:
                            n69wspa33y.append(n69wspa32b[k2])
                            n69wspa2ec.add(k2)
                        n69wspa385.remove(k2)
                n69wsp9oqx = n69wsp9oqx | n69wspa385
                return (n69wspa33y, n69wspa2ec, n69wsp9oqx)
            if n69wspa2pc:
                primekeys = {'nodes': 'uid', 'funcs': 'uid', 'classes': 'uid', 'params': 'def_id'}
                n69wspa2k5 = {}
                n69wspa2ds = {}
                n69wspa2sw = {}
                n69wspa2hv = []
                n69wspa36n = []
                n69wspa35x = []
                n69wspa340 = time.time()
                n69wspa2dx = None
                for key in ['nodes', 'funcs', 'classes', 'params']:
                    myups, myupids, mydels = b69wspa0yk(n69wspa35c['db'][key], n69wspa35i['db'][key], primekeys[key], key)
                    n69wspa2k5[key] = myups
                    n69wspa2ds[key] = myupids
                    n69wspa2sw[key] = mydels
                    n69wspa2gr = self.delete(key, conds=[{primekeys[key]: did} for did in mydels], conn=conn, _skiplock=True) if mydels else aidle()
                    n69wspa2zw = self.upsert(key, pd.DataFrame(myups), skipna=False, conn=conn, _skiplock=True)
                    n69wspa2hv.append(n69wspa2gr)
                    if key == 'funcs':
                        n69wspa2dx = pd.DataFrame(myups)
                    if key in ('classes', 'funcs'):
                        n69wspa36n.append(n69wspa2zw)
                    else:
                        n69wspa35x.append(n69wspa2zw)
                n69wspa36i = self.b69x8ynntp(n69wspa2dx, conn=conn, _skiplock=True)

                async def b69x8ynnw0():
                    await asyncio.gather(*n69wspa2hv)
                    await asyncio.gather(*n69wspa36n)
                    await asyncio.gather(*n69wspa35x)
                await asyncio.gather(b69x8ynnw0(), n69wspa36i)
                focus_defid, n69wspa2pz = n69wspa2ua.split(':')
                if n69wspa2pz == 'dag':
                    n69wspa5ld = n69wspa2k5['nodes']
                    n69wspa5ld = [n69wspa33x for n69wspa33x in n69wspa5ld if n69wspa33x['def_id'] == focus_defid and n69wspa33x['node_type'] != 'start']
                    if len(n69wspa5ld) == 1:
                        n69yxx8iwg = n69wspa5ld[0]['uid']
                elif n69wspa2pz in ('funcs', 'classes'):
                    n69wspa5ld = n69wspa2k5[n69wspa2pz]
                    n69yxx8iwk = n69wspa2ds[n69wspa2pz]
                    n69wspa5ld = [n69wspa33x for n69wspa33x in n69wspa5ld if n69wspa33x['uid'] in n69yxx8iwk]
                    if n69wspa5ld:
                        n69yxx8ivv = min(range(len(n69wspa5ld)), key=lambda i: len(n69wspa5ld[i]['def_id']))
                        n69yxx8iwg = n69wspa5ld[n69yxx8ivv]['uid']
                n69wspa2ug = time.time()
            else:
                n69wspa2li = {k: pd.DataFrame(v) for k, v in n69wspa35i['db'].items()}
                if n69wspa2jq in ('all', 'dag'):
                    assert x69xm5dtzx(n69wspa2wq) in ('folder', 'func')
                    n69wspa2cb = [self.b69x8ynnuu(n69wspa2wq, n69wspa30o=n69wspa2jq == 'dag', skipshell=n69wspa2jq == 'all', conn=conn, _skiplock=True)]
                    n69wspa2x0 = [self.upsert(k, v, skipna=False, conn=conn, _skiplock=True) for k, v in n69wspa2li.items() if k in ('classes', 'funcs')]
                    n69wspa378 = [self.upsert(k, v, skipna=False, conn=conn, _skiplock=True) for k, v in n69wspa2li.items() if not k in ('classes', 'funcs')]
                else:
                    n69wspa2rr = n69wspa2wq if x69xm5dtzx(n69wspa2wq) in ('folder', 'class') else await self.b69x8ynnvl(n69wspa2wq, _skiplock=True, conn=conn)
                    if n69wspa2jq == 'funcs':
                        n69wsp9p3w = '/'
                    elif n69wspa2jq == 'classes':
                        n69wsp9p3w = '^'
                    else:
                        raise
                    n69wspa2o5 = f"\n                    def_id LIKE '{n69wspa2rr}{n69wsp9p3w}%'\n                    "
                    n69wspa2cb = [self.delete(tb, cond_sql=n69wspa2o5, conn=conn, _skiplock=True) for tb in ['funcs', 'classes', 'nodes']]
                    n69wspa2x0 = [self.upsert(k, v, skipna=False, conn=conn, _skiplock=True) for k, v in n69wspa2li.items() if k in ('classes', 'funcs')]
                    n69wspa378 = [self.upsert(k, v, skipna=False, conn=conn, _skiplock=True) for k, v in n69wspa2li.items() if not k in ('classes', 'funcs')]
                n69wspa36i = self.b69x8ynntp(n69wspa2li['funcs'], conn=conn, _skiplock=True)

                async def b69x8ynnw0():
                    await asyncio.gather(*n69wspa2cb)
                    await asyncio.gather(*n69wspa2x0)
                    await asyncio.gather(*n69wspa378)
                await asyncio.gather(b69x8ynnw0(), n69wspa36i)
            b69wspa0y3()
            n69wspa2jb = time.time()
        n69wspa340 = time.time()
        await self._batch_write([b69x8ynnu4], _skiplock=_skiplock, conn=conn)
        n69wspa2ug = time.time()
        assert n69wspa2ua is not None
        if n69wspa2ua != '<EMPTY>':
            assert n69wspa2mh is not None
        return (n69wspa2ua, n69wspa2mh, n69wsp9oye, n69yxx8iwg)

    async def b69x8ynnv1(self, n69wspa2qn, n69wspa2wc=None, conn=None, _skiplock=False):
        if x69xm5dtzx(n69wspa2qn) == 'folder':
            return []
        assert x69xm5dtzx(n69wspa2qn) == 'func'
        if n69wspa2wc is None:
            n69wspa2wc = await self.b69x8ynnv5(n69wspa2qn, conn=conn, _skiplock=_skiplock)
        n69wspa2wc = [n69wspa2w8 for n69wspa2w8 in n69wspa2wc if not n69wspa2w8.startswith(n69wspa2qn + '^') and (not x69xm5dtzx(n69wspa2w8) == 'folder')]
        n69wspa330 = []
        for n69wspa2w8 in n69wspa2wc:
            assert '#' in n69wspa2w8 and '^' in n69wspa2w8
            n69wspa2ox = n69wspa2w8.rfind('#')
            n69wspa2vg = n69wspa2w8.rfind('^')
            assert n69wspa2vg < n69wspa2ox
            left = n69wspa2w8[:n69wspa2vg]
            n69wspa2hh = n69wspa2w8[n69wspa2vg + 1:n69wspa2ox]
            right = n69wspa2w8[n69wspa2ox + 1:].lower()
            n69wspa330.append((left, n69wspa2hh, right))
        n69wspa357 = "\n        def_id = '{def_id}' AND\n        (SUBSTRING(hid, LENGTH('{hid}.') + 1) NOT LIKE '%.%') AND\n        branch = {branch}\n        "
        n69wsp9omx = [self.select('nodes', cond_sql=n69wspa357.format(def_id=n69wspa2wq, hid=n69wsp9p51, branch=f"'{n69wspa2w8}'" if not n69wspa2w8.isdigit() and (not n69wspa2w8 in ('true', 'false')) else n69wspa2w8), targets=['uid', 'hid', 'pres'], conn=conn, _skiplock=_skiplock) for n69wspa2wq, n69wsp9p51, n69wspa2w8 in n69wspa330]
        n69wsp9omx = await asyncio.gather(*n69wsp9omx)
        n69wspa2h3 = []
        for i, n69wsp9oya in enumerate(n69wsp9omx):
            if i < len(n69wsp9omx) - 1:
                if n69wspa330[i][0] == n69wspa330[i + 1][0]:
                    n69wsp9ow5 = n69wspa330[i + 1][1].split('.')[-1]
                    if not n69wsp9ow5 in n69wsp9oya['uid'].tolist():
                        pass
                else:
                    n69wsp9ow5 = None
            else:
                n69wsp9ow5 = None
            if not n69wsp9ow5 is None:
                n69wspa2zz = []
                n69wspa2qy = n69wsp9oya.to_dict(orient='records')
                n69wspa2qy = {n['uid']: n for n in n69wspa2qy}
                n69wspa2yh = n69wsp9oya[n69wsp9oya['hid'] == '1.0']['uid'].tolist()
                assert len(n69wspa2yh) <= 1, (n69wspa2yh, n69wsp9oya)
                n69wspa2yh = n69wspa2yh[0] if n69wspa2yh else '<EMPTY>'

                def b69wspa0y7(n65d20cda3):
                    nonlocal n69wspa2zz
                    if n65d20cda3 in n69wspa2zz:
                        return
                    n69wspa2zz.insert(0, n65d20cda3)
                    n69wsp9orp = n69wspa2qy[n65d20cda3]['pres']
                    if not n69wsp9orp:
                        return
                    n69wsp9orp = [p.split('.')[-1] if not p == '1.0' else n69wspa2yh for p in n69wsp9orp]
                    for p in n69wsp9orp:
                        b69wspa0y7(p)
                b69wspa0y7(n69wsp9ow5)
                n69wspa2h3 = n69wspa2h3 + n69wspa2zz
            else:
                n69wspa2h3 = n69wspa2h3 + n69wsp9oya[~n69wsp9oya['uid'].str.contains('-end')]['uid'].tolist()
        n69wspa311 = [self.select('vars', conds=[{'uid': n65d20cda3}], targets=['name', 'def_id', 'uid', 'type'], conn=conn, _skiplock=_skiplock) for n65d20cda3 in n69wspa2h3]
        n69wspa311 = await asyncio.gather(*n69wspa311)
        if not n69wspa311:
            return pd.DataFrame(columns=['name', 'def_id', 'uid', 'type'])
        n69wspa311 = pd.concat(n69wspa311, ignore_index=True)
        return n69wspa311

    async def b69x8ynnto(self, n69wspa2de, n69wspa2nm=[], external_class_ids=[], inject_imports=True, return_visible_brs=False, conn=None, _skiplock=False):
        assert '^' in n69wspa2de
        n69wspa2df = 0
        if not any([s in n69wspa2de.split('/')[-1] for s in ('#', '/', '-', '^', '.', '*')]):
            n69wspa2df = 1
        elif not any([s in n69wspa2de.split('#')[-1] for s in ('#', '/', '-', '^', '.', '*')]):
            n69wspa2df = 2
            n69wspa2de = n69wspa2de + '/_FAKE_FUNC'
        if n69wspa2df == 0:
            raise ValueError(f'eh005{n69wspa2de}')
        n69wspa2wc = await self.b69x8ynnv5(n69wspa2de, conn=conn, _skiplock=_skiplock)
        if n69wspa2df == 1:
            assert not n69wspa2de.endswith('^1#_')
            n69wspa2wc.append(n69wspa2de + '^1#_')
        n69wspa2lm = n69wspa2de.find('*')
        n69wspa2sg = n69wspa2de.find('/', n69wspa2de.find('/') + 1)
        n69wspa2hk = min([x for x in (n69wspa2lm, n69wspa2sg) if x >= 0])
        n69wspa2ja = n69wspa2de[:n69wspa2hk]
        assert n69wspa2ja in n69wspa2wc, f'eh025-1{n69wspa2de}, eh025-3{n69wspa2wc},eh025-4{n69wspa2ja}'
        assert x69xm5dtzx(n69wspa2wc[0]) == 'folder'
        assert n69wspa2ja.split('^')[0] + '^1#_' == n69wspa2wc[1], f'eh025-2{n69wspa2de}, eh025-3{n69wspa2wc},eh025-4{n69wspa2ja}'
        n69wspa31t = [self.b69x8ynnvv(did, 'func', conn=conn, _skiplock=_skiplock) for did in n69wspa2nm] + [self.b69x8ynnu2(did, conn=conn, _skiplock=_skiplock) for did in n69wspa2wc]
        n69wspa374 = [self.b69x8ynnvv(did, 'class', conn=conn, _skiplock=_skiplock) for did in external_class_ids] + [self.b69x8ynntl(did, conn=conn, _skiplock=_skiplock) for did in n69wspa2wc[1:]]
        n69wspa2tf = await asyncio.gather(asyncio.gather(*n69wspa31t), asyncio.gather(*n69wspa374))
        n69wsp9p3q, n69wsp9p6q = n69wspa2tf
        n69wsp9p3q.reverse()
        n69wsp9opz = []
        n69wsp9p60 = {}
        n69wspa2yn = ''
        for i in range(len(n69wsp9p3q)):
            for j in range(len(n69wsp9p3q[i])):
                n69wspa2ob = n69wsp9p3q[i].loc[j, 'def_id']
                assert '/' in n69wspa2ob and (not any([s in n69wspa2ob.split('/')[-1] for s in ('#', '/', '-', '^', '.', '*')]))
                n69wsp9opz.append((n69wspa2ob[n69wspa2ob.rfind('/') + 1:], n69wspa2ob))
                n69wsp9ozv = n69wsp9p3q[i].loc[j, 'inputs']
                n69wsp9orh, raw_args = b69wsp9mon(n69wsp9ozv)
                n69wsp9p60[n69wspa2ob] = raw_args
                if n69wspa2de.startswith(n69wspa2ob + '^'):
                    n69wspa2y3 = n69wsp9p3q[i].loc[j, 'imports_code'] or ''
                    n69wspa2yn = n69wspa2y3 + '\n' + n69wspa2yn
        n69wsp9oxq = []
        n69wspa30f = []
        for i in range(len(n69wsp9p6q)):
            for j in range(len(n69wsp9p6q[i])):
                n69wspa2ob = n69wsp9p6q[i].loc[j, 'def_id']
                assert '*' in n69wspa2ob and (not any([s in n69wspa2ob.split('*')[-1] for s in ('#', '/', '-', '^', '.', '*')]))
                n69wsp9oxq.append((n69wspa2ob[n69wspa2ob.rfind('*') + 1:], n69wspa2ob))
                n69wspa30f.append(self.b69x8ynnu2(n69wspa2ob, conn=conn, _skiplock=_skiplock))
        n69wspa2zc = await asyncio.gather(*n69wspa30f)
        for clf in n69wspa2zc:
            for j in range(len(clf)):
                n69wspa2ob = clf.loc[j, 'def_id']
                n69wsp9ozv = clf.loc[j, 'inputs']
                n69wsp9orh, raw_args = b69wsp9mon(n69wsp9ozv)
                n69wsp9p60[n69wspa2ob] = raw_args
        if not inject_imports:
            n69wsp9onl = [n69wsp9opz, n69wsp9oxq, n69wsp9p60, n69wspa2yn]
        else:
            n69wsp9on4 = b69wsp9mrs(b69wsp9mq1(n69wspa2yn), names_only=True)
            n69wsp9opz, n69wsp9oxq = b69wsp9mp9(n69wsp9on4, n69wsp9opz, n69wsp9oxq)
            n69wsp9onl = [n69wsp9opz, n69wsp9oxq, n69wsp9p60]
        if return_visible_brs:
            n69wsp9onl.append(n69wspa2wc)
        return n69wsp9onl

    async def b69x8ynnw4(self, n69wspa2yk, x69xm5dtzv=None, n69x75d5wx='', conn=None, _skiplock=False):

        async def b69x8ynnsv(conn):
            n69wspa2wc = x69xm5dtzv or await self.b69x8ynnv5(n69wspa2yk, conn=conn, _skiplock=True)
            n69wspa35m = [n69wspa2w8.rsplit('^', 1)[0] for n69wspa2w8 in n69wspa2wc if not x69xm5dtzx(n69wspa2w8) == 'folder']
            n69wspa35m.append(n69wspa2yk)
            n69wspa35m = list(set(n69wspa35m))
            n69wspa35m = [f for f in n69wspa35m if not x69xm5dtzx(f) == 'folder']
            n69wspa2lw = await self.select('funcs', conds=[{'def_id': did} for did in n69wspa35m], targets=['imports_code'], conn=conn, _skiplock=True)
            return (n69wspa2lw, n69wspa35m)
        n69wspa2lw = await self._batch_read([b69x8ynnsv], conn=conn, _skiplock=_skiplock)
        n69wspa2lw, n69wspa35m = n69wspa2lw[0]
        n69wspa2lw = n69wspa2lw['imports_code'].tolist()
        n69wspa2lw = '\n'.join(n69wspa2lw)
        if n69x75d5wx:
            n69wspa2lw = n69wspa2lw + '\n' + remove_common_indents(n69x75d5wx)
        _, _, n69wspa2lw = b69wsp9mrs(b69wsp9mq1(n69wspa2lw), expand=True)
        n69wspa2lw = list(set(n69wspa2lw.split('\n')))
        n69wspa2yn = [x for x in n69wspa2lw if x.startswith('import')]
        n69wspa2zx = [x for x in n69wspa2lw if x.startswith('from')]
        return {'imports': '\n'.join(n69wspa2yn), 'froms': '\n'.join(n69wspa2zx), 'visibility': n69wspa35m}

    async def b69x8ynnv5(self, n69wspa2yk, conn=None, _skiplock=False):
        n69wspa2y4 = []
        n69wspa2kk = []
        n69wspa33l = []
        n69wspa2if = x69xm5dtzx(n69wspa2yk) == 'node'
        if n69wspa2if:
            n69wspa2yk = n69wspa2yk + '#999'
        if n69wspa2yk.rfind('#') > n69wspa2yk.rfind('/'):
            n69wspa2lz = n69wspa2yk
            if n69wspa2yk.rfind('*') > n69wspa2yk.rfind('#'):
                n69wspa2lz = n69wspa2yk[:n69wspa2yk.rfind('*')]
            n69wspa2lz = n69wspa2lz + '/_FAKE_FUNC'
            n69wspa2yk = n69wspa2lz
        n69wspa33m = False
        for i in range(len(n69wspa2yk)):
            if n69wspa2yk[i] == '#':
                n69wspa2kk.append(i)
                n69wspa33m = True
            if n69wspa2yk[i] in ('*', '/'):
                if n69wspa33m:
                    n69wspa33l.append(i)
                n69wspa33m = False
        assert len(n69wspa2kk) == len(n69wspa33l), (n69wspa2kk, n69wspa33l)
        n69wspa36r = [n69wspa2yk[:sd] for sd in n69wspa33l]
        n69wspa2lt = [n69wspa2yk[:wd] for wd in n69wspa2kk]
        assert all(['^' in bid for bid in n69wspa2lt])
        n69wspa2lu = [(bid[:bid.rfind('^')], bid[bid.rfind('^') + 1:]) for bid in n69wspa2lt]
        n69wspa2mq = await asyncio.gather(*[self.b69x8ynnv6(*pair, conn=conn, _skiplock=_skiplock) for pair in n69wspa2lu])
        for i, apair in enumerate(n69wspa2lu):
            n69wspa2y4 = n69wspa2y4 + [apair[0] + childbr for childbr in n69wspa2mq[i]]
            n69wspa2y4.append(n69wspa36r[i])
        n69wspa322 = n69wspa2yk.split('/')[0]
        if not n69wspa322 in n69wspa2y4:
            n69wspa2y4.insert(0, n69wspa322)
        if n69wspa2if:
            n69wspa2y4 = n69wspa2y4[:-1]
        return n69wspa2y4

    async def b69x8ynnu7(self, my_branch_id, src_branch_id, conn=None, _skiplock=False):
        assert '#' in my_branch_id and '#' in src_branch_id
        assert not any([s in my_branch_id.split('#')[-1] for s in ('#', '/', '-', '^', '.', '*')])
        assert not any([s in src_branch_id.split('#')[-1] for s in ('#', '/', '-', '^', '.', '*')])
        if my_branch_id == src_branch_id:
            return True
        if my_branch_id.startswith(src_branch_id + '/') or my_branch_id.startswith(src_branch_id + '*'):
            return True
        n69wspa2wi = my_branch_id[:my_branch_id.rfind('#')]
        n69wspa2e4 = src_branch_id[:src_branch_id.rfind('#')]
        if not (n69wspa2wi.startswith(n69wspa2e4 + '.') or n69wspa2wi.startswith(n69wspa2e4 + '#') or n69wspa2wi == n69wspa2e4):
            return False
        if my_branch_id.startswith(n69wspa2e4 + '#'):
            if not my_branch_id.startswith(src_branch_id):
                return False
        n69wspa2qu = src_branch_id.count('#')
        n69wspa38f = find_nth(my_branch_id, '#', n69wspa2qu)
        assert n69wspa38f > 0
        n69wspa30d = my_branch_id[:n69wspa38f]
        assert '^' in n69wspa30d
        my_def_id, my_hid = (n69wspa30d[:n69wspa30d.rfind('^')], n69wspa30d[n69wspa30d.rfind('^') + 1:])
        assert my_def_id == src_branch_id[:src_branch_id.rfind('^')]
        n69wspa33c = []
        try:
            n69wspa33c = await self.b69x8ynnv6(my_def_id, my_hid, conn=conn, _skiplock=_skiplock)
        except Exception as ex:
            pass
        n69wspa2zd = src_branch_id[src_branch_id.rfind('^'):]
        if n69wspa2zd in n69wspa33c:
            return True
        return False

    async def b69x8ynnsw(self, n65d20cda3, run_id=None, start_block_dotted=False, conn=None, _skiplock=False):
        n69wspa2z8 = self.select('nodes', conds=[{'uid': n65d20cda3}], targets=['node_type'], conn=conn, _skiplock=_skiplock) if start_block_dotted else aidle(default=[])
        n69wspa2iy = self.select(table_name='vars', conds=[{'uid': n65d20cda3}], conn=conn, _skiplock=_skiplock)
        n69wspa2z8, n69wspa2iy = await asyncio.gather(n69wspa2z8, n69wspa2iy)
        if len(n69wspa2z8) > 0:
            if n69wspa2z8.loc[0, 'node_type'] == 'start':
                n69wspa2iy = n69wspa2iy[~n69wspa2iy['name'].str.contains('\\.')]
        n69wspa2iy = n69wspa2iy.drop(['value', 'def_id'], axis=1, errors='ignore').to_dict(orient='records')
        if not run_id:
            return n69wspa2iy
        n69wspa2hl = oclient.get_n_outputs(run_id, n65d20cda3.split('-end')[0], choice='vars')
        n69wspa2hl = [v['content'] for v in list(n69wspa2hl.values())]
        for var in n69wspa2iy:
            n69wspa2rq = table_lambda_get(n69wspa2hl, lambda x: x.get('name') == var['name'])
            if n69wspa2rq:
                var['type'] = n69wspa2rq[0]['type']
                var['repr'] = str(n69wspa2rq[0]['repr'])
        n69wspa2yt = [v['name'] for v in n69wspa2iy]
        n69wsp9oql = [{'name': v['name'], 'type': v['type'], 'repr': str(v['repr']), 'from': None} for v in n69wspa2hl if not v['name'] in n69wspa2yt]
        return n69wspa2iy + n69wsp9oql

    async def b69x8ynnv6(self, n69wspa2wq, n69wsp9p51, conn=None, _skiplock=False):
        if n69wsp9p51 == '1':
            return []
        if n69wsp9p51.count('.') == 1:
            return ['^1#_']
        n69wspa30h = n69wsp9p51.split('.')
        n69wspa2s5 = ['.'.join(n69wspa30h[:2])]
        n69wspa35t = []
        for i in range(3, len(n69wspa30h) + 1):
            n69wspa2ii = '.'.join(n69wspa30h[:i])
            n69wspa357 = f"def_id = '{n69wspa2wq}' AND hid = '{n69wspa2ii}'"
            n69wspa37h = self.select('nodes', cond_sql=n69wspa357, targets=['branch'], conn=conn, _skiplock=_skiplock)
            n69wspa35t.append(n69wspa37h)
            n69wspa2s5.append(n69wspa2ii)
        n69wspa2me = await asyncio.gather(*n69wspa35t)
        n69wspa2uk = ['^1#_']
        for i, rdf in enumerate(n69wspa2me):
            if len(rdf) > 0:
                n69wspa2ch = f"^{n69wspa2s5[i]}#{rdf.loc[0, 'branch']}"
                n69wspa2uk.append(n69wspa2ch)
                if len(rdf) > 1:
                    pass
            else:
                pass
        return n69wspa2uk

    async def b69x8ynnvc(self, n65d20cda3, _skiplock=False, conn=None):
        n69wspa2mh = await self.select(table_name='nodes', conds=[{'uid': n65d20cda3}], targets=['hid', 'branch'], _skiplock=_skiplock, conn=conn)
        assert len(n69wspa2mh) == 1, f'eh025-5{n65d20cda3},{n69wspa2mh}'
        n69wsp9p51 = n69wspa2mh.loc[0, 'hid']
        n69wspa32d = n69wspa2mh.loc[0, 'branch']
        return (n69wsp9p51, n69wspa32d)

    async def b69x8ynnue(self, n65d20cda3, _skiplock=False, conn=None):
        n69wspa2wq = await self.select(table_name='nodes', conds=[{'uid': n65d20cda3}], targets=['def_id'], _skiplock=_skiplock, conn=conn)
        if len(n69wspa2wq) == 0:
            return None
        n69wspa2wq = n69wspa2wq.loc[0, 'def_id']
        return n69wspa2wq

    async def b69x8ynnsz(self, n69wspa2tb, _skiplock=False, conn=None):

        async def b69wsp9mrq(conn):
            n69wspa2re = f"\n            NOT SUBSTRING(def_id, LENGTH('{n69wspa2tb}') + 2) LIKE '%^%' AND\n            (\n                (\n                    def_id LIKE '{n69wspa2tb}/%' AND (NOT SUBSTRING(def_id, LENGTH('{n69wspa2tb}/') + 1) LIKE '%/%')\n                )\n                OR\n                (\n                    def_id LIKE '{n69wspa2tb}*%' AND (LENGTH(SUBSTRING(def_id, LENGTH('{n69wspa2tb}*') + 1)) - LENGTH(REPLACE(SUBSTRING(def_id, LENGTH('{n69wspa2tb}*') + 1), '/', ''))) = 1 \n                )\n            )\n            "
            n69wsp9p3q = self.select('funcs', cond_sql=n69wspa2re, conn=conn, _skiplock=True)
            n69wsp9p6q = self.b69x8ynntl(n69wspa2tb, conn=conn, _skiplock=True)
            n69wsp9osv = self.select('params', cond_sql=n69wspa2re, conn=conn, _skiplock=True)
            n69wsp9p3q, n69wsp9p6q, n69wsp9osv = await asyncio.gather(n69wsp9p3q, n69wsp9p6q, n69wsp9osv)
            return (n69wsp9p6q, n69wsp9p3q, n69wsp9osv)
        n69wspa2p9 = await self._batch_read([b69wsp9mrq], conn=conn, _skiplock=_skiplock)
        return n69wspa2p9[0]

    async def b69x8ynnuj(self, brids, namefilter=None, _skiplock=False, conn=None):
        n69wspa350 = ''
        n69wspa2nq = ''
        if namefilter:
            n69wspa2g1 = '%'.join(namefilter) + '%'
            n69wspa2g1 = n69wspa2g1.strip('"').strip("'").replace("'", "\\'")
            n69wspa350 = f" AND REGEXP_REPLACE(def_id, '^.*/', '') LIKE '{n69wspa2g1}'"
            n69wspa2nq = f" AND REGEXP_REPLACE(def_id, '^.*\\\\*', '') LIKE '{n69wspa2g1}'"

        async def b69wsp9mrq(conn):

            async def b69x8ynnt7(n69wspa2tb):
                n69wspa37y = f"\n                NOT SUBSTRING(def_id, LENGTH('{n69wspa2tb}') + 2) LIKE '%^%' AND\n                (\n                    (\n                        def_id LIKE '{n69wspa2tb}/%' AND (NOT SUBSTRING(def_id, LENGTH('{n69wspa2tb}/') + 1) LIKE '%/%')\n                    )\n                ) {n69wspa350}\n                "
                n69wspa2qa = f"\n                NOT SUBSTRING(def_id, LENGTH('{n69wspa2tb}') + 2) LIKE '%^%' AND\n                (\n                    (\n                        def_id LIKE '{n69wspa2tb}*%' AND (NOT SUBSTRING(def_id, LENGTH('{n69wspa2tb}*') + 1) LIKE '%*%')\n                    )\n                ) {n69wspa2nq}\n                "
                n69wsp9p3q = self.select('funcs', cond_sql=n69wspa37y, targets=['def_id'], conn=conn, _skiplock=True)
                n69wsp9p6q = self.select('classes', cond_sql=n69wspa2qa, targets=['def_id'], conn=conn, _skiplock=True)
                n69wsp9p3q, n69wsp9p6q = await asyncio.gather(n69wsp9p3q, n69wsp9p6q)
                n69wspa2lg = n69wsp9p3q['def_id'].tolist()
                n69wspa2j8 = n69wsp9p6q['def_id'].tolist()
                n69wspa2lg = [n.split('/')[-1] for n in n69wspa2lg]
                n69wspa2j8 = [n.split('*')[-1] for n in n69wspa2j8]
                return (n69wspa2lg, n69wspa2j8)
            n69wspa2gj = await asyncio.gather(*[b69x8ynnt7(n69wspa2tb) for n69wspa2tb in brids])
            n69wspa2lg = []
            n69wspa2j8 = []
            for names in n69wspa2gj:
                n69wspa2lg = n69wspa2lg + names[0]
                n69wspa2j8 = n69wspa2j8 + names[1]
            return (n69wspa2lg, n69wspa2j8)
        n69wsp9onl = await self._batch_read([b69wsp9mrq], conn=conn, _skiplock=_skiplock)
        return n69wsp9onl[0]

    async def b69x8ynnuw(self, impcode_line, n69wspa381, extpkgs=[], recur_objs=False, _skiplock=False, conn=None):
        assert x69xm5dtzx(n69wspa381) == 'folder'
        n69wspa2ga = b69wsp9mq1(impcode_line)['body'][0]
        n69wspa337 = ''
        n69wspa38h = []
        level = n69wspa2ga.get('level', 1)
        n69wspa2t0 = n69wspa381.strip().replace('>', '.').replace('/', '.').rstrip('.')
        n69wspa2g5 = n69wspa2t0.split('.')
        if level > 1:
            n69wspa2g5 = n69wspa2g5[:-(level - 1)]
        n69wspa2s8 = '.'.join(n69wspa2g5).rstrip('.')
        n69wspa2kg = {}
        if n69wspa2ga['ntype'] == 'Import':
            assert len(n69wspa2ga['names']) == 1, impcode_line
            n69wspa38w = (n69wspa2s8 + '.' if n69wspa2s8 else n69wspa2s8) + n69wspa2ga['names'][0]['name']
            n69wspa337 = n69wspa38w if not '.' in n69wspa38w else x69xm5du01(n69wspa38w)
            n69wspa38h = []
        elif n69wspa2ga['ntype'] == 'ImportFrom':
            n69wspa38h = [n['name'] for n in n69wspa2ga['names']]
            for n in n69wspa2ga['names']:
                if n.get('asname'):
                    n69wspa2kg[n['name']] = n['asname']
            n69wspa38w = (n69wspa2s8 + '.' if n69wspa2s8 else n69wspa2s8) + n69wspa2ga['module']
            n69wspa337 = n69wspa38w if not '.' in n69wspa38w else x69xm5du01(n69wspa38w)
        if x69xm5dtzx(n69wspa337) == 'folder':
            return ({'classes': pd.DataFrame(columns=list(n69wspa2dn.keys()) + ['source_file', 'direct', 'raw_def_id']), 'funcs': pd.DataFrame(columns=list(n69wspa2ts.keys()) + ['source_file', 'direct', 'raw_def_id']), 'params': pd.DataFrame(columns=list(n69wspa356.keys()) + ['source_file', 'direct']), 'objs': pd.DataFrame(columns=list(n69wspa2hw.keys()) + ['source_file', 'direct', 'rawname'])}, False)
        assert x69xm5dtzx(n69wspa337) == 'func' and n69wspa337.count('/') == 1
        n69wspa2rj = 'true'
        if n69wspa38h:
            n69wspa2rj = ' OR '.join([f"name = '{x}'" for x in n69wspa38h])

        async def b69wsp9mrq(conn):
            n69wspa2el = n69wspa337 + '^1#_'
            n69wspa2rg = f"\n            WITH \n                nodes_visible AS ( \n                    SELECT `uid`,'hid'\n                    FROM nodes\n                    WHERE def_id = '{n69wspa337}' AND (LENGTH(hid) - LENGTH(REPLACE(hid, '.', ''))) = 1 \n                ),\n                vars_visible AS ( \n                    SELECT \n                        vars.`name`,vars.`type`,vars.`def_id`,vars.`uid`,vars.`ctx`,\n                        nodes_visible.hid AS hid\n                    FROM \n                        vars\n                    INNER JOIN \n                        nodes_visible \n                    ON \n                        vars.uid = nodes_visible.uid\n                    WHERE \n                        vars.`type` IS NOT NULL AND {n69wspa2rj} \n                ),\n                vars_dup AS (\n                    SELECT \n                        ROW_NUMBER() OVER (PARTITION BY `name`, `def_id` ORDER BY LENGTH(`type`) DESC) AS rn, \n                        `name`,`def_id`,`type`,`uid`,`ctx`\n                    FROM vars_visible\n                )\n            SELECT * FROM vars_dup WHERE rn=1\n            "
            n69wspa2nl = conn.execute(text(n69wspa2rg))
            n69wspa2s0 = self.b69x8ynnsz(n69wspa2el, conn=conn, _skiplock=True)
            n69wspa2nl, n69wspa2s0 = await asyncio.gather(n69wspa2nl, n69wspa2s0)
            n69wspa2nl = n69wspa2nl.fetchall()
            n69wspa2nl = pd.DataFrame([{'name': x[1], 'def_id': x[2], 'type': x[3], 'uid': x[4], 'ctx': x[5], 'ethnic': None} for x in n69wspa2nl]) if n69wspa2nl else pd.DataFrame(columns=['name', 'def_id', 'type', 'uid', 'ctx', 'ethnic'])
            n69wsp9p6q, n69wsp9p3q, n69wsp9osv = n69wspa2s0
            n69wsp9p6q['raw_def_id'] = n69wsp9p6q['def_id']
            n69wsp9p3q['raw_def_id'] = n69wsp9p3q['def_id']
            n69wspa2nl['rawname'] = n69wspa2nl['name']
            n69wsp9p0m = n69wsp9p3q[n69wsp9p3q['def_id'].str.contains('\\*')]
            n69wspa2cv = n69wsp9osv[n69wsp9osv['def_id'].str.contains('\\*')]
            n69wsp9p3q = n69wsp9p3q[~n69wsp9p3q['def_id'].str.contains('\\*')]
            n69wsp9osv = n69wsp9osv[~n69wsp9osv['def_id'].str.contains('\\*')]
            n69wspa305 = n69wsp9p6q.copy()
            n69wspa2ev = n69wsp9p0m.copy()
            n69wspa2gw = n69wspa2cv.copy()
            if n69wspa38h:
                n69wspa2j0 = tuple(['/' + f for f in n69wspa38h])
                n69wspa2wx = tuple(['*' + f for f in n69wspa38h])
                n69wspa2fs = '|'.join(['\\*' + f + '/' for f in n69wspa38h])
                n69wsp9p3q = n69wsp9p3q[n69wsp9p3q['def_id'].str.endswith(n69wspa2j0)]
                n69wsp9osv = n69wsp9osv[n69wsp9osv['def_id'].str.endswith(n69wspa2j0)]
                n69wsp9p6q = n69wsp9p6q[n69wsp9p6q['def_id'].str.endswith(n69wspa2wx)]
                n69wsp9p0m = n69wsp9p0m[n69wsp9p0m['def_id'].str.contains(n69wspa2fs)]
                n69wspa2cv = n69wspa2cv[n69wspa2cv['def_id'].str.contains(n69wspa2fs)]
            n69wsp9p3q = pd.concat([n69wsp9p3q, n69wsp9p0m], ignore_index=True)
            n69wsp9osv = pd.concat([n69wsp9osv, n69wspa2cv], ignore_index=True)
            n69wspa2nm = {'objs': n69wspa2nl, 'funcs': n69wsp9p3q, 'classes': n69wsp9p6q, 'params': n69wsp9osv}
            for k in n69wspa2nm:
                n69wspa2nm[k] = n69wspa2nm[k].copy()
                if k != 'objs':
                    n69wspa2nm[k]['direct'] = True
                n69wspa2nm[k]['source_file'] = '<UNK>'
            if impcode_line.startswith('import'):
                n69wspa2nm = b69wspa5ah(n69wspa2nm, impcode_line, n69wspa381)
            if recur_objs:
                n69wspa35v = n69wspa2nl['type'].tolist()
                n69wspa2zi = [n69wsp9oq0 for n69wsp9oq0 in n69wspa35v if n69wsp9oq0.startswith('[ENV]')]
                n69wspa31b = [n69wsp9oq0 for n69wsp9oq0 in n69wspa35v if not n69wsp9oq0.startswith('[ENV]')]
                n69wspa2d7 = []
                n69wspa2sh = []
                n69wspa33d = []
                n69wspa2ct = {}
                n69wspa2du = {}
                for cid in n69wspa35v:
                    if not (cid.count('*') == 1 and cid.count('^') == 1 and (cid.find('^') < cid.find('*')) and (x69xm5dtzx(cid) == 'class')):
                        continue
                    n69wspa2ku = cid.split('^')[0]
                    n69wspa37l = cid.split('*')[-1]
                    if not n69wspa2ku.startswith('[ENV]'):
                        if not n69wspa2ku in n69wspa2d7:
                            n69wspa2d7.append(n69wspa2ku)
                        if not n69wspa2ku in n69wspa2ct:
                            n69wspa2ct[n69wspa2ku] = [n69wspa37l]
                        else:
                            n69wspa2ct[n69wspa2ku].append(n69wspa37l)
                    else:
                        n69wspa2ku = n69wspa2ku[6:]
                        if not n69wspa2ku in n69wspa2sh:
                            n69wspa2sh.append(n69wspa2ku)
                        if not n69wspa2ku in n69wspa2du:
                            n69wspa2du[n69wspa2ku] = [n69wspa37l]
                        else:
                            n69wspa2du[n69wspa2ku].append(n69wspa37l)
                n69wspa32q = [x.split('*')[-1] for x in n69wspa305['def_id'].tolist()]
                n69wspa2rh = [i for grp in list(n69wspa2ct.values()) for i in grp if not i in n69wspa32q]
                n69wspa2wa = [i for grp in list(n69wspa2du.values()) for i in grp if not i in n69wspa32q]
                n69wspa33q = [i for grp in list(n69wspa2ct.values()) for i in grp if i in n69wspa32q]
                n69wspa2ki = await self.select('funcs', conds=[{'def_id': n69wspa337}], targets=['def_id', 'imports_code'], conn=conn, _skiplock=True)
                n69wspa2t8 = n69wspa2ki.loc[0, 'imports_code'] if len(n69wspa2ki) > 0 else ''
                n69wspa2nz = []
                n69wspa2x7 = []
                _, _, impcode_washed = b69wsp9mrs(b69wsp9mq1(n69wspa2t8), expand=True)
                for impline in impcode_washed.split('\n'):
                    if impline.startswith('import'):
                        n69wspa2nz.append(impline)
                        n69wspa2x7.append(impline)
                    elif ' as ' in impline:
                        n69wsp9oul = impline.split(' as ')[-1].strip()
                        if n69wsp9oul in n69wspa2rh:
                            n69wspa2nz.append(impline)
                        elif n69wsp9oul in n69wspa2wa:
                            n69wspa2x7.append(impline)
                    else:
                        n69wsp9oul = impline.split(' import ')[-1].strip()
                        if n69wsp9oul in n69wspa2rh:
                            n69wspa2nz.append(impline)
                        if n69wsp9oul in n69wspa2wa:
                            n69wspa2x7.append(impline)
                n69wspa2nz = '\n'.join(n69wspa2nz)
                n69wspa2x7 = '\n'.join(n69wspa2x7)
                n69wspa2g3 = n69wspa5l3.b69x8ynrdt(n69wspa2x7, extpkgs, retype='df', recur_obj_cls=False, xform_imports=True)
                n69wspa2gb = self.b69x8ynnva(n69wspa2nz, n69wspa381, extpkgs=extpkgs, recur_objs=False, retype='df', conn=conn, _skiplock=True)
                n69wspa2g3, n69wspa2gb = await asyncio.gather(n69wspa2g3, n69wspa2gb)
                env_recdata, n69wspa2lc = n69wspa2g3
                cpx_recdata, _ = n69wspa2gb
                n69wspa2z1 = tuple(['*' + x for x in n69wspa33q])
                n69wspa2ee = '|'.join(['\\*' + c + '/' for c in n69wspa33q])
                n69wspa2o0 = n69wspa305[n69wspa305['def_id'].str.endswith(n69wspa2z1)].copy()
                n69wspa2l9 = n69wspa2gw[n69wspa2gw['def_id'].str.contains(n69wspa2ee)].copy()
                n69wspa2fw = n69wspa2ev[n69wspa2ev['def_id'].str.contains(n69wspa2ee)].copy()
                n69wspa2o0['doas'] = False
                n69wspa2l9['doas'] = False
                n69wspa2fw['doas'] = False
                n69wspa2vj = {'funcs': n69wspa2fw, 'params': n69wspa2l9, 'classes': n69wspa2o0}
                for k in n69wspa2nm.keys():
                    env_recdata[k]['direct'] = False
                    cpx_recdata[k]['direct'] = False
                    n69wspa2nm[k] = pd.concat([n69wspa2nm[k], env_recdata[k], cpx_recdata[k]], ignore_index=True)
                    if k != 'objs':
                        n69wspa2nm[k]['doas'] = True
                        n69wspa2vj[k]['direct'] = False
                        n69wspa2nm[k] = pd.concat([n69wspa2nm[k], n69wspa2vj[k]], ignore_index=True)
                if None in n69wspa2nm['funcs']['raw_def_id'].tolist() or np.nan in n69wspa2nm['funcs']['raw_def_id'].tolist():
                    raise
            else:
                for k in n69wspa2nm:
                    if k != 'objs':
                        n69wspa2nm[k]['doas'] = True
                if None in n69wspa2nm['funcs']['raw_def_id'].tolist() or np.nan in n69wspa2nm['funcs']['raw_def_id'].tolist():
                    raise
            if len(n69wspa2nm['funcs']) + len(n69wspa2nm['classes']) + len(n69wspa2nm['objs']) == 0:
                n69wspa34w = await self.select('funcs', conds=[{'def_id': n69wspa337}], targets=['def_id'], conn=conn, _skiplock=True)
                n69wsp9omi = len(n69wspa34w) > 0
            else:
                n69wsp9omi = True
            n69wsp9onl = n69wspa2nm
            return (n69wsp9onl, n69wsp9omi)

        def b69wspa0xh(n69wspa2el, n69wspa2xv, n69wspa2sz=None):
            if DOT_REPL in n69wspa2el:
                return n69wspa2el
            left, namepart = n69wspa2el.rsplit(n69wspa2xv, 1)
            if not n69wspa2sz:
                n69wspa2dd = n69wspa2kg.get(namepart, namepart)
                n69wspa2kx = left + n69wspa2xv + n69wspa2dd
                return n69wspa2kx
            else:
                if not n69wspa2sz in namepart:
                    return n69wspa2el
                n69wspa2hh, right = namepart.split(n69wspa2sz, 1)
                n69wspa2dd = n69wspa2kg.get(n69wspa2hh, n69wspa2hh)
                n69wspa2kx = left + n69wspa2xv + n69wspa2dd + n69wspa2sz + right
                return n69wspa2kx
        n69wsp9onl = await self._batch_read([b69wsp9mrq], conn=conn, _skiplock=_skiplock)
        n69wsp9onl, n69wsp9omi = n69wsp9onl[0]
        for k in n69wsp9onl:
            if k == 'objs':
                n69wsp9onl[k]['name'] = n69wsp9onl[k]['name'].apply(lambda x: n69wspa2kg.get(x, x))
            elif k in ('params', 'funcs'):
                n69wspa2i6 = n69wsp9onl[k]['doas'] == True
                n69wsp9onl[k].loc[n69wspa2i6, 'def_id'] = n69wsp9onl[k].loc[n69wspa2i6, 'def_id'].apply(lambda x: b69wspa0xh(x, '/') if not '*' in x else b69wspa0xh(x, '*', n69wspa2sz='/'))
            elif k == 'classes':
                n69wspa2i6 = n69wsp9onl[k]['doas'] == True
                n69wsp9onl[k].loc[n69wspa2i6, 'def_id'] = n69wsp9onl[k].loc[n69wspa2i6, 'def_id'].apply(lambda x: b69wspa0xh(x, '*'))
            n69wsp9onl[k] = n69wsp9onl[k].drop(['doas'], axis=1, errors='ignore')
        return (n69wsp9onl, n69wsp9omi)

    async def b69x8ynnva(self, n69wspa34s, n69wspa381, extpkgs=[], recur_objs=False, retype='dict', conn=None, _skiplock=False):
        n69wspa307 = n69wspa34s.split('\n')
        n69wspa307 = [i for i in n69wspa307 if i.startswith('import') or i.startswith('from')]

        async def b69wsp9mrq(conn):
            n69wspa399 = [self.b69x8ynnuw(aline, n69wspa381, extpkgs=extpkgs, recur_objs=recur_objs, conn=conn, _skiplock=_skiplock) for aline in n69wspa307]
            n69wspa2me = await asyncio.gather(*n69wspa399)
            return n69wspa2me
        n69wspa2me = await self._batch_read([b69wsp9mrq], conn=conn, _skiplock=_skiplock)
        n69wspa2me = n69wspa2me[0]
        n69wspa2lc = [(n69wspa307[i], 'Import from GraPy modules failed.') for i in range(len(n69wspa2me)) if not n69wspa2me[i][1]]
        n69wspa2me = [r[0] for r in n69wspa2me]
        if n69wspa2lc:
            pass
        n69wsp9p3q = pd.concat([x['funcs'] for x in n69wspa2me], ignore_index=True).groupby(['def_id'], as_index=False).agg({**{n69wspa2id: 'first' for n69wspa2id in list(n69wspa2ts.keys()) + ['source_file', 'direct', 'raw_def_id']}, 'direct': 'any'}) if n69wspa2me else pd.DataFrame(columns=list(n69wspa2ts.keys()) + ['source_file', 'direct', 'raw_def_id'])
        n69wsp9p6q = pd.concat([x['classes'] for x in n69wspa2me], ignore_index=True).groupby(['def_id'], as_index=False).agg({**{n69wspa2id: 'first' for n69wspa2id in list(n69wspa2dn.keys()) + ['source_file', 'direct', 'raw_def_id']}, 'direct': 'any'}) if n69wspa2me else pd.DataFrame(columns=list(n69wspa2dn.keys()) + ['source_file', 'direct', 'raw_def_id'])
        n69wsp9osv = pd.concat([x['params'] for x in n69wspa2me], ignore_index=True).groupby(['def_id', 'name', 'ctx'], as_index=False).agg({**{n69wspa2id: 'first' for n69wspa2id in list(n69wspa356.keys()) + ['source_file', 'direct']}, 'direct': 'any'}) if n69wspa2me else pd.DataFrame(columns=list(n69wspa356.keys()) + ['source_file', 'direct'])
        n69wspa2nl = pd.concat([x['objs'] for x in n69wspa2me], ignore_index=True).drop_duplicates(subset=['name', 'type']) if n69wspa2me else pd.DataFrame(columns=list(n69wspa2hw.keys()) + ['rawname'])
        n69wsp9onl = {'funcs': n69wsp9p3q, 'classes': n69wsp9p6q, 'params': n69wsp9osv, 'objs': n69wspa2nl}
        try:
            if None in n69wsp9p3q['raw_def_id'].tolist() or np.nan in n69wsp9p3q['raw_def_id'].tolist():
                raise
        except:
            raise
        if retype != 'df':
            for k in n69wsp9onl:
                n69wsp9onl[k] = n69wsp9onl[k].to_dict(orient='records')
        return (n69wsp9onl, n69wspa2lc)

    async def b69x8ynnus(self, n69wspa2wq, n69wspa381, choice, extpkgs=[], n69wspa2z4=None, recur_objs=False, retype='df', n69wspa2r3=None, n69x75d5wx='', conn=None, _skiplock=False):
        n69wspa2lw = n69wspa2r3 or await self.b69x8ynnw4(n69wspa2wq, x69xm5dtzv=n69wspa2z4, n69x75d5wx=n69x75d5wx, conn=conn, _skiplock=_skiplock)
        n69wspa2dr = {}
        n69wspa2kv = []
        n69wspa2r5 = []
        if choice in ('imports', 'both'):
            n69wspa371 = n69wspa5l3.b69x8ynrdt(n69wspa2lw['imports'], extpkgs, retype='df')
            n69wspa2dw = self.b69x8ynnva(n69wspa2lw['imports'], n69wspa381, extpkgs=extpkgs, recur_objs=recur_objs, retype='df', conn=conn, _skiplock=_skiplock)
            n69wspa371, n69wspa2dw = await asyncio.gather(n69wspa371, n69wspa2dw)
            n69wspa2dr, _fails = n69wspa371
            intermod_data, _cfails = n69wspa2dw
            n69wspa2kv = n69wspa2kv + _fails
            n69wspa2r5 = n69wspa2r5 + _cfails
            for k in n69wspa2dr.keys():
                n69wspa2dr[k] = pd.concat([n69wspa2dr[k], intermod_data[k]], ignore_index=True)
        n69wspa31q = {}
        if choice in ('froms', 'both'):
            n69wspa30k = n69wspa5l3.b69x8ynrdt(n69wspa2lw['froms'], extpkgs, retype='df')
            n69wspa2dw = self.b69x8ynnva(n69wspa2lw['froms'], n69wspa381, extpkgs=extpkgs, recur_objs=recur_objs, retype='df', conn=conn, _skiplock=_skiplock)
            n69wspa30k, n69wspa2dw = await asyncio.gather(n69wspa30k, n69wspa2dw)
            n69wspa31q, _fails = n69wspa30k
            intermod_data, _cfails = n69wspa2dw
            n69wspa2kv = n69wspa2kv + _fails
            n69wspa2r5 = n69wspa2r5 + _cfails
            for k in n69wspa31q.keys():
                n69wspa31q[k] = pd.concat([n69wspa31q[k], intermod_data[k]], ignore_index=True)
        if retype != 'df':
            for k in n69wspa2dr:
                n69wspa2dr[k] = n69wspa2dr[k].to_dict(orient='records')
            for k in n69wspa31q:
                n69wspa31q[k] = n69wspa31q[k].to_dict(orient='records')
        n69wspa32p = list(set([cf[0] for cf in n69wspa2r5]) & set([b69wsp9mp8[0] for b69wsp9mp8 in n69wspa2kv]))
        n69wspa2lc = [f for f in n69wspa2r5 + n69wspa2kv if f[0] in n69wspa32p and (not f[1] == 'Import from GraPy modules failed')]
        return (n69wspa2dr, n69wspa31q, n69wspa2lc, n69wspa2lw)

    async def b69x8ynnvd(self, n69wspa2wq, n69wspa381, choice, extpkgs=[], n69wspa2z4=None, n69wspa2r3=None, recur_objs=False, n69x75d5wx='', conn=None, _skiplock=False):
        n69wspa2nd = {'funcs': pd.DataFrame(columns=list(n69wspa2ts.keys()) + ['source_file', 'direct', 'raw_def_id']), 'classes': pd.DataFrame(columns=list(n69wspa2dn.keys()) + ['source_file', 'direct', 'raw_def_id']), 'objs': pd.DataFrame(columns=list(n69wspa2hw.keys()) + ['source_file', 'rawname']), 'params': pd.DataFrame(columns=list(n69wspa356.keys()) + ['source_file', 'direct'])}
        if choice in ('', 'None', 'none', None):
            return (n69wspa2nd, {'fails': [], 'imports_codes': {'imports': '', 'froms': ''}, 'visibility': n69wspa2z4 or []})

        async def b69wsp9mrq(conn):
            n69wspa2ub = await self.b69x8ynnus(n69wspa2wq, n69wspa381, choice, extpkgs=extpkgs, n69wspa2z4=n69wspa2z4, recur_objs=recur_objs, retype='df', n69wspa2r3=n69wspa2r3, n69x75d5wx=n69x75d5wx, conn=conn, _skiplock=True)
            return n69wspa2ub
        n69wsp9onl = await self._batch_read([b69wsp9mrq], conn=conn, _skiplock=_skiplock)
        n69wspa2dr, n69wspa31q, n69wspa2lc, n69wspa2r3 = n69wsp9onl[0]
        n69wspa2g9 = {}
        for k in ('funcs', 'params', 'classes', 'objs'):
            n69wspa36s = pd.concat([n69wspa2dr.get(k, n69wspa2nd[k]), n69wspa31q.get(k, n69wspa2nd[k])], ignore_index=True)
            n69wspa36s = n69wspa36s[~(n69wspa36s['def_id'] == '<UNK>')]
            if k == 'objs':
                n69wspa36s = n69wspa36s.drop_duplicates(['name', 'type', 'uid'])
            elif k == 'params':
                n69wspa36s = n69wspa36s.drop_duplicates(['name', 'def_id', 'ctx'])
            else:
                n69wspa36s = n69wspa36s.drop_duplicates(['def_id'])
            n69wspa36s = n69wspa36s.replace(np.nan, None)
            n69wspa2g9[k] = n69wspa36s
            if not 'source_file' in n69wspa36s.columns:
                n69wspa36s['source_file'] = '<UNK>'
        return (n69wspa2g9, {'fails': n69wspa2lc, 'imports_codes': n69wspa2r3, 'visibility': n69wspa2r3['visibility']})

    async def b69x8ynnth(self, n69wsp9oqn, n65d20cda3, conn=None, _skiplock=False):
        n69wsp9oqn = n69wsp9oqn.replace(DOT_REPL, '.')

        async def b69wsp9mrq(conn):
            n69wsp9oz8 = await self.b69x8ynnsw(n65d20cda3, conn=conn, _skiplock=True)
            n69wsp9owr = [v for v in n69wsp9oz8 if v['name'] == n69wsp9oqn]
            if not n69wsp9owr:
                return '<UNK>'
            return n69wsp9owr[0]['type']
        n69wspa2tc = await self._batch_read([b69wsp9mrq], conn=conn, _skiplock=_skiplock)
        n69wspa2tc = n69wspa2tc[0]
        return n69wspa2tc

    async def b69x8ynntd(self, n69wspa35p, n69wspa2wq, n69wspa381, _skiplock=False, conn=None):
        assert x69xm5dtzx(n69wspa381) == 'folder'
        n69wspa2it = trunk_to_func(n69wspa2wq)
        n69wspa35p = trunk_to_func(n69wspa35p)

        async def b69wsp9mrq(conn):
            extpkgs = self.select('misc', cond_sql='true', targets=['external_pkgs'], conn=conn, _skiplock=True)
            n69wspa2og = self.select('nodes', cond_sql='true', conds=[{'def_id': n69wspa2it, 'node_type': 'start'}], targets=['uid'], conn=conn, _skiplock=True)
            extpkgs, n69wspa2og = await asyncio.gather(extpkgs, n69wspa2og)
            assert len(n69wspa2og) == 1, f'eh026{n69wspa2og}'
            n69wspa2og = n69wspa2og.loc[0, 'uid']
            extpkgs = extpkgs.loc[0, 'external_pkgs'] if len(extpkgs) > 0 else []

            async def b69x8ynnss():
                n69wspa2z4 = await self.b69x8ynnv5(n69wspa35p, _skiplock=True, conn=conn)
                n69wspa2z4 = [v for v in n69wspa2z4 if not x69xm5dtzx(v) == 'folder']
                n69wspa2l3 = [self.b69x8ynntl(n69wspa2tb, _skiplock=True, conn=conn) for n69wspa2tb in n69wspa2z4]
                n69wspa2l3 = asyncio.gather(*n69wspa2l3)
                n69wspa33e = [self.b69x8ynnu2(n69wspa2tb, params_style='separate', _skiplock=True, conn=conn) for n69wspa2tb in n69wspa2z4]
                n69wspa33e = asyncio.gather(*n69wspa33e)
                n69wspa2pa = self.b69x8ynnsx(n69wspa2og, n69wspa2it, 'obj', '', helpinfo={'class': '', 'root': n69wspa381, 'desc_format': 'dict'}, consider_imports=False, _skiplock=True, conn=conn)
                n69wspa2l3, n69wspa33e, n69wspa2pa = await asyncio.gather(n69wspa2l3, n69wspa33e, n69wspa2pa)
                n69wspa2pa = n69wspa2pa[0]
                n69wspa2l3 = pd.concat(n69wspa2l3) if n69wspa2l3 else pd.DataFrame(columns=list(n69wspa2dn.keys()))
                n69wspa33f = [l[0] for l in n69wspa33e]
                n69wspa37d = [l[1] for l in n69wspa33e]
                n69wspa33f = pd.concat(n69wspa33f) if n69wspa33f else pd.DataFrame(columns=list(n69wspa2ts.keys()))
                n69wspa37d = pd.concat(n69wspa37d) if n69wspa37d else pd.DataFrame(columns=list(n69wspa356.keys()))
                n69wspa2pa = pd.DataFrame(n69wspa2pa) if n69wspa2pa else pd.DataFrame(columns=['name', 'type', 'def_id'])
                n69wspa2pa = n69wspa2pa.dropna(subset=['type'])
                n69wspa33f['source_file'] = '<UNK>'
                n69wspa37d['source_file'] = '<UNK>'
                n69wspa2l3['source_file'] = '<UNK>'
                n69wspa33f['raw_def_id'] = n69wspa33f['def_id']
                n69wspa2l3['raw_def_id'] = n69wspa2l3['def_id']
                return (n69wspa2l3, n69wspa33f, n69wspa37d, n69wspa2pa, n69wspa2z4)
            n69wspa384 = self.b69x8ynnvd(n69wspa2it, n69wspa381, 'both', extpkgs=extpkgs, recur_objs=True, conn=None, _skiplock=True)
            lvs, n69wspa384 = await asyncio.gather(b69x8ynnss(), n69wspa384)
            n69wspa371, misc = n69wspa384
            n69wspa2l3, n69wspa33f, n69wspa37d, n69wspa2pa, n69wspa2z4 = lvs
            if misc.get('fails'):
                pass
            n69wspa32e = {}
            for fline in misc['imports_codes']['froms'].split('\n'):
                if not ' as ' in fline:
                    continue
                left, right = fline.split(' as ')
                left = left.split('import ')[-1].strip()
                right = right.strip()
                n69wspa32e[right] = left
            n69wspa34s = misc['imports_codes']['imports']
            n69wspa2dv = misc['imports_codes']['froms']
            n69wspa371['funcs']['source_file'] = n69wspa371['funcs']['source_file'].fillna('<UNK>')
            n69wspa371['classes']['source_file'] = n69wspa371['classes']['source_file'].fillna('<UNK>')
            n69wspa371['objs']['source_file'] = n69wspa371['objs']['source_file'].fillna('<UNK>')
            n69wspa2ka = n69wspa371['funcs']
            n69wspa2q6 = n69wspa371['classes']
            n69wspa2es = n69wspa371['objs']
            n69wspa2cc = n69wspa371['params']
            n69wspa2rk = n69wspa2ka[(~n69wspa2ka['source_file'].str.startswith(tuple(extpkgs)) | n69wspa2ka['source_file'].str.contains('/python3\\.')) & n69wspa2ka['def_id'].str.startswith('[ENV]')]
            n69wspa2ka = n69wspa2ka[n69wspa2ka['source_file'].str.startswith(tuple(extpkgs)) & ~n69wspa2ka['source_file'].str.contains('/python3\\.') | ~n69wspa2ka['def_id'].str.startswith('[ENV]')]
            n69wspa2cf = n69wspa2rk[~n69wspa2rk['def_id'].str.contains('\\*')]['def_id'].to_list()
            n69wspa2r6 = n69wspa2q6[(~n69wspa2q6['source_file'].str.startswith(tuple(extpkgs)) | n69wspa2q6['source_file'].str.contains('/python3\\.')) & n69wspa2q6['def_id'].str.startswith('[ENV]')]
            n69wspa2q6 = n69wspa2q6[n69wspa2q6['source_file'].str.startswith(tuple(extpkgs)) & ~n69wspa2q6['source_file'].str.contains('/python3\\.') | ~n69wspa2q6['def_id'].str.startswith('[ENV]')]
            n69wspa2im = n69wspa2r6['def_id'].to_list()
            n69wspa2wo = n69wspa2es[(~n69wspa2es['source_file'].str.startswith(tuple(extpkgs)) | n69wspa2es['source_file'].str.contains('/python3\\.')) & (n69wspa2es['ethnic'] == '[ENV]')]
            n69wspa2es = n69wspa2es[n69wspa2es['source_file'].str.startswith(tuple(extpkgs)) & ~n69wspa2es['source_file'].str.contains('/python3\\.') | ~(n69wspa2es['ethnic'] == '[ENV]')]
            n69wspa2tt = n69wspa2wo['name'].to_list()
            n69wspa2ke = pd.concat([n69wspa2ka[~n69wspa2ka['def_id'].str.contains('\\*')], n69wspa33f[~n69wspa33f['def_id'].str.contains('\\*')]], ignore_index=True)
            n69wspa2tu = n69wspa2ka[n69wspa2ka['def_id'].str.contains('\\*')]
            n69wspa2iw = pd.concat([n69wspa2cc[~n69wspa2cc['def_id'].str.contains('\\*')], n69wspa37d[~n69wspa37d['def_id'].str.contains('\\*')]], ignore_index=True)
            n69wspa2wt = n69wspa2cc[n69wspa2cc['def_id'].str.contains('\\*')]
            n69wspa2gn = pd.concat([n69wspa2q6, n69wspa2l3], ignore_index=True)
            n69wspa2pe = pd.concat([n69wspa2es, n69wspa2pa], ignore_index=True)
            n69wspa2ya = {}
            n69wspa2gf = list(set(n69wspa2ke['def_id'].tolist()))
            n69wspa2rl = {key: sub_df for key, sub_df in n69wspa2iw[n69wspa2iw['ctx'] == 'input'].groupby('def_id', sort=False)}
            n69wspa2g0 = {key: sub_df for key, sub_df in n69wspa2iw[n69wspa2iw['ctx'] == 'return'].groupby('def_id', sort=False)}
            n69wspa2uf = {key: sub_df for key, sub_df in n69wspa2ke.groupby('def_id', sort=False)}
            for n69wspa2jx in n69wspa2uf:
                n69wspa392 = n69wspa2g0.get(n69wspa2jx, [])
                if len(n69wspa392) > 0:
                    n69wspa392 = n69wspa392.to_dict(orient='records')
                n69wsp9osv = {'inputs': n69wspa2rl.get(n69wspa2jx, pd.DataFrame(columns=list(n69wspa356.keys()))).sort_values(by='place').drop(['place'], axis=1).to_dict(orient='records'), 'return': n69wspa392[0] if n69wspa392 else {'name': 'return'}}
                n69wspa2l4 = n69wspa2uf[n69wspa2jx].to_dict(orient='records')[0]
                n69wspa2kn = x69xm5du02(n69wspa2l4, n69wsp9osv)
                n69wsp9p3n = n69wspa2jx.split('/')[-1].replace(DOT_REPL, '.')
                if not '.' in n69wsp9p3n and (not (n69wspa2l4['def_id'].startswith(n69wspa2it + '^') or n69wspa2l4['def_id'].startswith(n69wspa2it + '*'))):
                    n69wspa2kn['rawname'] = n69wspa32e.get(n69wsp9p3n, n69wsp9p3n)
                else:
                    n69wspa2kn['rawname'] = n69wsp9p3n.split('.')[-1]
                n69wspa2ya[n69wsp9p3n] = n69wspa2kn
            n69wspa30a = {key: sub_df for key, sub_df in n69wspa2gn.groupby('def_id', sort=False)}
            n69wspa2ho = []
            n69wspa31c = {}
            n69wsp9p6f = []
            n69wspa2dp = []
            for cid in n69wspa30a:
                n69wspa2us = n69wspa30a[cid].to_dict(orient='records')[0]
                n69wspa37g = n69wspa2us['doc']
                n69wspa37g = n69wspa37g if (n69wspa37g or '').strip() else 'This class lacks description.\n'
                n69wsp9ole = cid.split('*')[-1].replace(DOT_REPL, '.')
                n69wsp9p6f.append(n69wsp9ole)
                n69wspa2dp.append(cid)
                if not '.' in n69wsp9ole and (not n69wspa2us['def_id'].startswith(n69wspa2it + '^')):
                    n69wspa33h = n69wspa32e.get(n69wsp9ole, n69wsp9ole)
                else:
                    n69wspa33h = n69wsp9ole.split('.')[-1]
                n69wspa31c[n69wsp9ole] = {'doc': n69wspa37g, 'rawname': n69wspa33h, 'raw_def_id': n69wspa2us['raw_def_id']}
                n69wspa369 = self.b69x8ynnsx(n69wspa2og, n69wspa2it, 'func', '', helpinfo={'class': n69wsp9ole, 'hasObj': False, 'obj': '', 'root': n69wspa381, 'stop_at_sitepkg': True}, n69wspa30r=(n69wspa371, {'fails': []}), conn=conn, _skiplock=True)
                n69wspa2ho.append(n69wspa369)
            n69wspa38d = await asyncio.gather(*n69wspa2ho)
            n69wspa2qj = [cf[0]['spbases'] for cf in n69wspa38d]
            n69wspa38d = [cf[0]['funcs'] for cf in n69wspa38d]
            n69wspa390 = []
            assert len(n69wsp9p6f) == len(n69wspa38d)
            for i in range(len(n69wsp9p6f)):
                n69wspa2se = [self.b69x8ynnsx(n69wspa2og, n69wspa2it, 'func_desc', cfname, helpinfo={'hasObj': True, 'obj': '', 'root': n69wspa381, 'class': n69wsp9p6f[i], 'desc_format': 'dict'}, n69wspa30r=(n69wspa371, {'fails': []}), conn=conn, _skiplock=True) for cfname in n69wspa38d[i]]
                n69wspa390.append(asyncio.gather(*n69wspa2se))
            n69wspa2w7 = await asyncio.gather(*n69wspa390)
            assert len(n69wspa2dp) == len(n69wsp9p6f)
            for i in range(len(n69wsp9p6f)):
                n69wspa2k2 = n69wspa2w7[i]
                n69wspa369 = n69wspa38d[i]
                n69wspa303 = n69wspa2qj[i]
                n69wsp9ole = n69wsp9p6f[i]
                assert len(n69wspa369) == len(n69wspa2k2)
                n69wspa31c[n69wsp9ole]['def_id'] = n69wspa2dp[i]
                n69wspa31c[n69wsp9ole]['source_file'] = n69wspa30a[n69wspa2dp[i]]['source_file'].tolist()[0]
                n69wspa30w = []
                for n69wsp9ow8 in n69wspa303:
                    n69wspa380 = n69wsp9ow8[0]
                    assert n69wspa380.startswith('[ENV]')
                    n69wspa2x4 = n69wspa380[6:].split('^')[0]
                    n69wspa2zg = n69wspa380.split('*')[-1]
                    n69wspa2x4 = n69wspa2x4.replace('^', '.').replace('/', '.')
                    n69wspa2zg = n69wspa2zg.replace(DOT_REPL, '.')
                    n69wspa31k = {'module': n69wspa2x4, 'classname': n69wspa2zg}
                    n69wspa30w.append(n69wspa31k)
                n69wspa31c[n69wsp9ole]['sitepkg_bases'] = n69wspa30w
                n69wspa31c[n69wsp9ole]['direct'] = any(n69wspa30a[n69wspa2dp[i]]['direct'].tolist())
                n69wspa31c[n69wsp9ole]['funcs'] = {}
                for j in range(len(n69wspa2k2)):
                    n69wspa2nf = n69wspa2k2[j][0]
                    n69wspa31c[n69wsp9ole]['funcs'][n69wspa369[j]] = n69wspa2nf
            n69wspa2pb = n69wspa2pe.drop_duplicates(['name']).replace(np.nan, None).to_dict(orient='records')
            n69wspa351 = {}
            n69wspa2ro = set(n69wspa2pe['name'].tolist())
            for odic in n69wspa2pb:
                if not odic['type']:
                    continue
                n69wsp9p2h = False
                n69wspa2ce = odic['name'].replace(DOT_REPL, '.')
                if not '.' in n69wspa2ce and (not (odic['def_id'].startswith(n69wspa2it + '^') or odic['def_id'].startswith(n69wspa2it + '*'))):
                    n69wspa2vp = n69wspa32e.get(n69wspa2ce, n69wspa2ce)
                else:
                    n69wspa2vp = n69wspa2ce.split('.')[-1]
                for n69wsp9ole, n69wspa2cg in n69wspa31c.items():
                    if odic['type'] == n69wspa2cg['def_id']:
                        n69wspa351[odic['name'].replace(DOT_REPL, '.')] = {'type': n69wsp9ole, 'rawname': n69wspa2vp, 'class': n69wspa2cg}
                        n69wspa2ro.remove(odic['name'])
                        n69wsp9p2h = True
                        continue
                if not n69wsp9p2h:
                    if odic['type'] in n69wspa2r6['def_id'].tolist():
                        n69wspa380 = odic['type']
                        assert n69wspa380.startswith('[ENV]')
                        n69wspa2x4 = n69wspa380[6:].split('^')[0]
                        n69wspa2zg = n69wspa380.split('*')[-1]
                        n69wspa2x4 = n69wspa2x4.replace('^', '.').replace('/', '.')
                        n69wspa2zg = n69wspa2zg.replace(DOT_REPL, '.')
                        n69wspa343 = n69wspa2r6[n69wspa2r6['def_id'] == n69wspa380][['source_file', 'raw_def_id']].to_dict(orient='records')[0]
                        n69wspa34o = n69wspa343['source_file']
                        n69wspa2jj = n69wspa343['raw_def_id']
                        n69wspa2cg = {'doc': f'imported from module {n69wspa2x4} in site-packages.', 'def_id': n69wspa380, 'source_file': n69wspa34o, 'sitepkg_bases': [], 'direct': False, 'funcs': {}, 'raw_def_id': n69wspa2jj}
                        n69wspa351[odic['name'].replace(DOT_REPL, '.')] = {'type': n69wspa2zg, 'rawname': n69wspa2vp, 'class': n69wspa2cg}
                        n69wsp9p2h = True
                        n69wspa2ro.remove(odic['name'])
                        continue
            if n69wspa2ro:
                pass
            n69wspa31c = {k: v for k, v in n69wspa31c.items() if v.get('direct')}
            n69wspa2h1 = {'funcs': n69wspa2ya, 'classes': n69wspa31c, 'objs': n69wspa351, 'imports_codes': misc['imports_codes'], 'visibility': n69wspa2z4}
            return n69wspa2h1
        n69wsp9onl = await self._batch_read([b69wsp9mrq], conn=conn, _skiplock=_skiplock)
        n69wsp9onl = n69wsp9onl[0]
        return n69wsp9onl

    async def b69x8ynntr(self, n69wsp9p1l, n69wspa32f, n69wspa381, extpkgs, conn=None, _skiplock=True):

        async def b69wsp9mrq(conn):
            nonlocal n69wsp9p1l
            n69wspa33o = ({'funcs': pd.DataFrame(columns=list(n69wspa2ts.keys()) + ['source_file', 'direct', 'raw_def_id']), 'classes': pd.DataFrame(columns=list(n69wspa2dn.keys()) + ['source_file', 'direct', 'raw_def_id']), 'objs': pd.DataFrame(columns=list(n69wspa2hw.keys()) + ['source_file', 'rawname']), 'params': pd.DataFrame(columns=list(n69wspa356.keys()) + ['source_file', 'direct'])}, {'fails': [], 'imports_codes': {'imports': '', 'froms': ''}, 'visibility': []})
            n69wspa345 = 'pass'
            n69wspa2fa = None
            for i in range(len(n69wspa32f)):
                n69wspa2ik = n69wspa32f[i]
                if not '*' in n69wsp9p1l:
                    return (False, None, None)
                elif n69wsp9p1l.startswith('[ENV]'):
                    n69wspa2cx = 1
                    n69wspa2q2 = n69wsp9p1l[6:n69wsp9p1l.find('^1#_*')].replace('>', '.').replace('/', '.').strip('.')
                    n69wspa2ut = n69wsp9p1l.split('^1#_*')[-1]
                    n69wspa2d4 = []
                else:
                    n69wspa2cx = 0
                    n69wspa2q2 = n69wsp9p1l.split('^')[0].replace('>', '.').replace('/', '.').strip('.')
                    n69wspa2d4, n69wspa2ut = n69wsp9p1l.rsplit('*', 1)
                    n69wspa2d4 = [n69wspa2d4]
                n69wspa2h5 = False
                n69wspa2lk = '# <NOT_FOUND>'
                for lci in range(20):
                    n69wspa2lk = await self.b69x8ynnul(n69wspa2q2, '', '', n69wspa2d4, n69wspa381, current_space=n69wspa2cx, conn=conn, _skiplock=True)
                    if '<NOT_FOUND>' in n69wspa2lk.split('\n')[0]:
                        return (False, None, None)
                    n69wspa2k1 = b69wsp9mq1(n69wspa2lk, def_cutoff=True)

                    def b69wspa0yd(n69wspa2mh):
                        if isinstance(n69wspa2mh, dict):
                            if n69wspa2mh.get('ntype') == 'ClassDef':
                                if n69wspa2mh['name'] == n69wspa2ut.split('.')[-1]:
                                    return True
                        return False
                    _, n69wspa2vl = brutal_gets(n69wspa2k1, b69wspa0yd)
                    if not n69wspa2vl:
                        n69wspa38j = n69wspa2lk.split('\n')[0].split('(env_level=')[-1].split(')')[0]
                        n69wspa2cx = int(n69wspa38j) if n69wspa38j.isdigit() else n69wspa2cx
                        _, _, lcimps = b69wsp9mrs(n69wspa2k1, expand=True)
                        n69wspa2mw = [l for l in lcimps.split('\n') if l.startswith('from ')]
                        n69wspa2pq = [l for l in lcimps.split('\n') if l.startswith('import ')]
                        n69wspa2zp = False
                        for fl in n69wspa2mw:
                            n69wspa38w = fl[5:].split(' import ')[0].strip()
                            n69wspa2s6 = fl[5:].split(' import ')[-1].split(' as ')[0].strip()
                            n69wspa2kb = fl.split(' as ')[-1].strip() if ' as ' in fl else n69wspa2s6
                            if n69wspa2kb == n69wspa2ut.split('.')[-1]:
                                n69wspa2ut = n69wspa2s6
                                n69wspa2q2 = n69wspa38w
                                n69wspa2zp = True
                                break
                        if n69wspa2zp:
                            continue
                        for il in n69wspa2pq:
                            n69wspa38w = il[7:].split(' as ')[0].strip()
                            n69wspa2t6 = il.split(' as ')[-1].strip() if ' as ' in il else n69wspa38w
                            if n69wspa2t6 == n69wspa2q2:
                                n69wspa2q2 = n69wspa38w
                                n69wspa2zp = True
                                break
                        if not n69wspa2zp:
                            return (False, None, None)
                        else:
                            continue
                    else:
                        n69wspa2h5 = True
                        break
                if not n69wspa2h5:
                    return (False, None, None)
                n69wspa2vl = n69wspa2vl[0]
                n69wspa2o1 = [n for n in n69wspa2vl['body'] if n.get('ntype') == 'FunctionDef' and n.get('name') == '__init__']
                if not n69wspa2o1:
                    return (False, None, None)
                n69wspa2o1 = b69wsp9mq1(n69wspa2o1[0]['code'])
                _, n69wspa2dg = b65wsp9mrz({'ntype': 'Module', 'body': n69wspa2o1['body'][0]['body']})
                n69wspa345 = n69wspa2dg
                if n69wspa2ik == '[CODE]':
                    break
                n69wspa2fa = None
                n69wspa2dg = n69wspa2dg + f'\nself.{n69wspa2ik}()'
                n69wspa2dg = n69wspa2dg.replace('self.', 'self' + DOT_REPL)
                n69wspa2fa = b69wsp9mrr(n69wspa2dg, 'self' + DOT_REPL + n69wspa2ik, n69wspa2dg.count('\n'))
                n69wspa2td = ''
                n69wspa2r3 = {}
                if not DOT_REPL in n69wspa2fa:
                    n69wspa2wu = await self.b69x8ynnul(n69wspa2q2, n69wspa2fa, '', [], n69wspa381, current_space=n69wspa2cx, conn=conn, _skiplock=True)
                    if 'Code section of class' in n69wspa2wu.split('\n')[0]:
                        n69wspa2td = f'from {n69wspa2q2} import {n69wspa2fa}'
                    elif 'This tool is imported into module' in n69wspa2wu.split('\n')[0]:
                        n69wspa37o = n69wspa2wu.split('Import code:')[-1].strip().split(' import ')[0].split('from ')[1]
                        n69wspa2td = f'from {n69wspa37o} import {n69wspa2fa}'
                    else:
                        _, _, n69wspa38l = b69wsp9mrs(n69wspa2k1, expand=True)
                        n69wspa2fb = [l for l in n69wspa38l.split('\n') if l.startswith('from ')]
                        n69wspa2t9 = False
                        for f in n69wspa2fb:
                            if f.endswith(' as ' + n69wspa2fa):
                                n69wspa2t9 = True
                                n69wspa2hf = f.split(' import ')[1].split(' as ')[0].strip()
                                n69wspa2td = f'from {n69wspa37o} import {n69wspa2hf}'
                                n69wspa2fa = n69wspa2hf
                                break
                        if not n69wspa2t9:
                            return (False, None, None)
                    n69wspa2r3 = {'froms': n69wspa2td, 'imports': ''}
                else:
                    n69wspa32x = await self.b69x8ynnul(n69wspa2q2, '', '', n69wspa2d4, n69wspa381, current_space=n69wspa2cx, conn=conn, _skiplock=True)
                    n69wspa2j5 = b69wsp9mq1(n69wspa32x, def_cutoff=True)
                    _, _, lord_imports_code = b69wsp9mrs(n69wspa2j5, expand=True)
                    n69wspa2oh = [l for l in lord_imports_code.split('\n') if l.startswith('import ')]
                    n69wspa37o = n69wspa2fa.rsplit(DOT_REPL, 1)[0]
                    n69wspa2s1 = [l for l in n69wspa2oh if f"import {n69wspa37o.replace(DOT_REPL, '.')}" in l or f"as {n69wspa37o.replace(DOT_REPL, '.')}" in l]
                    if not n69wspa2s1:
                        return (False, None, None)
                    n69wspa2s1 = n69wspa2s1[0]
                    n69wspa2r3 = {'froms': '', 'imports': n69wspa2s1}
                n69wspa33o = await self.b69x8ynnvd(None, '', 'both', extpkgs=extpkgs, n69wspa2z4=[], n69wspa2r3={**n69wspa2r3, 'visibility': []}, recur_objs=True, conn=conn, _skiplock=True)
                if n69wspa2cx > 0:
                    for k in ('funcs', 'params', 'classes'):
                        n69wspa33o[0][k] = n69wspa33o[0][k][n69wspa33o[0][k]['def_id'].str.startswith('[ENV]')]
                n69wspa2dp = n69wspa33o[0]['classes'][n69wspa33o[0]['classes']['def_id'].str.endswith('*' + n69wspa2fa)]['raw_def_id'].tolist()
                if not n69wspa2dp:
                    return (False, None, None)
                n69wsp9p1l = n69wspa2dp[0]
            return (True, n69wspa345, n69wspa33o, n69wsp9p1l, n69wspa2fa)
        n69wsp9onl = await self._batch_read([b69wsp9mrq], conn=conn, _skiplock=_skiplock)
        return n69wsp9onl[0]

    async def b69x8ynnsx(self, n65d20cda3, n69wspa2wq, item, value, helpinfo={}, n69wspa30r=None, consider_imports=True, blind_vision=[], _skiplock=False, conn=None):
        n69wspa37a = []
        if not value:
            value = ''
        elif not value.strip():
            value = ''
        assert not any([s in value for s in ('"', "'", '`')]), 'Illegal character detected'
        n69wspa381 = helpinfo.get('root')
        if not x69xm5dtzx(n69wspa381) == 'folder':
            n69wspa37a.append(f'Project root path must be a folder, got: {n69wspa381}, which is a {x69xm5dtzx(n69wspa381)}.')
            n69wspa381 = '<UNK>'
        if not consider_imports:
            n69wspa30r = ({'funcs': pd.DataFrame(columns=list(n69wspa2ts.keys())), 'classes': pd.DataFrame(columns=list(n69wspa2dn.keys())), 'objs': pd.DataFrame(columns=list(n69wspa2hw.keys())), 'params': pd.DataFrame(columns=list(n69wspa356.keys()))}, {'fails': []})

        async def b69wsp9mrq(conn):
            nonlocal value, n69wspa37a, n69wspa30r
            n69wspa38m = n69wspa2wq.split('^')[0]
            n69wsp9p51, _ = await self.b69x8ynnvc(n65d20cda3, _skiplock=True, conn=conn) if n65d20cda3 != 'UNDEFINED' else ['1.' + n65d20cda3, '']
            n69wspa2e5 = n69wspa2wq + '^' + n69wsp9p51
            value = value.strip() if not value is None else None
            n69wspa2xe = self.select('funcs', cond_sql=f"(def_id LIKE '{n69wspa381}/%' OR def_id LIKE '{n69wspa381}>%') AND (LENGTH(def_id) - LENGTH(REPLACE(def_id, '/', ''))) = 1", targets=['def_id'], conn=conn, _skiplock=True) if n65d20cda3 != 'UNDEFINED' else aidle(default=pd.DataFrame(columns=['def_id']))
            n69wspa2z4, extpkgs, n69wspa2xe = await asyncio.gather(self.b69x8ynnv5(n69wspa2e5, _skiplock=True, conn=conn) if n65d20cda3 != 'UNDEFINED' else aidle(default=blind_vision), self.select('misc', cond_sql='true', targets=['external_pkgs'], conn=conn, _skiplock=True), n69wspa2xe)
            n69wspa2z4 = [v for v in n69wspa2z4 if not x69xm5dtzx(v) == 'folder']
            n69wspa2xe = [x for x in n69wspa2xe['def_id'].tolist() if x != n69wspa38m]
            extpkgs = extpkgs.loc[0, 'external_pkgs'] if len(extpkgs) > 0 else []
            n69wspa2z4.reverse()
            n69wspa2qr = []
            for v in n69wspa2z4:
                n69wspa2vg = v.rfind('^')
                n69wspa2ox = v.rfind('#')
                assert n69wspa2vg > 0 and n69wspa2ox > 0 and (n69wspa2vg < n69wspa2ox), (n69wspa2vg, n69wspa2ox)
                left = v[:n69wspa2vg]
                n69wspa2hh = v[n69wspa2vg + 1:n69wspa2ox]
                right = v[n69wspa2ox + 1:]
                if right.isdigit():
                    right = int(right)
                elif right.lower() == 'true':
                    right = True
                elif right.lower() == 'false':
                    right = False
                n69wspa2qr.append((left, n69wspa2hh, right))
            n69wspa2gi = [f"""\n                        (def_id = '{v[0]}' AND\n                         hid LIKE '{v[1]}.%' \n                         AND NOT SUBSTRING(hid,LENGTH('{v[1]}')+2) LIKE '%.%' \n                         AND branch = {("'" + v[2] + "'" if isinstance(v[2], str) else v[2])})\n                         """ for v in n69wspa2qr]
            n69wspa2gi = ' OR '.join(n69wspa2gi) if n69wspa2gi else 'FALSE'
            n69wspa2kh = None
            if item == 'obj' or (item in ('class', 'class_desc') and helpinfo.get('obj') is not None):
                if item == 'obj':
                    n69wspa32j = value
                else:
                    n69wspa32j = helpinfo['obj'].strip()
                n69wspa2kh = f"\n                WITH \n                    nodes_in_module AS ( \n                        SELECT `uid`,`hid`,`branch`,`def_id`\n                        FROM nodes\n                        WHERE def_id LIKE '{n69wspa38m}^%' OR def_id = '{n69wspa38m}'\n                    ),\n                    nodes_visible AS (\n                        SELECT * FROM nodes_in_module \n                        WHERE {n69wspa2gi}\n                    ),\n                    vars_visible AS ( \n                        SELECT \n                            vars.`name`,vars.`type`,vars.`def_id`,vars.`uid`,\n                            nodes_visible.hid AS hid\n                        FROM \n                            vars\n                        INNER JOIN \n                            nodes_visible \n                        ON \n                            vars.uid = nodes_visible.uid\n                    ),\n                    vars_unnull_priority AS ( -- CHANGE251227 \n\t\t\t\t\t\tSELECT `name`,`type`,`def_id`,`uid`,`hid`\n\t\t\t\t\t\tFROM (\n\t\t\t\t\t\t\tSELECT\n\t\t\t\t\t\t\t\t`name`,`type`,`def_id`,`uid`,`hid`,\n\t\t\t\t\t\t\t\tROW_NUMBER() OVER (\n\t\t\t\t\t\t\t\t\tPARTITION BY `name`, `def_id`\n\t\t\t\t\t\t\t\t\tORDER BY (`type` IS NOT NULL) DESC\n\t\t\t\t\t\t\t\t) AS rn\n\t\t\t\t\t\t\tFROM vars_visible\n\t\t\t\t\t\t) t\n\t\t\t\t\t\tWHERE rn = 1\n                    ),\n                    vars_dup AS ( \n                        SELECT \n                            ROW_NUMBER() OVER (PARTITION BY `name` ORDER BY LENGTH(`def_id`) DESC) AS rn, \n                            `name`,\n                            `def_id` ,\n                            `type`\n                        FROM vars_unnull_priority\n                        WHERE name LIKE '{n69wspa32j}%' \n                    )\n                "
            if item == 'obj':
                n69wspa2p3 = helpinfo.get('desc_format') == 'dict'
                n69wspa2d0 = '' if not n69wspa2p3 else ',`type`'
                sql = n69wspa2kh + f'\n                SELECT \n                    `name`,`def_id`{n69wspa2d0}\n                FROM vars_dup\n                WHERE rn=1'
                if helpinfo.get('class').strip():
                    sql = sql + f" AND type LIKE '%*{helpinfo['class'].strip()}'"
                n69wspa2v7 = self.select('params', cond_sql=f"def_id = '{n69wspa2wq}' AND (name = 'cls' OR name = 'self')", targets=['name'], _skiplock=True, conn=conn) if '*' in n69wspa2wq else aidle(default=pd.DataFrame(columns=['name']))
                n69wsp9oq8 = conn.execute(text(sql))
                n69wspa2v7, n69wsp9oq8 = await asyncio.gather(n69wspa2v7, n69wsp9oq8)
                n69wspa32z = n69wspa2v7['name'].tolist()
                n69wsp9oq8 = n69wsp9oq8.fetchall()
                n69wsp9onl = [x[0] for x in n69wsp9oq8] if not n69wspa2p3 else [{'name': x[0], 'def_id': x[1], 'type': x[2]} for x in n69wsp9oq8]
                if '*' in n69wspa2wq:
                    n69wspa2c9 = n69wspa2wq.rsplit('/', 1)[0] if n69wspa2wq.rfind('/') > n69wspa2wq.rfind('*') else n69wspa2wq
                    if value.strip() in ('', 's', 'se', 'sel', 'self') and 'self' in n69wspa32z:
                        n69wsp9onl = ['self'] + n69wsp9onl if not n69wspa2p3 else [{'name': 'self', 'type': n69wspa2c9}] + n69wsp9onl
                    elif value.strip() in ('', 'c', 'cl', 'cls') and 'cls' in n69wspa32z:
                        n69wsp9onl = ['cls'] + n69wsp9onl if not n69wspa2p3 else [{'name': 'self', 'type': 'Type'}] + n69wsp9onl
                if not n69wsp9oq8:
                    if '.' in value:
                        n69wspa371, n69wspa2lc = n69wspa30r or await self.b69x8ynnvd(n69wspa2wq, n69wspa381, 'imports', extpkgs=extpkgs, n69wspa2z4=n69wspa2z4, recur_objs=True, conn=conn, _skiplock=True)
                    else:
                        n69wspa371, n69wspa2lc = n69wspa30r or await self.b69x8ynnvd(n69wspa2wq, n69wspa381, 'froms', extpkgs=extpkgs, n69wspa2z4=n69wspa2z4, recur_objs=True, conn=conn, _skiplock=True)
                    n69wspa37a = n69wspa37a + [f"Failed to import '{f[0]}': {f[1]}" for f in n69wspa2lc['fails']]
                    n69wspa2es = n69wspa371['objs']
                    n69wsp9onl = n69wspa2es[n69wspa2es['name'].str.startswith(value.replace('.', DOT_REPL))]['name'].tolist() if not n69wspa2p3 else n69wspa2es[n69wspa2es['name'].str.startswith(value.replace('.', DOT_REPL))].to_dict(orient='records')
                    n69wsp9onl = [r.replace(DOT_REPL, '.') for r in n69wsp9onl] if not n69wspa2p3 else [{**r, 'name': r['name'].replace(DOT_REPL, '.')} for r in n69wsp9onl]
                return n69wsp9onl
            elif item in ('func', 'params', 'func_desc', 'argname'):
                if item == 'func':
                    n69wsp9p0f = value
                elif item == 'argname':
                    n69wsp9p0f = helpinfo['func'].strip()
                else:
                    n69wsp9p0f = value
                if helpinfo.get('hasObj') and helpinfo.get('obj') and ('.' in n69wsp9p0f) and helpinfo.get('class'):
                    helpinfo['class'] = None
                if not helpinfo.get('class'):
                    if helpinfo.get('hasObj') or helpinfo.get('obj') in ('self', 'cls'):
                        n69wsp9orc = None
                        if not helpinfo.get('obj') in ('self', 'cls'):
                            if not '.' in n69wsp9p0f:
                                if helpinfo.get('nodecode') and helpinfo.get('row'):
                                    try:
                                        n69wsp9orc = b69wsp9mrr(n69wsp9orc, helpinfo['obj'], helpinfo['row'])
                                    except Exception as ec:
                                        traceback.print_exc()
                                if not n69wsp9orc:
                                    n69wspa30r = n69wspa30r or await self.b69x8ynnvd(n69wspa2wq, n69wspa381, 'both', extpkgs=extpkgs, n69wspa2z4=n69wspa2z4, recur_objs=True, conn=conn, _skiplock=True)
                                    n69wsp9orc, warns0 = await self.b69x8ynnsx(n65d20cda3, n69wspa2wq, 'class', None, helpinfo={'obj': helpinfo['obj'], 'root': helpinfo['root']}, n69wspa30r=n69wspa30r, consider_imports=True, _skiplock=True, conn=conn)
                                    if warns0:
                                        for w in warns0:
                                            n69wspa37a.append(w)
                                    n69wsp9orc = n69wsp9orc[0] if n69wsp9orc else None
                                if not n69wsp9orc:
                                    pass
                        elif not '.' in n69wsp9p0f:
                            n69wspa30r = n69wspa30r or await self.b69x8ynnvd(n69wspa2wq, n69wspa381, 'none', extpkgs=extpkgs, n69wspa2z4=n69wspa2z4, recur_objs=True, conn=conn, _skiplock=True)
                            n69wsp9orc = n69wspa2wq.split('*')[-1].split('/')[0]
                        if not n69wsp9orc:
                            if '.' in n69wsp9p0f:
                                n69wspa2e6 = helpinfo['obj'] + '.' + n69wsp9p0f.split('.')[0]
                                if n69wspa2e6.startswith('cls.'):
                                    return '<UNK>'
                                else:
                                    n69wspa34f = helpinfo['obj']
                                    n69wspa32f = n69wsp9p0f.split('.')
                                    n69wsp9p1l, _ = await self.b69x8ynnsx(n65d20cda3, n69wspa2wq, 'class', None, helpinfo={'obj': n69wspa34f, 'root': helpinfo['root'], 'full': True, 'israw': True}, n69wspa30r=n69wspa30r, consider_imports=True, _skiplock=True, conn=conn)
                                    if not n69wsp9p1l:
                                        return '<UNK>'
                                    n69wsp9p1l = n69wsp9p1l[0]
                                    n69wspa38o = await self.b69x8ynntr(n69wsp9p1l, n69wspa32f[:-1], helpinfo['root'], extpkgs, conn=conn, _skiplock=True)
                                    if not n69wspa38o[0]:
                                        return '<UNK>'
                                    _, n69wspa2dg, n69wspa33o, n69wsp9p1l, n69wspa2fa = n69wspa38o
                                    n69wspa310 = helpinfo.copy()
                                    n69wspa2mp = value
                                    if item == 'argname':
                                        n69wspa310['func'] = n69wspa32f[-1]
                                    else:
                                        n69wspa2mp = n69wspa32f[-1]
                                    n69wspa310['class'] = n69wspa2fa.replace(DOT_REPL, '.')
                                    n69wspa310['obj'] = '[UNDEFINED]'
                                    n69wspa2dp = n69wspa33o[0]['classes']['def_id'].tolist()
                                    n69wspa313 = [c for c in n69wspa2dp if c.endswith('*' + n69wspa2fa)]
                                    if not n69wspa313:
                                        return '<UNK>'
                                    n69wspa313 = n69wspa313[0]
                                    n69wspa301 = [n69wspa313.rsplit('*', 1)[0]]
                                    n69wspa33z = await self.b69x8ynnsx('UNDEFINED', 'UNDEFINED/UNDEFINED', item, n69wspa2mp, n69wspa310, n69wspa33o, blind_vision=n69wspa301, _skiplock=True, conn=conn)
                                    for w in n69wspa33z[1]:
                                        n69wspa37a.append(w)
                                    return n69wspa33z[0]
                        if not n69wsp9orc:
                            return '<UNK>'
                        n69wspa310 = {**helpinfo, 'class': n69wsp9orc}
                        n69wsp9onl, warns1 = await self.b69x8ynnsx(n65d20cda3, n69wspa2wq, item, value, helpinfo=n69wspa310, n69wspa30r=n69wspa30r, consider_imports=consider_imports, _skiplock=True, conn=conn)
                        if warns1:
                            for w in warns1:
                                n69wspa37a.append(w)
                        return n69wsp9onl
                    n69wspa2me = [self.b69x8ynnu2(n69wspa2tb, params_style='skip', targets=['def_id'], _skiplock=True, conn=conn) for n69wspa2tb in n69wspa2z4]
                    n69wspa2me = await asyncio.gather(*n69wspa2me)
                    n69wspa2me = pd.concat(n69wspa2me, ignore_index=True)
                    n69wspa2me = n69wspa2me['def_id'].tolist()
                    if '.' in n69wsp9p0f:
                        n69wspa371, n69wspa2lc = n69wspa30r or await self.b69x8ynnvd(n69wspa2wq, n69wspa381, 'imports', extpkgs=extpkgs, n69wspa2z4=n69wspa2z4, recur_objs=False, conn=conn, _skiplock=True)
                        n69wspa2ka = n69wspa371['funcs']
                        n69wspa2ka = n69wspa2ka[~n69wspa2ka['def_id'].str.contains('\\*')]
                        n69wspa2pg = n69wspa2ka
                        n69wspa30l = n69wspa371['params']
                        n69wspa30l = n69wspa30l[~n69wspa30l['def_id'].str.contains('\\*')]
                        n69wspa37q = n69wspa2ka['def_id'].tolist()
                        n69wspa2me = n69wspa2me + n69wspa37q
                    else:
                        n69wspa371, n69wspa2lc = n69wspa30r or await self.b69x8ynnvd(n69wspa2wq, n69wspa381, 'froms', extpkgs=extpkgs, n69wspa2z4=n69wspa2z4, recur_objs=False, conn=conn, _skiplock=True)
                        n69wspa37m = n69wspa371['funcs'].to_dict(orient='records')
                        n69wspa2ka = n69wspa371['funcs']
                        n69wspa2ka = n69wspa2ka[~n69wspa2ka['def_id'].str.contains('\\*')]
                        n69wspa2pg = n69wspa2ka
                        n69wspa30l = n69wspa371['params']
                        n69wspa30l = n69wspa30l[~n69wspa30l['def_id'].str.contains('\\*')]
                        for f in n69wspa37m:
                            n69wspa2jx = f['def_id']
                            if '*' in n69wspa2jx:
                                continue
                            n69wspa2me.append(n69wspa2jx)
                    n69wspa37a = n69wspa37a + [f"Failed to import '{f[0]}':{f[1]}" for f in n69wspa2lc['fails']]
                    if not n69wspa2me:
                        return '<UNK>'
                    if item == 'func':
                        n69wspa2me = list(set([r.split('/')[-1] for r in n69wspa2me]))
                        n69wspa2me = [r.replace(DOT_REPL, '.') for r in n69wspa2me if r.startswith(n69wsp9p0f.replace('.', DOT_REPL))]
                        return n69wspa2me
                    else:
                        n69wspa2me = [r for r in n69wspa2me if r.endswith('/' + n69wsp9p0f.replace('.', DOT_REPL))]
                        n69wspa2it = replace_lastpart(n69wspa2me[0], '/', repfrom='.', repto=DOT_REPL) if n69wspa2me else '<UNK>'
                        if n69wspa2it == '<UNK>':
                            n69wspa37a.append(f'Cannot find func {n69wsp9p0f}.')
                else:
                    n69wspa2tc = aidle()
                    if helpinfo.get('obj') and helpinfo.get('obj') != '[UNDEFINED]':
                        n69wsp9oqn = helpinfo['obj']
                        n69wspa2tc = self.b69x8ynnth(n69wsp9oqn, n65d20cda3, conn=conn, _skiplock=True)
                    n69wspa334 = 'both'

                    async def b69x8ynnu1():
                        return n69wspa30r or await self.b69x8ynnvd(n69wspa2wq, n69wspa381, n69wspa334, extpkgs=extpkgs, n69wspa2z4=n69wspa2z4, recur_objs=True, conn=conn, _skiplock=True)
                    n69wspa32c = b69x8ynnu1()
                    n69wspa2tc, n69wspa32c = await asyncio.gather(n69wspa2tc, n69wspa32c)
                    n69wspa371, n69wspa2lc = n69wspa32c
                    n69wspa37a = n69wspa37a + [f"Failed to import '{f[0]}':{f[1]}" for f in n69wspa2lc['fails']]
                    n69wspa2tc = '' if not n69wspa2tc else n69wspa2tc
                    if helpinfo.get('obj') and helpinfo.get('obj') != '[UNDEFINED]':
                        if not '*' in n69wspa2tc:
                            n69wspa37a.append(f'Failed to fetch type of obj {n69wsp9oqn}. Recommendations are not guaranteed correct.')
                        else:
                            n69wspa2tc = replace_lastpart(n69wspa2tc, '*', repfrom='.', repto=DOT_REPL)
                    n69wspa2pg = n69wspa371['funcs']
                    n69wspa30s = n69wspa371['classes']
                    n69wspa30l = n69wspa371['params']
                    n69wspa2pg = n69wspa2pg[n69wspa2pg['def_id'].str.contains('\\*')]
                    n69wspa30l = n69wspa30l[n69wspa30l['def_id'].str.contains('\\*')]
                    n69wspa2yw = []
                    n69wspa2tv = []
                    n69wspa2f2 = 0
                    n69wspa303 = []

                    async def b69x8ynnw1(n69wsp9onk, vision, round1=False):
                        nonlocal n69wspa37a, n69wspa2yw, n69wspa2tv, n69wspa2f2, n69wspa2pg, n69wspa30l, n69wspa303
                        n69wspa2f2 = n69wspa2f2 + 1
                        n69wspa2o9 = False
                        if round1:
                            if n69wspa2tc:
                                if '*' in n69wspa2tc:
                                    n69wspa2o9 = True
                        n69wsp9onk = n69wsp9onk.replace('.', DOT_REPL)
                        n69wspa2z2 = False
                        if not vision:
                            pass
                        elif vision[0] in n69wspa2z4:
                            n69wspa2z2 = True
                        n69wspa2fu = await self.select('classes', cond_sql=f"def_id LIKE '%*{n69wsp9onk}'", conn=conn, _skiplock=True) if n69wspa2z2 else pd.DataFrame(columns=list(n69wspa2dn.keys()))
                        n69wspa2l5 = n69wspa2fu['def_id'].tolist()
                        n69wspa2sn = []
                        n69wspa2ic = ['<UNK>'] * 299
                        n69wspa36z = vision
                        n69wspa2ph = 'local' if n69wspa2z2 else 'env'
                        if n69wspa2z2:
                            for c in n69wspa2l5:
                                for n69wsp9p68, v in enumerate(vision):
                                    if c.startswith(v + '*'):
                                        if c[len(v) + 1:] == n69wsp9onk:
                                            n69wspa2sn.append(c)
                                            n69wspa36z = vision[n69wsp9p68:]
                                            break
                        if not n69wspa2sn:
                            n69wspa2ph = 'env'
                            n69wspa2z3 = n69wspa2pg
                            n69wspa2th = n69wspa30l
                            if n69wspa2z2:
                                n69wspa2i5 = n69wspa30s[n69wspa30s['def_id'].str.endswith('*' + n69wsp9onk)]
                                n69wspa2z3 = n69wspa2pg
                                n69wspa2th = n69wspa30l
                            else:
                                n69wspa2i5 = pd.DataFrame(columns=list(n69wspa2dn.keys()))
                                if vision:
                                    n69wspa2q3 = vision[0].split('^')[0]
                                    if not n69wspa2q3.startswith('[ENV]'):
                                        n69wspa2t4 = self.b69x8ynnsz(n69wspa2q3 + '^1#_', conn=conn, _skiplock=True)
                                        n69wspa2n1 = self.b69x8ynnvd(n69wspa2q3, n69wspa381, 'both', extpkgs=extpkgs, n69wspa2z4=[n69wspa2q3 + '^1#_'], recur_objs=False, conn=conn, _skiplock=True)
                                        n69wspa2t4, n69wspa2n1 = await asyncio.gather(n69wspa2t4, n69wspa2n1)
                                        slclasss, slfuncs, slparams = n69wspa2t4
                                        n69wspa2n1, n69wspa2lc = n69wspa2n1
                                        n69wspa37a = n69wspa37a + [f"Failed to import '{f[0]}':{f[1]}" for f in n69wspa2lc['fails']]
                                        slclasss['raw_def_id'] = slclasss['def_id']
                                        slfuncs['raw_def_id'] = slfuncs['def_id']
                                        n69wspa2i5 = pd.concat([n69wspa2n1['classes'], slclasss], ignore_index=True)
                                        n69wspa2i5 = n69wspa2i5[n69wspa2i5['def_id'].str.endswith('*' + n69wsp9onk)]
                                        n69wspa2z3 = pd.concat([n69wspa2n1['funcs'], slfuncs], ignore_index=True)
                                        n69wspa2th = pd.concat([n69wspa2n1['params'], slparams], ignore_index=True)
                                    else:
                                        pass
                                else:
                                    pass
                            n69wspa2sn = n69wspa2i5['def_id'].tolist()
                            n69wspa2ic = n69wspa2i5['source_file'].tolist()
                        if not n69wspa2sn:
                            return (None, [], [], '<UNK>')
                        if n69wspa2o9:
                            if n69wspa2sn:
                                n69wspa394 = [m for m in n69wspa2sn if m == n69wspa2tc]
                                if n69wspa394:
                                    n69wspa2sn = n69wspa394
                                else:
                                    n69wspa37a.append(f"Conflict found on type of obj {helpinfo['obj']}: {n69wspa2sn} vs {n69wspa2tc}. Recommendations are not guaranteed correct.")
                        n69wspa2vn = [len(c) for c in n69wspa2sn]
                        n69wspa2xx = np.argmax(n69wspa2vn)
                        n69wspa313 = n69wspa2sn[n69wspa2xx]
                        n69wspa2yp = n69wspa2ic[n69wspa2xx]
                        if n69wspa2ph == 'env':
                            n69wspa36z = [n69wspa313.rsplit('*', 1)[0]]
                        if helpinfo.get('stop_at_sitepkg'):
                            if x69xm5du07(n69wspa313, n69wspa2yp, extpkgs):
                                n69wspa303.append((n69wspa313, n69wspa2yp))
                                return (n69wspa313, [], n69wspa36z, n69wspa2yp)
                        if n69wspa2ph == 'local':
                            n69wsp9oxo = n69wspa2fu[n69wspa2fu['def_id'] == n69wspa313]['bases'].tolist()[0]
                            n69wspa2py = self.b69x8ynnu2(n69wspa313, params_style='skip', targets=['def_id'], _skiplock=True, conn=conn)
                            n69wspa2yw.append(n69wspa2py)
                        elif n69wspa2ph == 'env':
                            n69wsp9oxo = n69wspa2i5[n69wspa2i5['def_id'] == n69wspa313]['bases'].tolist()[0] or []
                            n69wspa2u6 = n69wspa2z3[n69wspa2z3['def_id'].str.startswith(n69wspa313 + '/')]
                            n69wspa2jf = n69wspa2th[n69wspa2th['def_id'].str.startswith(n69wspa313 + '/')]
                            n69wspa2mj = n69wspa2u6['def_id'].tolist()
                            n69wspa2pg = pd.concat([n69wspa2pg, n69wspa2u6], ignore_index=True).drop_duplicates(['def_id'])
                            n69wspa30l = pd.concat([n69wspa30l, n69wspa2jf], ignore_index=True).drop_duplicates(['def_id', 'name', 'ctx'])
                            n69wspa2tv = n69wspa2tv + n69wspa2mj
                        assert isinstance(n69wsp9oxo, list), n69wspa2fu
                        return (n69wspa313, n69wsp9oxo, n69wspa36z, n69wspa2yp)
                    n69wspa313, n69wsp9oxo, n69wspa36z, n69wspa2yp = await b69x8ynnw1(helpinfo['class'], n69wspa2z4, round1=True)
                    if not n69wspa313:
                        match item:
                            case 'func':
                                return [] if not helpinfo.get('stop_at_sitepkg') else {'funcs': [], 'spbases': n69wspa303}
                            case 'params':
                                return '<UNK>'
                    n69wspa2uj = [n69wspa313]

                    async def b69x8ynnvi(thisbases, thisvision, thisclass, thisorigin):
                        nonlocal n69wspa2uj, n69wspa303
                        if helpinfo.get('stop_at_sitepkg'):
                            if x69xm5du07(thisclass, thisorigin, extpkgs):
                                n69wspa303.append((thisclass, thisorigin))
                                return
                        for n69wspa380 in thisbases:
                            n69wspa5m7, newbases, n69wspa36z, neworigin = await b69x8ynnw1(n69wspa380, thisvision)
                            if n69wspa5m7 is not None:
                                n69wspa2uj.append(n69wspa5m7)
                                if newbases:
                                    await b69x8ynnvi(newbases, n69wspa36z, n69wspa5m7, neworigin)
                    await b69x8ynnvi(n69wsp9oxo, n69wspa36z, n69wspa313, n69wspa2yp)
                    n69wspa2uj = [c for c in n69wspa2uj if c not in n69wspa303]
                    n69wsp9p3q = await asyncio.gather(*n69wspa2yw)
                    n69wsp9p3q = pd.concat(n69wsp9p3q) if n69wsp9p3q else pd.DataFrame(columns=['def_id'])
                    n69wsp9p3q = n69wsp9p3q['def_id'].tolist()
                    n69wsp9p3q = n69wsp9p3q + n69wspa2tv
                    if not n69wsp9p3q:
                        if item != 'func':
                            return '<UNK>'
                        else:
                            return [] if not helpinfo.get('stop_at_sitepkg') else {'funcs': [], 'spbases': n69wspa303}
                    n69wspa36a = []
                    for f in n69wsp9p3q:
                        f = f.split('/')[-1]
                        if not f in n69wspa36a:
                            if f.startswith(n69wsp9p0f.strip()):
                                n69wspa36a.append(f)
                    if item == 'func':
                        if helpinfo.get('stop_at_sitepkg'):
                            return {'funcs': n69wspa36a, 'spbases': n69wspa303}
                        else:
                            return n69wspa36a
                    else:
                        n69wsp9p3q = [f for f in n69wsp9p3q if f.endswith('/' + n69wsp9p0f)]
                        n69wspa2it = replace_lastpart(n69wsp9p3q[0], '*', repfrom='.', repto=DOT_REPL) if n69wsp9p3q else '<UNK>'
                        if n69wspa2it == '<UNK>':
                            n69wspa37a.append(f"Cannot find func {helpinfo['class']}.{n69wsp9p0f}.")
                if item == 'params':
                    if n69wspa2it == '<UNK>':
                        return '<UNK>'
                    if not helpinfo.get('hasObj') and n69wsp9p0f in '__call__':
                        n69wspa2lr = '<UNK>'
                    else:
                        n69wsp9osv = {'inputs': pd.DataFrame(columns=list(n69wspa356.keys())), 'return': pd.DataFrame(columns=list(n69wspa356.keys()))}
                        if not n69wspa2it.startswith('[ENV]') and (not DOT_REPL in n69wspa2it):
                            n69wsp9osv = await self.b69x8ynnt2(n69wspa2it, _skiplock=True, conn=conn)
                        if len(n69wsp9osv['return']) == 0 and len(n69wsp9osv['inputs']) == 0:
                            n69wsp9osv = {'inputs': n69wspa30l[(n69wspa30l['def_id'] == n69wspa2it) & (n69wspa30l['ctx'] == 'input')].sort_values(by='place').drop(['place'], axis=1).to_dict(orient='records')}
                        n69wspa2lr = [[p['name'], p['default']] for p in n69wsp9osv['inputs'] if not p['name'] in ('self', 'cls')]
                    return n69wspa2lr
                elif item == 'argname':
                    if n69wspa2it == '<UNK>':
                        return '<UNK>'
                    n69wspa357 = f"\n                    def_id = '{n69wspa2it}' AND name LIKE '{value.strip()}%' AND ctx = 'input'\n                    "
                    if not helpinfo.get('hasObj') and n69wsp9p0f in '__call__':
                        return []
                    n69wspa2w2 = pd.DataFrame(columns=['name'])
                    if not n69wspa2it.startswith('[ENV]') and (not DOT_REPL in n69wspa2it):
                        n69wspa2w2 = await self.select('params', cond_sql=n69wspa357, targets=['name'], _skiplock=True, conn=conn)
                    if len(n69wspa2w2) == 0:
                        n69wspa2pr = await self.select('funcs', conds=[{'def_id': n69wspa2it}], targets=['def_id'], _skiplock=True, conn=conn)
                        if len(n69wspa2pr) == 0:
                            n69wspa2w2 = n69wspa30l[(n69wspa30l['def_id'] == n69wspa2it) & (n69wspa30l['ctx'] == 'input') & n69wspa30l['name'].str.startswith(value.strip())]
                    n69wspa2w2 = n69wspa2w2['name'].tolist()
                    n69wspa2w2 = [p for p in n69wspa2w2 if not p in ('cls', 'self')]
                    return n69wspa2w2
                elif item == 'func_desc':
                    if n69wspa2it == '<UNK>':
                        return '<UNK>'
                    if not helpinfo.get('hasObj') and n69wsp9p0f in '__call__':
                        return f"Function: class-level __call__ at {n69wspa2it.replace(DOT_REPL, '.')}"
                    n69wsp9p3c = pd.DataFrame(columns=list(n69wspa2ts.keys()))
                    if not n69wspa2it.startswith('[ENV]') and (not DOT_REPL in n69wspa2it):
                        if not n69wspa2it.startswith('[ENV]'):
                            n69wsp9osv = self.b69x8ynnt2(n69wspa2it, _skiplock=True, conn=conn)
                            n69wsp9p3c = self.select('funcs', conds=[{'def_id': n69wspa2it}], targets=['def_id', 'doc', 'is_async'], _skiplock=True, conn=conn)
                            n69wsp9osv, n69wsp9p3c = await asyncio.gather(n69wsp9osv, n69wsp9p3c)
                            n69wsp9p3c['raw_def_id'] = n69wsp9p3c['def_id']
                    if len(n69wsp9p3c) == 0:
                        n69wspa392 = n69wspa30l[(n69wspa30l['def_id'] == n69wspa2it) & (n69wspa30l['ctx'] == 'return')].to_dict(orient='records')
                        n69wsp9osv = {'inputs': n69wspa30l[(n69wspa30l['def_id'] == n69wspa2it) & (n69wspa30l['ctx'] == 'input')].sort_values(by='place').drop(['place'], axis=1).to_dict(orient='records'), 'return': n69wspa392[0] if n69wspa392 else {'name': 'return'}}
                        n69wsp9p3c = n69wspa2pg[n69wspa2pg['def_id'] == n69wspa2it].reset_index(drop=True)
                    if len(n69wsp9p3c) == 0:
                        n69wspa37a.append(f'Cannot find tool description for {n69wspa2it}.')
                        return '<UNK>'
                    n69wspa2l4 = n69wsp9p3c.to_dict(orient='records')[0]
                    n69wspa2l4['def_id'] = n69wspa2it
                    if helpinfo.get('desc_format') == 'dict':
                        n69wspa30m = x69xm5du02(n69wspa2l4, n69wsp9osv)
                    else:
                        n69wspa30m = x69xm5du03(n69wspa2l4, n69wsp9osv)
                    return n69wspa30m
            elif item == 'attr':
                n69wspa34f = helpinfo['obj']
                n69wspa32f = value.split('.')
                n69wspa2uu = n69wspa32f[-1]
                n69wspa32f[-1] = '[CODE]'
                n69wsp9p1l, _ = await self.b69x8ynnsx(n65d20cda3, n69wspa2wq, 'class', None, helpinfo={'obj': n69wspa34f, 'root': helpinfo['root'], 'full': True, 'israw': True}, n69wspa30r=n69wspa30r, consider_imports=True, _skiplock=True, conn=conn)
                if not n69wsp9p1l:
                    return []
                n69wsp9p1l = n69wsp9p1l[0]
                n69wspa2kf = await self.b69x8ynntr(n69wsp9p1l, n69wspa32f, n69wspa381, extpkgs, conn=None, _skiplock=True)
                if not n69wspa2kf[0]:
                    return []
                n69wspa2kf = n69wspa2kf[1]
                n69wsp9omn = 'self\\.([a-zA-Z_][a-zA-Z0-9_]*)'
                matches = re.findall(n69wsp9omn, n69wspa2kf)
                matches = [m for m in set(matches) if m.startswith(n69wspa2uu)]
                return matches
            elif item == 'class':
                if helpinfo.get('obj') is not None:
                    assert not value
                    if not helpinfo.get('obj').strip():
                        return []
                    if helpinfo['obj'].strip() in ('self', 'cls'):
                        n69wspa2v7 = await self.select('params', cond_sql=f"def_id = '{n69wspa2wq}' AND (name = '{helpinfo['obj'].strip()}')", targets=['name'], _skiplock=True, conn=conn)
                        if len(n69wspa2v7) == 0:
                            n69wspa37a.append(f'''This function does not have "{helpinfo['obj'].strip()}" as the zeroth parameter.''')
                            return []
                        if not helpinfo.get('full'):
                            n69wsp9orc = n69wspa2wq.split('*')[-1].split('/')[0]
                        else:
                            left, right = n69wspa2wq.rsplit('*', 1)
                            n69wspa2zg = right.split('/')[0]
                            n69wsp9orc = left + '*' + n69wspa2zg
                        return [n69wsp9orc]
                    sql = n69wspa2kh + f"\n                    SELECT \n                        `type`,`def_id` \n                    FROM vars_dup\n                    WHERE `name` = '{helpinfo['obj']}' AND rn=1\n                    ORDER BY LENGTH(`def_id`) DESC\n                    -- LIMIT 1 -- 可以这里limit也可以查出来看一下再取\n                    "
                    n69wsp9oq8 = await conn.execute(text(sql))
                    n69wsp9oq8 = n69wsp9oq8.fetchall()
                    n69wspa2dq = None
                    if n69wsp9oq8:
                        if len(n69wsp9oq8) == 1:
                            if not n69wsp9oq8[0][0]:
                                n69wsp9oq8 = []
                    if n69wsp9oq8 and helpinfo.get('israw'):
                        n69wspa37p = n69wsp9oq8[0][0].split('*')[-1]
                        if not '.' in n69wspa37p:
                            n69wspa30e = await self.b69x8ynnw4(n69wspa2wq, x69xm5dtzv=n69wspa2z4, conn=conn, _skiplock=_skiplock)
                            n69wspa2ej = n69wspa30e['froms'].split('\n')
                            for frline in n69wspa2ej:
                                if ' as ' in frline:
                                    if frline.split(' as ')[-1].strip() == n69wspa37p:
                                        n69wspa2dq = frline.split(' as ')[0].split(' import ')[-1].strip()
                                        n69wspa2dq = n69wsp9oq8[0][0].rsplit('*', 1)[0] + '*' + n69wspa2dq
                                        break
                        else:
                            n69wspa2dq = n69wsp9oq8[0][0].rsplit('*', 1)[0] + '*' + n69wspa37p.split('.')[-1]
                    if not n69wsp9oq8:
                        if '.' in helpinfo['obj']:
                            n69wspa371, n69wspa2lc = n69wspa30r or await self.b69x8ynnvd(n69wspa2wq, n69wspa381, 'imports', extpkgs=extpkgs, n69wspa2z4=n69wspa2z4, recur_objs=True, conn=conn, _skiplock=True)
                        else:
                            n69wspa371, n69wspa2lc = n69wspa30r or await self.b69x8ynnvd(n69wspa2wq, n69wspa381, 'froms', extpkgs=extpkgs, n69wspa2z4=n69wspa2z4, recur_objs=True, conn=conn, _skiplock=True)
                        n69wspa37a = n69wspa37a + [f"Failed to import '{f[0]}':{f[1]}" for f in n69wspa2lc['fails']]
                        n69wspa2es = n69wspa371['objs']
                        n69wsp9oq8 = n69wspa2es[n69wspa2es['name'] == helpinfo['obj'].replace('.', DOT_REPL)]
                        n69wsp9oq8 = n69wsp9oq8['type'].tolist()
                        n69wsp9oq8 = [r.replace(DOT_REPL, '.') for r in n69wsp9oq8]
                        n69wsp9oq8 = [n69wsp9oq8] if n69wsp9oq8 else []
                        if n69wsp9oq8 and helpinfo.get('israw'):
                            n69wspa2m0 = n69wspa371['classes']
                            n69wspa2qp = n69wspa2m0[n69wspa2m0['def_id'] == n69wsp9oq8[0][0].replace('.', DOT_REPL)]['raw_def_id'].tolist()
                            if n69wspa2qp:
                                n69wspa2dq = n69wspa2qp[0].replace(DOT_REPL, '.')
                    if not n69wsp9oq8:
                        n69wspa37a.append(f"Cannot find the class of object {helpinfo['obj']}.")
                        return []
                    else:
                        n69wspa2tc = n69wsp9oq8[0][0]
                        if helpinfo.get('israw'):
                            n69wspa2tc = n69wspa2dq or n69wspa2tc
                        if not n69wspa2tc:
                            n69wspa37a.append(f"Cannot find the class of obj {helpinfo['obj']}. This is normal if this object has never appeared before this node.")
                            return []
                        n69wspa2tc = n69wspa2tc.split('*')[-1] if not helpinfo.get('full') else n69wspa2tc
                        return [n69wspa2tc]
                elif '.' in value:
                    value = value.replace('.', DOT_REPL)
                    n69wspa371, n69wspa2lc = n69wspa30r or await self.b69x8ynnvd(n69wspa2wq, n69wspa381, 'imports', extpkgs=extpkgs, n69wspa2z4=n69wspa2z4, recur_objs=False, conn=conn, _skiplock=True)
                    n69wspa37a = n69wspa37a + [f"Failed to import '{f[0]}':{f[1]}" for f in n69wspa2lc['fails']]
                    n69wspa30s = n69wspa371['classes']
                    n69wspa2me = list(set([r.split('*')[-1] for r in n69wspa30s['def_id'].tolist()]))
                    n69wspa2me = [r.replace(DOT_REPL, '.') for r in n69wspa2me if r.startswith(value)]
                    return n69wspa2me
                else:
                    n69wspa2me = [self.b69x8ynntl(n69wspa2tb, targets=['def_id'], _skiplock=True, conn=conn) for n69wspa2tb in n69wspa2z4]
                    n69wspa2me = await asyncio.gather(*n69wspa2me)
                    n69wspa2me = pd.concat(n69wspa2me, ignore_index=True)
                    n69wspa2me = n69wspa2me['def_id'].tolist()
                    n69wspa2me = list(set([r.split('*')[-1] for r in n69wspa2me]))
                    n69wspa2me = [r for r in n69wspa2me if r.startswith(value)]
                    if not n69wspa2me:
                        n69wspa371, n69wspa2lc = n69wspa30r or await self.b69x8ynnvd(n69wspa2wq, n69wspa381, 'froms', extpkgs=extpkgs, n69wspa2z4=n69wspa2z4, recur_objs=False, conn=conn, _skiplock=True)
                        n69wspa37a = n69wspa37a + [f"Failed to import '{f[0]}':{f[1]}" for f in n69wspa2lc['fails']]
                        n69wspa30s = n69wspa371['classes']
                        n69wspa2me = list(set([r.split('*')[-1] for r in n69wspa30s['def_id'].tolist()]))
                        n69wspa2me = [r.replace(DOT_REPL, '.') for r in n69wspa2me if r.startswith(value)]
                    return n69wspa2me
            elif item == 'class_desc':
                if helpinfo.get('obj'):
                    if helpinfo['obj'].strip() in ('cls', 'self'):
                        if not '*' in n69wspa2wq:
                            return '<UNK>'
                        n69wspa2v7 = await self.select('params', cond_sql=f"def_id = '{n69wspa2wq}' AND (name = '{helpinfo['obj'].strip()}')", targets=['name'], _skiplock=True, conn=conn)
                        if len(n69wspa2v7) == 0:
                            return '<UNK>'
                        n69wspa2lm = n69wspa2wq.rfind('*')
                        n69wspa2sg = n69wspa2wq[n69wspa2lm:].find('/') + n69wspa2lm
                        assert n69wspa2sg > n69wspa2lm
                        n69wsp9ou1 = n69wspa2wq[:n69wspa2sg]
                        n69wspa2nr = await self.select('classes', conds=[{'def_id': n69wsp9ou1}], targets=['doc'])
                        if len(n69wspa2nr) == 0:
                            n69wspa37a.append(f'Cannot find class {n69wsp9ou1}.')
                            return '<UNK>'
                        n69wspa37g = n69wspa2nr.loc[0, 'doc']
                        n69wspa37g = n69wspa37g if n69wspa37g else 'This class lacks description.\n'
                        n69wspa30t = f"Type: {(n69wsp9ou1.split('*')[-1] if helpinfo['obj'].strip() != 'cls' else 'Type[' + n69wsp9ou1.split('*')[-1] + ']')} at {n69wsp9ou1}\n"
                        n69wspa37g = n69wspa30t + n69wspa37g
                        return n69wspa37g
                    else:
                        sql = n69wspa2kh + f",\n                        class_id AS (\n                            SELECT \n                                `type`,`def_id` \n                            FROM vars_dup\n                            WHERE `name` = '{helpinfo['obj']}' AND rn=1\n                            ORDER BY LENGTH(`def_id`) DESC\n                            LIMIT 1\n                        )\n                        SELECT class_id.`type`, classes.doc\n                        FROM class_id\n                        LEFT JOIN classes\n                        ON class_id.`type` = classes.def_id; \n                        "
                        n69wsp9oq8 = await conn.execute(text(sql))
                        n69wsp9oq8 = n69wsp9oq8.fetchall()
                        n69wspa2m7 = False
                        if not n69wsp9oq8:
                            n69wspa37a.append(f"Cannot find class of object {helpinfo['obj']}. This is normal if this object has never appeared before this node.")
                            n69wspa2m7 = True
                        if not n69wsp9oq8[0][0]:
                            n69wspa37a.append(f"Cannot find class of obj {helpinfo['obj']}. This is normal if this object has never appeared before this node.")
                            n69wspa2m7 = True
                        if n69wspa2m7:
                            clonly_ret, clonly_warns = await self.b69x8ynnsx(n65d20cda3, n69wspa2wq, item, value, helpinfo={**helpinfo, 'obj': None}, n69wspa30r=n69wspa30r, consider_imports=consider_imports, _skiplock=True, conn=conn)
                            for cw in clonly_warns:
                                n69wspa37a.append(cw)
                            return clonly_ret
                        n69wspa2ih = n69wsp9oq8[0][0]
                        if value:
                            n69wsp9oq8 = [r for r in n69wsp9oq8 if r[0].endswith('*' + value)]
                        if not n69wsp9oq8:
                            n69wspa37a.append(f"Cannot find matching class {value} of object {helpinfo['obj']}. We found its class might be {n69wspa2ih}.")
                            return '<UNK>'
                        if not n69wsp9oq8[0][0]:
                            n69wspa37a.append(f"Cannot find matching class {value} of obj {helpinfo['obj']}. We found its class might be {n69wspa2ih}.")
                            return '<UNK>'
                        n69wspa37g = n69wsp9oq8[0][1]
                        n69wspa37g = n69wspa37g if n69wspa37g else 'This class lacks description.\n'
                        n69wspa30t = f"Type: {n69wsp9oq8[0][0].split('*')[-1]} at {n69wsp9oq8[0][0]}\n"
                        n69wspa37g = n69wspa30t + n69wspa37g
                        return n69wspa37g
                else:
                    if not value:
                        return '<UNK>'
                    if not '.' in value:
                        n69wspa2me = [self.b69x8ynntl(n69wspa2tb, targets=['def_id', 'doc'], _skiplock=True, conn=conn) for n69wspa2tb in n69wspa2z4]
                        n69wspa2me = await asyncio.gather(*n69wspa2me)
                        n69wspa2me = pd.concat(n69wspa2me, ignore_index=True)
                        n69wspa2me = n69wspa2me.to_dict(orient='records')
                        n69wspa2me = [r for r in n69wspa2me if r['def_id'].endswith('*' + value)]
                        if not n69wspa2me:
                            n69wspa371, n69wspa2lc = n69wspa30r or await self.b69x8ynnvd(n69wspa2wq, n69wspa381, 'froms', extpkgs=extpkgs, n69wspa2z4=n69wspa2z4, recur_objs=True, conn=conn, _skiplock=True)
                            n69wspa37a = n69wspa37a + [f"Failed to import '{f[0]}':{f[1]}" for f in n69wspa2lc['fails']]
                            n69wspa30s = n69wspa371['classes']
                            n69wspa2me = [r for r in n69wspa30s.to_dict(orient='records') if r['def_id'].endswith(f'*{value}')]
                        if not n69wspa2me:
                            n69wspa37a.append(f'Cannot find class {value}.')
                            return '<UNK>'
                    else:
                        n69wspa371, n69wspa2lc = n69wspa30r or await self.b69x8ynnvd(n69wspa2wq, n69wspa381, 'imports', extpkgs=extpkgs, n69wspa2z4=n69wspa2z4, recur_objs=True, conn=conn, _skiplock=True)
                        n69wspa37a = n69wspa37a + [f"Failed to import '{f[0]}':{f[1]}" for f in n69wspa2lc['fails']]
                        n69wspa30s = n69wspa371['classes']
                        n69wspa30s['def_id'] = n69wspa30s['def_id'].apply(lambda x: x.replace(DOT_REPL, '.'))
                        n69wspa2me = [r for r in n69wspa30s.to_dict(orient='records') if r['def_id'].endswith(f"*{value.replace(DOT_REPL, '.')}")]
                    if len(n69wspa2me) == 0:
                        n69wspa37a.append(f'Cannot find class {value}.')
                        return '<UNK>'
                    n69wspa37g = n69wspa2me[0]['doc'] if n69wspa2me else '<UNK>'
                    n69wspa37g = n69wspa37g if n69wspa37g else 'This class lacks description.\n'
                    n69wspa37g = f"dir: {n69wspa2me[0]['def_id']}\n{n69wspa37g}" if n69wspa2me else '<UNK>'
                    return n69wspa37g
        n69wsp9onl = await self._batch_read([b69wsp9mrq], _skiplock=_skiplock, conn=conn)
        n69wsp9onl = n69wsp9onl[0]
        return (n69wsp9onl, n69wspa37a)

    def b69wspa0yf(self, code, pos, sugtype):
        n69wspa2xo = int(pos['lineNumber']) - 1
        n69wspa2id = int(pos['column']) - 1
        n69wspa2uu, targrow, already_params = parse_for_sugs(code, n69wspa2xo, n69wspa2id, sugtype)
        if sugtype == 'kernel_objs':
            n69wspa2or = requests_post(url=f'http://localhost:{configer.grapy.sandbox_port}/get_sugs', data={'objpart': n69wspa2uu})
            n69wspa2or = n69wspa2or['data']
        elif sugtype == 'kernel_params':
            n69wspa2or = requests_post(url=f'http://localhost:{configer.grapy.sandbox_port}/get_params', data={'funcpart': n69wspa2uu})
            n69wspa2or = n69wspa2or['data']
            assert isinstance(n69wspa2or, list)
            n69wspa2or = [{'name': s + '=' if not s.startswith('*') else '*' * s.count('*'), 'type': 'variable'} for s in n69wspa2or]
        else:
            raise ValueError(sugtype)
        return n69wspa2or

    async def b69x8ynnu3(self, n69wspa2wq, n65d20cda3, code, pos, word={'word': ''}, n69wspa381='', sugtype='objs', conn=None, _skiplock=False):
        n69wspa2xo = int(pos['lineNumber']) - 1
        n69wspa2id = int(pos['column']) - 1
        n69wspa2uu, targrow, already_params = parse_for_sugs(code, n69wspa2xo, n69wspa2id, sugtype)
        if '.' in n69wspa2uu:
            n69wspa2zs, value = n69wspa2uu.rsplit('.', 1)
        else:
            n69wspa2zs = ''
            value = n69wspa2uu

        async def b69wsp9mrq(conn):
            n69wspa38m = n69wspa2wq.split('^')[0]
            n69wspa34g = self.b69x8ynnvc(n65d20cda3, _skiplock=True, conn=conn)
            extpkgs = self.select('misc', cond_sql='true', targets=['external_pkgs'], conn=conn, _skiplock=True)
            n69wspa34g, extpkgs = await asyncio.gather(n69wspa34g, extpkgs)
            n69wsp9p51, _ = n69wspa34g
            extpkgs = extpkgs.loc[0, 'external_pkgs'] if len(extpkgs) > 0 else []
            n69wspa2e5 = n69wspa2wq + '^' + n69wsp9p51
            n69wspa2z4 = await self.b69x8ynnv5(n69wspa2e5, _skiplock=True, conn=conn)
            n69wspa2z4 = [v for v in n69wspa2z4 if x69xm5dtzx(v) == 'cond']
            n69wspa2z4.reverse()
            n69wspa2qr = []
            for v in n69wspa2z4:
                n69wspa2vg = v.rfind('^')
                n69wspa2ox = v.rfind('#')
                assert n69wspa2vg > 0 and n69wspa2ox > 0 and (n69wspa2vg < n69wspa2ox), (n69wspa2vg, n69wspa2ox)
                left = v[:n69wspa2vg]
                n69wspa2hh = v[n69wspa2vg + 1:n69wspa2ox]
                right = v[n69wspa2ox + 1:]
                if right.isdigit():
                    right = int(right)
                elif right.lower() == 'true':
                    right = True
                elif right.lower() == 'false':
                    right = False
                n69wspa2qr.append((left, n69wspa2hh, right))
            n69wspa2gi = [f"""\n                        (def_id = '{v[0]}' AND\n                         hid LIKE '{v[1]}.%' \n                         AND NOT SUBSTRING(hid,LENGTH('{v[1]}')+2) LIKE '%.%' \n                         AND branch = {("'" + v[2] + "'" if isinstance(v[2], str) else v[2])})\n                         """ for v in n69wspa2qr]
            n69wspa2gi = ' OR '.join(n69wspa2gi)
            if n69wspa2uu == '<UNDEFINED>':
                return []
            n69wspa2ps = 'both'
            if n69wspa2uu.startswith('self.') or n69wspa2uu.startswith('cls.'):
                n69wspa2ps = 'none'
            elif not '.' in n69wspa2uu:
                n69wspa2ps = 'froms'

            async def b69x8ynnux():
                n69wspa31x = []
                n69wspa371, impmisc = await self.b69x8ynnvd(n69wspa2wq, n69wspa381, n69wspa2ps, extpkgs=extpkgs, n69wspa2z4=n69wspa2z4, recur_objs=True, conn=None, _skiplock=True)
                n69wspa30r = (n69wspa371, impmisc)
                n69wspa2h7 = impmisc['imports_codes']['imports'].split('\n')
                n69wspa2zh = [c[6:].split(' as ')[-1].strip() for c in n69wspa2h7]
                return (n69wspa2zh, n69wspa30r)

            def b69wspa0xy(n69wspa2uu, n69wspa2zh):
                n69wspa2v3 = n69wspa2uu.split('.')
                n69wspa2jc = ''
                left = ''
                n69wspa2hh = ''
                right = ''
                for i in range(len(n69wspa2v3) - 1):
                    n69wspa36y = '.'.join(n69wspa2v3[:i + 1])
                    if n69wspa36y in n69wspa2zh:
                        n69wspa2jc = n69wspa36y
                        n69wspa2hh = n69wspa2v3[i + 1]
                        left = n69wspa2jc + '.' + n69wspa2hh
                        right = '.'.join(n69wspa2v3[i + 2:])
                if not n69wspa2jc:
                    n69wspa2hh = n69wspa2v3[0]
                    left = n69wspa2hh
                    right = '.'.join(n69wspa2v3[1:])
                else:
                    assert n69wspa2uu.startswith(n69wspa2jc + '.')
                return (left, right, left != n69wspa2uu)
            if sugtype == 'params':
                if not n69wspa2uu:
                    return []
                n69wspa31x = []
                n69wspa2zh, n69wspa30r = await b69x8ynnux()
                n69wspa35t = []
                if n69wspa2uu.count('.') == 0:
                    n69wspa368 = self.b69x8ynnsx(n65d20cda3, n69wspa2wq, 'params', n69wspa2uu, helpinfo={'hasObj': False, 'obj': '', 'root': n69wspa381, 'class': None}, n69wspa30r=n69wspa30r, _skiplock=True, conn=conn)
                    n69wspa2f5 = self.b69x8ynnsx(n65d20cda3, n69wspa2wq, 'params', '__init__', helpinfo={'hasObj': False, 'obj': '', 'root': n69wspa381, 'class': n69wspa2uu}, n69wspa30r=n69wspa30r, _skiplock=True, conn=conn)
                    n69wspa2jz = self.b69x8ynnsx(n65d20cda3, n69wspa2wq, 'params', '__call__', helpinfo={'hasObj': True, 'obj': n69wspa2uu, 'nodecode': code, 'row': targrow, 'root': n69wspa381, 'class': ''}, n69wspa30r=n69wspa30r, _skiplock=True, conn=conn)
                    if n69wspa2uu[0].lower() == n69wspa2uu[0]:
                        n69wspa35t = [n69wspa368, n69wspa2jz, n69wspa2f5]
                    else:
                        n69wspa35t = [n69wspa2f5, n69wspa368, n69wspa2jz]
                    for n69wsp9p6g in n69wspa35t:
                        n69wspa2w2, n69wspa37a = await n69wsp9p6g
                        if n69wspa2w2 != '<UNK>':
                            n69wspa31x = [p[0] for p in n69wspa2w2]
                            break
                else:
                    n69wspa31x = []
                    if not n69wspa2uu.startswith('self.'):
                        left, right, _ = b69wspa0xy(n69wspa2uu, n69wspa2zh)
                        n69wspa35t = []
                        if not right:
                            n69wspa2gc = self.b69x8ynnsx(n65d20cda3, n69wspa2wq, 'params', left, helpinfo={'hasObj': False, 'obj': '', 'root': n69wspa381, 'class': None}, n69wspa30r=n69wspa30r, _skiplock=True, conn=conn)
                            n69wspa2q0 = self.b69x8ynnsx(n65d20cda3, n69wspa2wq, 'params', '__init__', helpinfo={'hasObj': False, 'obj': '', 'root': n69wspa381, 'class': left}, n69wspa30r=n69wspa30r, _skiplock=True, conn=conn)
                            n69wspa35b = self.b69x8ynnsx(n65d20cda3, n69wspa2wq, 'params', '__call__', helpinfo={'hasObj': True, 'obj': left, 'nodecode': code, 'row': targrow, 'root': n69wspa381, 'class': ''}, n69wspa30r=n69wspa30r, _skiplock=True, conn=conn)
                            if n69wspa2hh[0].lower() == n69wspa2hh[0]:
                                n69wspa35t = [n69wspa2gc, n69wspa35b, n69wspa2q0]
                            else:
                                n69wspa35t = [n69wspa2q0, n69wspa2gc, n69wspa35b]
                        else:
                            n69wspa2q0 = self.b69x8ynnsx(n65d20cda3, n69wspa2wq, 'params', right, helpinfo={'hasObj': False, 'obj': '', 'root': n69wspa381, 'class': left}, n69wspa30r=n69wspa30r, _skiplock=True, conn=conn)
                            n69wspa35b = self.b69x8ynnsx(n65d20cda3, n69wspa2wq, 'params', right, helpinfo={'hasObj': True, 'obj': left, 'nodecode': code, 'row': targrow, 'root': n69wspa381, 'class': ''}, n69wspa30r=n69wspa30r, _skiplock=True, conn=conn)
                            if n69wspa2hh[0].lower() == n69wspa2hh[0]:
                                n69wspa35t = [n69wspa35b, n69wspa2q0]
                            else:
                                n69wspa35t = [n69wspa2q0, n69wspa35b]
                        for n69wsp9p6g in n69wspa35t:
                            n69wspa2w2, n69wspa37a = await n69wsp9p6g
                            if n69wspa2w2 != '<UNK>':
                                n69wspa31x = [p[0] for p in n69wspa2w2]
                                break
                    else:
                        if n69wspa2uu.startswith('self.'):
                            left = 'self'
                            right = n69wspa2uu[5:]
                        else:
                            left = 'cls'
                            right = n69wspa2uu[4:]
                        n69wspa2w2, n69wspa37a = await self.b69x8ynnsx(n65d20cda3, n69wspa2wq, 'params', right, helpinfo={'hasObj': left == 'self', 'obj': left, 'root': n69wspa381, 'class': ''}, n69wspa30r=n69wspa30r, _skiplock=True, conn=conn)
                        n69wspa31x = [p[0] for p in n69wspa2w2] if n69wspa2w2 != '<UNK>' else []
                n69wsp9onl = [{'name': n + '=' if not n.startswith('*') else '*' * n.count('*'), 'type': 'variable'} for n in n69wspa31x if not n in already_params]
                return n69wsp9onl
            n69wspa2or = []
            if not n69wspa2zs:
                if value:
                    n69wspa2g1 = '%'.join(value) + '%'
                    n69wspa2xn = f"\n                    WITH \n                        nodes_in_module AS ( \n                            SELECT `uid`,`hid`,`branch`,`def_id`\n                            FROM nodes\n                            WHERE def_id LIKE '{n69wspa38m}^%' OR def_id = '{n69wspa38m}'\n                        ),\n                        nodes_visible AS (\n                            SELECT * FROM nodes_in_module \n                            WHERE {n69wspa2gi}\n                        ),\n                        vars_visible AS ( \n                            SELECT \n                                vars.`name`,vars.`type`,vars.`def_id`,vars.`uid`,\n                                nodes_visible.hid AS hid\n                            FROM \n                                vars\n                            INNER JOIN \n                                nodes_visible \n                            ON \n                                vars.uid = nodes_visible.uid AND vars.name LIKE '{n69wspa2g1}'\n                        )\n                    SELECT `name` FROM vars_visible\n                    "
                    n69wsp9p5l = conn.execute(text(n69wspa2xn))
                    n69wspa2wd = self.b69x8ynnuj(n69wspa2z4, value, _skiplock=True, conn=conn)
                    n69wsp9p5l, n69wspa2wd = await asyncio.gather(n69wsp9p5l, n69wspa2wd)
                    n69wsp9p5l = n69wsp9p5l.fetchall()
                    n69wsp9p5l = [x[0] for x in n69wsp9p5l]
                    n69wspa2kj = [{'name': n, 'type': 'variable'} for n in n69wsp9p5l]
                    n69wspa347 = [{'name': n, 'type': 'function'} for n in n69wspa2wd[0]]
                    n69wspa2e7 = [{'name': n, 'type': 'class'} for n in n69wspa2wd[1]]
                    n69wspa2or = n69wspa2kj + n69wspa347 + n69wspa2e7
                else:
                    n69wspa2or = []
            if not n69wspa2or:
                n69wspa2zh, n69wspa30r = await b69x8ynnux()
                if not n69wspa2uu:
                    return []
                n69wspa2or = n69wspa2or + [{'name': m, 'type': 'module'} for m in n69wspa2zh if m.startswith(n69wspa2uu)]

                async def b69x8ynnv3(n69wspa2uu):
                    n69wspa34z = []
                    n69wspa368 = self.b69x8ynnsx(n65d20cda3, n69wspa2wq, 'func', n69wspa2uu, helpinfo={'hasObj': False, 'obj': '', 'root': n69wspa381, 'class': ''}, n69wspa30r=n69wspa30r, _skiplock=True, conn=conn)
                    n69wspa2iu = self.b69x8ynnsx(n65d20cda3, n69wspa2wq, 'class', n69wspa2uu, helpinfo={'root': n69wspa381}, n69wspa30r=n69wspa30r, _skiplock=True, conn=conn)
                    n69wspa2ib = self.b69x8ynnsx(n65d20cda3, n69wspa2wq, 'obj', n69wspa2uu, helpinfo={'root': n69wspa381, 'class': ''}, n69wspa30r=n69wspa30r, _skiplock=True, conn=conn)
                    n69wspa2nt = await asyncio.gather(n69wspa368, n69wspa2iu, n69wspa2ib)
                    n69wspa2hp = ['function', 'class', 'variable']
                    for ti, tips in enumerate(n69wspa2nt):
                        if tips[0] == '<UNK>':
                            continue
                        n69wspa34z = n69wspa34z + [{'name': n, 'type': n69wspa2hp[ti]} for n in tips[0]]
                    return n69wspa34z
                if n69wspa2uu.count('.') == 0:
                    n69wspa2or = n69wspa2or + await b69x8ynnv3(n69wspa2uu)
                else:
                    left, right, has_attr = b69wspa0xy(n69wspa2uu, n69wspa2zh)
                    if not has_attr:
                        n69wspa2or = n69wspa2or + await b69x8ynnv3(n69wspa2uu)
                    else:
                        n69wspa2jk = self.b69x8ynnsx(n65d20cda3, n69wspa2wq, 'func', right, helpinfo={'class': left, 'hasObj': False, 'obj': '', 'root': n69wspa381}, n69wspa30r=n69wspa30r, _skiplock=True, conn=conn)
                        n69wspa2u0 = self.b69x8ynnsx(n65d20cda3, n69wspa2wq, 'func', right, helpinfo={'class': '', 'hasObj': True, 'obj': left, 'root': n69wspa381}, n69wspa30r=n69wspa30r, _skiplock=True, conn=conn)
                        n69wspa2ql = self.b69x8ynnsx(n65d20cda3, n69wspa2wq, 'attr', right, helpinfo={'class': '', 'hasObj': True, 'obj': left, 'root': n69wspa381}, n69wspa30r=n69wspa30r, _skiplock=True, conn=conn)
                        n69wspa2nt = await asyncio.gather(n69wspa2jk, n69wspa2u0, n69wspa2ql)
                        for ti, tips in enumerate(n69wspa2nt):
                            if tips[0] == '<UNK>':
                                continue
                            n69wspa2or = n69wspa2or + [{'name': n, 'type': 'attribute' if ti == 2 else 'function'} for n in tips[0]]
                    n69wsp9p5g = n69wspa2uu.rsplit('.', 1)[0] + '.'
                    for sug in n69wspa2or:
                        if sug['name'].startswith(n69wsp9p5g):
                            sug['name'] = sug['name'][len(n69wsp9p5g):]
            return n69wspa2or
        n69wsp9onl = await self._batch_read([b69wsp9mrq], conn=conn, _skiplock=_skiplock)
        n69wsp9onl = n69wsp9onl[0]
        return n69wsp9onl

    async def b69x8ynnum(self, n69wspa2wq, n65d20cda3, code, pos, n69wspa381='', sugtype='objs', allowjedi=True, conn=None, _skiplock=False):
        n69wspa2gm = copy.deepcopy(self.n69wspa2gm)
        n69wspa2gs = False
        n69wspa2or = []
        if sugtype.startswith('kernel'):
            n69wspa2or = self.b69wspa0yf(code, pos, sugtype)
            return n69wspa2or
        if allowjedi and sugtype != 'params':
            if n69wspa2gm:
                n69wspa337 = n69wspa2wq.split('^')[0]
                if n69wspa2gm['entry_module'] == n69wspa337 and n69wspa2gm['target_uid'] == n65d20cda3:
                    n69wspa2gs = True
                    n69wspa2or = await self.b69x8ynnv7(n69wspa2gm, n69wspa2wq, n65d20cda3, code, pos, n69wspa381=n69wspa381, conn=conn, _skiplock=_skiplock)
        if not n69wspa2or:
            n69wspa2pv = '' if not n69wspa2gs else '，dc09'
            n69wspa2or = await self.b69x8ynnu3(n69wspa2wq, n65d20cda3, code, pos, {'word': ''}, n69wspa381=n69wspa381, sugtype=sugtype, conn=conn, _skiplock=_skiplock)
        return n69wspa2or

    async def b69x8ynnut(self, n69wspa2wq, n65d20cda3, n69wspa381='', conn=None, _skiplock=False):
        self.n69wspa2gm = {}
        n69wspa381 = n69wspa381.strip()
        n69wspa337 = n69wspa2wq.split('^')[0]
        n69wspa2i1 = await self.b69x8ynntt(n69wspa381, n69wspa337, 'allbelow', tolerance=2, style='pure', n69wspa2zf=None, codeswaps={n65d20cda3: SUGCODE_HOLDER}, import_range='allabove', conn=conn, _skiplock=_skiplock)
        self.n69wspa2gm = {'entry_module': n69wspa337, 'target_uid': n65d20cda3, 'codes': n69wspa2i1}

    async def b69x8ynnv7(self, n69wspa2gm, n69wspa2wq, n65d20cda3, code, pos, n69wspa381='', conn=None, _skiplock=False):
        n69wspa381 = n69wspa381.strip()
        n69wspa337 = n69wspa2wq.split('^')[0]
        n69wspa2gf = [n69wspa337]
        assert x69xm5dtzx(n69wspa337) == 'func'
        n69wspa2gm = copy.deepcopy(n69wspa2gm)
        assert n69wspa2gm['entry_module'] == n69wspa337 and n69wspa2gm['target_uid'] == n65d20cda3

        async def b69wsp9mrq(conn):

            async def b69x8ynnuv(n69wspa2it):
                n69wspa340 = time.time()
                n69wspa2i1 = n69wspa2gm['codes']
                n69wspa2ug = time.time()
                n69wspa2sm = n69wspa2i1['files'][n69wspa2i1['entry']]
                n69wspa2va = n69wspa2sm[:n69wspa2sm.find(SUGCODE_HOLDER)]
                n69wsp9p1r = len(n69wspa2va) - n69wspa2va.rfind('\n') - 1
                n69wspa2oo = code.replace('\n', '\n' + ' ' * n69wsp9p1r).strip() + '\n'
                n69wspa2sm = n69wspa2sm.replace(SUGCODE_HOLDER, n69wspa2oo)
                n69wsp9oyj = n69wspa2va.count('\n') + pos['lineNumber']
                n69wspa2id = n69wsp9p1r + pos['column'] - 1
                n69wspa2sx = {}
                for path, cod in n69wspa2i1['files'].items():
                    n69wspa2v4 = n69wspa381.replace('>', '/').strip('/')
                    if path == n69wspa2i1['entry']:
                        continue
                    if not n69wspa381:
                        n69wspa2sx[path] = cod
                    if path.startswith(n69wspa2v4 + '/'):
                        path = path[len(n69wspa2v4) + 1:]
                        n69wspa2sx[path] = cod
                extpkgs = n69wspa2i1['external_pkgs']
                n69wspa2ux = JediEnvManager()
                n69wspa2or = []
                try:
                    n69wspa2or = n69wspa2ux.suggest(n69wspa2sm, n69wsp9oyj, n69wspa2id, extpkgs=extpkgs, livecodes=n69wspa2sx)
                except Exception as e:
                    traceback.print_exc()
                return n69wspa2or
            n69wspa2or = []
            for n69wspa2jx in n69wspa2gf:
                n69wspa2or = await b69x8ynnuv(n69wspa2jx)
                if n69wspa2or:
                    break
            return n69wspa2or
        n69wspa2or = await self._batch_read([b69wsp9mrq], conn=conn, _skiplock=_skiplock)
        return n69wspa2or[0]

    async def b69x8ynnti(self, choice, n69wsp9omn, deffilter='', conn=None, _skiplock=False):
        if not n69wsp9omn.strip():
            return []
        if choice == 'nodes':
            n69wspa2tk = ['uid', 'code', 'toolcall', 'params_map', 'comments', 'expr', 'cases', 'iter', 'slice']
            n69wspa2r1 = ['def_id'] + n69wspa2tk
        elif choice == 'funcs':
            n69wspa2tk = ['globals', 'nonlocals', 'imports_code', 'deco_expr', 'doc']
            n69wspa2r1 = ['def_id'] + n69wspa2tk
        elif choice == 'classes':
            n69wspa2tk = ['bases', 'vars', 'deco_expr', 'doc']
            n69wspa2r1 = ['def_id'] + n69wspa2tk
        else:
            raise ValueError(choice)
        n69wsp9omn = n69wsp9omn.replace("'", "\\'")

        async def b69wsp9mrq(conn):
            n69wspa2rj = [f"`{s}` LIKE '%{n69wsp9omn}%'" for s in n69wspa2tk]
            n69wspa2rj = ' OR '.join(n69wspa2rj)
            n69wspa2pj = [f"CASE WHEN `{s}` LIKE '%{n69wsp9omn}%' THEN (CHAR_LENGTH(`{s}`) - CHAR_LENGTH(REPLACE(`{s}`, '{n69wsp9omn}', ''))) DIV CHAR_LENGTH('{n69wsp9omn}') ELSE 0 END AS `{s}`" if not s in ('def_id', 'uid') else s if s == 'def_id' else f"CASE WHEN `{s}` LIKE '%{n69wsp9omn}%' THEN CONCAT('y:',`{s}`) ELSE CONCAT('n:',`{s}`) END AS `{s}`" for s in n69wspa2tk]
            n69wspa2pj = ',\n'.join(n69wspa2pj)
            n69wspa2oq = aidle()
            if choice == 'nodes':
                sql = f"SELECT def_id, {n69wspa2pj} FROM nodes WHERE ({n69wspa2rj}) AND def_id LIKE '{deffilter}%' LIMIT 1000"
            else:
                n69wsp9p3w = '/' if choice == 'funcs' else '*'
                n69wspa2io = f"SUBSTRING_INDEX(def_id, '{n69wsp9p3w}', -1) LIKE '%{n69wsp9omn}%'"
                n69wspa36g = f"CASE WHEN `def_id` LIKE '%{n69wsp9omn}%' THEN CONCAT('y:',def_id) ELSE CONCAT('n:',def_id) END AS def_id"
                sql = f"SELECT {n69wspa36g},\n{n69wspa2pj} FROM {choice} WHERE ({n69wspa2rj} OR {n69wspa2io}) AND def_id LIKE '{deffilter}%' LIMIT 1000"
                if choice == 'funcs':
                    n69wspa2oq = self.select('params', cond_sql=f"name LIKE '%{n69wsp9omn}%'", targets=['def_id'], conn=conn, _skiplock=True)
            n69wspa2hj = conn.execute(text(sql))
            n69wsp9oq8, presult = await asyncio.gather(n69wspa2hj, n69wspa2oq)
            n69wsp9oq8 = n69wsp9oq8.fetchall()
            n69wsp9oq8 = pd.DataFrame(n69wsp9oq8, columns=n69wspa2r1)
            if choice == 'funcs':
                n69wspa2tj = n69wsp9oq8['def_id'].tolist()
                n69wspa2xi = [did[2:] for did in n69wspa2tj if did.startswith('y:')]
                n69wsp9oq8['def_id'] = n69wsp9oq8['def_id'].apply(lambda x: x[2:])
                n69wspa2tz = presult['def_id'].tolist()
                n69wspa2xt = [[p, n69wspa2tz.count(p)] for p in set(n69wspa2tz)]
                n69wspa2u7 = pd.DataFrame(n69wspa2xt, columns=['def_id', 'params'])
                n69wsp9oq8 = n69wsp9oq8.merge(n69wspa2u7, on='def_id', how='outer').fillna(0)
                n69wsp9oq8['def_id'] = n69wsp9oq8['def_id'].apply(lambda x: 'y:' + x if x in n69wspa2xi else 'n:' + x)
            n69wsp9oq8 = n69wsp9oq8.to_dict(orient='records')
            return n69wsp9oq8
        n69wsp9onl = await self._batch_read([b69wsp9mrq], _skiplock=_skiplock, conn=conn)
        return n69wsp9onl[0]

    async def b69x8ynnvf(self, conn=None, _skiplock=False):
        extpkgs = await self.select('misc', cond_sql='true', targets=['external_pkgs'], conn=conn, _skiplock=_skiplock)
        extpkgs = extpkgs.loc[0, 'external_pkgs'] if len(extpkgs) > 0 else []
        return extpkgs

    async def b69x8ynnvh(self, n69wsp9oxz, conn=None, _skiplock=False):
        assert isinstance(n69wsp9oxz, list), f'Paths must be a list, got {n69wsp9oxz}'
        for p in n69wsp9oxz:
            assert isinstance(p, str), f'Each path must be string, got {p}'
        n69wspa2hy = pd.DataFrame([{'user_id': self.x69xm5dtzt, 'external_pkgs': n69wsp9oxz}])
        await self.upsert('misc', n69wspa2hy, conn=conn, _skiplock=_skiplock)

    async def b69x8ynntz(self, conn=None, _skiplock=False):

        async def b69wspa5ai(conn):
            n69wspa2r3 = self.select('funcs', cond_sql='true', targets=['imports_code'], conn=conn, _skiplock=True)
            extpkgs = self.select('misc', cond_sql='true', targets=['external_pkgs'], conn=conn, _skiplock=True)
            n69wspa2r3, extpkgs = await asyncio.gather(n69wspa2r3, extpkgs)
            extpkgs = extpkgs.loc[0, 'external_pkgs'] if len(extpkgs) > 0 else []
            n69wspa2r3 = n69wspa2r3['imports_code'].tolist()
            n69wspa2p4 = ''
            for icode in n69wspa2r3:
                try:
                    b69wsp9mq1(icode)
                    n69wspa2p4 = n69wspa2p4 + ('\n' + icode)
                except Exception as e:
                    pass
            n69wspa2rx = await n69wspa5l3.b69x8ynrdt(n69wspa2p4, extpkgs)
            n69wspa30c = [{**n69wspa2qf, 'def_id': f'{UNIVERSAL_FAKER}/{UNIVERSAL_FAKER}', 'uid': UNIVERSAL_FAKER, 'ethnic': '[ENV]'}]
            n69wspa2h4 = [{**n69wspa2f8, 'uid': '_extuid', 'node_type': 'start', 'def_id': f'{UNIVERSAL_FAKER}/{UNIVERSAL_FAKER}', 'hid': '1.0'}]
            n69wspa2t1 = self.upsert('funcs', pd.DataFrame(n69wspa2rx['funcs'] + n69wspa30c), conn=conn, _skiplock=True)
            n69wspa308 = self.upsert('classes', pd.DataFrame(n69wspa2rx['classes']), conn=conn, _skiplock=True)
            n69wspa2vw = self.upsert('params', pd.DataFrame(n69wspa2rx['params']), conn=conn, _skiplock=True)
            n69wsp9ott = self.upsert('vars', pd.DataFrame(n69wspa2rx['objs']), conn=conn, _skiplock=True)
            n69wspa2yr = self.upsert('nodes', pd.DataFrame(n69wspa2h4), conn=conn, _skiplock=True)
            await asyncio.gather(n69wspa2t1, n69wspa308)
            await asyncio.gather(n69wspa2yr, n69wspa2vw)
            await n69wsp9ott
        await self._batch_write([b69wspa5ai], conn=conn, _skiplock=_skiplock)

    async def b69x8ynnty(self, conn=None, _skiplock=False):

        async def b69x8ynnu8(conn):
            n69wspa2le = self.delete('funcs', conds=[{'ethnic': '[ENV]'}], conn=conn, _skiplock=True)
            n69wspa2db = self.delete('classes', conds=[{'ethnic': '[ENV]'}], conn=conn, _skiplock=True)
            n69wspa377 = self.delete('vars', conds=[{'ethnic': '[ENV]'}], conn=conn, _skiplock=True)
            await asyncio.gather(n69wspa2le, n69wspa2db, n69wspa377)
        await self._batch_write([b69x8ynnu8], conn=conn, _skiplock=_skiplock)

    async def b69x8ynnt8(self, conn=None, _skiplock=False):

        async def b69x8ynnu4(conn):
            self.x69xm5dtzu = {}
            n69wspa2r7 = self.select('funcs', cond_sql='true', targets=['def_id', 'imports_code'], conn=conn, _skiplock=True)
            n69wspa2le = self.delete('funcs', cond_sql="ethnic = '[ENV]'", conn=conn, _skiplock=True)
            n69wspa2db = self.delete('funcs', cond_sql="ethnic = '[ENV]'", conn=conn, _skiplock=True)
            n69wspa377 = self.delete('vars', cond_sql="ethnic = '[ENV]'", conn=conn, _skiplock=True)
            n69wspa2r7, _, _, _ = await asyncio.gather(n69wspa2r7, n69wspa2le, n69wspa2db, n69wspa377)
            await self.b69x8ynntp(n69wspa2r7, conn=conn, _skiplock=True)
        await self._batch_write([b69x8ynnu4], conn=conn, _skiplock=_skiplock)

    async def b69x8ynntp(self, n69wsp9omb, conn=None, _skiplock=False):
        return
handler = A69wspa0yq()
if __name__ == '__main__':
    pass