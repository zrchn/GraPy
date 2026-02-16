
"""
Copyright (c) 2026 Zhiren Chen. Provided as-is for local use only.
"""

import re
import sys
import time
from fastapi.responses import StreamingResponse, JSONResponse
import json5
import requests
from core.cft.consts import FUNCID_COMMENT_LEFTLABEL, FUNCID_COMMENT_RIGHTLABEL, UID_COMMENT_LEFTLABEL, UID_COMMENT_RIGHTLABEL
from consts import LOGLEVEL
from loguru import logger
from basic.background_runner import AsyncTaskRunner
from basic.configer import configer
from fastapi import FastAPI, Request, WebSocket, HTTPException
import websockets
from starlette.websockets import WebSocketDisconnect
import asyncio
from asyncio import Queue
from basic.stream_handler import AsyncStreamHandler, SSEStream
import traceback
from utils.shared import enrich_by_type, generate_unique_id, tostr, pretty_repr, safe_default
import json
import orjson
from core.controller import ClientAbort, extract_roi, handler, x69xm5du01, nesttypes
from core.cft.utils import x69xm5dtzx, idgen
from core.output_client import client as output_client
import copy
from fastapi.middleware.cors import CORSMiddleware
import aiohttp

class ORJSONResponse(JSONResponse):
    media_type = 'application/json'

    def render(self, n69wsp9p72) -> bytes:
        if n69wsp9p72 is None:
            return b'null'
        return orjson.dumps(n69wsp9p72, option=orjson.OPT_NON_STR_KEYS | orjson.OPT_SERIALIZE_NUMPY)
app = FastAPI(default_response_class=ORJSONResponse)
ENRICH = None
if configer.grapy.enrich_vars_display == 'lazy':
    ENRICH = 'full'
elif configer.grapy.enrich_vars_display == 'off':
    ENRICH = 'debyte'
lazy_enrich_len = configer.grapy.vars_enrichable_maxlen_when_lazy
x69ydt6f7z = True
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_credentials=True, allow_methods=['*'], allow_headers=['*'])
save_lock = asyncio.Lock()
x69y6b52ru = {}

class X69y6b52rv(AsyncTaskRunner):

    def __init__(self):
        pass

    async def x69y6b52rw(self):
        while True:
            await asyncio.sleep(0.1)
            handler.x69xm5dtzq = True
            try:
                async with save_lock:
                    n69wspa2lt = list(x69y6b52ru.keys())
                    if n69wspa2lt:
                        pass
                    for n69wspa2qg in n69wspa2lt:
                        if x69y6b52ru.get(n69wspa2qg):
                            data = x69y6b52ru[n69wspa2qg]
                            x69y6b52ru[n69wspa2qg] = None
                            del x69y6b52ru[n69wspa2qg]
                            _ = await x69y6b52s2(data)
            except Exception as e:
                traceback.print_exc()
            finally:
                handler.x69xm5dtzq = False

    async def x69y6b52rx(self):
        self.start_monitor()
        n69wspa2q0 = asyncio.create_task(self.x69y6b52rw())
        self.submit_task(n69wspa2q0)
x69y6b52ry = X69y6b52rv()
inited = False

@app.post('/app/on_init')
async def on_init(data: dict):
    global inited
    try:
        if not inited:
            await handler.b69x8ynntf()
            await x69y6b52ry.x69y6b52rx()
        else:
            pass
        inited = True
        return {'error_code': 200}
    except Exception as e:
        traceback.print_exc()
        return {'error_code': 400, 'msg': str(e)}

@app.post('/appl1cl')
async def x69z63qyiz(data: dict):
    flow_id = data['workflowId']
    assert flow_id.count(':') == 1
    (left, right) = flow_id.split(':')
    nodedata = await handler.b69x8ynnvx(left, right, count_previews=True, to_bouncer=True)
    return nodedata

@app.post('/app/fjt')
async def b69x8ynnu6(data: dict):
    global x69ydt6f7z
    try:
        if x69xm5dtzx(data['moduleId']) == 'folder':
            return {'error_code': 400, 'msg': 'Cannot undo at folder level.'}
        (hist_scope, n69wspa2mh, n69wsp9oye, n69yxx8iwg) = await handler.b69x8ynnu6(data['moduleId'], data['xx'])
        if n69wspa2mh != '<EMPTY>':
            x69ydt6f7z = False
        return {'error_code': 200, 'scopeId': hist_scope, 'data': n69wspa2mh, 'alsoReload': n69wsp9oye, 'focusNodeId': n69yxx8iwg}
    except Exception as e:
        traceback.print_exc()
        return {'error_code': 400, 'msg': str(e)}

@app.post('/app/get_tool_counts')
async def get_tool_counts(data: dict):
    try:
        countinfo = await handler.b69x8ynnuh(data['node'])
        return {'error_code': 200, 'tool_counts': countinfo}
    except Exception as e:
        traceback.print_exc()
        return {'error_code': 400, 'msg': str(e)}

@app.post('/app/wkfym')
async def x69y6b52rz(data: dict):
    try:
        code = data['code']
        replace_info = data.get('upsertInfo', None)
        shell_br = data['shellId']
        if replace_info:
            if replace_info['mode'] == 'single':
                replace_info = {'mode': 'replace', 'section': [replace_info['uid'], replace_info['uid']]}
        await handler.b69x8ynnun(shell_br)
        location_now = data['locationNow']
        assert location_now.count(':') == 1
        (n69wspa2wq, n69wspa2jq) = location_now.split(':')
        assert n69wspa2jq == 'dag'
        shell_id = n69wspa2wq + '^' + shell_br
        if code != '<<TBY1>>':
            (_, n69wspa2nn, n69wspa31g) = await handler.b69x8ynnt6(code, shell_id, n69wspa2da=replace_info, cached=True, n69wspa381=data.get('rootpath'), tolerance=2, n69znp79nl=False)
        else:
            (_, n69wspa2nn, n69wspa31g) = await handler.b69x8ynnt4(shell_id, n69wspa2da=replace_info, cached=True, n69wspa381=data.get('rootpath'), tolerance=2)
        return {'error_code': 200, 'data': {**n69wspa2nn, **n69wspa31g}}
    except Exception as e:
        e = str(e)
        traceback.print_exc()
        n69wsp9ozf = traceback.format_exc()
        n69wsp9ovf = n69wsp9ozf.split('File "<unknown>",')[-1] if 'File "<unknown>",' in n69wsp9ozf else str(e)
        n69wsp9ovf = n69wsp9ovf.split('During handling of the above exception')[0] if 'During handling of the above exception' in n69wsp9ovf else n69wsp9ovf
        n69wsp9ovf = e if UID_COMMENT_LEFTLABEL in e else n69wsp9ovf
        return {'error_code': 400, 'msg': n69wsp9ovf}

@app.post('/app/wkfyj')
async def x69y6b52s0(data: dict):
    try:
        code = data['code']
        n69wspa2jq = data['scopeType']
        n69wsp9oqx = data.get('dels') or []
        if x69xm5dtzx(data['branchId']) == 'folder':
            raise RuntimeError('Cannot update modules as funcs. ')
        await handler.b69x8ynnun(data['branchId'])
        if x69xm5dtzx(data['branchId']) != 'class':
            replace_info = {'mode': 'tools', 'scope': n69wspa2jq, 'dels': n69wsp9oqx}
            if code != '<<TBY1>>':
                (_, n69wspa2nn, n69wspa31g) = await handler.b69x8ynnt6(code, data['branchId'], n69wspa2da=replace_info, cached=True, n69wspa381=data.get('rootpath'), tolerance=2, n69znp79nl=False)
            else:
                (_, n69wspa2nn, n69wspa31g) = await handler.b69x8ynnt4(data['branchId'], n69wspa2da=replace_info, cached=True, n69wspa381=data.get('rootpath'), tolerance=2)
        elif code != '<<TBY1>>':
            assert n69wspa2jq == 'funcs'
            (_, n69wspa2nn, n69wspa31g) = await handler.b69x8ynntj(code, data['branchId'], cached=True, n69wspa381=data.get('rootpath'), del_funcs=n69wsp9oqx, tolerance=2, n69znp79nl=False)
        else:
            (_, n69wspa2nn, n69wspa31g) = await handler.b69x8ynnt4(data['branchId'], cached=True, n69wspa381=data.get('rootpath'), del_funcs=n69wsp9oqx, tolerance=2)
        n69wspa2nn = {**n69wspa2nn, **n69wspa31g}
        return {'error_code': 200, 'data': n69wspa2nn}
    except Exception as e:
        traceback.print_exc()
        return {'error_code': 400, 'msg': str(e)}

@app.post('/app/reshell_by_llm')
async def reshell_by_llm(data: dict):
    try:
        n69wspa35p = data['llmScope']
        upsert_shell = data['shellId']
        replace_info = data['upsertInfo']
        n69wspa381 = data['rootpath']
        if replace_info:
            if replace_info['mode'] == 'single':
                replace_info = {'mode': 'replace', 'section': [replace_info['uid'], replace_info['uid']]}
        await handler.b69x8ynnun(upsert_shell)
        (_, n69wspa2nn, n69wspa31g) = await handler.b69x8ynnvz(data['prompt'], n69wspa35p, upsert_shell, n69wspa381=n69wspa381, n69wspa2zf=replace_info)
        n69wspa2nn = {**n69wspa2nn, **n69wspa31g}
        return {'error_code': 200, 'data': n69wspa2nn}
    except Exception as e:
        e = str(e)
        traceback.print_exc()
        exc = traceback.format_exc()
        excs = exc.split('\n')
        localdex = 0
        for (localdex, n69wsp9ou8) in enumerate(excs):
            if 'File "<unknown>"' in n69wsp9ou8:
                break
        appexc = '\n'.join(excs[localdex + 1:]) if localdex + 1 < len(excs) else str(e)
        appexc = appexc.split('During handling of the above exception')[0] if 'During handling of the above exception' in appexc else appexc
        appexc = e if UID_COMMENT_LEFTLABEL in e else appexc
        return {'error_code': 400, 'msg': appexc}

@app.websocket('/app/llm_reshell')
async def llm_reshell(websocket: WebSocket):
    streamer = websocket
    await streamer.accept()
    data = await streamer.receive_text()
    handler.x69xm5dtzq = True
    try:
        async with save_lock:
            await streamer.send_text(json.dumps({'event': 'start'}, ensure_ascii=False))
            try:
                data = json.loads(data)
            except:
                data = eval(data)
            n69wspa35p = data['llmScope']
            upsert_shell = data['shellId']
            replace_info = data['upsertInfo']
            n69wspa381 = data['rootpath']
            prompt = data['prompt']
            if replace_info:
                if replace_info['mode'] == 'single':
                    replace_info = {'mode': 'replace', 'section': [replace_info['uid'], replace_info['uid']]}
            await handler.b69x8ynnun(upsert_shell)
            (_, n69wspa2nn, n69wspa31g) = await handler.b69x8ynnvz(prompt, n69wspa35p, upsert_shell, n69wspa381=n69wspa381, n69wspa2zf=replace_info, ws=streamer)
            n69wspa2nn = {**n69wspa2nn, **n69wspa31g}
            await streamer.send_text(json.dumps({'event': 'reshell', 'data': n69wspa2nn}, ensure_ascii=False))
            await streamer.send_text(json.dumps({'event': 'end'}, ensure_ascii=False))
    except ClientAbort as e:
        pass
    except WebSocketDisconnect:
        pass
    except asyncio.CancelledError:
        await streamer.close()
    except Exception as e:
        traceback.print_exc()
        await streamer.send_text(json.dumps({'event': 'error', 'msg': f'Failed to generate: {str(e)}'}, ensure_ascii=False))
        await streamer.close()
    finally:
        handler.x69xm5dtzq = False

@app.post('/app/reset_agent_memory')
async def reset_agent_memory(data: dict):
    try:
        n69wspa337 = data['moduleId'].split('^')[0]
        handler.coder.delete_memory(lambda x: True, session_id=n69wspa337)
        return {'error_code': 200}
    except Exception as e:
        traceback.print_exc()
        return {'error_code': 400, 'msg': str(e)}

@app.post('/app/lhbc')
async def x69y6b52s1(data: dict):
    n69wspa2wq = data['defId']
    n69wsp9p51 = data['hid']
    parents = await handler.b69x8ynnv6(n69wspa2wq, n69wsp9p51)
    if not parents:
        return {'error_code': 400, 'msg': 'no parent found.'}
    return {'data': parents[-1]}

@app.post('/app/xjbh')
async def b69x8ynnst(data: dict):
    try:
        handler.x69xm5dtzq = True
        async with save_lock:
            await handler.b69x8ynnst(data['scopeId'], data['scopeType'])
        return {'error_code': 200}
    except Exception as e:
        traceback.print_exc()
        return {'error_code': 400, 'msg': str(e)}
    finally:
        handler.x69xm5dtzq = False

async def x69y6b52s2(data: dict):
    quiet = data.get('quiet')
    try:
        n69wsp9ore = data['repose']
        n69wspa2qg = data.get('scopeId')
        cached = data.get('cached')
        n69wspa381 = data.get('rootpath')
        n69wspa2f3 = data.get('revars', True)
        n69wspa2sf = data.get('preCache', False)
        compare_level = data.get('debounceLevel', 0)
        timestamp = data.get('timestamp', 0)
        n69wspa2wq = n69wspa2qg.split(':')[0]
        lazy = data.get('lazy')
        data = data['data']
        if '^' in n69wspa2wq:
            n69wspa38m = n69wspa2wq.split('^')[0]
            assert '/' in n69wspa38m, n69wspa38m
            handler.b69wspa0xr(n69wspa38m)
        if cached and n69wspa2sf:
            await handler.b69x8ynnun(n69wspa2qg)
        if data['scopeType'] == 'dag':
            n69wsp9onl = await handler.b69x8ynntg(data['nodes'], data['edges'], n69wspa2wq=n69wspa2wq, n69wsp9ore=n69wsp9ore, cached=x69xm5dtzx(n69wspa2wq) != 'folder', n69wspa2sf=False, n69wspa2f3=n69wspa2f3, revars_failure_behavior='debug', level='adlvt', compare_kwargs={'level': compare_level}, n69wspa381=n69wspa381, timestamp=timestamp)
            n69wsp9onl = n69wsp9onl if not quiet else data
            n69wsp9onl['error_code'] = 200
            if lazy:
                n69wsp9onl['saved'] = True
            return n69wsp9onl
        elif data['scopeType'] == 'funcs':
            (n69wspa2qg, n69wspa2jq) = n69wspa2qg.split(':')
            assert n69wspa2jq == 'funcs'
            n69wsp9onl = await handler.b69x8ynnvr(n69wspa2qg, data['nodes'], n69wsp9ore=n69wsp9ore, cached=x69xm5dtzx(n69wspa2wq) != 'folder', n69wspa2sf=False, timestamp=timestamp)
            n69wsp9onl = {'nodes': n69wsp9onl if not quiet else data['nodes'], 'error_code': 200}
            if lazy:
                n69wsp9onl['saved'] = True
            return n69wsp9onl
        elif data['scopeType'] == 'classes':
            (n69wspa2qg, n69wspa2jq) = n69wspa2qg.split(':')
            assert n69wspa2jq == 'classes'
            n69wsp9onl = await handler.b69x8ynntk(n69wspa2qg, data['nodes'], n69wsp9ore=n69wsp9ore, cached=x69xm5dtzx(n69wspa2wq) != 'folder', n69wspa2sf=False, timestamp=timestamp)
            n69wsp9onl = {'nodes': n69wsp9onl if not quiet else data['nodes'], 'error_code': 200}
            if lazy:
                n69wsp9onl['saved'] = True
            return n69wsp9onl
        else:
            return {'error_code': 400, 'msg': f"wrong scopeType: {data['scopeType']}"}
    except Exception as e:
        traceback.print_exc()
        ecode = 400
        n69wsp9ovf = str(e)
        if str(e).startswith('[404]'):
            ecode = 404
            n69wsp9ovf = str(e)[5:].strip()
            return {'error_code': ecode, 'msg': n69wsp9ovf}
        return {'error_code': ecode, 'msg': n69wsp9ovf} if not quiet else {'nodes': data['nodes'], 'edges': data.get('edges', []), 'error_code': 200}

async def x69y6b52s3(data: dict):
    try:
        n69wsp9ore = data['repose']
        n69wspa2qg = data.get('scopeId')
        n69wspa2wq = n69wspa2qg.split(':')[0]
        check_grammar = data.get('checkGrammar', False)
        data = data['data']
        if data['scopeType'] == 'dag':
            try:
                n69wsp9onl = await handler.b69x8ynntg(data['nodes'], data['edges'], n69wspa2wq=n69wspa2wq, n69wsp9ore=n69wsp9ore, level='adlvv' if check_grammar else 'adlvl', _skiplock=True)
                n69wsp9onl['error_code'] = 200
            except Exception as e:
                traceback.print_exc()
                n69wsp9onl = {'error_code': 300, **data, 'msg': f'Failed to validate: {e}'}
            await asyncio.sleep(0.5)
            return n69wsp9onl
        else:
            return {'error_code': 400, 'msg': f"wrong scopeType: {data['scopeType']}"}
    except Exception as e:
        traceback.print_exc()
        return {'error_code': 400, 'msg': str(e)}

@app.post('/app/crldm')
async def x69y6b52s4(data: dict):
    global x69ydt6f7z
    if not x69ydt6f7z and data.get('label') == 'byEnterSubs':
        return {'error_code': 200, 'nodes': data['data']['nodes'], 'edges': data['data'].get('edges', [])}
    x69ydt6f7z = True
    try:
        if data.get('lazy') == True:
            assert data.get('quiet'), 'em001'
            if handler.x69xm5dtzq and (not data.get('caserelas')):
                x69y6b52ru[data['scopeId']] = data
                return {'error_code': 200}
        handler.x69xm5dtzq = True
        try:
            async with save_lock:
                x69y6b52ru[data['scopeId']] = None
                del x69y6b52ru[data['scopeId']]
                n69wsp9onl = await x69y6b52s2(data)
            return n69wsp9onl
        finally:
            handler.x69xm5dtzq = False
    finally:
        pass

@app.post('/app/jrldm')
async def x69y6b52s5(data: dict):
    true = True
    false = False
    null = 'null'
    n69wsp9onl = await x69y6b52s3(data)
    for todel in ['width', 'height', 'selected', 'dragging']:
        for n69wspa2mh in n69wsp9onl['nodes']:
            if todel in n69wspa2mh:
                del n69wspa2mh[todel]
    return n69wsp9onl

@app.post('/app/wchern')
async def x69y6b52s6(data: dict):
    try:
        n69wsp9oya = data['nodes']
        n69wspa34y = data['edges']
        n69wspa2wq = data['defId']
        target = data['targetUid']
        n69wsp9onl = await handler.b69x8ynntg(n69wsp9oya, n69wspa34y, n69wspa2wq=n69wspa2wq, n69wsp9ore=False, level='adlvx', _skiplock=True)
        return {'error_code': 200, 'data': n69wsp9onl}
    except Exception as e:
        traceback.print_exc()
        return {'error_code': 400, 'msg': str(e)}

@app.post('/app/ljm')
async def b69x8ynntv(data: dict):
    try:
        n69wspa2zf = None
        n69wspa2um = False
        is_shell = False
        if data.get('labelInfo'):
            if data['labelInfo']['mode'] == 'single':
                if data['labelInfo'].get('node_type') == 'start':
                    is_shell = True
                    n69wspa2zf = data['labelInfo']
                elif data['labelInfo']['node_type'] not in nesttypes:
                    n69wspa2um = True
                    n69wspa2zf = data['labelInfo']
                else:
                    n69wspa2zf = {'mode': 'replace', 'section': (data['labelInfo']['uid'], data['labelInfo']['uid'])}
            else:
                n69wspa2zf = data['labelInfo']
        if is_shell:
            (code, xcode) = await handler.b69x8ynnu5(data['baseId'], n69wspa2k7=n69wspa2zf['node_br'], style='pure', tolerance=2)
        elif not n69wspa2um:
            (code, xcode) = await handler.b69x8ynntv(data['baseId'], data['choice'], n69wspa2zf=n69wspa2zf, style='pure', tolerance=2)
        else:
            (code, xcode) = await handler.b69x8ynnv8(n69wspa2zf['uid'], n69wspa2zf['node_type'], style='pure', tolerance=2)
        code = extract_roi(code)
        return {'error_code': 200, 'data': code}
    except Exception as e:
        traceback.print_exc()
        return {'error_code': 400, 'msg': str(e)}

@app.post('/app/cxlg')
async def x69y6b52s7(data: dict):
    data = data['data']
    for n69wspa2ww in data['edges']:
        n69wspa2ww['id'] = idgen.generate('l')
    return data

@app.post('/app/gyzh')
async def x69y6b52s8(data: dict):
    try:
        await handler.b69x8ynnvk(data['uid'], data['handleMap'], todel=data.get('todel', None))
        return {'error_code': 200}
    except Exception as e:
        traceback.print_exc()
        return {'error_code': 400, 'msg': str(e)}

@app.post('/app/bgdcl')
async def b69wsp9moz(data: dict):
    try:
        (n69wsp9oz2, newedges) = await handler.b69x8ynnvo(data['nodeType'], data['defId'], data['srcNodeId'], data['srcHandleId'], data['srcX'], data['srcY'])
        return {'error_code': 200, 'data': {'nodes': n69wsp9oz2, 'edges': newedges}}
    except Exception as e:
        traceback.print_exc()
        return {'error_code': 400, 'msg': str(e)}

@app.post('/app/gggg')
async def x69y6b52sd(data: dict):
    try:
        if data['nodeType'] == 'func':
            n69wsp9opc = await handler.b69x8ynnuz(data['scopeId'], data['srcX'], data['srcY'])
            return {'error_code': 200, 'data': n69wsp9opc}
        elif data['nodeType'] == 'class':
            n69wsp9opc = await handler.b69x8ynnvy(data['scopeId'], data['srcX'], data['srcY'])
            return {'error_code': 200, 'data': n69wsp9opc}
    except Exception as e:
        traceback.print_exc()
        return {'error_code': 400, 'data': str(e)}

@app.post('/app/lhh')
async def b69x8ynnvc(data: dict):
    n65d20cda3 = data['uid']
    try:
        (n69wsp9p51, n69wspa2w8) = await handler.b69x8ynnvc(n65d20cda3)
        return {'error_code': 200, 'hid': n69wsp9p51}
    except Exception as e:
        return {'error_code': 400, 'msg': str(e)}

@app.post('/app/lgkk')
async def x69y6b52se(data: dict):
    n65d20cda3 = data['uid']
    try:
        (n69wsp9p51, n69wspa2w8) = await handler.b69x8ynnvc(n65d20cda3)
        assert '.' in n69wsp9p51, f'uid {n65d20cda3}查出的hid {n69wsp9p51}不带.'
        n69wspa2ii = n69wsp9p51[:n69wsp9p51.rfind('.')]
        return {'error_code': 200, 'parentId': n69wspa2ii, 'branch': str(n69wspa2w8)}
    except Exception as e:
        return {'error_code': 400, 'msg': str(e)}

@app.post('/app/ndbs')
async def b69x8ynnsw(data: dict):
    try:
        n69wspa2f3 = await handler.b69x8ynnsw(data['uid'], data.get('runId'), start_block_dotted=True, _skiplock=True)
        for v in n69wspa2f3:
            if not v.get('from_node') and (not v.get('from_def')):
                v['from'] = '[local]'
            else:
                v['from'] = (v.get('from_def') or '') + ' ' + (v.get('from_node') or '')
            if isinstance(v.get('repr'), str) and ENRICH == 'full':
                v['repr'] = enrich_by_type(v['repr'], dtype=v.get('type'), enrichable_len=lazy_enrich_len)
        n69wspa2f3 = [{k: v for (k, v) in n69wsp9oq0.items() if not k in ('ctx', 'uid', 'ethnic', 'from_def', 'from_node')} for n69wsp9oq0 in n69wspa2f3]
        urvs = []
        urvset = set()
        for v in n69wspa2f3:
            if v['name'] in urvset:
                continue
            urvset.add(v['name'])
            urvs.append(v)
        return {'error_code': 200, 'data': urvs}
    except Exception as e:
        return {'error_code': 400, 'msg': str(e)}

def is_solid(string):
    if not string:
        return False
    if not string.strip():
        return False
    return True

@app.post('/app/nddlu')
async def x69y6b52sf(data: dict):
    try:
        (rsp, n69wspa37a) = await handler.b69x8ynnsx(data['uid'], data['defId'], 'class', None, helpinfo={'obj': data['objName'], 'root': data['rootpath']})
        if n69wspa37a:
            pass
        if not rsp:
            return {'error_code': 200, 'data': None, 'warns': n69wspa37a}
        return {'error_code': 200, 'data': rsp[0], 'warns': n69wspa37a}
    except Exception as e:
        traceback.print_exc()
        e = str(e)
        tb = traceback.format_exc()
        if 'File "<unknown>",' in tb:
            e = tb.split('File "<unknown>",')[1] + '\n' + e
        if '有bug，一个uid查出0个或好几个hid' in tb:
            e = 'Target node not found in DB. If this node is newly added, it takes a few seconds for suggestions to be available.'
        return {'error_code': 400, 'msg': str(e)}

@app.post('/app/lwjs')
async def b69x8ynnvq(data: dict):
    n69wspa381 = data.get('rootpath', '')
    try:
        n69wspa36m = await handler.b69x8ynnvq(n69wspa381=n69wspa381)
        return {'data': n69wspa36m, 'error_code': 200}
    except Exception as e:
        traceback.print_exc()
        return {'error_code': 400, 'msg': str(e)}

@app.post('/app/nfzm')
async def x69y6b52sg(data: dict):
    try:
        if data['metadata']['type'] == 'tool':
            helpinfo = {}
            if data['metadata']['item'] == 'func':
                helpinfo = {'class': data['metadata'].get('class'), 'hasObj': data['metadata'].get('hasObj'), 'obj': (data['metadata'].get('obj') or '').strip()}
            elif data['metadata']['item'] == 'argname':
                n69wsp9p0f = data['metadata'].get('func')
                if not n69wsp9p0f:
                    if not data['metadata'].get('class'):
                        return {'error_code': 200, 'data': []}
                    if data['metadata'].get('hasObj'):
                        n69wsp9p0f = '__call__'
                    else:
                        n69wsp9p0f = '__init__'
                helpinfo = {'class': data['metadata'].get('class'), 'func': n69wsp9p0f, 'hasObj': data['metadata'].get('hasObj'), 'obj': (data['metadata'].get('obj') or '').strip()}
            elif data['metadata']['item'] == 'obj':
                helpinfo = {'class': data['metadata'].get('class')}
            helpinfo['root'] = data['metadata'].get('root')
            (n69wsp9onl, n69wspa37a) = await handler.b69x8ynnsx(data['metadata']['uid'], data['metadata']['def_id'], data['metadata']['item'], data['value'].strip(), helpinfo=helpinfo, _skiplock=True)
            if n69wspa37a:
                pass
            if data['metadata']['item'] == 'func' and helpinfo.get('hasObj') and helpinfo.get('obj') and ('.' in data['value']) and isinstance(n69wsp9onl, list):
                n69wsp9onl = [data['value'][:data['value'].rfind('.') + 1] + r for r in n69wsp9onl]
            n69wspa2n4 = data['metadata'].get('exists', [])
            return {'error_code': 200, 'data': [{'value': v} for v in n69wsp9onl if not v in n69wspa2n4] if n69wsp9onl != '<UNK>' else [], 'warns': '\n'.join(n69wspa37a)}
        raise
    except Exception as e:
        traceback.print_exc()
        e = str(e)
        tb = traceback.format_exc()
        if 'File "<unknown>",' in tb:
            e = tb.split('File "<unknown>",')[1] + '\n' + e
        if '有bug，一个uid查出0个或好几个hid' in tb:
            e = 'Target node not found in DB. If this node is newly added, it takes a few seconds for suggestions to be available.'
        return {'error_code': 400, 'msg': str(e)}

@app.post('/app/get_external_paths')
async def get_external_paths(data: dict):
    try:
        n69wsp9onl = await handler.b69x8ynnvf()
        n69wspa2ky = json.dumps(n69wsp9onl, ensure_ascii=False).strip().strip('[').strip(']')
        return {'error_code': 200, 'data': n69wspa2ky}
    except Exception as e:
        traceback.print_exc()
        return {'error_code': 400, 'msg': str(e)}

@app.post('/app/set_external_paths')
async def set_external_paths(data: dict):
    try:
        xpaths = data['paths']
        xpaths = xpaths.strip().strip('[').strip(']').strip(',')
        xpaths = json5.loads('[' + xpaths + ']')
        await handler.b69x8ynnvh(xpaths)
        return {'error_code': 200, 'data': None}
    except Exception as e:
        traceback.print_exc()
        return {'error_code': 400, 'msg': str(e)}

@app.post('/app/get_untracked_vars')
async def b69x8ynnuf(data: dict):
    try:
        n69wspa2vq = await handler.b69x8ynnuf(data['defId'])
        n69wspa2vq = [v.strip("'").strip('"') for v in n69wspa2vq]
        n69wspa2vq = ', '.join(n69wspa2vq)
        return {'error_code': 200, 'data': n69wspa2vq}
    except Exception as e:
        traceback.print_exc()
        return {'error_code': 400, 'msg': str(e)}

@app.post('/app/set_untracked_vars')
async def b69x8ynntm(data: dict):
    try:
        n69wsp9p5l = data['varNames'].strip().strip('[').strip(']').strip(',').split(',')
        n69wsp9p5l = [v.strip().strip('"').strip("'") for v in n69wsp9p5l if v.strip()]
        await handler.b69x8ynntm(data['defId'], n69wsp9p5l)
        return {'error_code': 200, 'data': None}
    except Exception as e:
        traceback.print_exc()
        return {'error_code': 400, 'msg': str(e)}

@app.post('/app/lctk')
async def x69y6b52sh(data: dict):
    try:
        if not is_solid(data['funcName']):
            if is_solid(data['className']):
                if data.get('hasObj'):
                    data['funcName'] = '__call__'
                else:
                    data['funcName'] = '__init__'
        (n69wsp9onl, n69wspa37a) = await handler.b69x8ynnsx(data['uid'], data['defId'], 'params', data['funcName'], helpinfo={'hasObj': data.get('hasObj'), 'obj': (data.get('obj') or '').strip(), 'root': data['rootpath'], 'class': data['className'].strip() if isinstance(data['className'], str) else data['className']}, _skiplock=True)
        if n69wspa37a:
            pass
        return {'error_code': 200, 'data': n69wsp9onl, 'warns': '\n'.join(n69wspa37a)}
    except Exception as e:
        traceback.print_exc()
        e = str(e)
        tb = traceback.format_exc()
        if 'File "<unknown>",' in tb:
            e = tb.split('File "<unknown>",')[1] + '\n' + e
        if '有bug，一个uid查出0个或好几个hid' in tb:
            e = 'Target node not found in DB. If this node is newly added, it takes a few seconds for suggestions to be available.'
        return {'error_code': 400, 'msg': str(e)}

@app.post('/app/nhxgs')
async def x69y6b52si(data: dict):
    try:
        if not (data['funcName'].strip() if isinstance(data['funcName'], str) else data['funcName']) and (not (data['className'].strip() if isinstance(data['className'], str) else data['className'])):
            return {'error_code': 200, 'data': '<UNK>'}
        if is_solid(data['className']):
            if not (data['funcName'].strip() if isinstance(data['funcName'], str) else data['funcName']):
                if data.get('hasObj'):
                    data['funcName'] = '__call__'
                else:
                    data['funcName'] = '__init__'
        (n69wsp9onl, n69wspa37a) = await handler.b69x8ynnsx(data['uid'], data['defId'], 'func_desc', data['funcName'], helpinfo={'hasObj': data.get('hasObj'), 'obj': (data.get('obj') or '').strip(), 'root': data['rootpath'], 'class': data['className'].strip() if isinstance(data['className'], str) else data['className']})
        if n69wspa37a:
            pass
        return {'error_code': 200, 'data': n69wsp9onl, 'warns': '\n'.join(n69wspa37a)}
    except Exception as e:
        traceback.print_exc()
        e = str(e)
        tb = traceback.format_exc()
        if 'File "<unknown>",' in tb:
            e = tb.split('File "<unknown>",')[1] + '\n' + e
        if '有bug，一个uid查出0个或好几个hid' in tb:
            e = 'Target node not found in DB. If this node is newly added, it takes a few seconds for suggestions to be available.'
        return {'error_code': 400, 'msg': str(e)}

@app.post('/app/lludc')
async def x69y6b52sj(data: dict):
    try:
        if not (data['className'].strip() if isinstance(data['className'], str) else data['className']) and (not (data['objName'].strip() if isinstance(data['objName'], str) else data['objName'])):
            return {'error_code': 200, 'data': '<UNK>'}
        (n69wsp9onl, n69wspa37a) = await handler.b69x8ynnsx(data['uid'], data['defId'], 'class_desc', data['className'], helpinfo={'root': data['rootpath'], 'obj': data['objName'].strip() if isinstance(data['objName'], str) else data['objName']})
        if n69wspa37a:
            pass
        return {'error_code': 200, 'data': n69wsp9onl, 'warns': '\n'.join(n69wspa37a)}
    except Exception as e:
        traceback.print_exc()
        e = str(e)
        tb = traceback.format_exc()
        if 'File "<unknown>",' in tb:
            e = tb.split('File "<unknown>",')[1] + '\n' + e
        if '有bug，一个uid查出0个或好几个hid' in tb:
            e = 'Target node not found in DB. If this node is newly added, it takes a few seconds for suggestions to be available.'
        return {'error_code': 400, 'msg': str(e)}

@app.post('/app/lmfz')
async def b69x8ynnu3(data: dict):
    try:
        channel = data.get('channel') or 'hybrid'
        n69wsp9onl = await handler.b69x8ynnum(data['defId'], data['uid'], data['code'], data['pos'], data['rootpath'], sugtype=data.get('sugtype', 'objs'), allowjedi=channel == 'hybrid', _skiplock=True)
        return {'error_code': 200, 'data': n69wsp9onl}
    except Exception as e:
        traceback.print_exc()
        if '有bug，一个uid查出0个或好几个hid' in traceback.format_exc():
            e = 'Target node not found in DB. If this node is newly added, it takes a few seconds for suggestions to be available.'
        return {'error_code': 400, 'msg': str(e)}

@app.post('/app/bigxj')
async def x69y6b52sk(data: dict):
    try:
        await handler.b69x8ynnut(n69wspa2wq=data['defId'], n65d20cda3=data['uid'], n69wspa381=data['rootpath'])
        return {'error_code': 200}
    except Exception as e:
        traceback.print_exc()
        return {'error_code': 400, 'msg': str(e)}

@app.post('/app/xdkl')
async def x69y6b52sl(data: dict):
    try:
        upsert_info = data['upsertInfo']
        if upsert_info['mode'] == 'replace':
            await handler.b69x8ynnv4(upsert_info['section'][0], upsert_info['section'][1], data['nodes'], data['edges'])
        elif upsert_info['mode'] == 'insert':
            await handler.b69x8ynnu0(upsert_info['after'], data['nodes'], data['edges'], shell=data['shell'])
        elif upsert_info['mode'] == 'allbelow':
            await handler.b69x8ynnui(data['shell'], data['nodes'], data['edges'])
        elif upsert_info['mode'] == 'single':
            await handler.b69x8ynnv4(upsert_info['uid'], upsert_info['uid'], data['nodes'], data['edges'])
        else:
            raise ValueError(f"Unsupported mode: {upsert_info['mode']}")
        return {'error_code': 200}
    except Exception as e:
        traceback.print_exc()
        return {'error_code': 400, 'msg': str(e)}

@app.post('/app/czgg')
async def b69x8ynnts(data: dict):
    try:
        if not '^' in data['defId']:
            raise RuntimeError('Cannot copy module as a func. Open the module (dag) and copy the main branch instead.')
        await handler.b69x8ynnts(data['defId'])
        return {'error_code': 200}
    except Exception as e:
        traceback.print_exc()
        return {'error_code': 400, 'msg': str(e)}

@app.post('/app/director/module_get_target')
async def b69x8ynntx(data: dict):
    try:
        n69wspa38m = data['moduleId']
        target = data['target']
        n69wsp9onl = await handler.b69x8ynntx(n69wspa38m, target)
        return {'error_code': 200, 'data': n69wsp9onl}
    except Exception as e:
        traceback.print_exc()
        return {'error_code': 400, 'msg': str(e)}

@app.post('/app/director/module_edit_target')
async def b69x8ynnvn(data: dict):
    try:
        n69wspa38m = data['moduleId']
        target = data['target']
        n69wsp9p72 = data['content']
        await handler.b69x8ynnvn(n69wspa38m=n69wspa38m, target=target, n69wsp9p72=n69wsp9p72)
        return {'error_code': 200}
    except Exception as e:
        traceback.print_exc()
        return {'error_code': 400, 'msg': str(e)}

@app.post('/app/director/add_new_module')
async def b69x8ynnvj(data: dict):
    try:
        n69wspa38m = data['moduleId'].split('.')[0]
        rootfilter = data.get('rootfilter', '')
        if x69xm5dtzx(n69wspa38m) == 'folder':
            n69wspa38m = 'grapy/' + n69wspa38m
        n69wspa38n = await handler.b69x8ynnvj(n69wspa38m, rootfilter=rootfilter)
        return {'error_code': 200}
    except Exception as e:
        traceback.print_exc()
        return {'error_code': 400, 'msg': str(e)}

@app.post('/app/director/del_module')
async def b69x8ynnt0(data: dict):
    try:
        n69wspa38m = data['moduleId']
        rootfilter = data.get('rootfilter', '')
        n69wspa38n = await handler.b69x8ynnt0(n69wspa38m, rootfilter=rootfilter)
        return {'error_code': 200}
    except Exception as e:
        traceback.print_exc()
        return {'error_code': 400, 'msg': str(e)}

@app.post('/app/director/module_rename')
async def b69x8ynnw3(data: dict):
    try:
        old_module_id = data['oldModuleId']
        n69wspa2o4 = data['newModuleId']
        rootfilter = data.get('rootfilter', '')
        n69wspa38n = await handler.b69x8ynnw3(old_module_id, n69wspa2o4, rootfilter=rootfilter)
        return {'error_code': 200}
    except Exception as e:
        traceback.print_exc()
        return {'error_code': 400, 'msg': str(e)}

@app.post('/app/yxssy')
async def b69x8ynnti(data: dict):
    try:
        n69wsp9oq8 = await handler.b69x8ynnti(data['choice'], data['pattern'], deffilter=data.get('deffilter', ''))
        return {'error_code': 200, 'data': n69wsp9oq8}
    except Exception as e:
        traceback.print_exc()
        return {'error_code': 400, 'msg': str(e)}
opened_scopes = {}

@app.post('/app/dccbj')
async def x69y6b52sm(data: dict):
    try:
        opened_scopes[data['flowId']] = data['openScopes']
        if not data['openScopes']:
            del opened_scopes[data['flowId']]
            n69wspa2qg = data['flowId'].split('=')[1].split(':')[0]
            todels = []
            for n69wspa2jx in opened_scopes:
                if f'flow={n69wspa2qg}:' in n69wspa2jx or f'flow={n69wspa2qg}*' in n69wspa2jx or f'flow={n69wspa2qg}/' in n69wspa2jx or (f'flow={n69wspa2qg}^' in n69wspa2jx):
                    todels.append(n69wspa2jx)
            if todels:
                pass
            for td in todels:
                del opened_scopes[td]
        targscope = data['openScopes'][-1] if data['openScopes'] else 'UNDEFINED'
        allopens = []
        for (n69wspa2jx, fscopes) in opened_scopes.items():
            allopens = allopens + fscopes
        data = False
        if allopens.count(targscope) > 1:
            data = True
        return {'error_code': 200, 'data': data}
    except Exception as e:
        traceback.print_exc()
        return {'error_code': 400, 'msg': str(e)}

@app.post('/app/kernel/delvar')
async def n69wspa2xq(data: dict):
    try:
        n69wsp9p6e = requests.post(url=f'http://localhost:{configer.grapy.sandbox_port}/delvar', json=data)
        if n69wsp9p6e.status_code == 200:
            n69wsp9p6e = n69wsp9p6e.json()
            return {'error_code': 200}
        else:
            return {'error_code': n69wsp9p6e.status_code, 'msg': 'failed to delete.'}
    except Exception as e:
        traceback.print_exc()
        return {'error_code': 400, 'msg': str(e)}

@app.delete('/app/kernel/delmod/{name}')
async def delmod(name: str):
    response = requests.delete(f'http://localhost:{configer.grapy.sandbox_port}/delmod/{name}')

@app.delete('/app/kernel/refreshmod/{name}')
async def refreshmod(name: str):
    try:
        response = requests.delete(f'http://localhost:{configer.grapy.sandbox_port}/refreshmod/{name}')
        if response.status_code != 200:
            error_detail = response.json().get('detail', 'Unknown error')
            raise RuntimeError(f'Failed to refresh module: {error_detail}')
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))
SANDBOX_URL = f'http://localhost:{configer.grapy.sandbox_port}'

async def proxy_sandbox_run(streamer, start_info, timetag=None):
    async with aiohttp.ClientSession() as session:
        try:
            run_id = start_info['run_id']
            async with session.post(SANDBOX_URL + '/run', headers={'Accept': 'text/event-stream'}, data=json.dumps(start_info, ensure_ascii=False)) as resp:
                if resp.status != 200:
                    await streamer.send_text({'event': 'error', 'msg': f'沙盒服务错误代码{resp.status}', 'run_id': run_id})
                    return
                if 'text/event-stream' not in resp.content_type:
                    await streamer.send_text({'event': 'error', 'msg': f'data: [ERROR] Backend did not return SSE stream', 'run_id': run_id})
                    return
                async for n69wsp9ou8 in resp.n69wsp9p72:
                    if n69wsp9ou8:
                        n69wsp9ou8 = n69wsp9ou8.decode('utf-8')
                        if n69wsp9ou8.startswith('data:'):
                            print('line.startswith("b\'data:")')
                            n69wsp9ou8 = n69wsp9ou8[5:].strip()
                        if n69wsp9ou8.strip():
                            try:
                                n69wspa34v = json.loads(n69wsp9ou8)
                                if n69wspa34v.get('event') == 'output':
                                    n69wspa34v['timetag'] = timetag

                                    async def trans_err(err_content):
                                        if err_content['node_prop'] in ('task', 'func'):
                                            hier_funcid = await handler.b69x8ynnue(err_content['node_id'])
                                            if not hier_funcid:
                                                err_content['node_id'] = None
                                                err_content['node_prop'] = 'general'
                                                err_content['scope'] = None
                                            else:
                                                assert '/' in hier_funcid
                                                if err_content['node_prop'] == 'func':
                                                    err_content['node_id'] = hier_funcid
                                                    hierscope = hier_funcid.rsplit('/', 1)[0]
                                                    err_content['scope'] = hierscope + ':funcs'
                                                elif err_content['node_prop'] == 'task':
                                                    err_content['scope'] = hier_funcid + ':dag'
                                    if start_info['output_choice'] == 'stateonly':
                                        for (n69wspa2e5, n69wsp9oxl) in n69wspa34v['content'].items():
                                            for n69wsp9p5e in n69wsp9oxl['errors']:
                                                await trans_err(n69wsp9p5e['content'])
                                    else:
                                        for (n69wspa2e5, n69wsp9oxl) in n69wspa34v['content'].items():
                                            for out_id in n69wsp9oxl.keys():
                                                if n69wsp9oxl[out_id].get('content_type') == 'error':
                                                    await trans_err(n69wsp9oxl[out_id]['content'])
                                elif n69wspa34v.get('event') == 'start':
                                    continue
                                await streamer.send_text(n69wspa34v)
                            except Exception as e:
                                traceback.print_exc()
                                await streamer.send_text(n69wsp9ou8)
        except asyncio.CancelledError:
            print('客户端断开')
            raise
        except Exception as e:
            await streamer.send_text({'event': 'error', 'msg': f'[PROXY ERROR] {str(e)}', 'run_id': run_id})
        finally:
            await streamer.close()

@app.post('/app/run')
async def run(request: Request):
    data = await request.json()
    streamer = SSEStream()
    start_info = {}
    reload_records = []
    try:
        (run_id, timetag) = output_client.gen_run_id_pair()
        output_choice = data.get('outputChoice', 'all')
        codedata = await handler.b69x8ynntt(data['projectRoot'], data['baseId'], choice=data['choice'], n69wspa2zf=data['labelInfo'])
        reload_records = codedata['reloads']
        start_info = {'codedata': codedata, 'run_id': run_id, 'output_choice': output_choice}
        await streamer.send_text({'event': 'start', 'run_id': run_id, 'timetag': timetag})
    except Exception as e:
        traceback.print_exc()
        await streamer.send_text({'event': 'error', 'msg': f'节点转代码失败:{str(e)}', 'run_id': run_id})
        await streamer.close()
        return StreamingResponse(streamer.stream(), media_type='text/event-stream')
    srsp = await AsyncStreamHandler.run_func(proxy_sandbox_run, streamer, start_info, timetag=timetag)
    recorded = []
    for n69wspa2x4 in reload_records:
        if x69xm5du01(n69wspa2x4) in handler.x69xm5dtzr:
            handler.x69xm5dtzr.remove(x69xm5du01(n69wspa2x4))
            recorded.append(n69wspa2x4)
    return srsp
SANDBOX_WS_URL = f'ws://localhost:{configer.grapy.sandbox_port}'

async def proxy_sandbox_ws_run(streamer, start_info, timetag=None):
    run_id = 0
    n69wsp9omi = False
    try:
        run_id = start_info['run_id']
        async with websockets.connect(SANDBOX_WS_URL + '/run') as ws:
            await ws.send(json.dumps(start_info, ensure_ascii=False))
            while True:
                n69wsp9ou8 = await ws.recv()
                if n69wsp9ou8.strip():
                    try:
                        n69wspa34v = json.loads(n69wsp9ou8)
                    except:
                        traceback.print_exc()
                        continue
                    if n69wspa34v.get('event') == 'output':
                        try:
                            n69wspa34v['timetag'] = timetag

                            async def trans_err(err_content):
                                if err_content['node_prop'] in ('task', 'func'):
                                    hier_funcid = start_info.get('dag_funcid') or await handler.b69x8ynnue(err_content['node_id'])
                                    if not hier_funcid:
                                        err_content['node_id'] = None
                                        err_content['node_prop'] = 'general'
                                        err_content['scope'] = None
                                    else:
                                        assert '/' in hier_funcid
                                        if err_content['node_prop'] == 'func':
                                            err_content['node_id'] = hier_funcid
                                            hierscope = hier_funcid.rsplit('/', 1)[0]
                                            err_content['scope'] = hierscope + ':funcs'
                                        elif err_content['node_prop'] == 'task':
                                            err_content['scope'] = hier_funcid + ':dag'
                            if start_info['output_choice'] == 'stateonly':
                                for (n69wspa2e5, n69wsp9oxl) in n69wspa34v['content'].items():
                                    for n69wsp9p5e in n69wsp9oxl['errors']:
                                        await trans_err(n69wsp9p5e['content'])
                            else:
                                for (n69wspa2e5, n69wsp9oxl) in n69wspa34v['content'].items():
                                    for out_id in n69wsp9oxl.keys():
                                        if n69wsp9oxl[out_id].get('content_type') == 'error':
                                            await trans_err(n69wsp9oxl[out_id]['content'])
                            await streamer.send_text(json.dumps(n69wspa34v, ensure_ascii=False))
                        except Exception as e:
                            traceback.print_exc()
                            await streamer.send_text(n69wsp9ou8)
                    elif n69wspa34v.get('event') == 'prompt':
                        await streamer.send_text(n69wsp9ou8)
                        clrsp = await streamer.receive_text()
                        await ws.send(clrsp)
                    elif n69wspa34v.get('event') == 'start':
                        continue
                    elif n69wspa34v.get('event') == 'end':
                        await streamer.send_text(n69wsp9ou8)
                    else:
                        await streamer.send_text(n69wsp9ou8)
        n69wsp9omi = True
    except WebSocketDisconnect:
        pass
    except asyncio.CancelledError:
        pass
    except websockets.exceptions.ConnectionClosedOK:
        pass
    except Exception as e:
        await streamer.send_text(json.dumps({'event': 'error', 'msg': f'[PROXY ERROR] {str(e)}', 'run_id': run_id}, ensure_ascii=False))
        traceback.print_exc()
    finally:
        await streamer.close()
    return n69wsp9omi

@app.websocket('/app/run')
async def run(websocket: WebSocket):
    streamer = websocket
    await streamer.accept()
    data = await streamer.receive_text()
    try:
        data = json.loads(data)
    except:
        data = eval(data)
    start_info = {}
    reload_records = []
    n69wsp9omi = False
    (run_id, timetag) = output_client.gen_run_id_pair()
    try:
        output_choice = data.get('outputChoice', 'all')
        n69wspa39a = lambda : True
        codedata = await handler.b69x8ynntt(data['projectRoot'], data['baseId'], choice=data['choice'], n69wspa2zf=data['labelInfo'], n69wspa2wz=data.get('appdag'), n69wspa39a=n69wspa39a, tolerance=2, x69xm5dtzp=True)
        reload_records = codedata['reloads']
        codedata['loglevel'] = data.get('loglevel') or 'TRACE'
        start_info = {'codedata': codedata, 'run_id': run_id, 'output_choice': output_choice}
        if data.get('appdag'):
            if data['appdag'].get('nodes'):
                start_info['dag_funcid'] = data['appdag']['nodes'][0]['data'].get('def_id')
        await streamer.send_text(json.dumps({'event': 'start', 'run_id': run_id, 'timetag': timetag}, ensure_ascii=False))
    except Exception as e:
        is_syntax = isinstance(e, SyntaxError)
        e = str(e)
        traceback.print_exc()
        n69wsp9ozf = traceback.format_exc()
        n69wsp9ovf = n69wsp9ozf.split('File "<unknown>",')[-1] if 'File "<unknown>",' in n69wsp9ozf else str(e)
        n69wsp9ovf = n69wsp9ovf.split('During handling of the above exception')[0] if 'During handling of the above exception' in n69wsp9ovf else n69wsp9ovf
        if is_syntax:
            try:
                n69wsp9ovf = e if UID_COMMENT_LEFTLABEL in e else n69wsp9ovf
                nodeid = re.findall(f'{UID_COMMENT_LEFTLABEL}(.*?){UID_COMMENT_RIGHTLABEL}', n69wsp9ovf, re.DOTALL)
                nodeid = nodeid[0].strip() if nodeid else None
                n69wspa2jx = await handler.b69x8ynnue(nodeid) if nodeid else None
                if not n69wspa2jx:
                    n69wspa2jx = '0'
                n69wspa337 = n69wspa2jx.split('^')[0] if n69wspa2jx else None
                node_prop = 'task' if nodeid else 'general'
                errdic = {'node_id': nodeid, 'error': n69wsp9ovf.split(FUNCID_COMMENT_RIGHTLABEL)[-1], 'node_prop': node_prop, 'module': n69wspa337}
                response = requests.post(url=f'http://localhost:{configer.grapy.sandbox_port}/add_error', json={'run_id': run_id, 'err': errdic})
                reply = {'event': 'output', 'content': {nodeid or '0': {'runned': 0, 'errors': [{'content': {'node_id': (nodeid or n69wspa2jx) or '0', 'error': n69wsp9ovf.split(FUNCID_COMMENT_RIGHTLABEL)[-1], 'node_prop': node_prop, 'module': n69wspa337, 'scope': n69wspa2jx + ':dag' if nodeid else None}, 'content_type': 'error'}], 'prompts': []}}, 'run_id': run_id, 'timetag': timetag}
                await streamer.send_text(json.dumps(reply, ensure_ascii=False))
            except Exception as e2:
                traceback.print_exc()
                await streamer.send_text(json.dumps({'event': 'error', 'msg': f'Failed to complie flow: {e2}', 'run_id': run_id}, ensure_ascii=False))
        else:
            await streamer.send_text(json.dumps({'event': 'error', 'msg': f'Failed to complie flow: {n69wsp9ovf}', 'run_id': run_id}, ensure_ascii=False))
        await streamer.close()
        return
    n69wsp9omi = await proxy_sandbox_ws_run(streamer, start_info, timetag=timetag)
    if n69wsp9omi:
        recorded = []
        for n69wspa2x4 in reload_records:
            if x69xm5du01(n69wspa2x4) in handler.x69xm5dtzr:
                handler.x69xm5dtzr.remove(x69xm5du01(n69wspa2x4))
                recorded.append(n69wspa2x4)

@app.post('/app/get_kernel_var_infos')
async def get_kernel_var_infos(data: dict):
    try:
        glovars = requests.post(url=f'http://localhost:{configer.grapy.sandbox_port}/get_kernel_var_infos', json={'def_id': data.get('defId')})
        if glovars.status_code == 200:
            glovars = glovars.json()
        else:
            raise RuntimeError(f'kernal vars returned status code {glovars.status_code}')
        if isinstance(glovars, str):
            glovars = json5.loads(glovars)
        return glovars
    except Exception as e:
        traceback.print_exc()
        return {'error_code': 400, 'msg': str(e)}

@app.post('/app/get_kernel_module_names')
async def get_kernel_module_names(data: dict):
    try:
        n69wsp9p1v = requests.post(url=f'http://localhost:{configer.grapy.sandbox_port}/get_kernel_module_names', json={'nothing': None})
        if n69wsp9p1v.status_code == 200:
            n69wsp9p1v = n69wsp9p1v.json()
        else:
            raise RuntimeError(f'kernal modules returned status code {n69wsp9p1v.status_code}')
        if isinstance(n69wsp9p1v, str):
            n69wsp9p1v = json5.loads(n69wsp9p1v)
        return n69wsp9p1v
    except Exception as e:
        traceback.print_exc()
        return {'error_code': 400, 'msg': str(e)}

@app.post('/app/get_kernel_var_value')
async def get_kernel_var_value(data: dict):
    try:
        n69wsp9p6e = requests.post(url=f'http://localhost:{configer.grapy.sandbox_port}/get_kernel_var_value', json=data)
        if n69wsp9p6e.status_code == 200:
            n69wsp9p6e = n69wsp9p6e.json()
        else:
            raise RuntimeError(f'kernal vars returned status code {n69wsp9p6e.status_code}')
        if isinstance(n69wsp9p6e, str):
            n69wsp9p6e = json5.loads(n69wsp9p6e)
        if isinstance(n69wsp9p6e.get('data'), str) and ENRICH == 'full':
            n69wsp9p6e['data'] = enrich_by_type(n69wsp9p6e['data'], dtype=n69wsp9p6e.get('dtype'), enrichable_len=lazy_enrich_len)
        return n69wsp9p6e
    except Exception as e:
        traceback.print_exc()
        return {'error_code': 400, 'msg': str(e)}

@app.delete('/app/kill_kernel/{level}')
async def kill_kernel(level: int):
    response = requests.delete(f'http://localhost:{configer.grapy.sandbox_port}/kill/{level}')
    if response.status_code == 200:
        if level > 1:
            requests.post(url=f'http://localhost:{configer.grapy.sandbox_port}/get_kernel_var_infos', json={'nothing': None})
    elif response.status_code == 404:
        raise HTTPException(status_code=400, detail='resource not found')
    else:
        raise HTTPException(status_code=400, detail=f'Failed to kill: {response.status_code}: {response.text}')

@app.delete('/app/reset_kernel/{nothing}')
async def reset_kernel(nothing: int):
    response = requests.delete(f'http://localhost:{configer.grapy.sandbox_port}/reset_namespace/0')
    if response.status_code == 200:
        pass
    elif response.status_code == 404:
        pass
    else:
        pass

@app.post('/app/get_nodes_run_ids')
async def get_nodes_run_ids(data: dict):
    try:
        node_ids = data['uids']
        node_ids = [n.split('-')[0] for n in node_ids]
        n69wspa2xu = asyncio.get_running_loop()
        n69wsp9onl = await n69wspa2xu.run_in_executor(None, lambda : output_client.format_nodes_run_ids(node_ids))
        return {'error_code': 200, 'data': n69wsp9onl}
    except Exception as e:
        traceback.print_exc()
        return {'error_code': 400, 'msg': str(e)}

@app.post('/app/get_node_output')
async def get_node_output(data: dict):
    try:
        n69wspa2e5 = str(data['uid']).split('-end')[0]
        run_id = int(data['runId'])
        (n69wsp9onl, count) = output_client.format_n_outputs(run_id, n69wspa2e5)
        print('getNodeOutput ret:', n69wsp9onl, count)
        return {'error_code': 200, 'data': {'md': n69wsp9onl, 'runcount': count}}
    except Exception as e:
        traceback.print_exc()
        return {'error_code': 400, 'msg': str(e)}
if __name__ == '__main__':
    import uvicorn
    import argparse
    parser = argparse.ArgumentParser(description='Run the FastAPI app')
    parser.add_argument('-p', '--port', type=int, default=18808, help='Port to run the server on')
    args = parser.parse_args()
    uvicorn.run(app='serve:app', host='0.0.0.0', port=args.port, reload=True)