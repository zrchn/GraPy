import sys
from fastapi.responses import StreamingResponse
from kernel_basic.configer import configer
from fastapi import FastAPI, Request, WebSocket, HTTPException, status
import asyncio
from asyncio import Queue
from loguru import logger
from kernel_basic.stream_handler import AsyncStreamHandler, SSEStream
import traceback
import json
import copy
from fastapi.middleware.cors import CORSMiddleware
from kernel_runner import runner
from kernel_cache_handler import _disp_to_cache, cache
from _sbutils import statics
app = FastAPI()
recursion = 3
refer_in_recurs = True
reserve_primary_triggers = False
dedup = 8
edited_after_undo = True
run_lock = asyncio.Lock()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_credentials=True, allow_methods=['*'], allow_headers=['*'])

@app.post('/add_error')
async def add_error(data: dict):
    logger.info('添加报错：{}', data)
    try:
        run_id = data['run_id']
        err = data['err']
        _disp_to_cache(run_id, err, node_id=err['node_id'] if err['node_prop'] == 'task' else '0', content_type='error')
        return {'error_code': 200}
    except Exception as e:
        return {'error_code': 400, 'msg': f'请求沙盒报错失败：{e}'}

@app.post('/get_kernel_var_infos')
async def get_kernel_var_infos(data: dict):
    logger.info('get_kernel_var_infos data: {}', data)
    try:
        glovars = runner.get_kernel_var_infos(cnskey=data.get('def_id'))
        logger.info('get_kernel_var_infos glovars: {}', glovars)
        return {'error_code': 200, 'data': glovars}
    except Exception as e:
        traceback.print_exc()
        return {'error_code': 400, 'msg': f'get_kernel_var_infos failed: {e}'}

@app.post('/get_kernel_module_names')
async def get_kernel_module_names(data: dict):
    logger.info('get_kernel_module_names data: {}', data)
    try:
        mods = runner.get_kernel_modules()
        logger.debug('get_kernel_module_names mods: {}', [m['name'] for m in mods])
        return {'error_code': 200, 'data': mods}
    except Exception as e:
        traceback.print_exc()
        return {'error_code': 400, 'msg': f'get_kernel_module_names failed: {e}'}

@app.post('/get_sugs')
async def get_sugs(data: dict):
    logger.info('get_sugs data: {}', data)
    try:
        v = runner.get_sugs(data['objpart'])
        logger.info('get_sugs sugs: {}', v)
        return {'error_code': 200, 'data': v}
    except Exception as e:
        traceback.print_exc()
        return {'error_code': 400, 'msg': f'get_sugs failed: {e}'}

@app.post('/get_params')
async def get_params(data: dict):
    logger.info('get_params data: {}', data)
    try:
        v = runner.get_params(data['funcpart'])
        logger.info('get_params params: {}', v)
        return {'error_code': 200, 'data': v}
    except Exception as e:
        traceback.print_exc()
        return {'error_code': 400, 'msg': f'get_params failed: {e}'}

@app.post('/get_kernel_var_value')
async def get_kernel_var_value(data: dict):
    logger.info('get_kernel_var_value data: {}', data)
    try:
        v, vtype = runner.get_kernel_var_value(data['name'], cnskey=data.get('defId'))
        logger.info('get_kernel_var_value val, type: {}', v, vtype)
        return {'error_code': 200, 'data': v, 'dtype': vtype}
    except Exception as e:
        traceback.print_exc()
        return {'error_code': 400, 'msg': f'get_kernel_var_value failed: {e}'}

@app.websocket('/run')
async def run(websocket: WebSocket):
    await websocket.accept()
    start_info = await websocket.receive_text()
    logger.debug('ws run start_info:', start_info)
    try:
        start_info = json.loads(start_info)
    except:
        start_info = eval(start_info)
    codedata = start_info['codedata']
    run_id = start_info['run_id']
    output_choice = start_info.get('output_choice', 'all')
    logger.debug(f'ws /run start_info:{start_info}')
    async with run_lock:
        await websocket.send_text(json.dumps({'event': 'start', 'run_id': run_id}, ensure_ascii=False))
        statics.run_id = run_id
        try:
            await runner.start_running_codes(codedata, run_id)
            await runner.astream_outputs(run_id, websocket, choice=output_choice)
        except Exception as e:
            logger.error(f'执行失败：{e}')
            await websocket.send_text(json.dumps({'event': 'error', 'msg': str(e)}, ensure_ascii=False))
            traceback.print_exc()

@app.delete('/kill/{level}')
async def kill(level: int):
    cache.sigkill_inputer()
    if level > 1:
        logger.info('收到Kill指令，调用runner.rebirth()')
        runner.rebirth()
    await asyncio.sleep(1)

@app.delete('/reset_namespace/{nothing}')
async def reset_namespace(nothing: int):
    async with run_lock:
        runner.reset_iworker_namespace()

@app.post('/delvar')
async def delvar(data: dict):
    logger.info('delvar data:', data)
    try:
        runner.delvar(data['name'], cnskey=data.get('def_id'))
        return {'error_code': 200}
    except Exception as e:
        logger.error(f'删除变量失败。data：{data}')
        return {'error_code': 400, 'msg': str(e)}

@app.delete('/delmod/{name}')
async def delmod(name: str):
    runner.del_kernel_modules([name])

@app.delete('/refreshmod/{name}')
async def refreshmod(name: str):
    errs = runner.refresh_kernel_modules([name])
    if errs:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(errs))
runner.reset_builtin_modules()
if __name__ == '__main__':
    import uvicorn
    import argparse
    parser = argparse.ArgumentParser(description='Run the FastAPI app')
    parser.add_argument('-p', '--port', type=int, default=configer.grapy.sandbox_port, help='Port to run the server on')
    parser.add_argument('-l', '--loglevel', type=str, default='TRACE', help='logger level')
    args = parser.parse_args()
    logger.add(sys.stdout, level=args.loglevel)
    logger.info(f'starting app on port {args.port} with log level {args.loglevel}...')
    uvicorn.run(app='sandbox_serve:app', host='0.0.0.0', port=args.port, workers=1, reload=False)