import asyncio
from asyncio import Queue
from loguru import logger
import traceback
from fastapi.responses import StreamingResponse
import json

class SSEStream:

    def __init__(self):
        self._is_closed = False
        self._queue = Queue()

    async def send(self, data: str):
        if self._is_closed:
            logger.error(f'SSEStream is closed, can no longer send.')
            return
        await self._queue.put(data)

    async def send_text(self, data: str):
        if isinstance(data, dict) or isinstance(data, list):
            try:
                data = json.dumps(data, ensure_ascii=False)
            except Exception as e:
                logger.error(f'send_text()收到格式异常的data，报错：{e}')
                logger.debug(f'异常的data：{data}')
        elif not isinstance(data, str):
            logger.error(f'send_text()收到类型异常的data，忽略')
            logger.debug(f'异常的data：{data}')
            return
        logger.info(f'Trying to send data')
        await self.send(data)

    async def close(self):
        await self._queue.put('<<<STREAM_CLOSE>>>')
        self._is_closed = True

    async def stream(self):
        try:
            while True:
                data = await self._queue.get()
                logger.debug(f'SSEStream data: {data}')
                if data == '<<<STREAM_CLOSE>>>':
                    logger.info('Closing SSEStream streamer')
                    break
                yield f'data: {data}\n\n'
        except asyncio.CancelledError:
            await self.close()
            logger.warning('Client disconnected.')
        except Exception as e:
            logger.error(f'Error in stream: {e}')

class AsyncStreamHandler:

    @staticmethod
    def locate_streamer(*args, **kwargs):
        for arg in args:
            if isinstance(arg, SSEStream):
                return arg
        for k, v in kwargs.items():
            if isinstance(v, SSEStream):
                return v
        return None

    @staticmethod
    async def run_func(func, *args, error_formater=lambda e: {'code': 400, 'message': str(e)}, **kwargs):
        streamer = AsyncStreamHandler.locate_streamer(*args, **kwargs)
        if not streamer:
            logger.error(f'SSEStream对象不在函数{func.__name__}的入参中，不要使用AsyncStreamHandler类')
            raise ValueError(f'SSEStream对象不在函数{func.__name__}的入参中，不要使用AsyncStreamHandler类')

        async def func_adapter(func, *args, **kwargs):
            try:
                await func(*args, **kwargs)
            except Exception as e:
                logger.error(f'函数{func.__name__}出错：{e}')
                traceback.print_exc()
                await streamer.send(error_formater(e))
                await streamer.close()
        asyncio.create_task(func_adapter(func, *args, **kwargs))
        return StreamingResponse(streamer.stream(), media_type='text/event-stream')