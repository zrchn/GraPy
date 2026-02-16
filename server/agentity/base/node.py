import sys
import traceback
from basic.configer import configer
import pandas as pd
from loguru import logger
from functools import wraps
from abc import ABC, abstractmethod
from agentity.base.memory import Memory
from asyncio import Queue
from basic.background_runner import AsyncTaskRunner
import copy
import asyncio
import inspect
from typing import Optional, Union

def execution(conflict_behavior='wait', msgs_to_memory=True, submit_final_rsp=False, rsp_handler_callback=None):

    def decorator(func):

        @wraps(func)
        async def wrapper(node, *args, **kwargs):
            waitcount = 0
            if conflict_behavior == 'wait':
                while node.is_executing:
                    await asyncio.sleep(1)
                    waitcount = waitcount + 1
                    if waitcount % 10 == 0:
                        logger.debug(f'该节点当前正在执行，需等待执行完毕才能再次开始执行。已等待{waitcount}秒')
            node.is_executing = True
            print(f'### decorator kwargs ({node.node_type}):', kwargs)
            session_id = kwargs.get('session_id')
            if session_id is None:
                session_id = '<DEFAULT>'
                logger.debug(f'execute()未提供session_id参数，@execution()装饰器默认使用会话<DEFAULT>')
            msgs = kwargs.get('msgs')
            if msgs is None:
                msgs = args[0]
            if msgs_to_memory:
                await node.submit_rsps(msgs, session_id=session_id, to_memory=True, to_rsp_queue=False)
            logger.info(f'开始执行节点：node_id={node.node_id}, node_type={node.node_type}, session_id={session_id}')
            logger.debug(f'输入：args:{args}; kwargs:{kwargs}')
            try:
                result = await func(node, *args, **kwargs)
                if submit_final_rsp:
                    node.valid_msgs(result)
                    await node.submit_rsps(result, session_id=session_id, extra_callback=rsp_handler_callback)
                logger.info(f'结束执行节点：node_id={node.node_id}, node_type={node.node_type}, session_id={session_id}')
                logger.debug(f'输出：{result}')
            except Exception as e:
                result = [{'role': 'system', 'content': f'<EXECTUION_ERROR>{e}</EXECTUION_ERROR>'}]
                node.valid_msgs(result)
                logger.error(f'执行报错：{e}')
                traceback.print_exc()
                await node.submit_rsps(result, session_id=session_id)
            node.result = result
            node.is_executing = False
            node.execute_count = node.execute_count + 1
            return result
        return wrapper
    return decorator

class Node(AsyncTaskRunner, ABC):
    is_executing: bool = False
    execute_count: int = 0
    start_time: float = 0
    end_time: float = 0
    node_type: str = 'basenode'
    memory: Memory
    msg_queue: Queue
    rsp_queues: dict
    duty_cycle_running: bool = False

    def __init__(self, nodedict, memory_fields=('role', 'content'), existing_context=None, memory_n=1999, max_memories=1999, llm_max_tokens=-1, important_patterns=[], error_behavior='raise'):
        assert 'role' in memory_fields and 'content' in memory_fields, 'role和content是上下文必有字段'
        super().__init__()
        self.error_behavior = error_behavior
        nodedict = copy.deepcopy(nodedict)
        nodedict['node_type'] = nodedict.get('node_type') or nodedict.get('type') or self.__class__
        assert 'node_id' in nodedict or 'task_id' in nodedict
        nodedict['node_id'] = nodedict.get('node_id') if 'node_id' in nodedict else nodedict['task_id']
        self.is_executing = False
        self.execute_count = 0
        for (k, v) in nodedict.items():
            setattr(self, k, v)
        assert not self.node_type == 'basenode', f"Node初始化的nodedict未提供有效的node_type：{nodedict.get('node_type')}"
        assert not '_session_id' in memory_fields, '_session_id是内置键，不得使用'
        sid_field = []
        if isinstance(memory_fields, list):
            sid_field = ['_session_id']
        elif isinstance(memory_fields, tuple):
            sid_field = ('_session_id',)
        else:
            raise ValueError(f'无法识别的memory_fields：{memory_fields}')
        self.llm_max_tokens = llm_max_tokens if llm_max_tokens > 0 else configer.llm.max_tokens
        self.memory = Memory(fields=memory_fields + sid_field, context=existing_context, max_memories=max_memories, llm_max_tokens=self.llm_max_tokens, important_patterns=important_patterns)
        self.msg_queue = Queue()
        self.rsp_queues = {'<DEFAULT>': Queue()}
        self.memory_n = memory_n

    def create_session(self, session_id):
        if session_id in self.rsp_queues:
            logger.warning(f'会话{session_id}已存在，无法新建')
        else:
            logger.info(f'创建新会话{session_id}')
            self.rsp_queues[session_id] = Queue()

    def delete_session(self, session_id):
        if session_id not in self.rsp_queues:
            logger.warning(f'会话{session_id}不存在，无法删除')
        else:
            logger.info(f'删除会话{session_id}，暂不删除memory')
            del self.rsp_queues[session_id]

    def delete_memory(self, filter_func, session_id='<DEFAULT>'):
        self.memory.delete(filter_func, session_id=session_id)

    async def start_asso(self):
        pass

    async def stop_asso(self):
        pass

    @execution
    @abstractmethod
    async def execute(self, msgs=[], session_id='<DEFAULT>'):
        raise NotImplementedError('必须提供执行函数')

    async def submit_rsps(self, msgs, session_id='<DEFAULT>', to_memory=True, to_rsp_queue=True, extra_callback=None, log=True):
        assert session_id in self.rsp_queues, f'会话{session_id}不存在，无法提交响应'
        if log:
            logger.info(f'{self.node_type} {self.node_id} submitting rsps')
            logger.debug(f'submit_rsps() msgs: {msgs}, session_id: {session_id}')
        self.valid_msgs(msgs)
        if to_rsp_queue:
            await self.rsp_queues[session_id].put(msgs)
        if to_memory:
            memsgs = [{**msg, **{'_session_id': session_id}} for msg in msgs]
            self.memory.appends(memsgs)
        if extra_callback:
            if inspect.iscoroutinefunction(extra_callback):
                ersp = await extra_callback(msgs, session_id=session_id)
                return ersp
            elif isinstance(extra_callback, callable):
                return extra_callback(msgs, session_id=session_id)
            else:
                logger.warning(f'extra_callback必须是同步或异步函数，收到了{type(extra_callback)}')

    async def acquiring_rsps(self, session_id='<DEFAULT>', break_cond: callable=lambda x: False, yield_at_break=False, error_behavior=None):
        if not error_behavior:
            error_behavior = self.error_behavior
        try:
            while True:
                if not session_id in self.rsp_queues:
                    logger.warning(f'跳过不存在的session_id，2秒后再尝试获取：{session_id}')
                    await asyncio.sleep(2)
                    continue
                rsp = await self.rsp_queues[session_id].get()
                self.rsp_queues[session_id].task_done()
                logger.debug(f'acquiring_rsps() session_id: {session_id}, rsp: {rsp}')
                if rsp == '<STOP>':
                    break
                if rsp:
                    do_break = False
                    for msg in rsp:
                        if msg['role'] == 'system':
                            if isinstance(msg.get('content'), str):
                                if msg['content'].startswith('<EXECTUION_ERROR>') and msg['content'].endswith('</EXECTUION_ERROR>'):
                                    err = msg['content']
                                    logger.error(f'agent输出消息中有报错信息：{err}')
                                    if error_behavior == 'raise':
                                        raise RuntimeError(f'Error throwed by agent: {err}')
                                    elif error_behavior == 'break':
                                        do_break = True
                                        break
                    if do_break:
                        break
                    if break_cond(rsp[-1]):
                        logger.info(f'acquiring_rsps()暂时退出')
                        if yield_at_break:
                            yield rsp
                        break
                yield rsp
        finally:
            dirty_msgs = []
            dirty_rsps = []
            try:
                while True:
                    d = self.msg_queue.get_nowait()
                    dirty_msgs.append(d)
                    self.msg_queue.task_done()
            except asyncio.QueueEmpty:
                pass
            try:
                while True:
                    d = self.rsp_queues[session_id].get_nowait()
                    dirty_rsps.append(d)
                    self.rsp_queues[session_id].task_done()
            except asyncio.QueueEmpty:
                pass
            if dirty_msgs:
                logger.warning(f'输出循环退出时清空尚未使用的msgs：{dirty_msgs}')
            if dirty_rsps:
                logger.warning(f'输出循环退出时清空尚未使用的rsps，session_id={session_id}：{dirty_rsps}')

    async def start_duty_cycle(self, msgs_processor=lambda msgs: msgs):
        logger.info(f'node {self.node_type} {self.node_id} starting duty cycle')
        self.start_monitor()
        await self.start_asso()

        async def _start_cycle():
            self.duty_cycle_running = True
            try:
                while True:
                    try:
                        (msg_dicts, session_id) = await self.msg_queue.get()
                    finally:
                        self.msg_queue.task_done()
                    logger.debug(f'({self.node_type}) msgs in start_duty_cycle(): {msg_dicts}, session_id: {session_id}')
                    if msg_dicts == '<STOP>':
                        logger.info(f'stopping node {self.node_type} {self.node_id}')
                        if not session_id == '<DEFAULT>':
                            logger.warning(f'<STOP>将停止整个Node所有session的会话，指定session_id无效')
                        for (k, que) in self.rsp_queues.items():
                            await que.put('<STOP>')
                        break
                    task = asyncio.create_task(self.execute(msgs_processor(msg_dicts), session_id=session_id))
                    await task
            finally:
                await self.stop_asso()
                self.duty_cycle_running = False
        if not self.duty_cycle_running:
            cycle = asyncio.create_task(_start_cycle())
            self.submit_task(cycle)
        else:
            logger.info('agent的duty cycle已经启动，不能重复启动')

    async def listen_msgs(self, msgs, session_id='<DEFAULT>'):
        logger.info(f"node {self.node_type} {self.node_id} session {session_id} listening msgs {('(<STOP> received)' if msgs == '<STOP>' else '')}")
        logger.debug(f'msgs: {msgs}, session_id: {session_id}')
        if not session_id in self.rsp_queues:
            self.create_session(session_id)
        if msgs == '<STOP>':
            await self.msg_queue.put((msgs, session_id))
            return
        self.valid_msgs(msgs)
        await self.msg_queue.put((msgs, session_id))

    def valid_msg(self, msg):
        assert set(list(msg.keys())).issubset(set(self.memory.columns)), f'msg里不得有node的memory中没有的键：{set(list(msg.keys()))} vs {set(self.memory.columns)}'
        assert not '_session_id' in msg, '_session_id是内置键，不得使用'

    def valid_msgs(self, msgs):
        assert isinstance(msgs, list), f'msgs必须是list，收到了{type(msgs)}'
        for msg in msgs:
            self.valid_msg(msg)

    def get_n_memory(self, session_id='<DEFAULT>', n=None, skip_p_newest=0, ignore_fields=[], filter_func=None, max_tokens=-1, important_patterns=[]):
        if n is None:
            n = -1
        return self.memory.get_formated(n=n, session_id=session_id, skip_p_newest=skip_p_newest, ignore_fields=ignore_fields, filter_func=filter_func, return_session_id=False, max_tokens=max_tokens, important_patterns=important_patterns)

    def __repr__(self, excludes=[]):
        attributes = vars(self)
        _excludes = ['msg_queue', 'rsp_queue', 'memory']
        nodedict = {name: value for (name, value) in attributes.items() if not callable(value) and (not name in excludes) and (not name in _excludes)}
        return f'node {self.node_type} {self.node_id} information: {nodedict}'
if __name__ == '__main__':
    logger.add(sys.stdout, level='DEBUG')

    class TestNode(Node):

        @execution(submit_final_rsp=True)
        async def execute(self, x, session_id='0'):
            print('TestNode x:', x)
            return [{'role': 'assistant', 'content': '......'}]
    n = TestNode({'node_type': 'test', 'node_id': '2.3'}, existing_context=[{'role': 'tool', 'content': 'aloha', '_session_id': 2}])
    import asyncio
    asyncio.run(n.execute(888))