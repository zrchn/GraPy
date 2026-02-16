from typing import Optional, Union, Any, Callable, List, Dict
import asyncio
import traceback
from loguru import logger

class AsyncTaskRunner:
    task_monitor_task: Optional[asyncio.Task] = None
    monitor_started: bool = False
    tasks: set = set()

    def __init__(self):
        pass

    async def _monitor_tasks(self):
        while True:
            if self.tasks:
                (done, pending) = await asyncio.wait(self.tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    try:
                        await task
                        self.tasks.remove(task)
                    except Exception as e:
                        self.tasks.remove(task)
                        logger.critical(f'异步任务出错：{e}')
                        traceback.print_exc()
            else:
                await asyncio.sleep(0.2)
        self.monitor_started = False

    def submit_task(self, task):
        self.tasks.add(task)

    def start_monitor(self):
        if self.monitor_started:
            logger.info('后台运行已经启动，不能重复启动')
            return
        self.monitor_started = True
        if not self.task_monitor_task or self.task_monitor_task.done():
            self.task_monitor_task = asyncio.create_task(self._monitor_tasks())