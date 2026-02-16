import sys
import time
import redis
from basic.configer import configer
from loguru import logger
import json
import json5
from utils.shared import time14_to_readable
from core.cft.utils import idgen
REDIS_HOST = configer.grapy.redis_host
REDIS_PORT = int(configer.grapy.redis_port)
NODE_MAX_RECORDS = int(configer.grapy.node_max_prints) - 1
assert NODE_MAX_RECORDS > 0

class OutputClient:

    def __init__(self):
        self.cache = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        self.node_max_records = NODE_MAX_RECORDS

    def get_n_outputs(self, run_id, node_id, n=None, choice='prints', dolog=True) -> dict[int, dict[str, str | dict]]:
        n = n or self.node_max_records
        run_id = int(run_id)
        node_id = str(node_id)
        node_id = node_id.replace('<', '\\<').replace('>', '\\>').replace('-', '\\-').replace('.', '\\.')
        typeselector = ''
        if choice == 'prints':
            typeselector = ' @content_type:{text|error|node_runned|inputed|prompt}'
        elif choice == 'vars':
            typeselector = ' @content_type:{var}'
        elif choice == 'stateonly':
            typeselector = ' @content_type:{node_runned|error}'
        elif choice == 'all':
            typeselector = ''
        elif choice == 'prompt':
            typeselector = ' @content_type:{prompt}'
        elif choice == 'inputed':
            typeselector = ' @content_type:{inputed}'
        result = self.cache.execute_command('FT.SEARCH', 'idx:output', f'@run_id:[{run_id} {run_id}] @node_id:{{{node_id}}} {typeselector}', 'SORTBY', 'record_id', 'DESC', 'LIMIT', '0', str(n), 'RETURN', '3', '$.content', 'record_id', '$.content_type')
        if result == [0] and dolog:
            logger.info('output记录查不到，run_id={}, node_id={},choice={}', run_id, node_id, choice)
            return {}
        retdata = {}
        for i in range(2, len(result), 2):
            record_id = result[i][1]
            content = result[i][3]
            content_type = result[i][5]
            try:
                content = json.loads(content)
                content = content[0]
            except:
                logger.info('content无法用json解析，尝试json5，run_id={}, node_id={}', run_id, node_id)
                print('buggy content:', content)
                try:
                    content = json5.loads(content)
                    content = content[0]
                except Exception as e:
                    logger.error('json5也无法解析，run_id={}, node_id={}，error:{}', run_id, node_id, e)
            retdata[record_id] = {'content': content, 'content_type': content_type}
        return retdata

    def format_n_outputs(self, run_id, node_id, n=None) -> str:
        n = n or self.node_max_records
        data = self.get_n_outputs(run_id, node_id, n=10000, choice='prints')
        count_data = self.get_n_outputs(run_id, node_id, n=10000, choice='stateonly')
        md = []
        runcount = len([1 for x in count_data.values() if x.get('content_type') == 'node_runned'])
        overflowed = False
        count = 0
        for (i, entity) in enumerate(data.values()):
            if count > n:
                overflowed = True
            if entity.get('content_type') in 'var':
                continue
            elif entity.get('content_type') == 'node_runned':
                if not overflowed:
                    md = md + ['————————————']
            elif entity.get('content_type') == 'text':
                if not overflowed:
                    pluser = [str(entity['content'])]
                    if any([str(entity['content']).startswith(prefix) for prefix in ['[ERROR]', '[EXCEPTION]', '[CRITICAL]', '[WARNING]', '[INFO]', '[DEBUG]', '[TRACE]']]):
                        pluser[0] = '\r' + pluser[0]
                    md = md + pluser
                    count = count + 1
            elif entity.get('content_type') == 'prompt':
                md = md + [f"[DEBUG] Prompt: {str(entity['content'])}"]
            elif entity.get('content_type') == 'inputed':
                md = md + [f"[DEBUG] Input: {str(entity['content']['input'])}"]
            elif entity.get('content_type') == 'error':
                if entity['content']['node_prop'] == 'task':
                    md = md + [f"\r[EXCEPTION]{str(entity['content']['error'])}"]
                    runcount = runcount + 1
            else:
                logger.info(f"Unknown的content_type，默认直接变成str渲染：{entity.get('content_type')}")
                if not overflowed:
                    md = md + [str(entity['content'])]
        md.reverse()
        md = '\n'.join(md)
        if overflowed:
            logger.info(f'节点可能运行次数太多，显示条数溢出({n})。run_id={run_id},node_id={node_id}')
            md = md + f'\n<span style="color:blue">Only showing {n} items</span>'
        return (md, runcount)

    def get_all_by_run_id(self, run_id, n_per_node=None, choice='prints') -> dict[str, dict[int, dict[str, str | dict]]]:
        n_per_node = n_per_node or self.node_max_records
        run_id = int(run_id)
        typeselector = ''
        if choice == 'prints':
            typeselector = ''
        elif choice == 'vars':
            typeselector = ' @content_type:{var}'
        elif choice == 'stateonly':
            typeselector = ' @content_type:{node_runned|error}'
        result = self.cache.execute_command('FT.SEARCH', 'idx:output', f'@run_id:[{run_id} {run_id}]{typeselector}', 'LIMIT', '0', '9999', 'RETURN', '4', '$.node_id', '$.content', '$.record_id', '$.content_type')
        data = {}
        if result == [0]:
            return data
        for i in range(2, len(result), 2):
            node_id = result[i][1]
            record_id = result[i][5]
            content = result[i][3]
            content_type = result[i][7]
            if choice == 'prints' and content_type == 'var':
                continue
            try:
                content = json.loads(content)
                content = content[0]
            except:
                logger.info('content无法用json解析，尝试json5，run_id={}, node_id={}', run_id, node_id)
                print('buggy content:', content)
                try:
                    content = json5.loads(content)
                    content = content[0]
                except Exception as e:
                    logger.warning('json5也无法解析，run_id={}, node_id={}，error:{}', run_id, node_id, e)
            if not node_id in data:
                data[node_id] = {}
            data[node_id][record_id] = {'content': content, 'content_type': content_type}
        return data

    def get_all_available_run_ids(self, node_id=None, limit=9999):
        selector = '*'
        if node_id:
            selector = f'@node_id:{{{node_id}}}'
        result = self.cache.execute_command('FT.AGGREGATE', 'idx:output', selector, 'GROUPBY', '1', '@run_id', 'REDUCE', 'COUNT', '0', 'SORTBY', '2', '@run_id', 'DESC')
        if result == [0]:
            return []
        unique_run_ids = [int(r[1]) for r in result[1:]]
        assert sum(unique_run_ids) >= 0, unique_run_ids
        return unique_run_ids[:limit]

    def format_nodes_run_ids(self, node_ids):
        run_ids = [self.get_all_available_run_ids(node_id=node_id) for node_id in set(node_ids)]
        run_ids = [item for sublist in run_ids for item in sublist]
        run_ids = list(set(run_ids))
        run_ids.sort(reverse=True)
        ret = [{'label': time14_to_readable(t)[:-5], 'value': t} for t in run_ids]
        return ret

    def gen_run_id_pair(self):
        run_id = idgen.generate(return_type='int')
        timetag = time14_to_readable(run_id)[:-5]
        return (run_id, timetag)
client = OutputClient()
if __name__ == '__main__':
    outs = client.format_n_outputs(1210, '68wdhxsns')
    print('outs:', outs)