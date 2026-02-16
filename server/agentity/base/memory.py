import re
import sys
from basic.configer import configer
from utils.shared import estimate_tokens
import pandas as pd
from loguru import logger
from consts import vbs
import copy

def _rank4del(mycontext, important_patterns=[]):
    syscontext = mycontext[mycontext['role'] == 'system']
    sysavoids = syscontext.index[1:2].tolist() + syscontext.index[0:1].tolist() + syscontext.index[-1:].tolist()
    idxs = mycontext.index.tolist()
    frontidxs = [i for i in idxs if not i in sysavoids]
    idxs = frontidxs + sysavoids
    importants = {}

    def add_imports(row):
        nonlocal importants
        for (i, ipp) in enumerate(important_patterns):
            if ipp[0] == row['role']:
                matchs = re.findall(ipp[1], row['content'], re.DOTALL)
                if matchs:
                    if not i in importants:
                        importants[i] = []
                    importants[i].insert(0, row.name)
    mycontext.apply(add_imports, axis=1)
    for (i, imps) in importants.items():
        mode = 'all' if len(important_patterns[i]) < 3 else important_patterns[i][2]
        if mode == 'newest':
            importants[i] = [imps[0]]
        elif mode == 'oldest':
            importants[i] = [imps[-1]]
    impordexs = []
    for (imlv, imdexs) in importants.items():
        for imdex in imdexs:
            if not imdex in impordexs:
                impordexs.insert(0, imdex)
    idxs = [x for x in idxs if not x in impordexs] + impordexs
    return idxs

def suppress_by_tokens(context, max_tokens, session_id='<DEFAULT>', important_patterns=[]):
    if not isinstance(context, pd.DataFrame):
        context = pd.DataFrame(context)
    context = context.reset_index(drop=True)
    mycontext = context[context['_session_id'] == session_id]
    deldexs = []
    delusers = 0
    delagents = 0
    delsyss = 0
    delothers = 0
    deltokens = 0
    pre_numtokens = estimate_tokens(str(mycontext.to_dict(orient='records')))
    numtokens2del = pre_numtokens - max_tokens
    if pre_numtokens > max_tokens:
        idxs = _rank4del(mycontext, important_patterns=important_patterns)
        for (idxi, i) in enumerate(idxs):
            if i in deldexs:
                continue
            deldexs.append(i)
            if mycontext.loc[i, 'role'] in 'user':
                delusers = delusers + 1
            elif mycontext.loc[i, 'role'] in 'system':
                delsyss = delsyss + 1
            elif mycontext.loc[i, 'role'] in ('agent', 'assistant'):
                delagents = delagents + 1
            else:
                delothers = delothers + 1
            deltokens = deltokens + estimate_tokens(str(mycontext.loc[i].to_dict()))
            if idxi < len(mycontext.index) - 1:
                nexti = mycontext.index[idxi + 1]
                if mycontext.loc[nexti, 'role'] in ('agent', 'assistant') and (not nexti in deldexs):
                    deldexs.append(nexti)
                    delagents = delagents + 1
                    deltokens = deltokens + estimate_tokens(str(mycontext.loc[nexti].to_dict()))
            if deltokens >= numtokens2del:
                break
    context = context.drop(deldexs)
    logger.debug('限制字数压缩掉{}条记忆，session_id={}，其中user {}条，assistant {}条，system {}条，字符数最大{}，实际从{}压缩到{}', len(deldexs), session_id, delusers, delagents, delsyss, max_tokens, pre_numtokens, estimate_tokens(str(context.to_dict(orient='records'))))
    return context

def suppress_by_records(context, max_records, session_id='<DEFAULT>', important_patterns=[]):
    logger.trace('suppress_by_records important_patterns: {}', important_patterns)
    if not isinstance(context, pd.DataFrame):
        context = pd.DataFrame(context)
    context = context.reset_index(drop=True)
    mycontext = context[context['_session_id'] == session_id]
    maxrecs = max_records
    deldexs = []
    delusers = 0
    delagents = 0
    delsyss = 0
    if len(mycontext) - len(deldexs) > maxrecs:
        idxs = _rank4del(mycontext, important_patterns=important_patterns)
        for (idxi, i) in enumerate(idxs):
            if i in deldexs:
                continue
            deldexs.append(i)
            if mycontext.loc[i, 'role'] in 'user':
                delusers = delusers + 1
            elif mycontext.loc[i, 'role'] in 'system':
                delsyss = delsyss + 1
            elif mycontext.loc[i, 'role'] in ('agent', 'assistant'):
                delagents = delagents + 1
            else:
                delothers = delothers + 1
            if idxi < len(mycontext.index) - 1:
                nexti = mycontext.index[idxi + 1]
                if mycontext.loc[nexti, 'role'] in ('agent', 'assistant') and (not nexti in deldexs):
                    deldexs.append(nexti)
                    delagents = delagents + 1
            if len(mycontext) - len(deldexs) <= maxrecs:
                break
    logger.debug('限制数量压缩掉{}条记忆，session_id={}，其中user {}条，assistant {}条，system {}条', len(deldexs), session_id, delusers, delagents, delsyss)
    context = context.drop(deldexs)
    return context

class Memory:
    context: pd.DataFrame

    def __init__(self, fields=[], context=None, max_memories=9999, llm_max_tokens=-1, important_patterns=[]):
        self.max_records = max_memories
        self.llm_max_tokens = llm_max_tokens if llm_max_tokens > 0 else configer.llm.max_tokens
        self.important_patterns = important_patterns
        if isinstance(context, pd.DataFrame):
            if fields:
                logger.warning(f'创建memory在有context的情况下忽略fields')
            assert '_session_id' in context.columns, '提供的上下文必须包含_session_id列'
            logger.info(f'用提供的context创建memory')
            logger.debug(f'context: {context}')
            self.context = context
        elif isinstance(context, list):
            if fields:
                logger.warning(f'创建memory在有context的情况下忽略fields')
            try:
                self.context = pd.DataFrame(context)
                assert '_session_id' in self.context.columns, '提供的上下文必须包含_session_id列'
                logger.info(f'用提供的context创建memory')
                logger.debug(f'context: {context}')
            except:
                logger.error(f'Cannot format context to dataframe: {context}')
                raise ValueError(f'Cannot format context to dataframe: {context}')
        elif context is not None:
            logger.error(f'context格式不支持: {context}')
            raise ValueError(f'context格式不支持: {context}')
        else:
            if not fields:
                fields = ('role', 'content')
            myfields = copy.deepcopy(fields)
            if not '_session_id' in myfields:
                logger.info(f'memory自动添加_session_id列')
                myfields = [f for f in myfields] + ['_session_id']
            logger.info(f'创建空memory：{myfields}')
            self.context = pd.DataFrame([{f: '<INIT>' for f in myfields}])

    def append(self, infodic):
        assert set(list(infodic.keys())).issubset(set(self.context.columns.tolist())), f'输入信息字段与现有记忆不匹配：{infodic.keys()}'
        assert '_session_id' in infodic, '需把_session_id放在infodic里'
        self.appends([infodic])

    def appends(self, infos, del_overflow=True, important_patterns=None):
        logger.debug('infos to append len: {}, important_patterns:{}', len(infos), important_patterns)
        important_patterns = important_patterns if important_patterns is not None else self.important_patterns
        if not isinstance(infos, pd.DataFrame):
            infos = pd.DataFrame(infos)
        assert set(infos.columns.tolist()).issubset(set(self.context.columns.tolist())), f'输入信息字段与现有记忆不匹配：{infos.columns.tolist()}'
        self.context = pd.concat([self.context, infos], ignore_index=True)
        if del_overflow:
            sids = set(infos['_session_id'].tolist())
            for sid in sids:
                self.context = suppress_by_records(self.context, self.max_records, session_id=sid, important_patterns=important_patterns)

    def delete(self, filter_func, session_id='<DEFAULT>'):
        sid_cond = self.context['_session_id'] == session_id
        filter_func = self.context.apply(filter_func, axis=1)
        combined = sid_cond & filter_func
        self.context = self.context[~combined]

    def filter(self, filter_func, session_id='<DEFAULT>'):
        sid_cond = self.context['_session_id'] == session_id
        filter_func = self.context.apply(filter_func, axis=1)
        combined = sid_cond & ~filter_func
        self.context = self.context[~combined]

    def inserts(self, infos, position):
        if position < 0:
            position = len(self.context) + position
        assert position >= 0 and position < len(self.context), f'不恰当的position {position}，对于一个长度为{len(self.context)}的上下文'
        if not isinstance(infos, pd.DataFrame):
            infos = pd.DataFrame(infos)
        assert set(infos.columns.tolist()).issubset(set(self.context.columns.tolist())), f'输入信息字段与现有记忆不匹配：{infos.columns.tolist()}'
        self.context = pd.concat([self.context.iloc[:position], infos, self.context.iloc[position:]], ignore_index=True)

    def get_formated(self, session_id='<DEFAULT>', n=-1, skip_p_newest=0, ignore_fields=[], filter_func=None, return_session_id=True, max_tokens=-1, important_patterns=[]):
        if n == -1:
            n = 999999
        if max_tokens <= 0:
            max_tokens = self.llm_max_tokens
        columns_to_keep = [col for col in self.context.columns if col not in ignore_fields]
        if not filter_func:
            context = self.context[self.context['_session_id'] == session_id][columns_to_keep]
        else:
            context = self.context[self.context['_session_id'] == session_id][self.context.apply(filter_func, axis=1)][columns_to_keep]
        i0 = max(0, len(context) - skip_p_newest - n)
        i1 = max(i0, len(context) - skip_p_newest)
        context = context[i0:i1]
        context = suppress_by_tokens(context, max_tokens, session_id=session_id, important_patterns=important_patterns)
        if not return_session_id:
            context = context.drop(columns=['_session_id'])
        return copy.deepcopy(context.to_dict(orient='records'))

    def __repr__(self):
        return f"All memory: {self.context.to_dict(orient='records')}"

    @property
    def columns(self):
        return self.context.columns.tolist()

    def __len__(self):
        return len(self.context)
if __name__ == '__main__':
    m = Memory(['role', 'content'])
    print('m.get_formated():', m.get_formated())
    m.append({'role': 'user', 'content': 'hello'})
    print('m.get_formated():', m.get_formated())
    m.append({'role': 'agent', 'content': 'fubar'})
    m.append({'role': 'system', 'content': 1234})
    print('m.get_formated():', m.get_formated())
    print('m.get_formated(filter_func=lambda row:row["role"]!="agent"):', m.get_formated(filter_func=lambda row: row['role'] != 'agent'))
    print('m.get_formated(n=2,ignore_fields="role"):', m.get_formated(n=2, ignore_fields='role'))