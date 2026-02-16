
"""
Copyright (c) 2026 Zhiren Chen. Provided as-is for local use only.
"""

import sys
import time
import re
import ast
import json
import numpy as np
import pandas as pd
import copy
import traceback
from loguru import logger
import pymysql
import asyncio
from dbutils.pooled_db import PooledDB
import sqlalchemy as sa
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncConnection
from typing import Dict, List, Any, Set, Union
import inspect as pyinspect

def gen_create_sql(table_name, typesdict, primes=[], need_new_mapping=False):
    type_mapping = {'int64': 'INT', 'int': 'INT', 'float64': 'FLOAT', 'float': 'FLOAT', 'str': 'VARCHAR(255)', 'bool': 'TINYINT(1)', 'object': 'JSON', 'dict': 'JSON', 'list': 'JSON', 'json': 'JSON', 'datetime64[ns]': 'DATETIME', 'timedelta[ns]': 'BIGINT'}
    new_mapping = {}
    sqls = []
    for (vname, vtypes) in typesdict.items():
        allownull = False
        if 'NoneType' in vtypes:
            allownull = True
        vtypes = [n69wsp9p6g for n69wsp9p6g in vtypes if n69wsp9p6g != 'NoneType']
        if len(vtypes) > 1:
            vtypes = ['json']
        vtype = vtypes[0]
        qtype = type_mapping.get(vtype, vtype)
        asql = f'{vname} {qtype} '
        if not allownull:
            asql = asql + 'NOT NULL'
        sqls.append(asql)
        new_mapping[vname] = asql[len(vname) + 1:]
    sqls = ',\n'.join(sqls)
    unique_clause = ''
    if primes:
        unique_key_name = f"uk_{'_'.join(primes)}"
        unique_clause = f", UNIQUE KEY `{unique_key_name}` (`{'`,`'.join(primes)}`)"
    sqls = f'CREATE TABLE {table_name} ({sqls}{unique_clause});'
    if need_new_mapping:
        return (sqls, new_mapping)
    return sqls

def fill_nan_from_other_df(n69wsp9p0u: pd.DataFrame, n69wsp9olz: pd.DataFrame, cond_cols: list, inplace=False, nan_behavior='raise'):
    if not inplace:
        n69wsp9p0u = n69wsp9p0u.copy()
    nan_mask = n69wsp9p0u.isna()
    nan_positions = nan_mask.stack()[nan_mask.stack()].index
    for (row_idx, n69wspa2id) in nan_positions:
        n69wspa2xo = n69wsp9p0u.loc[row_idx]
        if not cond_cols:
            cond_cols = n69wspa2xo.dropna().index.tolist()
            if n69wspa2id in cond_cols:
                cond_cols.remove(n69wspa2id)
            if len(cond_cols) == 0:
                raise ValueError(f"Row {row_idx} has NaN in '{n69wspa2id}' but no other non-NaN columns to form condition.")
        cond_mask = pd.Series([True] * len(n69wsp9olz), index=n69wsp9olz.index)
        for c in cond_cols:
            cond_val = n69wspa2xo[c]
            cond_mask = cond_mask & (n69wsp9olz[c] == cond_val)
        candidates = n69wsp9olz.loc[cond_mask, n69wspa2id]
        fill_value = np.nan
        if len(candidates) == 0:
            if nan_behavior == 'raise':
                raise KeyError(f"No matching row in df2 for filling df1.loc[{row_idx}, '{n69wspa2id}'] using conditions {n69wspa2xo[cond_cols].to_dict()}")
            else:
                pass
        else:
            fill_value = candidates.iloc[0]
        n69wsp9p0u.at[row_idx, n69wspa2id] = fill_value
    return n69wsp9p0u

def isnan(x, vbs=False):
    if x in (np.nan, None):
        return True
    if isinstance(x, float) and str(x).lower() == 'nan':
        return True
    return False

class DBHandler:

    def __init__(self, db_url: str, table_definitions: Dict[str, Dict]):
        self.sync_url = db_url
        self.async_url = db_url.replace('pymysql', 'aiomysql')
        self.table_definitions = table_definitions
        self.engine = None
        self.async_engine = None
        self.inspector = None
        self._write_lock = asyncio.Lock()
        self._read_lock = asyncio.Lock()
        self._read_count = 0
        self._initialize()
        if self.async_engine is None:
            self.async_engine = create_async_engine(self.async_url, echo=False)

    def _initialize(self):
        try:
            self.engine = create_engine(self.sync_url, echo=False)
            with self.engine.connect() as conn:
                conn.execute(text('SELECT 1'))
            self.inspector = inspect(self.engine)
            for (table_name, config) in self.table_definitions.items():
                self._check_or_create_table(table_name, config)
        except Exception as e:
            raise ConnectionError(f'❌ 数据库初始化失败: {e}')

    def _check_or_create_table(self, table_name: str, config: Dict):
        fields = config.get('fields', {})
        primes = config.get('primes', [])
        uniques = config.get('uniques', [])
        foreigns = config.get('foreigns', [])
        if not isinstance(fields, dict):
            raise ValueError(f"表 '{table_name}' 的 fields 必须是字典")
        if not isinstance(primes, list) or not primes:
            raise ValueError(f"表 '{table_name}' 必须指定非空的 primes 字段列表")
        if self.inspector.has_table(table_name):
            self._validate_table_structure(table_name, fields, primes)
        else:
            self._create_table_with_unique_key(table_name, fields, primes, uniques, foreigns)

    def _validate_table_structure(self, table_name: str, fields: Dict, primes: List[str]):
        existing_columns = {n69wspa2id['name']: n69wspa2id for n69wspa2id in self.inspector.get_columns(table_name)}
        expected_columns = set(fields.keys())
        existing_column_names = set(existing_columns.keys())
        if existing_column_names != expected_columns:
            missing = expected_columns - existing_column_names
            extra = existing_column_names - expected_columns
            msg = f"表 '{table_name}' 字段不匹配"
            if missing:
                msg = msg + f'，缺少: {missing}'
            if extra:
                msg = msg + f'，多余: {extra}'
            raise ValueError(msg)

    def _create_table_with_unique_key(self, table_name: str, fields: Dict[str, str], primes: List[str], uniques: List[Union[List[str], str]], foreigns: List[Dict]):
        column_defs = []
        for (col_name, col_type) in fields.items():
            column_defs.append(f'`{col_name}` {col_type.strip()}')
        assert primes, f'表{table_name}缺乏primes'
        primary_clause = f", PRIMARY KEY ({','.join(primes)})"
        unique_clauses = []
        for ugroup in uniques:
            if isinstance(ugroup, str):
                ugroup = [ugroup]
            if isinstance(ugroup, list):
                unique_key_name = f"uk_{'_'.join(ugroup)}"
                unique_clause = f", UNIQUE KEY `{unique_key_name}` (`{'`,`'.join(ugroup)}`)"
                unique_clauses.append(unique_clause)
            else:
                raise ValueError(f'表{table_name}设的唯一字段中，每个分组格式必须是list或str，收到：{ugroup}')
        foreign_clauses = []
        if foreigns:
            for finfo in foreigns:
                clause = f", \nFOREIGN KEY ({finfo['foreigns']}) REFERENCES {finfo['references']} "
                if finfo.get('behaviors'):
                    clause = clause + finfo['behaviors']
                foreign_clauses.append(clause)
        columns_sql = ', '.join(column_defs) + primary_clause + ''.join(unique_clauses) + ''.join(foreign_clauses)
        create_sql = f'CREATE TABLE `{table_name}` ({columns_sql}) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;'
        with self.engine.connect() as conn:
            conn.execute(text(create_sql))
            conn.commit()

    async def _acquire_read(self):
        async with self._read_lock:
            if self._read_count == 0:
                await self._write_lock.acquire()
            self._read_count = self._read_count + 1

    async def _release_read(self):
        async with self._read_lock:
            self._read_count = self._read_count - 1
            if self._read_count == 0:
                self._write_lock.release()

    def _form_sql_selector(self, table_name: str, conds: list[dict]=[], cond_sql: str='', targets=[]):
        assert conds or cond_sql
        sql_query = []
        if conds:
            for n69wsp9p3g in conds:
                subquery = []
                for k in n69wsp9p3g:
                    real_k = k
                    kpath = ''
                    if '->' in k:
                        assert k.count('->') == 1, f'使用json路径筛选时只允许出现一个->，收到：{k}'
                        (real_k, kpath) = k.split('->')
                        real_k = real_k.strip()
                        kpath = ' -> ' + f"'{kpath.strip()}'"
                        assert 'JSON' in self.table_definitions[table_name]['fields'][real_k], f'字段{real_k}不是JSON类型'
                    v = n69wsp9p3g[k]
                    if isinstance(v, str):
                        v = f"'{v}'"
                    elif type(v) in (int, float, bool):
                        pass
                    else:
                        raise ValueError(f'用于查询的变量不能是非常规类型，收到的变量：{v}')
                    subquery.append(f'`{real_k}`{kpath} = {v}')
                subquery = ' AND '.join(subquery)
                subquery = f'({subquery})'
                sql_query.append(subquery)
            sql_query = ' OR '.join(sql_query)
        else:
            sql_query = 'true'
        if cond_sql:
            cond_sql = cond_sql.strip()
            sql_query = f'({sql_query}) AND ({cond_sql})'
        sql_query = f'FROM {table_name} WHERE {sql_query}'
        return sql_query

    async def delete(self, table_name: str, conds: list[dict]=[], cond_sql: str='', conn=None, _skiplock=False):
        assert conds or cond_sql
        sql_query = self._form_sql_selector(table_name, conds=conds, cond_sql=cond_sql)
        sql_query = f'DELETE {sql_query}'
        try:
            if not _skiplock:
                async with self._write_lock:
                    if conn is None:
                        async with self.async_engine.connect() as conn:
                            await conn.execute(text(sql_query))
                            await conn.commit()
                    else:
                        await conn.execute(text(sql_query))
            elif conn is None:
                async with self.async_engine.connect() as conn:
                    await conn.execute(text(sql_query))
                    await conn.commit()
            else:
                await conn.execute(text(sql_query))
        except Exception as ex:
            raise ex

    async def select(self, table_name: str, conds: list[dict]=[], cond_sql: str='', targets=[], avoids=[], post=None, conn=None, _skiplock=False, json_repairer=None, debuglabel=''):
        assert conds or cond_sql
        assert isinstance(targets, list)
        if not targets:
            targets = list(self.table_definitions[table_name]['fields'].keys())
        if avoids:
            targets = [n69wsp9p6g for n69wsp9p6g in targets if not n69wsp9p6g in avoids]
        sql_query = self._form_sql_selector(table_name, conds=conds, cond_sql=cond_sql)
        if targets:
            mytargets = ', '.join([f'`{n69wsp9p6g}`' for n69wsp9p6g in targets])
        else:
            mytargets = '*'
        sql_query = f'SELECT {mytargets} {sql_query}'
        if debuglabel:
            sql_query = f'-- debuglabel: {debuglabel} \n' + sql_query
        try:
            if not _skiplock:
                await self._acquire_read()
            if conn is None:
                async with self.async_engine.connect() as conn:
                    n69wsp9oq8 = await conn.execute(text(sql_query))
            else:
                n69wsp9oq8 = await conn.execute(text(sql_query))
        except Exception as ex:
            print(f'!!! select() buggy conds,cond_sql: {conds},{cond_sql}')
            print(f'!!! DBHandler select() buggy sql_query: {sql_query}')
            traceback.print_exc()
            raise ex
        finally:
            if not _skiplock:
                await self._release_read()
        n69wsp9oq8 = n69wsp9oq8.fetchall()
        n69wspa2r1 = targets
        if not n69wspa2r1:
            columns_info = self.inspector.get_columns(table_name)
            n69wspa2r1 = [n69wspa2id['name'] for n69wspa2id in columns_info]
        if n69wsp9oq8:

            def maybe_load(x, jkey):
                if x not in (None, np.nan):
                    if isinstance(x, str):
                        x = json.loads(x)
                        if json_repairer is not None:
                            x = json_repairer(x, table_name, jkey)
                    else:
                        raise ValueError(f'本应是json的数据，查出来既不是str也不是None:{x}')
                return x
            n69wsp9oq8 = [{n69wspa2r1[i]: n69wspa2hl[i] for i in range(len(n69wspa2r1))} for n69wspa2hl in n69wsp9oq8]
            n69wsp9oq8 = pd.DataFrame(n69wsp9oq8)
            jkeys = []
            for (k, v) in self.table_definitions[table_name]['fields'].items():
                if 'JSON' in v.upper():
                    jkeys.append(k)
            jkeys = set(jkeys) & set(n69wsp9oq8.columns.tolist())
            for jk in jkeys:
                n69wsp9oq8[jk] = n69wsp9oq8[jk].apply(maybe_load, jkey=jk)
            if post is not None:
                n69wsp9oq8 = n69wsp9oq8[n69wsp9oq8.apply(post, axis=1)]
            n69wsp9oq8 = n69wsp9oq8.replace(np.nan, None)
            return n69wsp9oq8
        else:
            result_df = pd.DataFrame(columns=n69wspa2r1)
            return result_df

    async def _prefill_unnull_values(self, conn: AsyncConnection, df: pd.DataFrame, table_name: str, primes: List[str]):
        df = df.copy()
        columns = self.inspector.get_columns(table_name)
        not_null_columns = [n69wspa2id['name'] for n69wspa2id in columns if not n69wspa2id.get('nullable', True)]
        all_columns = list(set([n69wspa2id['name'] for n69wspa2id in columns]) & set(df.columns.tolist()).union(set(not_null_columns)))
        missed_cols = set(not_null_columns) - set(df.columns.tolist())
        if missed_cols:
            for mc in missed_cols:
                df[mc] = None
        if not df[all_columns].isna().any().any():
            return df
        nans = []
        all_nanfields = []
        for i in range(len(df)):
            nanfields = []
            for n69wspa2id in df.columns.tolist():
                if n69wspa2id in all_columns:
                    if isnan(df.iloc[i][n69wspa2id]):
                        nanfields.append(n69wspa2id)
                        all_nanfields.append(n69wspa2id)
            if nanfields:
                nans.append({p: df.iloc[i][p] for p in primes})
        all_nanfields = list(set(all_nanfields))
        supp = await self.select(table_name, conds=nans, targets=list(set(primes + all_nanfields)), conn=conn, _skiplock=True)
        _origlen = len(supp)
        supp = supp.drop_duplicates(subset=primes, keep='last')
        if len(supp) < _origlen:
            pass
        assert not supp[list(set(primes) | set(all_nanfields) & set(not_null_columns))].isna().any().any(), f'有bug，数据库里查出来用于填充的数据，而且筛选过非空字段，竟然有nan。supp: {supp}'
        n69wspa340 = time.time()
        df = fill_nan_from_other_df(df, supp, primes, nan_behavior='ignore')
        n69wspa2ug = time.time()
        if df[list(set(primes + not_null_columns))].isna().any().any():
            nadf = df[list(set(primes + not_null_columns))]
            for i in range(len(nadf)):
                if nadf.iloc[i].isna().any():
                    pass
            raise ValueError(f"缺失非null字段无法自动补回，检查是否提交的df有缺失必要字段: {df.to_dict(orient='records')}")
        return df

    async def upsert(self, table_name, df, primekeys=None, skipna=True, force_update=0, conn=None, _skiplock=False):
        n69wspa340 = time.time()
        if len(df) == 0:
            return
        if table_name not in self.table_definitions:
            raise ValueError(f'未知表名: {table_name}')
        config = self.table_definitions[table_name]
        fields = config['fields']
        primes = config['primes'] if not primekeys else primekeys
        df = df.copy()
        _origlen = len(df)
        try:
            df = df.drop_duplicates(subset=primes, keep='last')
        except Exception as e:
            raise RuntimeError(f'按primes去重失败({table_name})：{e}')
        if len(df) < _origlen:
            pass
        dfcols = df.columns.tolist()

        def maybe_dump(x):
            if x not in (None, np.nan):
                x = json.dumps(x, ensure_ascii=False)
            return x
        if len(set(dfcols) - set(fields.keys())) > 0:
            pass
        df = df[list(set(dfcols) & set(fields.keys()))]
        if df[primes].isna().any().any():
            raise ValueError(f'提供的prime有缺失')
        if self.async_engine is None:
            self.async_engine = create_async_engine(self.async_url, echo=False)

        async def _form_records(conn, df):
            if skipna:
                df = await self._prefill_unnull_values(conn, df, table_name, primes)
            dfcols = df.columns.tolist()
            for n69wspa2id in dfcols:
                df[n69wspa2id] = df[n69wspa2id].apply(lambda x: None if isnan(x) else x)
            for n69wspa2id in dfcols:
                if 'JSON' in fields[n69wspa2id]:
                    df[n69wspa2id] = df[n69wspa2id].apply(lambda x: maybe_dump(x))
            df = df.replace(np.nan, None)
            records = df.to_dict(orient='records')
            table_cols = [f'`{c}`' for c in dfcols]
            placeholders = ', '.join([f':{c}' for c in dfcols])
            update_cols = [n69wspa2id for n69wspa2id in dfcols if not n69wspa2id in primes]
            if not force_update:
                insert_cols = ', '.join(table_cols)
                if not update_cols:
                    update_clause = 'DO NOTHING'
                else:
                    update_clause = ', '.join([f'`{n69wspa2id}` = new_data.`{n69wspa2id}`' for n69wspa2id in update_cols])
                sql = f'\n                INSERT INTO `{table_name}` ({insert_cols})\n                VALUES ({placeholders})\n                AS new_data  -- ← 定义别名\n                ON DUPLICATE KEY UPDATE\n                {update_clause}\n                '
            else:
                where_clause = ' AND '.join([f'`{n69wspa2id}` = :{n69wspa2id}' for n69wspa2id in primes])
                if force_update > 0:
                    check_sql = f'SELECT 1 FROM `{table_name}` WHERE {where_clause} LIMIT 1'
                    n69wspa2n4 = await asyncio.gather(*[conn.execute(text(check_sql), adata) for adata in records])
                    n69wspa2n4 = [e.first() for e in n69wspa2n4]
                    nonexists = [xi for xi in range(len(n69wspa2n4)) if not n69wspa2n4[xi]]
                    if nonexists:
                        mylog = logger.warning if force_update <= 2 else logger.error
                        mylog(f'在禁用INSERT模式下，{len(nonexists)}条需更新的记录在表{table_name}中找不到，共{len(records)}条。')
                        if force_update > 3:
                            raise ValueError('需更新的记录找不到')
                update_clause = ', '.join([f'`{n69wspa2id}` = :{n69wspa2id}' for n69wspa2id in update_cols])
                sql = f'\n                UPDATE `{table_name}`\n                SET {update_clause}\n                WHERE {where_clause}\n                '
            return (records, sql)
        if not _skiplock:
            async with self._write_lock:
                if conn is None:
                    async with self.async_engine.begin() as conn:
                        (records, sql) = await _form_records(conn, df)
                        if len(records) == 0:
                            return
                        await conn.execute(text(sql), records)
                        await conn.commit()
                else:
                    (records, sql) = await _form_records(conn, df)
                    if len(records) == 0:
                        return
                    await conn.execute(text(sql), records)
        elif conn is None:
            async with self.async_engine.begin() as conn:
                (records, sql) = await _form_records(conn, df)
                if len(records) == 0:
                    return
                await conn.execute(text(sql), records)
                await conn.commit()
        else:
            (records, sql) = await _form_records(conn, df)
            if len(records) == 0:
                return
            await conn.execute(text(sql), records)
        n69wspa2ug = time.time()

    async def _batch_read(self, gatherables, _skiplock, conn=None):
        try:
            if not _skiplock:
                await self._acquire_read()
            if conn is None:
                async with self.async_engine.connect() as conn:
                    n69wspa2p9 = await asyncio.gather(*[g(conn=conn) for g in gatherables])
            else:
                n69wspa2p9 = await asyncio.gather(*[g(conn=conn) for g in gatherables])
            return n69wspa2p9
        finally:
            if not _skiplock:
                await self._release_read()

    async def _batch_write(self, gatherables, _skiplock, conn=None):
        if not _skiplock:
            async with self._write_lock:
                if conn is None:
                    async with self.async_engine.begin() as conn:
                        n69wspa2p9 = await asyncio.gather(*[g(conn=conn) for g in gatherables])
                        await conn.commit()
                else:
                    n69wspa2p9 = await asyncio.gather(*[g(conn=conn) for g in gatherables])
        elif conn is None:
            async with self.async_engine.begin() as conn:
                n69wspa2p9 = await asyncio.gather(*[g(conn=conn) for g in gatherables])
                await conn.commit()
        else:
            n69wspa2p9 = await asyncio.gather(*[g(conn=conn) for g in gatherables])
        return n69wspa2p9