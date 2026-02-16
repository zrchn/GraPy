import sys
import re
from agentity.base.llm import OpenAILLM
from loguru import logger
import asyncio
from typing import Optional, Union, Any
import json5

class JsonFormater(OpenAILLM):
    PROMPT_TEMPLATE: str = '\nGiven the text:\n\n{text}\n\nFormat it into accurate json format. Do not output anything else. Your output should be able to be loaded by json directly.\n\nNotes:\n- Do not over-compile "\\" into "\\\\", "\\\'" into "\\\\\'", "\\"" into "\\\\"", etc, as these characters are already compiled once.\n\nYour output:\n\n    '
    name: str = 'JsonFormater'

    async def run(self, text: str):
        prompt = self.PROMPT_TEMPLATE.format(text=text)
        rsp = await self._aask(prompt)
        return rsp

async def jsonformat(text, sur='json'):
    try:
        pattern = f'```{sur}\\n(.*?)\\n```'
        jsn = text
        jsn = '\n'.join([j.split('//')[0] if j else '' for j in jsn.split('\n')])
        if f'```{sur}' in jsn:
            jsn = re.search(pattern, text, re.DOTALL).group(1).strip()
        try:
            try:
                jsn = json5.loads(jsn)
            except Exception as e:
                logger.warning(f'{e}, trying using eval instead of json5')
                jsnp = jsn.replace('true', 'True').replace('false', 'False').replace('null', 'None')
                jsn = eval(jsnp)
        except:
            jsn = jsn.replace('\n', '')
            try:
                jsn = json5.loads(jsn)
            except Exception as e:
                logger.warning(f'{e}, trying using eval instead of json5')
                jsn = eval(jsn.replace('true', 'True').replace('false', 'False').replace('null', 'None'))
        return jsn
    except Exception as e:
        logger.warning(f'{e}, trying loading after AI reformatting')
        jsn = await JsonFormater().run(text)
        pattern = f'```{sur}\\n(.*?)\\n```'
        jsn = re.search(pattern, jsn, re.DOTALL).group(1).strip()
        jsn = json5.loads(jsn)
        return jsn
if __name__ == '__main__':
    workflow = "\n    每日分析和报告生成：系统每天定时对生产数据按下列步骤进行分析：\n    - 日维度分析： 主要包括以下四个因子：频率、分析颗粒度、分析时长花费、分析维度。\n    - 分析维度：\n        - 日计划达成率： 计算方式为完成数/计划数*100%, 数据需要先提取。\n        - 重点工序的产能发挥率： 每小时实际生产件数/MCT件数*100%，去除7:00-7:30之间的特定时间段。产能发挥率低于设定值时，提示异常并提取、整理和分类异常原因。\n        - 库存情况： WMS内的库存数量-使用日计划数量，如果大于零反馈“库存正常”，小于零反馈差值。\n        - 根据以上数据，用自然语言写出报告。\n    {'tasks need modification': ['2'],\n        'tasks must not be modified': ['1', '3', '4', '5'],\n        'how to modify': '根据用户建议，需要移除第二个步骤，即移除杭州银行2023年收益数据的收集任务。',\n        'all_tasks after modification': [{'instruction': '从宁波银行的官方渠道或财务报告中收集2023年的收益数据。',\n        'dependent_task_ids': [],\n        'assigned_tools': 'retrieve_business_info',\n        'task_id': '1',\n        'title': '收集宁波银行2023年收益数据'},\n        {'instruction': '对收集到的宁波银行2023年的收益数据进行详细分析，包括但不限于收入、利润、资产等关键指标。',\n        'dependent_task_ids': ['1'],\n        'assigned_tools': 'retrieve_business_info',\n        'task_id': '3',\n        'title': '分析宁波银行2023年收益数 据'},\n        {'instruction': '对收集到的杭州银行2023年的收益数据进行详细分析，包括但不限于收入、利润、资产等关键指标。',\n        'dependent_task_ids': [],\n        'assigned_tools': 'retrieve_business_info',\n        'task_id': '4',\n        'title': '分析杭州银行2023年收益数据'},\n        {'instruction': '将宁波银行和杭州银行2023年的收益数据进行对比分析，找出差异和共同点，总结两家银行的收益表现。',\n        'dependent_task_ids': ['3', '1'],\n        'assigned_tools': '<NONE>',\n        'task_id': '5',\n        'title': '对比两家银行的收益'}]}\n    This are the extracted tasks.\n    "
    text = '\n    ```json\n{\n    "answer": "基于指定的组织、产品和工序信息，MES系统返回的计划完成数的自然语言描述",\n    "data": "包含组织名称、产品型号、工序代码及对应计划完成数的结构化数据字典"\n}\n```\n    '
    j = asyncio.run(jsonformat(text))
    print(j)