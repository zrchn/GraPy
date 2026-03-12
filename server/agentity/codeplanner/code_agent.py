import sys
from utils.shared import estimate_tokens, suppress_tokens
import asyncio
import json
import time
from loguru import logger
from typing import Any, Optional, Union
import re
import traceback
import copy
from core.cft.consts import SECTION_START_LABEL, SECTION_END_LABEL, INSERT_LABEL
import os
import json5
from utils.jsonformater import jsonformat
from datetime import datetime
import pandas as pd
from agentity.base.node import Node, execution
from agentity.base.llm import OpenAILLM
sys_pattern = '你是一名编程助手。用户会让你帮忙生成、修改或补全python的任务流代码，或根据已有代码向你提问。'
corrector = '\n你的上轮输出不合法。请严格按照要求输出。重申一遍，你的输出必须按照以下自定义格式：\n[<ACTION>]: ...\n[<EXPLAIN>]: ...\n[<EXTRA_PARAM1>]: ...\n... \n\n比如，当action为orchestrate时，你需要填写[<ACTION>]、[<EXPLAIN>]、[<CODE>]，当action为check_codes时，你需要填写[<ACTION>]、[<EXPLAIN>]、[<SELECTIONS>]，当action为talk_to_user时，你需要填写[<ACTION>]、[<EXPLAIN>]、[<MESSAGE>]，等等以此类推，一开始都告诉你了。\n报错信息：{err}\n'
base_sys_prompt = sys_pattern + f"""\n背景：我们的执行环境是一种新型的交互式REPL内核，会维护变量状态，支持查看节点历史变量值、查看内核变量当前值等高级功能。当用户提到产生任务流或pipeline等相关内容时，你可以理解为产生python代码。用户也可能就已有代码和状态向你提问，让你讲解。除此之外，如果由于缺乏必要信息而难以一次性产生代码或回答，你还有一些其他选择，例如向用户提问、查看包的源代码、查看变量记录或实时值等。另外，对你的输出还有一些结构化格式要求。请看下面详细说明。\n\n用户在每次提问时，会给你发送的内容包括：\n- 用户的问题或者对任务流的描述；\n- 当前已有的代码，里面标注了需重点关注的位置；\n- 在当前环境下可用的函数、类、对象等可用的变量以及各自的描述，描述可能有缺失，但你可以使用查看代码的action（后面会介绍）自己去看源代码。    描述里直接给出的函数、类、对象等都是直接可用的，不用重复声明或导入。\n\n关于用户发来的现有代码以及你该如何应对的详细解释：\n现有代码中，会用'# {INSERT_LABEL}'标注出需重点关注的位置，或者用'# {SECTION_START_LABEL}'和'# {SECTION_END_LABEL}'标注出需重点关注的的区间。根据用户的语义，有两种可能的情况：\n** 如果你认为用户希望你生成或修改任务流代码：**\n'# {INSERT_LABEL}'代表需要你插入新代码的位置，而'# {SECTION_START_LABEL}'和'# {SECTION_END_LABEL}'代表需要你替换的区间。这两种格式的标记不会同时出现，每次只会出现一种。说白了就是看到'# {INSERT_LABEL}'时需要你插入新代码，看到'# {SECTION_START_LABEL}'和'# {SECTION_END_LABEL}'时需要你替换一个区间的代码。你仍然可以使用其他行动收集所需的信息，但最后一个行动需要是orchestrate，除非fail_to_generate。不管是插入还是替换，你只需要产生新的用于插入或替换的代码，不要重复已有的代码。系统会直接把你产生的代码插入或替换到已有代码中标记的地方去。\n** 如果你认为用户希望你根据已有代码解答他的疑问：**\n这种情况则不需要使用orchestrate行动产生代码。你在回答用户问题之前仍然可以使用各种行动来查找信息，但最终需要使用talk_to_user来回答用户。\n\n以下会介绍你的一些行动（action）。注意每次允许选择一个或多个行动（直接平铺、用换行隔开每个行动选项即可）。只有在确保各个行动无任何相互依赖关系时才能同时选择多个行动。而当你不确定时，分成多次执行、每次只选择单个行动总是保险的。使用各行动的总体格式设计为自定义复合格式，并不是纯json，每个行动因为参数不一样所以格式有区别。请务必遵守使用每个行动时所要求的格式。\n\n# ------ 你的所有行动选项 ------\norchestrate:\n介绍：选该行动来输出你最终的局部任务流代码。当你选择orchestrate输出代码之后，这轮任务也会自动结束。\n你的输出格式：\n[<ACTION>]: orchestrate\n[<EXPLAIN>]: 用自然语言表达你的思路，以及对任务流代码的解释。\n[<CODE>]: ```python\n# 这里放你的代码。不管是插入还是替换区间，你只需要产生新的用于插入或替换的局部代码，不要重复已有的代码。\n```\n\nfail_to_generate:\n介绍：当你认定无法产生任务流时，可以选择失败，放弃生成任务流代码。失败的场景例如发现环境里缺少必要的依赖（调用check_pkg_exist后得知）、反复报错无法突破、或者其他你认为无法生成任务流的情况。\n你的输出格式：\n[<ACTION>]: fail_to_generate\n[<EXPLAIN>]: 说出你的理由，为什么放弃？\n\ncheck_codes:\n介绍：由于用户提供的对可用的函数、类、对象等的描述可能不足，在你对其中任何项目有疑问时可使用，一次可批量查看多个代码，且允许多次甚至连环使用。\n你的输出格式：\n[<ACTION>]: check_codes\n[<EXPLAIN>]: 用自然语言表达你的思路，为什么需要像这样查看代码\n[<SELECTIONS>]: ```json\n// 请使用json格式来表达一个list[dict]的结构。每个dict都是一组用来查看一块代码的参数。格式为：\n[\n    {{\n        "module": "(str) 必须提供module原名（注意不是alias），例如'pandas'、'dbutils.pooled_db'等。",\n        "class": "(str|None) 不提供class仅提供func则表示需要查找独立函数的代码，若提供class而不提供func则是查找整个class的代码，若提供class和func则是查找类内函数的代码。都需要使用原名而不是alias。比如，如果有个类alias名叫DF，而它的原名叫DataFrame，是通过代码'from pandas import DataFrame as DF'而来，那你在任务流代码中需使用DF、但在该查询源代码工具中需使用DataFrame。",\n        "func": "(str|None) 上面已经解释过了，和class搭配、可以提供也可以不提供，会起到完全不同的效果。也需要用原名。",\n        "env_level": // (int) 从0、1、2中选一个。因为sys.path中有3层不同的导入来源，每个来源优先级不一样，且是单向可见的，所以需要记录你现在所查看的代码位于第几层环境。用户给你的代码和每次check_codes工具返出的代码里都应该有标注env_level，直接抄过来就行。如果没标的话就选0。\n    }},\n    ... // 其他更多查看代码的参数组\n]\n// 备注：尽可能少查看整个module的代码，因为可能会很长。可以优先查看单个类或单个函数的代码。\n```\n\ntalk_to_user:\n介绍：当你对用户的需求不完全理解、或者感觉到有必要向用户告知或请示时，可以使用这个行动。当用自然语言回答用户问题时用的也是这个行动。\n你的输出格式：\n[<ACTION>]: talk_to_user\n[<EXPLAIN>]: 用自然语言表达你的思路，为什么需要联系用户\n[<MESSAGE>]: 你向用户的提问或通报或回答，格式为自然语言。\n\ncheck_pkg_exist:\n介绍：用于查看一个包是否存在，以及它的版本号。注意：用户在Tools中有提供的模块、类、函数等，默认都是已经存在的，不需要查看！只有当你想要使用用户未提到的包时才需查看是否存在。一般用于检查pip安装在site-packages里的包。如果存在，会返回版本号，否则返回'package does not exist'。\n你的输出格式：\n[<ACTION>]: check_pkg_exist\n[<PKG_NAME>]: 你想查看的包名，注意不是类名、函数名，而是整个依赖的名称，例如pandas、numpy这种。\n\ncheck_var_values:\n介绍：主动查看变量在某代码块上的历史值、或变量在全局的最新值。会返回一组含type、value的字典数组。如果查询某个变量时出现问题，会为它多返回一个msg字段把出现的问题告知你。\n[<ACTION>]: check_var_values\n[<EXPLAIN>]: 用自然语言表达你的思路，为什么需要查看这个变量，要用它来做什么\n[<SELECTIONS>]: ```json\n[\n    {{\n        "block_uid": "(str) 给你提供的代码里可能会使用<codeblock uid="xxx"></codeblock>标记出一些代码块，说的就是这个uid。如果你需要查看的不是变量的最新值、而是变量经过这个代码块时留下的快照，就请提供对应的uid。如果你本来就希望查看变量的最新值、或提供的代码里没有可用的uid，请在这一栏填写'<LATEST>'。",\n        "var_name": "(str) 变量名称。允许含点的attribute，但是暂不允许含中括号的索引。"\n    }},\n    // 可以批量查看多个变量值\n]\n```\n\ncheck_block_logs:\n介绍：主动查看某代码块上次执行时产生的日志（通常由print、log函数、报错等产生）。如果查询某个日志时出现问题，会为它多返回一个msg字段把出现的问题告知你。\n[<ACTION>]: check_block_logs\n[<EXPLAIN>]: 用自然语言表达你的思路，为什么需要查看这个节点的日志\n[<SELECTIONS>]: ```json\n[\n    {{\n        "block_uid": "(str) 给你提供的代码里可能会使用<codeblock uid="xxx"></codeblock>标记出一些代码块，说的就是这个uid。"\n    }},\n    // 可以批量查看多个节点的日志\n]\n```\n\n# ------------------\n\n# 注意事项\n关于orchestrate时code格式的注意事项：\n- 每行代码尽可能简洁，不要把好几个逻辑挤到一行代码里，比如：\n    y = func1(func2(x))\n  应该拆成：\n    _temp = func2(x) # 可按实际情况合理为临时变量命名\n    y = func1(_temp)\n- 在函数定义和类定义下，可使用doc格式来写介绍；\n- 定义函数参数时，尽可能使用 typing/typing_extensions中的Annotated和Doc （不需要import，直接用）来为每个参数以及返回值标注类型和描述。\n- 在代码逻辑中，可以使用'#'开头的注释，并且提倡在逻辑较复杂的地方多用注释，以利于用户理解。每条注释最好独占一行，不要放在一行代码后面。但是，千万不要在函数定义参数的括号内、dict或list之中放置注释，因为解析器不支持。\n- 代码中不需要写 if __name__ == '__main__' ，直接写代码就行。\n- 由于执行器使用了一种支持异步的REPL交互编程技术，当在顶层逻辑中调用异步函数时，不需要写asyncio.run()，直接用await或async for就行。\n\n一些提示：\n- 当用户提到工具（tool）时，泛指类、函数、或类的实例化对象。\n- 用户提供的工具介绍里，可能会包含类、函数、对象等的原名和假名。比如，import pandas as pd 会导致一个类的原名为DataFrame、假名为pd.DataFrame，而from pandas import DataFrame as DF 会导致原名为DataFrame的类的假名变成DF。在代码中，显然应该使用假名（这是python的规则），而当使用check_codes工具时，则必须使用原名。\n- 可能会把前序任务的历史也在上下文中给到你，供参考。\n- 你在生成任务流代码时，只需产生用于插入到'# {INSERT_LABEL}'处或用于替换掉'# {SECTION_START_LABEL}'和'# {SECTION_END_LABEL}'区间的局部代码，不要重复其他代码。\n"""
base_sys_tokens = estimate_tokens(base_sys_prompt)
user_qpartten = 'Please generate your output based on the following information.'
user_prompt_template = user_qpartten + '\n' + "\n========== Existing code ==========\n{existing_code}\n\n========== Tools ========== \n{tools_desc}\n\n========== User's query ========== \n{user_input}\n\n========== Notes ========== \n{note}\n\nPlease use the same language (English/Chinese) as the user's query for any natural language parts. You must comply with the format criteria!\n"

class Coder(Node):

    def __init__(self, nodedict, error_behavior='raise'):
        super().__init__(nodedict=nodedict, memory_fields=('role', 'content'), important_patterns=[('system', sys_pattern, 'oldest')], error_behavior=error_behavior)
        self.llm = OpenAILLM()
        self.base_tokens = 0
        assert self.llm_max_tokens > base_sys_tokens + 1000, f'LLM max tokens not enough. Should be at least {base_sys_tokens + 1000}'

    @execution(msgs_to_memory=False, submit_final_rsp=False)
    async def execute(self, start_msgs, session_id='<DEFAULT>'):
        assert self.llm_max_tokens > 1000, f'LLM must be able to intake >1000 tokens for this application to run.'
        pregen_max_tokens = int(self.llm_max_tokens * 0.67)
        event = start_msgs[0]['content']
        formated_context = []
        llm_sys_input = None
        llm_user_input = ''
        newtokens = 0
        is_started = False
        temp_ctxt = self.get_n_memory(session_id=session_id, important_patterns=[('system', sys_pattern)])
        if len(temp_ctxt) > 0:
            is_started = True
        if len([tc for tc in temp_ctxt if sys_pattern in tc['content']]) > 1:
            logger.error(f"有bug，记忆中出现{len([tc for tc in temp_ctxt if sys_pattern in tc['content']])}个基础sys prompt")
        if event in 'userquery':
            if not is_started:
                llm_sys_input = base_sys_prompt
                formated_context.append({'role': 'system', 'content': base_sys_prompt})
            user_data = start_msgs[1]['content']
            cate4llm = {'dag': 'logic', 'funcs': 'function', 'classes': 'class'}[user_data['category']]
            action4llm = {'insert': 'insert a', 'replace': 'replace the', 'allbelow': 'replace the', 'single': 'replace the', 'append': 'insert a'}[user_data['mode']]
            note = f"Please {action4llm} {cate4llm} or answer the user's question, depending on what they are asking for."
            if user_data.get('class_above'):
                assert cate4llm in ('logic', 'function'), f"nested class not allowed: {user_data['class_above']}"
                note = note + f"\nCaution: the target function is an attribute under class {user_data['class_above']}."
            existing_code = user_data['existing_code']
            user_dead_tokens = estimate_tokens(user_prompt_template) + estimate_tokens(note)
            xst_code_tokens = estimate_tokens(existing_code)
            if SECTION_END_LABEL in existing_code and SECTION_START_LABEL in existing_code:
                leftdex = existing_code.find(SECTION_START_LABEL)
                rightdex = existing_code.rfind(SECTION_END_LABEL) + len(SECTION_END_LABEL)
                assert rightdex > leftdex
            elif INSERT_LABEL in existing_code:
                leftdex = existing_code.find(INSERT_LABEL)
                rightdex = leftdex + len(INSERT_LABEL)
            else:
                leftdex = int(len(existing_code) / 2)
                rightdex = leftdex
            left = existing_code[:leftdex]
            mid = existing_code[leftdex:rightdex]
            right = existing_code[rightdex:]
            coretokens = estimate_tokens(mid)
            lefttokens = estimate_tokens(left)
            righttokens = estimate_tokens(right)
            code_max_tokens = max(pregen_max_tokens - base_sys_tokens - user_dead_tokens - estimate_tokens(user_data['user_input']) - estimate_tokens(user_data['tools_desc']), coretokens + min(lefttokens + righttokens, 400))
            if coretokens + min(lefttokens + righttokens, 400) + estimate_tokens(user_data['user_input']) > pregen_max_tokens - base_sys_tokens - user_dead_tokens:
                raise RuntimeError(f"Unable to generate, due to {coretokens + min(lefttokens + righttokens, 400) + estimate_tokens(user_data['user_input']) - (pregen_max_tokens - base_sys_tokens - user_dead_tokens)} tokens exceeding. Try narrow your selection.")
            totalovfl = xst_code_tokens - code_max_tokens
            if totalovfl > 0:
                logger.debug('需要压缩代码{}-{}={}个token', xst_code_tokens, code_max_tokens, totalovfl)
                leftovfl = int(lefttokens / (lefttokens + righttokens) * totalovfl)
                rightovfl = int(righttokens / (lefttokens + righttokens) * totalovfl)
                newleft = suppress_tokens(left, max(lefttokens - leftovfl, 0))
                newright = suppress_tokens(right, max(righttokens - rightovfl, 0))
                existing_code = newleft + mid + newright
            llm_user_input = user_prompt_template.format(existing_code=existing_code, user_input=user_data['user_input'], tools_desc=user_data['tools_desc'], note=note)
            if estimate_tokens(llm_user_input) + base_sys_tokens > pregen_max_tokens:
                ovfl = estimate_tokens(llm_user_input) + base_sys_tokens - pregen_max_tokens
                logger.warning(f'用户提问时字数超限了{ovfl}个，压缩工具介绍。')
                tools_desc = user_data['tools_desc']
                user_query = user_data['user_input']
                if estimate_tokens(tools_desc) > ovfl:
                    tools_desc = suppress_tokens(tools_desc, estimate_tokens(tools_desc) - ovfl)
                else:
                    user_query_old = user_query
                    user_query = suppress_tokens(user_query, estimate_tokens(user_query) + estimate_tokens(tools_desc) - ovfl)
                    tools_desc = ''
                    logger.info('用户问题也被压缩: {} -> {}', user_query_old, user_query)
                llm_user_input = user_prompt_template.format(existing_code=existing_code, user_input=user_query, tools_desc=tools_desc, note=note)
            formated_context.append({'role': 'user', 'content': llm_user_input})
            newtokens = estimate_tokens(llm_user_input) + estimate_tokens(llm_sys_input)
            self.base_tokens = base_sys_tokens + estimate_tokens(llm_user_input)
        elif event == 'tool':
            assert is_started
            tool_text = ''
            for stmsg in start_msgs[1:]:
                tool_data = stmsg['content']
                tool_name = tool_data['action']
                tool_content = str(tool_data.get('rsp'))
                tool_err = tool_data.get('error')
                tool_msg = tool_data.get('extra_msg')
                assert tool_content or tool_err
                if tool_content:
                    tool_text = tool_text + (f'Action {tool_name} output:\n' + tool_content + '\n')
                if tool_err:
                    tool_text = tool_text + f'{tool_name} encountered errors: {tool_err}\n'
                if tool_msg:
                    tool_text = tool_text + (tool_msg + '\n')
                tool_text = tool_text + '\n'
            toolout_tokens = estimate_tokens(tool_text)
            logger.debug('self.base_tokens={},toolout_tokens={},self.llm_max_tokens={}', self.base_tokens, toolout_tokens, self.llm_max_tokens)
            if self.base_tokens + toolout_tokens > self.llm_max_tokens:
                tool_text = suppress_tokens(tool_text, self.llm_max_tokens - self.base_tokens)
                logger.info('tool_text suppressed due to overtokening.')
                logger.debug('Suppress outcome: {}', tool_text)
            formated_context.append({'role': 'user', 'content': tool_text})
            newtokens = estimate_tokens(tool_text)
        elif event == 'illegal':
            formated_context.append({'role': 'user', 'content': corrector.format(err=start_msgs[1]['content'])})
            newtokens = 20
        else:
            raise
        input_context = self.get_n_memory(session_id=session_id, max_tokens=self.llm_max_tokens - newtokens, important_patterns=[('system', sys_pattern, 'oldest'), ('user', user_qpartten, 'newest')])
        input_context = input_context + formated_context
        await self.submit_rsps(formated_context, session_id=session_id, to_rsp_queue=False)
        if not (input_context[0]['content'].startswith(base_sys_prompt[:10]) and input_context[0]['role'] == 'system'):
            logger.warning('有bug，input_context开始不是system prompt。roles: {}', [c['role'] for c in input_context])
            maybe_sys_rownum = [i for i in range(len(input_context)) if (input_context[i]['content'].startswith(base_sys_prompt[:10]) and input_context[i]['role']) == 'system']
            if maybe_sys_rownum:
                logger.warning('发现system prompt被移到后面了，移回前面。maybe_sys_rownum：{}', maybe_sys_rownum)
                input_context = [input_context[maybe_sys_rownum[0]]] + [input_context[i] for i in range(len(input_context)) if not i in maybe_sys_rownum]
            else:
                input_context = [{'role': 'system', 'content': base_sys_prompt}] + input_context
                logger.warning('发现system prompt缺失，补上，但是注意可能有超token风险：{}（风险不大，前面是算了sysprompt的字数来缩减过上下文的', estimate_tokens(input_context))
        rsp = await self.llm.aanswer(input_context)
        await self.submit_rsps([{'role': 'assistant', 'content': rsp}], session_id=session_id)
        return