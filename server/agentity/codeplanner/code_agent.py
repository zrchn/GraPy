import sys
from utils.shared import estimate_tokens, suppress_text, suppress_tokens
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
exec(f'from core.cft.processors{sys.version_info.minor}{"w" if sys.platform == "win32" else ""} import *')
from core.cft.utils import brutal_gets, access_nested_data, set_nested_data, all_desc2nl_in_tokens

def find_func_paths_in_roi(tree, startlabel, endlabel):
    inroi_paths = []
    inroi = False

    def _find(node, path):
        print(path)
        nonlocal inroi
        print(node.get('code', ''))
        if startlabel in node.get('code', ''):
            inroi = True
        elif endlabel in node.get('code', ''):
            inroi = False
        if inroi:
            if node['ntype'] in ('FunctionDef', 'AsyncFunctionDef'):
                inroi_paths.append(path)
        ckeys = ['body', 'orelse', 'handlers', 'finalbody', 'cases']
        for ck in ckeys:
            if ck in node:
                for i in range(len(node[ck])):
                    _find(node[ck][i], path + [ck, i])
    _find(tree, [])
    return inroi_paths

def suppress_other_funcs(code, max_tokens, avoid_patterns=[SECTION_START_LABEL, SECTION_END_LABEL, INSERT_LABEL], retain_roi=True):
    origcode = code
    labelreps = ['_GRAPY_INNER_LABEL_' + str(i) for i in range(len(avoid_patterns))]
    labelmap = {avoid_patterns[i]: labelreps[i] for i in range(len(avoid_patterns))}
    labelinvmap = {labelreps[i]: avoid_patterns[i] for i in range(len(avoid_patterns))}
    for k, v in labelmap.items():
        code = code.replace(k, v)
    origtokens = estimate_tokens(code)
    tokens2supp = origtokens - max_tokens
    if tokens2supp < 0:
        return (origcode, tokens2supp)
    tree = code2tree(code=code, def_cutoff=False, keep_comments=True)
    paths, _ = brutal_gets(node=tree, condfunc=lambda x: x.get('ntype') in ('FunctionDef', 'AsyncFunctionDef') if isinstance(x, dict) else False, blocked_chains=[], advance_blockers=[])
    paths.sort(key=lambda x: len(x), reverse=True)
    paths_in_roi = find_func_paths_in_roi(code2tree(code), startlabel=labelmap[SECTION_START_LABEL], endlabel=labelmap[SECTION_END_LABEL]) if retain_roi else []
    suppressed = 0
    newcode = code
    newtokens = origtokens
    unfulfilled = tokens2supp
    for path in paths:
        if path in paths_in_roi:
            continue
        node = access_nested_data(tree, keychain=path)
        if any([o in node['code'] for o in labelreps]):
            continue
        newcode = '略'
        omiter = {'ntype': 'Expr', 'value': {'ntype': 'Constant', 'value': newcode}}
        newtokens = 3
        subtree = code2tree(node['code'], def_cutoff=False)
        if subtree['body'][0]['body']:
            if subtree['body'][0]['body'][0]['ntype'] == 'Expr':
                if subtree['body'][0]['body'][0]['value']['ntype'] == 'Constant':
                    subtree['body'][0]['body'] = [subtree['body'][0]['body'][0], copy.deepcopy(omiter)]
                else:
                    subtree['body'][0]['body'] = [copy.deepcopy(omiter)]
            set_nested_data(tree, path, subtree, inplace=True)
            _, newcode = tree2ast(tree)
            newtokens = estimate_tokens(newcode)
            unfulfilled = newtokens - max_tokens
            if unfulfilled <= 0:
                break
    for k, v in labelinvmap.items():
        newcode = newcode.replace(k, v)
    return (newcode, unfulfilled)

def suppress_code_section(code, max_tokens):
    holder = '# 省略{n}行，详见源代码'
    holder_tokens = 18
    if estimate_tokens(code) <= max_tokens:
        return code
    lines = code.split('\n')
    nums = list(range(len(lines)))
    xnums = []
    xlines = {}
    for i in range(len(nums) // 2):
        xnums.append(nums[i])
        xnums.append(nums[-i - 1])
    if len(nums) % 2 != 0:
        xnums.append(nums[len(nums) // 2])
    tokens = 0
    for x in xnums:
        tokens = tokens + (estimate_tokens(lines[x]) + 1)
        if tokens > max_tokens - holder_tokens:
            indents = 0 if x == 0 else len(lines[x]) - len(lines[x].lstrip())
            xlines[x] = holder.format(n=len(lines) - len(xlines))
            break
        xlines[x] = lines[x]
    xlines = sorted(xlines.items(), key=lambda item: item[0])
    xlines = [l[1] for l in xlines]
    return '\n'.join(xlines)

def suppress_code_inserter(code, max_tokens):
    if not INSERT_LABEL in code:
        logger.warning(f'没insert标别用suppress_code_inserter')
        return suppress_code_section(code, max_tokens)
    lines = code.split('\n')
    holder = '# ...'
    holder_tokens = 6
    idx = [l for l in range(len(lines)) if INSERT_LABEL in lines[l]][0]
    n = len(lines)
    if n == 0:
        return []
    if not 0 <= idx < n:
        raise ValueError('idx must be in [0, len(lines))')
    result = {}
    result[idx] = lines[idx]
    d = 1
    leftfull = False
    rightfull = False
    tokens = estimate_tokens(lines[idx])
    if tokens > max_tokens:
        logger.warning(f'才放一行就超了{tokens - max_tokens}个字数')
    while len(result) < n:
        left = idx - d
        right = idx + d
        added = False
        if left >= 0:
            tokens = tokens + (estimate_tokens(lines[left]) + 1)
            if tokens > max_tokens - 2 * holder_tokens:
                break
            result[left] = lines[left]
            added = True
        if left <= 0:
            leftfull = True
        if right < n:
            tokens = tokens + (estimate_tokens(lines[right]) + 1)
            if tokens > max_tokens - 2 * holder_tokens:
                break
            result[right] = lines[right]
            added = True
        if right >= len(lines) - 1:
            rightfull = True
        if not added:
            break
        d = d + 1
    xlines = sorted(result.items(), key=lambda item: item[0])
    xlines = [l[1] for l in xlines]
    recode = '\n'.join(xlines)
    if not leftfull:
        recode = holder + '\n' + recode
    if not rightfull:
        recode = recode + '\n' + holder
    return recode
sys_pattern = '你是一名编程助手。用户会让你帮忙生成、修改或补全python的任务流代码，或根据已有代码向你提问。'
corrector_tem = '\n你的上轮输出不合法。请严格按照要求输出。重申一遍，你的输出必须按照以下自定义格式：\n[<ACTION>]: ...{corr_explainer}\n[<EXTRA_PARAM1>]: ...\n... \n\n比如，当action为orchestrate时，你需要填写[<ACTION>]{explain_labeler}、[<CODE>]，当action为check_codes时，你需要填写[<ACTION>]{explain_labeler}、[<SELECTIONS>]，当action为talk_to_user时，你需要填写[<ACTION>]{explain_labeler}、[<MESSAGE>]，等等以此类推，一开始都告诉你了。\n报错信息：<<<ERR_REP>>>\n'
corr_explainer = '\n[<EXPLAIN>]: ...'
explain_labeler = '、[<EXPLAIN>]'
base_sys_prompt_tem = sys_pattern + '\n背景：我们的执行环境是一种新型的交互式REPL内核，会维护变量状态，支持查看节点历史变量值、查看内核变量当前值等高级功能。当用户提到产生任务流或pipeline等相关内容时，你可以理解为产生python代码。用户也可能就已有代码和状态向你提问，让你讲解。除此之外，如果由于缺乏必要信息而难以一次性产生代码或回答，你还有一些其他选择，例如向用户提问、查看包的源代码、查看变量记录或实时值等。另外，对你的输出还有一些结构化格式要求。请看下面详细说明。\n\n用户在每次提问时，会给你发送的内容包括：\n- 用户的问题或者对任务流的描述；\n- 当前已有的代码，里面标注了需重点关注的位置；\n- 在当前环境下可用的函数、类、对象等可用的变量以及各自的描述，描述可能有缺失，但你可以使用查看代码的action（后面会介绍）自己去看源代码。    描述里直接给出的函数、类、对象等都是直接可用的，不用重复声明或导入。\n\n关于用户发来的现有代码以及你该如何应对的详细解释：\n现有代码中，会用\'{cmt_bf_label}{INSERT_LABEL}\'标注出需重点关注的位置，或者用\'{cmt_bf_label}{SECTION_START_LABEL}\'和\'{cmt_bf_label}{SECTION_END_LABEL}\'标注出需重点关注的的区间（可能被用户称作选区、选中区域、高亮部分等近义词，或者英文selection、selected area等）。根据用户的语义，有两种可能的情况：\n** 如果你认为用户希望你生成或修改任务流代码：**\n\'{cmt_bf_label}{INSERT_LABEL}\'代表需要你插入新代码的位置，而\'{cmt_bf_label}{SECTION_START_LABEL}\'和\'{cmt_bf_label}{SECTION_END_LABEL}\'代表需要你替换的区间。这两种格式的标记不会同时出现，每次只会出现一种。说白了就是看到\'{cmt_bf_label}{INSERT_LABEL}\'时需要你插入新代码，看到\'{cmt_bf_label}{SECTION_START_LABEL}\'和\'{cmt_bf_label}{SECTION_END_LABEL}\'时需要你替换一个区间的代码。你仍然可以使用其他行动收集所需的信息，但最后一个行动需要是orchestrate，除非fail_to_generate。不管是插入还是替换，你只需要增量地产生新代码，不要重复已有的代码。系统会直接把你产生的代码插入或替换到已有代码中标记的地方去。\n** 如果你认为用户希望你根据已有代码解答他的疑问：**\n这种情况则不需要使用orchestrate行动产生代码。你在回答用户问题之前仍然可以使用各种行动来查找信息，但最终需要使用talk_to_user来回答用户。\n\n以下会介绍你的一些行动（action）。注意每次允许选择一个或多个行动（直接平铺、用换行隔开每个行动选项即可）。只有在确保各个行动无任何相互依赖关系时才能同时选择多个行动。而当你不确定时，分成多次执行、每次只选择单个行动总是保险的。使用各行动的总体格式设计为自定义复合格式，并不是纯json，每个行动因为参数不一样所以格式有区别。请务必遵守使用每个行动时所要求的格式。\n\n# ------ 你的所有行动选项 ------\norchestrate:\n介绍：选该行动来输出你最终的局部任务流代码。当你选择orchestrate输出代码之后，这轮任务也会自动结束。\n你的输出格式：\n[<ACTION>]: orchestrate{orc_explainer}\n[<CODE>]: ```python\n# 这里放你的代码。不管是插入还是替换区间，你只需要产生新的用于插入或替换的局部代码，不要重复已有的代码。基础缩进可以按0处理，也就是说你不用考虑目标位置已有的缩进。\n```\n\nfail_to_generate:\n介绍：当你认定无法产生任务流时，可以选择失败，放弃生成任务流代码。失败的场景例如发现环境里缺少必要的依赖（调用check_pkg_exist后得知）、反复报错无法突破、或者其他你认为无法生成任务流的情况。\n你的输出格式：\n[<ACTION>]: fail_to_generate{fail_explainer}\n\ncheck_signatures:\n介绍：如果对可用的函数或类的描述中有使用\'略\'或\'...\'省略的部分，你可以使用该行动来获取完整的介绍。但是本行动不会返回源代码。\n你的输出格式：\n[<ACTION>]: check_signatures{base_explainer}\n[<SELECTIONS>]: ```json\n// 请使用json格式来表达一个list[dict]的结构。每个dict都是一组用来查看一个工具signature的参数。格式为：\n[\n    {{\n        "module": "(str) 必须提供module原名（注意不是alias），例如\'pandas\'、\'dbutils.pooled_db\'等。",\n        "class": "(str|None) 不提供class仅提供func则表示需要查找独立函数的介绍，若提供class而不提供func则是查找类的介绍、以及类内所有函数的介绍，若提供class和func则是查找类内单个函数的介绍。都需要使用原名而不是alias。",\n        "func": "(str|None) 上面已经解释过了，和class搭配、可以提供也可以不提供，会起到完全不同的效果。也需要用原名。"\n    }},\n    ... // 其他更多查看signature的参数组\n]\n```\n\ncheck_codes:\n介绍：由于用户提供的对可用的函数、类、对象等的描述可能不足，在你对其中任何项目有疑问时可使用，一次可批量查看多个代码，且允许多次甚至连环使用。\n你的输出格式：\n[<ACTION>]: check_codes{checkcode_explainer}\n[<SELECTIONS>]: ```json\n// 请使用json格式来表达一个list[dict]的结构。每个dict都是一组用来查看一块代码的参数。格式为：\n[\n    {{\n        "module": "(str) 必须提供module原名（注意不是alias），例如\'pandas\'、\'dbutils.pooled_db\'等。",\n        "class": "(str|None) 不提供class仅提供func则表示需要查找独立函数的代码，若提供class而不提供func则是查找整个class的代码，若提供class和func则是查找类内函数的代码。都需要使用原名而不是alias。比如，如果有个类alias名叫DF，而它的原名叫DataFrame，是通过代码\'from pandas import DataFrame as DF\'而来，那你在任务流代码中需使用DF、但在该查询源代码工具中需使用DataFrame。",\n        "func": "(str|None) 上面已经解释过了，和class搭配、可以提供也可以不提供，会起到完全不同的效果。也需要用原名。",\n        "env_level": // (int) 从0、1、2中选一个。因为sys.path中有3层不同的导入来源，每个来源优先级不一样，且是单向可见的，所以需要记录你现在所查看的代码位于第几层环境。用户给你的代码和每次check_codes工具返出的代码里都应该有标注env_level，直接抄过来就行。如果没标的话就选0。\n    }},\n    ... // 其他更多查看代码的参数组\n]\n// 备注：尽可能少查看整个module的代码，因为可能会很长。可以优先查看单个类或单个函数的代码。\n```\n{talk_to_user_action}\ncheck_pkg_exist:\n介绍：用于查看一个包是否存在，以及它的版本号。注意：用户在Tools中有提供的模块、类、函数等，默认都是已经存在的，不需要查看！只有当你想要使用用户未提到的包时才需查看是否存在。一般用于检查pip安装在site-packages里的包。如果存在，会返回版本号，否则返回\'package does not exist\'。\n你的输出格式：\n[<ACTION>]: check_pkg_exist\n[<PKG_NAME>]: 你想查看的包名，注意不是类名、函数名，而是整个依赖的名称，例如pandas、numpy这种。\n\ncheck_var_values:\n介绍：主动查看变量在某代码块上的历史值、或变量在全局的最新值。会返回一组含type、value的字典数组。如果查询某个变量时出现问题，会为它多返回一个msg字段把出现的问题告知你。\n[<ACTION>]: check_var_values{checkvar_explainer}\n[<SELECTIONS>]: ```json\n[\n    {{\n        "block_uid": "(str) 给你提供的代码里可能会使用<codeblock uid="xxx"></codeblock>标记出一些代码块，说的就是这个uid。如果你需要查看的不是变量的最新值、而是变量经过这个代码块时留下的快照，就请提供对应的uid。如果你本来就希望查看变量的最新值、或提供的代码里没有可用的uid，请在这一栏填写\'<LATEST>\'。",\n        "var_name": "(str) 变量名称。允许含点的attribute，但是暂不允许含中括号的索引。"\n    }},\n    // 可以批量查看多个变量值\n]\n```\n\ncheck_block_logs:\n介绍：主动查看某代码块上次执行时产生的日志（通常由print、log函数、报错等产生）。如果查询某个日志时出现问题，会为它多返回一个msg字段把出现的问题告知你。\n[<ACTION>]: check_block_logs{checklog_explainer}\n[<SELECTIONS>]: ```json\n[\n    {{\n        "block_uid": "(str) 给你提供的代码里可能会使用<codeblock uid="xxx"></codeblock>标记出一些代码块，说的就是这个uid。"\n    }},\n    // 可以批量查看多个节点的日志\n]\n```\n\n# ------------------\n\n# 注意事项\n关于orchestrate时code格式的注意事项：\n- 每行代码尽可能简洁，不要把好几个逻辑挤到一行代码里，比如：\n    y = func1(func2(x))\n  应该拆成：\n    _temp = func2(x) # 可按实际情况合理为临时变量命名\n    y = func1(_temp)\n- 在函数定义和类定义下，可使用doc格式来写介绍；\n- 定义函数参数时，尽可能使用 typing/typing_extensions中的Annotated和Doc （不需要import，直接用）来为每个参数以及返回值标注类型和描述。\n- 在代码逻辑中，可以使用\'#\'开头的注释，并且提倡在逻辑较复杂的地方多用注释，以利于用户理解。每条注释最好独占一行，不要放在一行代码后面。但是，千万不要在函数定义参数的括号内、dict或list之中放置注释，因为解析器不支持。\n- 代码中不需要写 if __name__ == \'__main__\' ，直接写代码就行。\n- 由于执行器使用了一种支持异步的REPL交互编程技术，当在顶层逻辑中调用异步函数时，不需要写asyncio.run()，直接用await或async for就行。\n\n一些提示：\n- 当用户提到工具（tool）时，泛指类、函数、或类的实例化对象。\n- 用户提供的工具介绍里，可能会包含类、函数、对象等的原名和假名。比如，import pandas as pd 会导致一个类的原名为DataFrame、假名为pd.DataFrame，而from pandas import DataFrame as DF 会导致原名为DataFrame的类的假名变成DF。在代码中，显然应该使用假名（这是python的规则），而当使用check_codes工具时，则必须使用原名。\n- 可能会把前序任务的历史也在上下文中给到你，供参考。\n- 你在生成任务流代码时，只需产生用于插入到\'{cmt_bf_label}{INSERT_LABEL}\'处或用于替换掉\'{cmt_bf_label}{SECTION_START_LABEL}\'和\'{cmt_bf_label}{SECTION_END_LABEL}\'区间的局部代码，不要重复其他代码。切记，不要重复其他代码！\n'
base_explainer = '\n[<EXPLAIN>]: 用自然语言表达你的思路'
orc_explainer = '\n[<EXPLAIN>]: 用自然语言表达你的思路，以及对任务流代码的解释。'
fail_explainer = '\n[<EXPLAIN>]: 说出你的理由，为什么放弃？'
checkcode_explainer = '\n[<EXPLAIN>]: 用自然语言表达你的思路，为什么需要像这样查看代码'
talk_to_user_action = '\ntalk_to_user:\n介绍：当你对用户的需求不完全理解、或者感觉到有必要向用户告知或请示时，可以使用这个行动。当用自然语言回答用户问题时用的也是这个行动。\n你的输出格式：\n[<ACTION>]: talk_to_user\n[<EXPLAIN>]: 用自然语言表达你的思路，为什么需要联系用户\n[<MESSAGE>]: 你向用户的提问或通报或回答，格式为自然语言。\n'
checkvar_explainer = '\n[<EXPLAIN>]: 用自然语言表达你的思路，为什么需要查看这个变量，要用它来做什么'
checklog_explainer = '\n[<EXPLAIN>]: 用自然语言表达你的思路，为什么需要查看这个节点的日志'
cmt_bf_label = '# '
user_qpartten = 'Please generate your output based on the following information.'
user_prompt_template = user_qpartten + '\n' + "\n========== Existing code ==========\n{existing_code}\n\n========== Tools ========== \n{tools_desc}\n\n========== User's query ========== \n{user_input}\n\n========== Notes ========== \n{note}\n\nPlease use the same language (English/Chinese) as the user's query for any natural language parts. You must comply with the format criteria!\n"

class Coder(Node):

    def __init__(self, nodedict, error_behavior='raise', role='module', base_url=None, api_key=None, default_model='', llm_max_tokens=-1):
        super().__init__(nodedict=nodedict, memory_fields=('role', 'content'), important_patterns=[('system', sys_pattern, 'oldest')], error_behavior=error_behavior, llm_max_tokens=llm_max_tokens)
        self.llm = OpenAILLM(base_url=base_url, api_key=api_key, default_model=default_model)
        self.role = role
        self.base_tokens = 0
        my_base_explainer = ''
        my_orc_explainer = ''
        my_fail_explainer = ''
        my_checkcode_explainer = ''
        my_talk_to_user_action = ''
        my_checkvar_explainer = ''
        my_checklog_explainer = ''
        my_corr_explainer = ''
        my_explain_labeler = ''
        my_cmt_bf_label = ''
        if role != 'sug':
            my_base_explainer = base_explainer
            my_orc_explainer = orc_explainer
            my_fail_explainer = fail_explainer
            my_checkcode_explainer = checkcode_explainer
            my_talk_to_user_action = talk_to_user_action
            my_checkvar_explainer = checkvar_explainer
            my_checklog_explainer = checklog_explainer
            my_corr_explainer = corr_explainer
            my_explain_labeler = explain_labeler
            my_cmt_bf_label = cmt_bf_label
        self.base_sys_prompt = base_sys_prompt_tem.format(INSERT_LABEL=INSERT_LABEL, SECTION_END_LABEL=SECTION_END_LABEL, SECTION_START_LABEL=SECTION_START_LABEL, orc_explainer=my_orc_explainer, fail_explainer=my_fail_explainer, checkcode_explainer=my_checkcode_explainer, base_explainer=my_base_explainer, talk_to_user_action=my_talk_to_user_action, checkvar_explainer=my_checkvar_explainer, checklog_explainer=my_checklog_explainer, cmt_bf_label=my_cmt_bf_label)
        self.corrector = corrector_tem.format(corr_explainer=my_corr_explainer, explain_labeler=my_explain_labeler)
        self.base_sys_tokens = estimate_tokens(self.base_sys_prompt)
        assert self.llm_max_tokens > self.base_sys_tokens + 1000, f'LLM max tokens not enough. Should be at least {self.base_sys_tokens + 1000}'

    @execution(msgs_to_memory=False, submit_final_rsp=False)
    async def execute(self, start_msgs, session_id='<DEFAULT>'):
        assert self.llm_max_tokens > 1000, f'LLM must be able to intake >1000 tokens for this application to run.'
        pregen_max_tokens = int(self.llm_max_tokens * 0.67)
        roi_echo_max = 140
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
                llm_sys_input = self.base_sys_prompt
                formated_context.append({'role': 'system', 'content': self.base_sys_prompt})
            user_data = start_msgs[1]['content'].copy()
            nl_tools_desc = all_desc_to_nl(user_data['tools_desc'])
            orig_tools_desc = user_data['tools_desc']
            user_data['tools_desc'] = nl_tools_desc
            cate4llm = {'dag': 'logic', 'funcs': 'function', 'classes': 'class'}[user_data['category']]
            action4llm = {'insert': 'insert a', 'replace': 'replace the selected', 'allbelow': 'replace the selected', 'single': 'replace the selected', 'append': 'insert a'}[user_data['mode']]
            if self.role == 'sug':
                note = f"Please insert one or a few lines of code. The 'talk_to_user' action is temporarily disabled. Try to orchestrate as soon as possible."
                assert INSERT_LABEL in user_data['existing_code']
            else:
                note = f"Please {action4llm} {cate4llm} or answer the user's question, depending on what they are asking for."
            if user_data.get('class_above'):
                assert cate4llm in ('logic', 'function'), f"nested class not allowed: {user_data['class_above']}"
                note = note + f"\nCaution: the target function is an attribute under class {user_data['class_above']}."
            roi = ''
            if SECTION_START_LABEL in user_data['existing_code']:
                clines = user_data['existing_code'].split('\n')
                roidexs = [x for x in range(len(clines)) if SECTION_START_LABEL in clines[x] or SECTION_END_LABEL in clines[x]]
                roi = '\n'.join(clines[roidexs[0]:roidexs[-1] + 1])
                if not roi.count(SECTION_START_LABEL) == 1 or not roi.count(SECTION_END_LABEL) == 1:
                    logger.error(f'有问题的roi，缺label：{roi}')
                roi_echo_rsv_codeonly = min(roi_echo_max, estimate_tokens(roi))
                roi_echo_rsv = roi_echo_rsv_codeonly + 20
            elif INSERT_LABEL in user_data['existing_code']:
                roi = user_data['existing_code']
                roi_echo_rsv_codeonly = min(roi_echo_max, estimate_tokens(roi))
                roi_echo_rsv = roi_echo_rsv_codeonly + 20
            else:
                roi_echo_rsv = 0
            user_dead_tokens = estimate_tokens(user_prompt_template) + estimate_tokens(note)
            existing_code = user_data['existing_code']
            code_max_dry_tokens = pregen_max_tokens - self.base_sys_tokens - user_dead_tokens - estimate_tokens(user_data['user_input']) - estimate_tokens(user_data['tools_desc'])
            try:
                existing_code, unfulfilled = suppress_other_funcs(existing_code, code_max_dry_tokens - roi_echo_rsv)
                if unfulfilled > 0:
                    if SECTION_START_LABEL in user_data['existing_code'] and SECTION_END_LABEL in user_data['existing_code']:
                        logger.debug('光压roi外的函数body还不够，欠{}个token，连roi里的函数也压', unfulfilled)
                        existing_code, unfulfilled = suppress_other_funcs(user_data['existing_code'], code_max_dry_tokens - roi_echo_rsv, retain_roi=False)
                    if unfulfilled > 0:
                        logger.debug('光压含roi的函数body都不够，欠{}个token，等后续压缩ROI两侧或工具介绍', unfulfilled)
            except Exception as e:
                logger.warning(f'压缩其他函数失败：{e}')
                traceback.print_exc()
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
            code_max_tokens = max(code_max_dry_tokens, coretokens + min(lefttokens + righttokens, 400))
            if coretokens + min(lefttokens + righttokens, 400) + estimate_tokens(user_data['user_input']) > pregen_max_tokens - self.base_sys_tokens - user_dead_tokens:
                raise RuntimeError(f"Unable to generate, due to {coretokens + min(lefttokens + righttokens, 400) + estimate_tokens(user_data['user_input']) - (pregen_max_tokens - self.base_sys_tokens - user_dead_tokens)} tokens exceeding. Try narrow your selection.")
            totalovfl = xst_code_tokens + roi_echo_rsv - code_max_tokens
            if totalovfl > 0:
                logger.debug('需要压缩代码{}个token', totalovfl)
                leftovfl = int(lefttokens / (lefttokens + righttokens) * totalovfl)
                rightovfl = int(righttokens / (lefttokens + righttokens) * totalovfl)
                newleft = suppress_tokens(left, max(lefttokens - leftovfl, 0))
                newright = suppress_tokens(right, max(righttokens - rightovfl, 0))
                existing_code = newleft + mid + newright
            llm_user_input = user_prompt_template.format(existing_code=existing_code, user_input=user_data['user_input'], tools_desc=user_data['tools_desc'], note=note)
            if estimate_tokens(llm_user_input) + self.base_sys_tokens > pregen_max_tokens - roi_echo_rsv:
                ovfl = estimate_tokens(llm_user_input) + roi_echo_rsv + self.base_sys_tokens - pregen_max_tokens
                logger.warning(f'用户提问时字数超限了{ovfl}个，压缩工具介绍。')
                tools_desc = user_data['tools_desc']
                user_query = user_data['user_input']
                tdtokens = estimate_tokens(tools_desc)
                if tdtokens > ovfl:
                    tools_desc = all_desc2nl_in_tokens(orig_tools_desc, tdtokens - ovfl)
                    td0 = tools_desc
                    tools_desc = suppress_text(tools_desc, tdtokens - ovfl, omitpatterns=['------ Functions ------', '------ Objects ------', '------ Classes ------'])
                    if td0 != tools_desc:
                        logger.info(f'警告：用户选中部分长度逼近极限或工具太多导致没能给每个工具保留最少一个路径')
                else:
                    querymax = estimate_tokens(user_query) + estimate_tokens(tools_desc) - ovfl + roi_echo_rsv
                    user_query_old = user_query
                    user_query = suppress_tokens(user_query, max(50, querymax))
                    tools_desc = ''
                    if user_query_old != user_query:
                        logger.info('用户问题也被压缩: {} -> {}', user_query_old, user_query)
                llm_user_input = user_prompt_template.format(existing_code=existing_code, user_input=user_query, tools_desc=tools_desc, note=note)
            vacancy = pregen_max_tokens - (estimate_tokens(llm_user_input) + self.base_sys_tokens)
            if vacancy > 40:
                if INSERT_LABEL in roi:
                    roi = suppress_code_inserter(roi, roi_echo_max)
                    if roi:
                        llm_user_input = llm_user_input + ('\nPlease notice the insert position in the existing code at:\n' + roi)
                        if self.role == 'sug':
                            llm_user_input = llm_user_input + '\n\nOnly generate the code to be inserted in between. Do NOT repeat existing code.'
                elif SECTION_END_LABEL in roi:
                    roi = suppress_code_section(roi, roi_echo_max)
                    if roi:
                        llm_user_input = llm_user_input + ('\nPlease focus on the selected area in the existing code:\n' + roi)
            formated_context.append({'role': 'user', 'content': llm_user_input})
            newtokens = estimate_tokens(llm_user_input) + estimate_tokens(llm_sys_input)
            self.base_tokens = self.base_sys_tokens + estimate_tokens(llm_user_input)
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
            formated_context.append({'role': 'user', 'content': self.corrector.replace('<<<ERR_REP>>>', start_msgs[1]['content'])})
            newtokens = 20
        else:
            raise
        input_context = self.get_n_memory(session_id=session_id, max_tokens=self.llm_max_tokens - newtokens, important_patterns=[('system', sys_pattern, 'oldest'), ('user', user_qpartten, 'newest')])
        input_context = input_context + formated_context
        await self.submit_rsps(formated_context, session_id=session_id, to_rsp_queue=False)
        if not (input_context[0]['content'].startswith(self.base_sys_prompt[:10]) and input_context[0]['role'] == 'system'):
            logger.warning('有bug，input_context开始不是system prompt。roles: {}', [c['role'] for c in input_context])
            maybe_sys_rownum = [i for i in range(len(input_context)) if (input_context[i]['content'].startswith(self.base_sys_prompt[:10]) and input_context[i]['role']) == 'system']
            if maybe_sys_rownum:
                logger.warning('发现system prompt被移到后面了，移回前面。maybe_sys_rownum：{}', maybe_sys_rownum)
                input_context = [input_context[maybe_sys_rownum[0]]] + [input_context[i] for i in range(len(input_context)) if not i in maybe_sys_rownum]
            else:
                input_context = [{'role': 'system', 'content': self.base_sys_prompt}] + input_context
                logger.warning('发现system prompt缺失，补上，但是注意可能有超token风险：{}（风险不大，前面是算了sysprompt的字数来缩减过上下文的', estimate_tokens(input_context))
        rsp = await self.llm.aanswer(input_context)
        await self.submit_rsps([{'role': 'assistant', 'content': rsp}], session_id=session_id)
        return