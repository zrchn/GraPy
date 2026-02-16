import json
import re
import sys
import traceback
from _sbconsts import UID_COMMENT_LEFTLABEL, UID_COMMENT_RIGHTLABEL

def parse_client_error(stacktrace, srccodes={}) -> list[dict]:
    tlines = stacktrace.split('\n')
    errors = []
    for (i, line) in enumerate(tlines):
        node_ids = re.findall(f'{UID_COMMENT_LEFTLABEL}(.*?){UID_COMMENT_RIGHTLABEL}', line, re.DOTALL)
        if not node_ids:
            continue
        node_id = node_ids[-1]
        err = ''
        module = ''
        if i > 0:
            maybe_module_line = tlines[i - 1]
            modnames = re.findall('File "<virtual:(.*?)>",', stacktrace, re.DOTALL)
            if modnames:
                module = modnames[0].strip()
        if i < len(tlines) - 1:
            errline = tlines[i + 1].strip()
            if errline.startswith('File "'):
                err = stacktrace
            else:
                err = '\n'.join(tlines[i:])
        node_prop = 'task'
        if line.strip().startswith('def') or line.strip().startswith('async def'):
            node_prop = 'func'
        errinfo = {'node_id': node_id, 'error': err, 'node_prop': node_prop, 'module': module}
        errors.append(errinfo)
    if not errors:
        node_id = None
        node_prop = 'general'
        module = None
        if srccodes and (not errors) and ('File "<virtual:' in stacktrace):
            errlines = [l for l in tlines if 'File "<virtual:' in l and ', line' in l]
            if errlines:
                errline = errlines[0]
                modnames = re.findall('File "<virtual:(.*?)>",', errline, re.DOTALL)
                if modnames:
                    module = modnames[0].strip()
                    codelineno = errline.split(', line ')[-1]
                    if codelineno.isdigit():
                        codelineno = int(codelineno)
                        if module and codelineno >= 0:
                            modpath = module.replace('.', '/') + '.py'
                            if modpath in srccodes:
                                try:
                                    srccode = srccodes[modpath]
                                    codeline = srccode.split('\n')[codelineno - 1]
                                    stacktrace = stacktrace.rstrip() + f'\n(buggy code: {codeline})'
                                    node_ids = re.findall(f'{UID_COMMENT_LEFTLABEL}(.*?){UID_COMMENT_RIGHTLABEL}', codeline, re.DOTALL)
                                    if node_ids:
                                        node_id = node_ids[0]
                                        node_prop = 'task'
                                    if codeline.strip().startswith('def') or codeline.strip().startswith('async def'):
                                        node_prop = 'func'
                                except Exception as e:
                                    print(f'报错栈不带目标代码，翻找过程中二次报错：{e}', file=sys.__stderr__)
                                    print(traceback.format_exc(), file=sys.__stderr__)
        errors = [{'node_id': node_id, 'error': stacktrace, 'node_prop': node_prop, 'module': module}]
    return errors

class ClientParsedError(Exception):

    def __init__(self, stacktrace, srccodes={}):
        errors = parse_client_error(stacktrace, srccodes=srccodes)
        msg = json.dumps(errors, ensure_ascii=False)
        super().__init__(msg)

class KernelInterrupted(Exception):

    def __init__(self, msg):
        super().__init__(msg)