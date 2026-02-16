
"""
Copyright (c) 2026 Zhiren Chen. Provided as-is for local use only.
"""

import os
from pathlib import Path
from loguru import logger
CONFIG_PATH = os.getcwd().rsplit('/',1)[0]+'/configs.yaml'
VARSEND_FUNC = '_se8N_dv742_arS' + '_safe'
CONSOLE_PRINTER = '_d87I3s08P9'
CONSOLE_INPUTER = '_input_via_cache'
INPUT_TIMEOUT = 300
RUN_START_LABEL = '<<<RUN-START>>>'
RUN_FINISH_LABEL = '<<<RUN-FINISH>>>'
UID_COMMENT_LEFTLABEL = '<<NODE-UID>>'
UID_COMMENT_RIGHTLABEL = '<</NODE-UID>>'
FUNCID_COMMENT_LEFTLABEL = '<<FUNC-ID>>'
FUNCID_COMMENT_RIGHTLABEL = '<</FUNC-ID>>'
REDIS_LIMIT_MAX = 10000
KEEP_HISTORY_RUNS = 10
BUILTIN_NAMES = ['_', 'sys', '_mods', '_disp_to_cache', '_send_vars_to_cache', '_input_via_cache', '_global_var_infos', '_get_kernel_var_infos', 'Doc', 'Any', 'pprint', 'plt', 'logger', 'matplotlib', '_run_id', 'Annotated', '_autocomplete', '_autocompleted', '_get_function_arg_names', '_function_arg_names', '_name_filter', '_global_var', '_local_var', '_var_name', '_local_vnames', '_global_vnames', '_local_var_infos', '_mod_name', '_mods', '_mod_errs', '_errs', '_mod_e', '_vars_tracking']
NAMESPACE_PICKLE_FILE = 'discarded'
EXTRA_BUILTIN_MODS = ['sys', 'importlib', 'PIL', 'PIL.ExifTags', 'PIL.GimpGradientFile', 'PIL.GimpPaletteFile', 'PIL.Image', 'PIL.ImageChops', 'PIL.ImageColor', 'PIL.ImageFile', 'PIL.ImageMode', 'PIL.ImagePalette', 'PIL.ImageSequence', 'PIL.PaletteFile', 'PIL.PngImagePlugin', 'PIL.TiffTags', 'PIL._binary', 'PIL._deprecate', 'PIL._imaging', 'PIL._util', 'PIL._version', 'cycler', 'dateutil.rrule', 'importlib._adapters', 'importlib._common', 'importlib.resources', 'kiwisolver', 'kiwisolver._cext', 'kiwisolver.exceptions', 'matplotlib', 'matplotlib._afm', 'matplotlib._api', 'matplotlib._api.deprecation', 'matplotlib._blocking_input', 'matplotlib._c_internal_utils', 'matplotlib._cm', 'matplotlib._cm_bivar', 'matplotlib._cm_listed', 'matplotlib._cm_multivar', 'matplotlib._color_data', 'matplotlib._constrained_layout', 'matplotlib._docstring', 'matplotlib._enums', 'matplotlib._fontconfig_pattern', 'matplotlib._image', 'matplotlib._layoutgrid', 'matplotlib._mathtext', 'matplotlib._mathtext_data', 'matplotlib._path', 'matplotlib._pylab_helpers', 'matplotlib._text_helpers', 'matplotlib._tight_bbox', 'matplotlib._tight_layout', 'matplotlib._version', 'matplotlib.artist', 'matplotlib.axes', 'matplotlib.axes._axes', 'matplotlib.axes._base', 'matplotlib.axes._secondary_axes', 'matplotlib.axis', 'matplotlib.backend_bases', 'matplotlib.backend_managers', 'matplotlib.backend_tools', 'matplotlib.backends', 'matplotlib.backends.registry', 'matplotlib.bezier', 'matplotlib.category', 'matplotlib.cbook', 'matplotlib.cm', 'matplotlib.collections', 'matplotlib.colorbar', 'matplotlib.colorizer', 'matplotlib.colors', 'matplotlib.container', 'matplotlib.contour', 'matplotlib.dates', 'matplotlib.dviread', 'matplotlib.figure', 'matplotlib.font_manager', 'matplotlib.ft2font', 'matplotlib.gridspec', 'matplotlib.hatch', 'matplotlib.image', 'matplotlib.inset', 'matplotlib.layout_engine', 'matplotlib.legend', 'matplotlib.legend_handler', 'matplotlib.lines', 'matplotlib.markers', 'matplotlib.mathtext', 'matplotlib.mlab', 'matplotlib.offsetbox', 'matplotlib.patches', 'matplotlib.path', 'matplotlib.projections', 'matplotlib.projections.geo', 'matplotlib.projections.polar', 'matplotlib.pyplot', 'matplotlib.quiver', 'matplotlib.rcsetup', 'matplotlib.scale', 'matplotlib.spines', 'matplotlib.stackplot', 'matplotlib.streamplot', 'matplotlib.style', 'matplotlib.style.core', 'matplotlib.table', 'matplotlib.texmanager', 'matplotlib.text', 'matplotlib.textpath', 'matplotlib.ticker', 'matplotlib.transforms', 'matplotlib.tri', 'matplotlib.tri._triangulation', 'matplotlib.tri._tricontour', 'matplotlib.tri._trifinder', 'matplotlib.tri._triinterpolate', 'matplotlib.tri._tripcolor', 'matplotlib.tri._triplot', 'matplotlib.tri._trirefine', 'matplotlib.tri._tritools', 'matplotlib.units', 'matplotlib.widgets', 'mpl_toolkits', 'mpl_toolkits.mplot3d', 'mpl_toolkits.mplot3d.art3d', 'mpl_toolkits.mplot3d.axes3d', 'mpl_toolkits.mplot3d.axis3d', 'mpl_toolkits.mplot3d.proj3d', 'packaging', 'packaging._structures', 'packaging.version', 'plistlib', 'pyexpat', 'pyexpat.errors', 'pyexpat.model', 'pyparsing', 'pyparsing.actions', 'pyparsing.common', 'pyparsing.core', 'pyparsing.exceptions', 'pyparsing.helpers', 'pyparsing.results', 'pyparsing.testing', 'pyparsing.unicode', 'pyparsing.util', 'redis.commands.json', 'redis.commands.json._util', 'redis.commands.json.commands', 'redis.commands.json.decoders', 'redis.commands.json.path', 'kernel_virtual_importer', 'xml', 'xml.parsers', 'xml.parsers.expat', 'xml.parsers.expat.errors', 'xml.parsers.expat.model']
vbs = False
stream = 1