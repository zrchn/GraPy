pickle = None
from kernel_cache_handler import _disp_to_cache, _send_vars_to_cache, _input_via_cache
from _sbconsts import NAMESPACE_PICKLE_FILE
default_namespace = {'_disp_to_cache': _disp_to_cache, '_send_vars_to_cache': _send_vars_to_cache, '_input_via_cache': _input_via_cache, '_': None}

def clean_unpicklables(dic, avoids=set()):
    cleaned_dic = {}
    unp = []
    for (k, v) in dic.items():
        if k in avoids:
            unp.append(k)
            continue
        try:
            v = pickle.dumps(v)
            cleaned_dic[k] = v
        except:
            unp.append(k)
    return (cleaned_dic, unp)

def reset_pickled_namespace():
    return
    with open(NAMESPACE_PICKLE_FILE, 'wb') as f:
        pickle.dump(default_namespace, f)