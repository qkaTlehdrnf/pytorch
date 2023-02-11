import sys
import torch
from contextlib import contextmanager
from torch.backends import ContextProp, PropModule, __allow_nonbracketed_mutation

def set_flags(_enabled):
    orig_flags = (torch._C._get_transparent_hugepages_enabled(),)
    torch._C._set_transparent_hugepages_enabled(_enabled)
    return orig_flags

@contextmanager
def flags(enabled=False):
    with __allow_nonbracketed_mutation():
        orig_flags = set_flags(enabled)
    try:
        yield
    finally:
        with __allow_nonbracketed_mutation():
            set_flags(orig_flags[0])

class TransparentHugePagesModule(PropModule):
    def __init__(self, m, name):
        super(TransparentHugePagesModule, self).__init__(m, name)
    enabled = ContextProp(torch._C._get_transparent_hugepages_enabled, torch._C._set_transparent_hugepages_enabled)

# This is the sys.modules replacement trick, see
# https://stackoverflow.com/questions/2447353/getattr-on-a-module/7668273#7668273
sys.modules[__name__] = TransparentHugePagesModule(sys.modules[__name__], __name__)

# Add type annotation for the replaced module
enabled: bool
