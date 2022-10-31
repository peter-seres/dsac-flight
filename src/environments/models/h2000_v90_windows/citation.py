# Import the low-level C/C++ module
if __package__ or "." in __name__:
    # noinspection PyUnresolvedReferences
    from . import _citation
else:
    # noinspection PyUnresolvedReferences
    import _citation

import builtins as __builtin__


# noinspection PyBroadException
def _swig_repr(self):
    try:
        strthis = "proxy of " + self.this.__repr__()
    except __builtin__.Exception:
        strthis = ""
    return "<%s.%s; %s >" % (
        self.__class__.__module__,
        self.__class__.__name__,
        strthis,
    )


# noinspection PyShadowingBuiltins
def _swig_setattr_nondynamic_instance_variable(set):
    def set_instance_attr(self, name, value):
        if name == "thisown":
            self.this.own(value)
        elif name == "this":
            set(self, name, value)
        elif hasattr(self, name) and isinstance(getattr(type(self), name), property):
            set(self, name, value)
        else:
            raise AttributeError("You cannot add instance attributes to %s" % self)

    return set_instance_attr


# noinspection PyShadowingBuiltins
def _swig_setattr_nondynamic_class_variable(set):
    def set_class_attr(cls, name, value):
        if hasattr(cls, name) and not isinstance(getattr(cls, name), property):
            set(cls, name, value)
        else:
            raise AttributeError("You cannot add class attributes to %s" % cls)

    return set_class_attr


def _swig_add_metaclass(metaclass):
    """Class decorator for adding a metaclass to a SWIG wrapped class - a slimmed down version of six.add_metaclass"""

    def wrapper(cls):
        return metaclass(cls.__name__, cls.__bases__, cls.__dict__.copy())

    return wrapper


class _SwigNonDynamicMeta(type):
    """Metaclass to enforce nondynamic attributes (no new attributes) for a class"""

    __setattr__ = _swig_setattr_nondynamic_class_variable(type.__setattr__)


def initialize():
    return _citation.initialize()


def terminate():
    return _citation.terminate()


def step(cmd):
    return _citation.step(cmd)
