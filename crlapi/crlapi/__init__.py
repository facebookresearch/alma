

def instantiate_class(arguments):
    from importlib import import_module

    d = dict(arguments)
    if "classname" in d:
        classname = d["classname"]
        del d["classname"]
    else:
        classname = d["class_name"]
        del d["class_name"]
    module_path, class_name = classname.rsplit(".", 1)
    module = import_module(module_path)
    c = getattr(module, class_name)
    return c(**d)


def get_class(arguments):
    from importlib import import_module
    d = dict(arguments)
    if "classname" in d:
        classname = d["classname"]
        del d["classname"]
    else:
        classname = d["class_name"]
        del d["class_name"]

    module_path, class_name = classname.rsplit(".", 1)
    module = import_module(module_path)
    c = getattr(module, class_name)
    return c


def get_arguments(arguments):
    from importlib import import_module

    d = dict(arguments)
    if "classname" in d:
        classname = d["classname"]
        del d["classname"]
    else:
        classname = d["class_name"]
        del d["class_name"]
    return d
