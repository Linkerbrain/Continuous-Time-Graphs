"""
    TODO:

    Parameters module idea.

    @callable
    def func_accepts_pars_object(pars):
        pass

    translates to

    def func_accepts_pars_object(*args, **kwargs):
        if len(args) == 1 and isinstance(args[0], Parameters):
            return original_func_accepts_pars_object(args[0])
        else:
            return original_func_accepts_pars_object(Parameters.parse(*args, **kwargs)))

"""