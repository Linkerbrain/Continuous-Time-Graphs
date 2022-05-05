import traceback

def no_traceback(func, params):
    try:
        func(params)
    except Exception as e:
        exceptiondata = traceback.format_exc().splitlines()
        exceptionarray = [exceptiondata[-1]] + exceptiondata[1:-1]

        class b:
            BLUE = '\033[34m'
            WEIRD = '\033[41m'
            HEADER = '\033[95m'
            OKBLUE = '\033[94m'
            OKCYAN = '\033[96m'
            OKGREEN = '\033[92m'
            WARNING = '\033[93m'
            FAIL = '\033[91m'
            ENDC = '\033[0m'
            BOLD = '\033[1m'
            UNDERLINE = '\033[4m'

        def fix_fileline(string):
            parts = string.split('"')

            if len(parts) < 3:
                return string

            return '' + parts[0] + b.ENDC \
             + b.FAIL + "\\".join(parts[1].split("\\")[-3:]) + b.ENDC \
             + '' + parts[2] + b.ENDC

        def fix_errorline(string):
            parts = string.split(':')

            return b.FAIL + parts[0] + b.ENDC + ': '

        print(b.WEIRD + "- - - - - - - - - " + b.ENDC)
        print(fix_fileline(exceptiondata[-3]))
        print(b.BLUE + exceptiondata[-2] + "       " + b.ENDC)
        print()
        print(fix_errorline(exceptiondata[-1]))
        print("   ", e)
        print(b.WEIRD + "- - - - - - - - - " + b.ENDC)