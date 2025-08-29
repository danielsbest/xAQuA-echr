BC_HEADER = "\033[95m"
BC_OK_BLUE = "\033[94m"
BC_OK_CYAN = "\033[96m"
BC_OK_GREEN = "\033[92m"
BC_WARNING = "\033[93m"
BC_FAIL = "\033[91m"
BC_END_C = "\033[0m"
BC_BOLD = "\033[1m"
BC_UNDERLINE = "\033[4m"


def print_error(msg):
    print(BC_FAIL + msg + BC_END_C)


def print_warning(msg):
    print(BC_WARNING + msg + BC_END_C)


def print_info(msg):
    print(BC_OK_CYAN + msg + BC_END_C)


def print_success(msg):
    print(BC_OK_GREEN + msg + BC_END_C)


def print_info_highlighted(msg):
    print(BC_OK_BLUE + BC_BOLD + msg + BC_END_C)
