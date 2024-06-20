import pretty_errors

pretty_errors.configure(
    separator_character = '-',
    line_number_first   = True,
    display_link        = True,
    line_color          = pretty_errors.RED + '> ' \
                        + pretty_errors.RED_BACKGROUND \
                        + pretty_errors.BRIGHT_WHITE,
)