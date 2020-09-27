
def is_float(s):
    '''
    Tests if the input string, s, can be converted to a float number.
    '''
    try:
        x = float(s)
    except ValueError:
        return False
    else:
        return True

def load_parameters_from_txt(file_path):
    '''
    This helper function reads the lines of the given input txt file,
    and converts the content of the input file to a Python dict.
    The input txt file must be formatted as required:
    a. Each line starts with the name of a parameter, and is followed by
       a string of values split by comma. There should be a space between the
       parameter name and the values.
    b. Values in the string that can be converted to numbers will be converted
       to integers by defaults, unless they contain a point character '.' or
       the exp character 'e'. They will be converted to floats in this case.
       Otherwise the values will be kept as strings in the output dict.
    c. No space or tab is allowed at the end of each line.
    --
    Example:

    In example.txt:
    float_list 0.7,0.1,2e1
    int_list 200,12
    str_list thisisstring,0..1
    mixted_list 542,2e-1,.1,adf1,12w

    In [8]: params = load_parameters_from_txt('example.txt')
    In [9]: params
    Out[9]:
    {'float_list': [0.7, 0.1, 20.0],
     'int_list': [200, 12],
     'mixted_list': [542, 0.2, 0.1, 'adf1', '12w'],
     'str_list': ['thisisstring', '0..1']}
    '''
    with open(file_path, 'r') as f:
        params = dict()
        for line in f.read().splitlines():
            name, values_string = line.split(' ')
            values = []
            for s in values_string.split(','):
                if is_float(s):
                    if '.' in s or 'e' in s:
                        values.append(float(s))
                    else:
                        values.append(int(s))
                else:
                    values.append(s)
            params[name] = values
        return params
