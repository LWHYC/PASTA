# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import configparser
def is_int(valid_str):
    start_digit = 0
    if(valid_str[0] =='-'):
        start_digit = 1
    flag = True
    for i in range(start_digit, len(valid_str)):
        if(str(valid_str[i]) < '0' or str(valid_str[i]) > '9'):
            flag = False
            break
    return flag

def is_float(valid_str):
    flag = False
    if('.' in valid_str and len(valid_str.split('.'))==2):
        if(is_int(valid_str.split('.')[0]) and is_int(valid_str.split('.')[1])):
            flag = True
        else:
            flag = False
    elif('e' in valid_str and len(valid_str.split('e'))==2):
        if(is_int(valid_str.split('e')[0]) and is_int(valid_str.split('e')[1])):
            flag = True
        else:
            flag = False       
    else:
        flag = False
    return flag 

def is_bool(var_str):
    if( var_str=='True' or var_str == 'true' or var_str =='False' or var_str=='false'):
        return True
    else:
        return False
    
def parse_bool(var_str):
    if(var_str=='True' or var_str == 'true' ):
        return True
    else:
        return False
     
def is_list(valid_str):
    if(valid_str[0] == '[' and valid_str[-1] == ']'):
        return True
    else:
        return False
    
def parse_list(valid_str):
    sub_str = valid_str[1:-1]
    splits = sub_str.split(',')
    output = []
    for item in splits:
        item = item.strip()
        if(is_int(item)):
            output.append(int(item))
        elif(is_float(item)):
            output.append(float(item))
        elif(is_bool(item)):
            output.append(parse_bool(item))
        else:
            output.append(item)
    return output

def parse_value_from_string(valid_str):
#     valid_str = valid_str.encode('ascii','ignore')
    if(is_int(valid_str)):
        val = int(valid_str)
    elif(is_float(valid_str)):
        val = float(valid_str)
    elif(is_list(valid_str)):
        val = parse_list(valid_str)
    elif(is_bool(valid_str)):
        val = parse_bool(valid_str)
    else:
        val = valid_str
    return val

def parse_config(filename, logger=False):
    config = configparser.ConfigParser()
    config.read(filename)
    output = {}
    for section in config.sections():   # Return a list of section names
        output[section] = {}
        for key in config[section]:
            valid_str = str(config[section][key])
            if(len(valid_str)>0):
                val = parse_value_from_string(valid_str) 
            else:
                val = None
            if logger:
                logger.print(section, key, valid_str, val)
            else:
                print(section, key, valid_str, val)
            output[section][key] = val
    return output
            
if __name__ == "__main__":
    print(is_int('555'))
    print(is_float('555.10'))
    a='[1 ,2 ,3 ]'
    print(a)
    print(parse_list(a))
    
    