import sys
import yaml
import os
import argparse
import json

################################################################################
################################################################################
def u_fileList2array(file_name):
    print('Loading data from: ' + file_name)
    F = open(file_name,'r') 
    lst = []
    for item in F:
        item = item.replace('\\', '/').rstrip()
        lst.append(item)
    F.close()
    return lst

################################################################################
################################################################################
#read yml files in opencv format, does not suport levels 
def u_readYAMLFile(fileName):
    ret = {}
    skip_lines=1    # Skip the first line which says "%YAML:1.0". Or replace it with "%YAML 1.0"
    with open(fileName) as fin:
        for i in range(skip_lines):
            fin.readline()
        yamlFileOut = fin.read()
        #myRe = re.compile(r":([^ ])")   # Add space after ":", if it doesn't exist. Python yaml requirement
        #yamlFileOut = myRe.sub(r': \1', yamlFileOut)
        ret = yaml.load(yamlFileOut)
    return ret

################################################################################
################################################################################
def u_save2File(file_name, data):
    print('Saving data in: ' + file_name)
    F = open(file_name,'w') 
    F.write(data)
    F.close()

################################################################################
################################################################################
def u_saveList2File(file_name, data):
    print('Saving data in: ' + file_name)
    F = open(file_name,'w') 
    for item in data:
        item = item.strip()
        F.write(item + '\n')
    F.close()
################################################################################
################################################################################
def u_fileNumberList2array(file_name):
    print('Loading data from: ' + file_name)
    F = open(file_name,'r') 
    lst = []
    for item in F:
        if len(item) > 0:
            lst.append(float(item))
    F.close()
    return lst
################################################################################
################################################################################
def u_saveArray2File(file_name, data):
    print('Saving data in: ' + file_name)
    F = open(file_name,'w') 
    for item in data:
        F.write(str(item))
        F.write('\n')
    F.close()
################################################################################
################################################################################
def u_saveArrayTuple2File(file_name, data):
    print('Saving data in: ' + file_name)
    F = open(file_name,'w') 
    for item in data:
        for tup in item:
            F.write(str(tup))
            F.write(' ')
        F.write('\n')
    F.close()
################################################################################
################################################################################
'''
Save dict into file, recommendably [.json]
'''
def u_saveDict2File(file_name, data):
    print ('Saving data in: ', file_name)
    with open(file_name, 'w') as outfile:  
        json.dump(data, outfile)

################################################################################
################################################################################
def u_mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

################################################################################
################################################################################
def u_listFileAll(directory, token):
    list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(token):
                 list.append(os.path.join(root, file))
    return list

################################################################################
################################################################################
def u_getPath(file):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('inputpath', nargs='?', 
                        help='The input path. Default = conf.json')
    args = parser.parse_args()
    return args.inputpath if args.inputpath is not None else file

################################################################################
################################################################################
def u_loadFileManager(directive, token = ''):
    print(directive)
    if os.path.isfile(directive):
        file_list = []
        file = open(directive)
        for item in file:
            item  = item.strip()
            file_list.append(item)
    else:
        file_list   = u_listFileAll(directive, token)
    return file_list

################################################################################
################################################################################
def u_progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush() 
    
################################################################################
################################################################################
def u_init_list_of_objects(size):
    list_of_objects = list()
    for i in range(0,size):
        list_of_objects.append( list() ) #different object reference each time
    return list_of_objects

################################################################################
################################################################################
def u_replaceStrList(str_list, token1, token2):
    for i in range(len(str_list)):
        str_list = str_list.replace(token1, token2)
    return str_list