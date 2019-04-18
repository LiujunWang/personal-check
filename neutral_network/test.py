import numpy as np
import os 


def dense_to_one_hot(label_list, num_examples):
    
    length = len(label_list)
    result_label = np.zeros((length, num_examples))
    for x in range(length):
        temp = label_list[x]
        result_label[x][temp] = 1

    return np.array(result_label)

 
      
def file_name(file_dir):

    path_name = []
    for root, dirs, files in os.walk(file_dir):
        path_name.append(files[0])

    return path_name

if __name__ == '__main__':
    
    label_list = [0, 1, 2, 3, 4]

    print(label_list.index(max(label_list)))
    #temp = dense_to_one_hot(label_list, 900)
    #print(temp.shape)
    '''
    path = "E:\\WorkPlace\\Software\\data\\person_reid\\test_query"

    path_name = file_name(path)
    print(path_name[0])
    '''