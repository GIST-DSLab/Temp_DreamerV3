import pickle
import numpy as np
from tqdm import tqdm
import copy
import random

# train_list = ['train_179.pkl', 'train_241.pkl' ,'train_380.pkl', 'train_150.pkl']
# eval_list = ['eval_179.pkl', 'eval_241.pkl' ,'eval_380.pkl', 'eval_150.pkl']
train_list = ['new_train_A.pkl', 'new_train_B.pkl']
eval_list = ['new_eval_A.pkl', 'new_eval_B.pkl']

train_input_list = []
train_output_list = []
eval_input_list = []
eval_output_list = []

train_grid_list = []
train_answer_list = []
train_example_input_list = []
train_example_output_list = []

eval_grid_list = []
eval_answer_list = []
eval_example_input_list = []
eval_example_output_list = []

def rotate_CW(state):
    temp_state = np.array(copy.deepcopy(state['grid'] if 'grid' in state else state))
    return np.rot90(temp_state, 3)

# rotate_right function is a clockwise rotation about the given state.
def rotate_CCW(state):
    temp_state = np.array(copy.deepcopy(state['grid'] if 'grid' in state else state))
    return np.rot90(temp_state)

# vertical_flip function is a flip by y-axis about the given state.
def vertical_flip(state):
    temp_state = np.array(copy.deepcopy(state['grid'] if 'grid' in state else state))
    return np.flipud(temp_state)

# horizontal_flip function is a flip by x-axis about the given state.
def horizontal_flip(state): 
    temp_state = np.array(copy.deepcopy(state['grid'] if 'grid' in state else state))
    return np.fliplr(temp_state)

def make(mode=380):
    if mode == 380:
        train_input_list = [np.array(np.random.randint(0, 10, size=(3, 3)).tolist()) for i in range(1000)]
        train_output_list = [np.array(rotate_CCW(target)) for target in train_input_list]
        
        train_set = [str(x) for x in train_input_list]
        eval_input_list = []
        for _ in range(100):
            while True:
                temp = np.random.randint(0, 10, size=(3, 3)).tolist()
                if train_set == None or str(temp) not in train_set:
                    eval_input_list.append(np.array(temp))
                    break
        eval_output_list = [np.array(rotate_CCW(target)) for target in eval_input_list]
    
    elif mode == 'B':
        train_input_list = [np.array(np.random.randint(0, 10, size=(3, 3)).tolist()) for i in range(1000)]
        train_output_list = [np.array(horizontal_flip(target)) for target in train_input_list]
        
        train_set = [str(x) for x in train_input_list]
        eval_input_list = []
        for _ in range(100):
            while True:
                temp = np.random.randint(0, 10, size=(3, 3)).tolist()
                if train_set == None or str(temp) not in train_set:
                    eval_input_list.append(np.array(temp))
                    break
        eval_output_list = [np.array(horizontal_flip(target)) for target in eval_input_list]
        
    with open(f'dataset/train_{mode}.pkl', 'wb') as f:
        pickle.dump([train_input_list, train_output_list], f, pickle.HIGHEST_PROTOCOL)
    
    with open(f'dataset/eval_{mode}.pkl', 'wb') as f:
        pickle.dump([eval_input_list, eval_output_list], f, pickle.HIGHEST_PROTOCOL)

def make_new_dataset(mode='A'):
    if mode == 'A':
        with open(f"dataset/train_380.pkl", "rb") as f:
            train_dataset = pickle.load(f)
        with open(f"dataset/eval_380.pkl", "rb") as f:
            eval_dataset = pickle.load(f)

        count = 0
        exist_dataset = set(tuple(map(tuple, array)) for array in train_dataset[0] + eval_dataset[0])
        example_input_list = []

        while count < 3300:
            temp = np.random.randint(0, 10, size=(3, 3))
            temp_tuple = tuple(map(tuple, temp))
            
            if temp_tuple not in exist_dataset:
                example_input_list.append(temp)
                exist_dataset.add(temp_tuple)
                count += 1

        example_train_input = example_input_list[:3000]
        example_train_output = [np.array(rotate_CCW(target)) for target in example_train_input]

        example_eval_input = example_input_list[3000:]
        example_eval_output = [np.array(rotate_CCW(target)) for target in example_eval_input]

    elif mode == 'B':
        make('B')

        with open(f"dataset/train_B.pkl", "rb") as f:
            train_dataset = pickle.load(f)
        with open(f"dataset/eval_B.pkl", "rb") as f:
            eval_dataset = pickle.load(f)

        count = 0

        exist_dataset = set(tuple(map(tuple, array)) for array in train_dataset[0] + eval_dataset[0])
        example_input_list = []

        while count < 3300:
            temp = np.random.randint(0, 10, size=(3, 3))
            temp_tuple = tuple(map(tuple, temp))
            
            if temp_tuple not in exist_dataset:
                example_input_list.append(temp)
                exist_dataset.add(temp_tuple)
                count += 1
        

        example_train_input = example_input_list[:3000]
        example_train_output = [np.array(rotate_CCW(target)) for target in example_train_input]

        example_eval_input = example_input_list[3000:]
        example_eval_output = [np.array(rotate_CCW(target)) for target in example_eval_input]
    
    elif mode == 'AB':
        with open(f"dataset/train_179.pkl", "rb") as f:
            train_dataset = pickle.load(f)
        with open(f"dataset/eval_179.pkl", "rb") as f:
            eval_dataset = pickle.load(f)

        count = 0

        exist_dataset = set(tuple(map(tuple, array)) for array in train_dataset[0] + eval_dataset[0])
        example_input_list = []

        while count < 3300:
            temp = np.random.randint(0, 10, size=(3, 3))
            temp_tuple = tuple(map(tuple, temp))
            
            if temp_tuple not in exist_dataset:
                example_input_list.append(temp)
                exist_dataset.add(temp_tuple)
                count += 1
        

        example_train_input = example_input_list[:3000]
        example_train_output = [np.array(horizontal_flip(rotate_CCW(target))) for target in example_train_input]

        example_eval_input = example_input_list[3000:]
        example_eval_output = [np.array(horizontal_flip(rotate_CCW(target))) for target in example_eval_input]
    
    
    new_train_dataset = {
            'example': [example_train_input, example_train_output],
            'grid': train_dataset[0],
            'answer': train_dataset[1]
        }

    new_eval_dataset = {
        'example': [example_eval_input, example_eval_output],
        'grid': train_dataset[0],
        'answer': train_dataset[1]
    }
    
    with open(f'dataset/new_train_{mode}.pkl', 'wb') as f:
        pickle.dump(new_train_dataset, f, pickle.HIGHEST_PROTOCOL)
    
    with open(f'dataset/new_eval_{mode}.pkl', 'wb') as f:
        pickle.dump(new_eval_dataset, f, pickle.HIGHEST_PROTOCOL)


def merge(mode='', num=1000):
    if mode == 'new':
        for target_train in train_list:
            temp_list = []
            with open(f'dataset/{target_train}', 'rb') as f:
                temp_list = pickle.load(f)
            train_grid_list.append(temp_list['grid'])
            train_answer_list.append(temp_list['answer'])
            train_example_input_list.append(temp_list['example'][0])
            train_example_output_list.append(temp_list['example'][1])

        for i in tqdm(range(len(train_list)-1)):
            for j in range(i+1,len(train_list)):
                print(f'i: {train_list[i]}, j: {train_list[j]}')
                merge_train_grid_list = train_grid_list[i][:num] + train_grid_list[j][:num]
                merge_train_answer_list = train_answer_list[i][:num] + train_answer_list[j][:num]
                merge_train_example_input_list = train_example_input_list[i][:num*3] + train_example_input_list[j][:num*3]
                merge_train_example_output_list =  train_example_output_list[i][:num*3] + train_example_output_list[j][:num*3]

                indices = list(range(len(merge_train_grid_list)))
                random.shuffle(indices)
                merge_train_grid_list = [merge_train_grid_list[i] for i in indices]
                merge_train_answer_list = [merge_train_answer_list[i] for i in indices]
                merge_train_example_input_list = [merge_train_example_input_list[i*3:i*3+3] for i in indices]
                merge_train_example_output_list =  [merge_train_example_output_list[i*3:i*3+3] for i in indices]
                merge_train = {
                    'example': [merge_train_example_input_list, merge_train_example_output_list],
                    'grid': merge_train_grid_list,
                    'answer': merge_train_answer_list
                }

                with open(f"dataset/new_train_{train_list[i].split('.')[0].split('_')[-1]}-{train_list[j].split('.')[0].split('_')[-1]}_{num*2}.pkl", "wb") as f:
                    pickle.dump(merge_train, f, pickle.HIGHEST_PROTOCOL)
    else:
        for target_train in train_list:
            temp_list = []
            with open(f'dataset/{target_train}', 'rb') as f:
                temp_list = pickle.load(f)
            train_input_list.append(temp_list[0])
            train_output_list.append(temp_list[1])

        for i in tqdm(range(len(train_list)-1)):
            for j in range(i+1,len(train_list)):
                print(f'i: {train_list[i]}, j: {train_list[j]}')
                merge_train_input_list = train_input_list[i][:num] + train_input_list[j][:num]
                merge_train_output_list = train_output_list[i][:num] + train_output_list[j][:num]

                indices = list(range(len(merge_train_input_list)))
                random.shuffle(indices)
                merge_train_input_list = [merge_train_input_list[i] for i in indices]
                merge_train_output_list = [merge_train_output_list[i] for i in indices]
                merge_train = [merge_train_input_list, merge_train_output_list]

                with open(f"dataset/train_{train_list[i].split('.')[0].split('_')[1]}-{train_list[j].split('.')[0].split('_')[1]}_{num*2}.pkl", "wb") as f:
                    pickle.dump(merge_train, f, pickle.HIGHEST_PROTOCOL)

def merge_check(mode='', num=1000):
    if mode == 'new':
        train_A_B_list = []
        total_list = [train_A_B_list] 
        count = 0
        for i in tqdm(range(len(train_list)-1)):
            for j in range(i+1,len(train_list)):
                name = f"new_train_{train_list[i].split('.')[0].split('_')[-1]}-{train_list[j].split('.')[0].split('_')[-1]}_{num*2}.pkl"
                with open(f"dataset/{name}", "rb") as f:
                    total_list[count] = pickle.load(f)
                    print(name)
                
                for target, result in zip(total_list[count]['grid'], total_list[count]['answer']):
                    if 'A' in name and 'B' in name:
                        predict1 = rotate_CCW(target)
                        predict2 = horizontal_flip(target)
                        
                    if (str(result) == str(np.array(predict1))) or (str(result) == str(np.array(predict2))):
                        pass
                    else:
                        print(f'{count}: 불통과')
        
                count += 1
    else:
        train_179_150_list = []
        train_179_241_list = []
        train_179_380_list = []
        train_241_150_list = []
        train_241_380_list = []
        train_380_150_list = []
        total_list = [train_179_241_list, train_179_380_list,  train_241_380_list, train_179_150_list, train_241_150_list, train_380_150_list] 
        count = 0
        for i in tqdm(range(len(train_list)-1)):
            for j in range(i+1,len(train_list)):
                name = f"train_{train_list[i].split('.')[0].split('_')[-1]}-{train_list[j].split('.')[0].split('_')[-1]}_2000.pkl"
                with open(f"dataset/{name}", "rb") as f:
                    total_list[count] = pickle.load(f)
                    print(name)
                
                for target, result in zip(total_list[count][0], total_list[count][1]):
                    if '179' in name and '241' in name:
                        predict1 = horizontal_flip(rotate_CCW(target))
                        predict2 = horizontal_flip(rotate_CCW(target))
                    elif '179' in name and '380' in name:
                        predict1 = horizontal_flip(rotate_CCW(target))
                        predict2 = rotate_CCW(target)
                    elif '179' in name and '150' in name:
                        predict1 = horizontal_flip(rotate_CCW(target))
                        predict2 = horizontal_flip(target)
                    elif '241' in name and '380' in name: 
                        predict1 = horizontal_flip(rotate_CCW(target))
                        predict2 = rotate_CCW(target) # rotate_CCW(target)
                    elif '241' in name and '150' in name:
                        predict1 = horizontal_flip(rotate_CCW(target))
                        predict2 = horizontal_flip(target)
                    elif '150' in name and '380' in name:
                        predict1 = horizontal_flip(target)
                        predict2 = rotate_CCW(target)
                    if (str(result) == str(np.array(predict1))) or (str(result) == str(np.array(predict2))):
                        pass
                    else:
                        print(f'{count}: 불통과')
        
                count += 1
def single_check():
    train_179_list = []
    train_241_list = []
    train_380_list = []
    train_150_list = []

    eval_179_list = []
    eval_241_list = []
    eval_380_list = []
    eval_150_list = []

    total_train_list = [train_179_list, train_241_list,  train_380_list, train_150_list] 
    total_eval_list = [eval_179_list, eval_241_list,  eval_380_list, eval_150_list] 
    count = 0
    for i in tqdm(range(len(train_list))):
        name = train_list[i]
        with open(f"dataset/{name}", "rb") as f:
            total_train_list[count] = pickle.load(f)
            print(name)
        
        for target, result in zip(total_train_list[count][0], total_train_list[count][1]):
            if '179' in name:
                predict1 = horizontal_flip(rotate_CCW(target))
            elif '380' in name:
                predict2 = rotate_CCW(target)
            elif '150' in name:
                predict2 = horizontal_flip(target)
            elif '241' in name: 
                predict1 = horizontal_flip(rotate_CCW(target))
            if (str(result) == str(np.array(predict1))) or (str(result) == str(np.array(predict2))):
                pass
            else:
                print(f'{count}: 불통과')
        count += 1
    
    count = 0
    for i in tqdm(range(len(eval_list))):
        name = eval_list[i]
        with open(f"dataset/{name}", "rb") as f:
            total_eval_list[count] = pickle.load(f)
            print(name)
        
        for target, result in zip(total_eval_list[count][0], total_eval_list[count][1]):
            if '179' in name:
                predict1 = horizontal_flip(rotate_CCW(target))
            elif '380' in name:
                predict2 = rotate_CCW(target)
            elif '150' in name:
                predict2 = horizontal_flip(target)
            elif '241' in name: 
                predict1 = horizontal_flip(rotate_CCW(target))
            if (str(result) == str(np.array(predict1))) or (str(result) == str(np.array(predict2))):
                pass
            else:
                print(f'{count}: 불통과')
        count += 1




if __name__ == "__main__":
    # make_new_dataset('AB')
    merge(mode='new', num=500)
    merge_check(mode='new', num=500)
        


