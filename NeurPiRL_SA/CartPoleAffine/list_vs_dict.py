import time
SIZE = 1000



def compare(SIZE):

    # Initialize
    my_list = list(range(SIZE))
    # DICT
    my_dict = {}
    for i in range(SIZE):
        my_dict[str(i)] = i

    # Compare
    t0 = time.time()
    for i in range(SIZE):
        q = my_list.pop()
    t_list = time.time() - t0


    t0 = time.time()
    for i in range(SIZE):
        q = my_dict[str(i)]
    t_dict = time.time() - t0


    print("Size: ", SIZE,  "\nList: ", t_list, "\nDict: ", t_dict, '\n', t_dict/t_list, '\n')


compare(10)
compare(20)
compare(100)
compare(1000)

