import numpy as np


def load_data(data_path, row, column) -> list:
    """turn input .txt file to a list of np array"""
    zero_symbol = " "
    one_symbol = "1"
    return_list = list()
    count = 0

    with open(data_path, 'r') as fp:
        tmp_list = list()
        for line in fp.readlines():
            # each data is separate by a blank line
            if count == row:
                count = 0
                return_list.append(np.array(tmp_list))
                tmp_list.clear()
                continue
            for i in range(column):
                if line[i] == zero_symbol:
                    tmp_list.append(-1)
                elif line[i] == one_symbol:
                    tmp_list.append(1)
            count += 1
        # 最後一筆訓練資料結束時沒有空一行
        return_list.append(np.array(tmp_list))
    return return_list


# todo try wider input
def load_data_1(data_path, row, column) -> list:
    """turn input .txt file to a list of np array"""
    zero_symbol = " "
    one_symbol = "1"
    return_list = list()
    count = 0

    with open(data_path, 'r') as fp:
        tmp_list = list()
        for line in fp.readlines():
            # each data is separate by a blank line
            if count == row:
                count = 0

                # add two row to improve rewind
                if row == 10:
                    for i in range(1):
                        for j in range(10 * column):
                            tmp_list.append(tmp_list[j])

                return_list.append(np.array(tmp_list))
                tmp_list.clear()
                continue
            for i in range(column):
                if line[i] == zero_symbol:
                    tmp_list.append(-1)
                elif line[i] == one_symbol:
                    tmp_list.append(1)
            count += 1
        # 最後一筆訓練資料結束時沒有空一行

        # add two row to improve rewind
        if row == 10:
            for i in range(1):
                for j in range(10 * column):
                    tmp_list.append(tmp_list[j])
        return_list.append(np.array(tmp_list))
    return return_list


def load_data_2(data_path, row, column) -> list:
    """turn input .txt file to a list of np array"""
    zero_symbol = " "
    one_symbol = "1"
    return_list = list()
    count = 0
    with open(data_path, 'r') as fp:
        tmp_list = list()
        for line in fp.readlines():
            # each data is separate by a blank line
            if count == row:
                count = 0
                return_list.append(np.array(tmp_list))
                tmp_list.clear()
                continue
            for i in range(column):
                if line[i] == zero_symbol:
                    tmp_list.append(-1)
                    tmp_list.append(-1)
                elif line[i] == one_symbol:
                    tmp_list.append(1)
                    tmp_list.append(1)
            count += 1
        # 最後一筆訓練資料結束時沒有空一行
        # add two row to improve rewind
        return_list.append(np.array(tmp_list))
    return return_list


def get_model(data_path="./data/Basic_Training.txt", row=12, column=9):
    data_list = load_data(data_path, row, column)
    p = data_list[0].shape[0]
    # print(p)
    n = len(data_list)
    w = np.zeros((p, p), dtype=float)
    # print(data_list[0])
    for i in range(n):
        w += np.matmul(np.expand_dims(data_list[i], axis=1), np.transpose(np.expand_dims(data_list[i], axis=1)))
    w = (w - n * np.eye(p, dtype=float)) / p
    theta = np.sum(w, axis=1)

    return w, theta


def predict(w, theta, input_vec) -> np.ndarray:
    """"""
    old_input = np.copy(input_vec)
    # next_input = list()
    while True:
        for i in range(input_vec.size):
            temp = np.matmul(w, input_vec)[i]
            if temp > 0:
                # next_input.append(1)
                input_vec[i] = 1
            elif temp < 0:
                # next_input.append(-1)
                input_vec[i] = -1
            # else:
                # next_input.append(input_vec[i])
        # input_vec = np.array(next_input)
        # next_input.clear()
        if (old_input == input_vec).all():
            break
        else:
            old_input = np.copy(input_vec)

    return input_vec


def predict_2(w, theta, input_vec) -> np.ndarray:
    """non syn, more neuron, no theta"""
    old_input = np.copy(input_vec)
    next_input = list()
    while True:
        for i in range(input_vec.size):
            temp = np.matmul(w, input_vec)[i]
            if temp > 0:
                next_input.append(1)
                # input_vec[i] = 1
            elif temp < 0:
                next_input.append(-1)
                # input_vec[i] = -1
            else:
                next_input.append(input_vec[i])
        input_vec = np.array(next_input)
        next_input.clear()
        if (old_input == input_vec).all():
            break
        else:
            old_input = np.copy(input_vec)

    return input_vec


def predict_3(w, theta, input_vec) -> np.ndarray:
    """non syn, more neuron, with delta -> trash"""
    old_input = np.copy(input_vec)
    next_input = list()
    while True:
        for i in range(input_vec.size):
            temp = np.matmul(w, input_vec)[i] - theta[i]
            if temp > 0:
                next_input.append(1)
                # input_vec[i] = 1
            elif temp < 0:
                next_input.append(-1)
                # input_vec[i] = -1
            else:
                next_input.append(input_vec[i])
        input_vec = np.array(next_input)
        next_input.clear()
        if (old_input == input_vec).all():
            break
        else:
            old_input = np.copy(input_vec)

    return input_vec


if __name__ == "__main__":
    test_input = "./data/for_test.txt"
    get_model(test_input, 1, 3)

    input_vectors = load_data(test_input, 1, 3)
    print(predict(np.array([1, 1, -1]), "./model/model.npy"))

    # with open("./model/model.npy", "rb") as f:
    #     w = np.load(f)
    #     print(w)
    #     theta = np.load(f)
    #     print(theta)

