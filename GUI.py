from tkinter import Tk, Canvas, Button, PhotoImage
from tkinter import filedialog as fd
import Model
import numpy as np
import random
# -------------- global variable ----------------
my_config = {
    'train_path': '',
    'test_path': ''
}

row = 0
column = 0
now_showing_index = 0
input_vectors = list()
output_vectors = list()
test_case_amount = 0
# ----------------- function -------------------


def add_noise():
    """read testing data and add noise"""
    global row, column, input_vectors, output_vectors, test_case_amount
    with open(my_config['test_path']) as f:
        if len(f.readline()) == 10:
            row = 12
            column = 9
        else:
            row = 10
            column = 10
    zero_symbol = " "
    one_symbol = "1"
    return_list = list()
    count = 0
    with open(my_config['test_path'], 'r') as fp:
        for line in fp.readlines():
            tmp_str = ""
            # each data is separate by a blank line
            if count == row:
                count = 0
                return_list.append("")
                continue
            for i in range(column):
                if line[i] == zero_symbol:
                    if random.random() >= 0.1:
                        tmp_str += " "
                    else:
                        tmp_str += "1"
                    # tmp_list.append(-1)
                elif line[i] == one_symbol:
                    if random.random() >= 0.1:
                        tmp_str += "1"
                    else:
                        tmp_str += " "
                    # tmp_list.append(1)
            return_list.append(tmp_str)
            count += 1
        # 最後一筆訓練資料結束時沒有空一行
    # print(return_list)
    with open("./data/noised_testing_data.txt", 'w') as f:
        for line in return_list:
            f.write(line + "\n")


def train():
    global row, column, input_vectors, output_vectors, test_case_amount
    output_vectors.clear()
    with open(my_config['train_path']) as f:
        if len(f.readline()) == 10:
            row = 12
            column = 9
        else:
            row = 10
            column = 10
    w, theta = Model.get_model(my_config['train_path'], row, column)
    # input_vectors used to show ground truth
    input_vectors = Model.load_data(my_config['train_path'], row, column)
    testing_data = Model.load_data(my_config['test_path'], row, column)
    # testing_data = Model.load_data_2(my_config['test_path'], row, column)
    for i in range(len(testing_data)):
        output_vectors.append(Model.predict(w, theta, np.copy(testing_data[i])))
    test_case_amount = len(output_vectors)
    set_window(0)


def turn_np_to_output_text(tmp_vector):
    global row, column
    tmp = ""
    for i in range(row):
        for j in range(column):
            if tmp_vector[i*column + j] == 1:
                tmp += "1"
            else:
                tmp += " "
        tmp += "\n"
    return tmp


def turn_np_to_output_text_2(tmp_vector):
    """input 向量變兩倍寬"""
    global row, column
    tmp = ""
    for i in range(row):
        for j in range(column):
            if tmp_vector[i*column + j] == 1:
                tmp += "1"
            else:
                tmp += " "
            j += 1
        tmp += "\n"
    return tmp


def set_window(index):
    global input_vectors, output_vectors
    canvas.itemconfig(input_window, text=turn_np_to_output_text(input_vectors[index]))
    canvas.itemconfig(output_window, text=turn_np_to_output_text(output_vectors[index]))


def show_previous():
    global now_showing_index
    if now_showing_index >= 1:
        now_showing_index -= 1
        set_window(now_showing_index)


def show_next():
    global now_showing_index, test_case_amount
    print(now_showing_index, test_case_amount)
    if now_showing_index < test_case_amount - 1:
        now_showing_index += 1
        set_window(now_showing_index)


def select_train():
    my_config['train_path'] = select_file()


def select_test():
    my_config['test_path'] = select_file()


def select_file():
    filetypes = (
        ('text files', '*.txt'),
    )
    file_path = fd.askopenfilename(
        title='Open a file',
        initialdir='./data',
        filetypes=filetypes)
    return file_path


window = Tk()
window.geometry("800x600")
window.configure(bg = "#FFFFFF")
window.title("hopfield network")

canvas = Canvas(
    window,
    bg = "#FFFFFF",
    height = 600,
    width = 800,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge"
)

canvas.place(x = 0, y = 0)
canvas.create_rectangle(
    0.0,
    0.0,
    800.0,
    120.0,
    fill="#000000",
    outline="")

# show input
output_window = canvas.create_text(
    500.0,
    175.0,
    anchor="nw",
    text="",
    fill="#000000",
    font=("Consolas", 18 * -1)
)

# show result
input_window = canvas.create_text(
    180.0,
    175.0,
    anchor="nw",
    text="",
    fill="#000000",
    font=("Consolas", 18 * -1)
)

# start run(show first result)
button_image_1 = PhotoImage(
    file="assets/button_1.png")
button_1 = Button(
    image=button_image_1,
    borderwidth=0,
    highlightthickness=0,
    command=train,
    relief="flat"
)
button_1.place(
    x=428.0,
    y=35.0,
    width=130.0,
    height=50.0
)

# choose training data
btnTrain_image = PhotoImage(
    file="assets/button_5.png")
btnTrain = Button(
    image=btnTrain_image,
    borderwidth=0,
    highlightthickness=0,
    command=select_train,
    relief="flat"
)
btnTrain.place(
    x=56.0,
    y=35.0,
    width=130.0,
    height=50.0
)

btnNoise = Button(
    # image=button_image_1,
    borderwidth=0,
    highlightthickness=0,
    command=add_noise,
    relief="flat",
    text="Noise",
    background="white"
)

btnNoise.place(
    x=614.0,
    y=35.0,
    width=130.0,
    height=50.0
)


# choose testing data
button_image_2 = PhotoImage(
    file="assets/button_2.png")
button_2 = Button(
    image=button_image_2,
    borderwidth=0,
    highlightthickness=0,
    command=select_test,
    relief="flat"
)
button_2.place(
    x=242.0,
    y=35.0,
    width=130.0,
    height=50.0
)

# show next result
button_image_3 = PhotoImage(
    file="assets/button_3.png")
button_3 = Button(
    image=button_image_3,
    borderwidth=0,
    highlightthickness=0,
    command=show_next,
    relief="flat"
)
button_3.place(
    x=480.0,
    y=509.0,
    width=130.0,
    height=50.0
)

# previous one
button_image_4 = PhotoImage(
    file="assets/button_4.png")
button_4 = Button(
    image=button_image_4,
    borderwidth=0,
    highlightthickness=0,
    command=show_previous,
    relief="flat"
)
button_4.place(
    x=190.0,
    y=509.0,
    width=130.0,
    height=50.0
)


window.resizable(False, False)
window.mainloop()
