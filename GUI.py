# the GUI layout code was generated by the Tkinter Designer by Parth Jadhav
# https://github.com/ParthJadhav/Tkinter-Designer

from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage
from tkinter import filedialog as fd
import Model
import numpy as np
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
    Model.get_model(my_config['train_path'], row, column)
    input_vectors = Model.load_data(my_config['test_path'], row, column)
    for i in range(len(input_vectors)):
        output_vectors.append(Model.predict(np.copy(input_vectors[i])))
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
    font=("Roboto", 18 * -1)
)

# show result
input_window = canvas.create_text(
    180.0,
    175.0,
    anchor="nw",
    text="",
    fill="#000000",
    font=("Roboto", 18 * -1)
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
    x=545.0,
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
    x=335.0,
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

# choose training data
button_image_5 = PhotoImage(
    file="assets/button_5.png")
button_5 = Button(
    image=button_image_5,
    borderwidth=0,
    highlightthickness=0,
    command=select_train,
    relief="flat"
)
button_5.place(
    x=125.0,
    y=35.0,
    width=130.0,
    height=50.0
)
window.resizable(False, False)
window.mainloop()
