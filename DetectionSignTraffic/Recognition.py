import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy
import cv2
#load the trained model to classify sign
from keras.models import load_model
model = load_model('./model/traffic_classifier02.h5')
#dictionary to label all traffic signs class.
classes = { 1:'20',
            2:'30', 
            3:'50', 
            4:'60', 
            5:'70', 
            6:'80', 
            7:'Unknown',
            8:'100', 
            9:'Unknown', 
            10:'Unknown', 
            11:'Unknown', 
            12:'Unknown', 
            13:'Unknown', 
            14:'Unknown', 
            15:'stop', 
            16:'Unknown', 
            17:'Unknown', 
            18:'Unknown', 
            19:'Unknown', 
            20:'Unknown', 
            21:'Unknown', 
            22:'Unknown', 
            23:'Unknown', 
            24:'Unknown', 
            25:'Unknown', 
            26:'Unknown', 
            27:'Unknown', 
            28:'Unknown', 
            29:'Unknown', 
            30:'Unknown', 
            31:'Unknown',
            32:'Unknown', 
            33:'Unknown', 
            34:'turn_right', 
            35:'turn_left', 
            36:'ahead_only', 
            37:'Unknown', 
            38:'Unknown', 
            39:'Unknown', 
            40:'Unknown', 
            41:'Unknown', 
            42:'Unknown', 
            43:'Unknown',
        }
#initialise GUI
top=tk.Tk()
top.geometry('800x600')
top.title('Traffic sign classification')
top.configure(background='#CDCDCD')
label=Label(top,background='#CDCDCD', font=('arial',15,'bold'))
sign_image = Label(top)

def detection(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = image.resize((30,30))
    image = numpy.expand_dims(image, axis=0)
    image = numpy.array(image)
    pred = model.predict_classes([image])[0]
    sign = classes[pred+1]
    return sign

def classify(file_path):
    global label_packed
    image = Image.open(file_path)
    print("IMAGE:",image)
    image = image.resize((30,30))
    print("IMAGE.resize:",image)
    image = numpy.expand_dims(image, axis=0)
    print("IMAGE.expand:",image)
    image = numpy.array(image)
    print("IMAGE.array:",image)
    pred = model.predict_classes([image])[0]
    sign = classes[pred+1]
    label.configure(foreground='#011638', text=sign) 
def show_classify_button(file_path):
    classify_b=Button(top,text="Classify Image",command=lambda: classify(file_path),padx=10,pady=5)
    classify_b.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
    classify_b.place(relx=0.79,rely=0.46)
def upload_image():
    try:
        file_path=filedialog.askopenfilename()
        uploaded=Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25),(top.winfo_height()/2.25)))
        im=ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image=im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass
# upload=Button(top,text="Upload an image",command=upload_image,padx=10,pady=5)
# upload.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
# upload.pack(side=BOTTOM,pady=50)
# sign_image.pack(side=BOTTOM,expand=True)
# label.pack(side=BOTTOM,expand=True)
# heading = Label(top, text="Know Your Traffic Sign",pady=20, font=('arial',20,'bold'))
# heading.configure(background='#CDCDCD',foreground='#364156')
# heading.pack()
# top.mainloop()