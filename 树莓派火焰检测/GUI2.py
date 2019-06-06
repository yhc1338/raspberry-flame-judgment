# coding=utf-8

# https://blog.csdn.net/yeshankuangrenaaaaa/article/details/85703572
# https://www.cnblogs.com/pertor/p/9664344.html
# https://blog.csdn.net/a1_a1_a/article/details/79981788
# http://www.cnblogs.com/zhangpengshou/p/3626137.html
# https://www.cnblogs.com/hhh5460/p/6664021.html?utm_source=itdadao&utm_medium=referral
# https://blog.csdn.net/u013468614/article/details/58689735
# https://blog.csdn.net/qq_27825451/article/details/81565760
# https://blog.csdn.net/Poul_henry/article/details/82590392
# http://www.mamicode.com/info-detail-2323554.html
# https://bbs.csdn.net/topics/392447100
# https://www.jb51.net/article/133163.htm
import tensorflow as tf
import Tkinter as tk
import tkFont  # https://blog.csdn.net/u013180459/article/details/82625908
import cv2
from PIL import Image, ImageTk  # sudo apt-get install python-imaging-tk
import numpy as np
import tkMessageBox as messagebox
import time
import DAC8532
import RPi.GPIO as GPIO
import serial

'''松耦合'''


# 弹窗
class MyDialog(tk.Toplevel, object):
    def __init__(self):
        super(MyDialog, self).__init__()
        self.title('参数设置')

        # 弹窗界面
        self.setup_UI()

    def setup_UI(self):
        # 第一行（两列）
        row1 = tk.Frame(self)
        row1.pack(fill="x")
        tk.Label(row1, text='亮延时：', width=8).pack(padx=10, pady=10, side=tk.LEFT)
        self.delayH = tk.IntVar()
        tk.Entry(row1, textvariable=self.delayH, width=15).pack(side=tk.LEFT)
        tk.Label(row1, text='秒', width=8).pack(side=tk.LEFT)
        # 第二行
        row2 = tk.Frame(self)
        row2.pack(fill="x", ipadx=1, ipady=1)
        tk.Label(row2, text='灭延时：', width=8).pack(padx=10, pady=10, side=tk.LEFT)
        self.delayL = tk.IntVar()
        tk.Entry(row2, textvariable=self.delayL, width=15).pack(side=tk.LEFT)
        tk.Label(row2, text='秒', width=8).pack(side=tk.LEFT)

        # 第三行
        row3 = tk.Frame(self)
        row3.pack(fill="x")
        tk.Button(row3, text=" 取消 ", command=self.cancel).pack(side=tk.RIGHT)
        tk.Button(row3, text=" 确定 ", command=self.ok).pack(side=tk.RIGHT)

    def ok(self):
        self.userinfo = [self.delayH.get(), self.delayL.get()]  # 设置数据
        self.destroy()  # 销毁窗口

    def cancel(self):
        self.userinfo = None  # 空
        self.destroy()


# 弹窗
class Myquit(tk.Toplevel, object):
    def __init__(self):
        super(Myquit, self).__init__()
        self.password = tk.StringVar()
        self.title('退出')

        # 弹窗界面
        self.setup_UI()

    def setup_UI(self):
        # 第一行（两列）
        row1 = tk.Frame(self)
        row1.pack(fill="x")
        tk.Label(row1, text='输入密码', width=8).pack(padx=10, pady=10, side=tk.LEFT)
        tk.Entry(row1, textvariable=self.password, show='*', width=15).pack(side=tk.LEFT)

        # 第二行
        row2 = tk.Frame(self)
        row2.pack(fill="x")
        tk.Button(row2, text=" 取消 ", command=self.cancel).pack(side=tk.RIGHT)
        tk.Button(row2, text=" 退出 ", command=self.ok).pack(side=tk.RIGHT)

    def ok(self):
        if self.password.get() == pw:
            ser.close()
            GPIO.cleanup()
            file.close()
            camera.release()
            cv2.destroyAllWindows()
            self.destroy()
            self.quit()
        else:
            messagebox.showwarning('警告', '密码错误')

    def cancel(self):
        self.destroy()


# 主窗
class MyApp(tk.Tk, object):

    def __init__(self):
        super(MyApp, self).__init__()
        # self.pack() # 若继承 tk.Frame ，此句必须有！
        self.resizable(width=False, height=False)
        self.protocol("WM_DELETE_WINDOW", self.callback)
        self.title('火焰检测')
        # 程序参数/数据
        self.delayH = 1
        self.delayL = 0
        self.britemp = [0 for _ in range(52)]
        self.R = tk.StringVar()  # 着火状态
        self.L = tk.StringVar()  # 亮度
        self.fft = tk.StringVar()  # 亮度
        self.size = tk.StringVar()
        self.t = [0]
        self.savedata = []
        self.savetime = [" "]
        # 程序界面
        self.setupUI()

    def setupUI(self):
        # 显示图像
        ft = tkFont.Font(family='Fixdsys', size=10, weight=tkFont.BOLD)
        self.panel = tk.Label(self)  # initialize image panel
        self.panel.pack(padx=10, pady=10, side=tk.LEFT)
        self.config(cursor="arrow")
        self.canvas = tk.Canvas()  # 创建一块显示图形的画布
        self.video_loop()

        # 第一行
        row1 = tk.Frame(self)
        row1.pack(fill="x")
        tk.Button(row1, text='   退出   ', command=self.take_snapshot).pack(padx=10, pady=10, side=tk.RIGHT)
        tk.Button(row1, text=" 参数设置 ", command=self.setup_config).pack(side=tk.RIGHT)

        # 第二行
        row2 = tk.Frame(self)
        row2.pack(fill="x")
        tk.Label(row2, text='着火状态：', width=8).pack(padx=10, pady=10, side=tk.LEFT)
        tk.Label(row2, textvariable=str(self.R), width=10).pack(side=tk.LEFT)
        # 第三行
        row3 = tk.Frame(self)
        row3.pack(fill="x")
        tk.Label(row3, text='火焰强度：', width=8).pack(padx=10, pady=10, side=tk.LEFT)
        tk.Label(row3, textvariable=str(self.L), width=5).pack(side=tk.LEFT)
        tk.Label(row3, text='%', width=5).pack(side=tk.LEFT)

        # 第四行
        row4 = tk.Frame(self)
        row4.pack(fill="x")
        tk.Label(row4, text='火焰面积：', width=8).pack(padx=10, pady=10, side=tk.LEFT)
        self.l1 = tk.Label(row4, textvariable=str(self.size), width=5)
        # self.l1 = tk.Label(row4, text=self.delayH, width=5) #亮延时
        self.l1.pack(side=tk.LEFT)
        tk.Label(row4, text='%', width=5).pack(side=tk.LEFT)

        # 第五行
        row5 = tk.Frame(self)
        row5.pack(fill="x")
        tk.Label(row5, text='闪烁频率:', width=8).pack(padx=10, pady=10, side=tk.LEFT)
        self.l2 = tk.Label(row5, textvariable=str(self.fft), width=5)
        self.l2.pack(side=tk.LEFT)
        tk.Label(row5, text='Hz', width=5).pack(side=tk.LEFT)

        # 第六行
        row6 = tk.Frame(self)
        row6.pack(fill="x")
        tk.Label(row6, text='西安理工大学\n\n杨海川\n\nSN:1283505225', justify=tk.LEFT, font=ft, width=25).pack(
            padx=10, pady=10, side=tk.LEFT)

    def take_snapshot(self):
        Myquit()

    def video_loop(self):
        success, img = camera.read()  # 从摄像头读取照片
        Fs = 15.0
        if success:
            cv2.waitKey(10)
            cv2image = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)  # 转换颜色从BGR到RGBA
            current_image = Image.fromarray(cv2image)  # 将图像转换成Image对象
            imgtk = ImageTk.PhotoImage(image=current_image)
            self.panel.imgtk = imgtk
            self.panel.config(image=imgtk)
            tk.Tk.update_idletasks(self)
            self.after(1, self.video_loop)
            # self.L = current_image.getpixel((4, 4))[0]
            img = Image.fromarray(img)
            img = np.array(img.convert('L').resize((28, 28)), dtype=np.float32)
            img = img.reshape((1, 28 * 28))
            img = img / 255.0
            sizetemp = 0
            for i in range(28 * 28):
                if img[0][i] > 0.5:
                    sizetemp = sizetemp + 1
            sizetemp = (int)(100 * sizetemp / 768)
            self.size.set(str(sizetemp))
            prediction = sess.run(y, feed_dict={x: img, keep_prob: 1.0})
            index = np.argmax(prediction)
            probability = prediction[0][index]
            pan = ""

            if sum(self.britemp[0:(int)(self.delayH * 10 + 1)]) == self.delayH * 10 + 1:
                self.R.set("亮")
                GPIO.output(21, True)
            if sum(self.britemp[0:(int)(self.delayL * 10 + 1)]) == 0:
                self.R.set("灭")
                GPIO.output(21, False)

            if index == 0 and probability > 0.8:
                self.britemp.pop()
                self.britemp.insert(0, 1)
                # self.R.set("high")
                pan = "high"
                # print('burning=high')
                # GPIO.output(21,True)

            elif index == 1 and probability > 0.8:
                self.britemp.pop()
                self.britemp.insert(0, 1)
                # self.R.set("low")
                pan = "low"
                # print('burning=low')
                # GPIO.output(21,True)

            elif index == 2 and probability > 0.8:
                self.britemp.pop()
                self.britemp.insert(0, 1)
                # self.R.set("too high")
                pan = "too high"
                # print('burning=too high')
                # GPIO.output(21,True)

            elif index == 3 and probability > 0.8:
                self.britemp.pop()
                self.britemp.insert(0, 0)
                # self.R.set("zero")
                pan = "zero"
                # print('burning=zero')
                # GPIO.output(21,False)
            else:
                # self.R.set("No")
                pan = "No"
                # print('burning=zero')
                # GPIO.output(21,False)

            brightness = (int)(100 * img.sum() / 784 * 1.2)

            Voltage = 0
            Voltage = brightness * 1.0 / 100
            if Voltage > 1:
                Voltage = 1
            DAC.DAC8532_Out_Voltage(DAC8532.channel_A, 3 * Voltage)
            DAC.DAC8532_Out_Voltage(DAC8532.channel_B, 3 - 3 * Voltage)

            self.L.set(str(brightness))
            # if brightness > 10:
            #    self.R.set("亮")
            # else:
            #    self.R.set("灭")

            t = time.localtime()
            hour = str('%02d' % t.tm_hour)
            min = str('%02d' % t.tm_min)
            sec = str('%02d' % t.tm_sec)
            time1 = hour + ":" + min + ":" + sec + " "
            self.savedata.append(brightness)
            if len(self.savedata) > 50:
                self.savedata.remove(self.savedata[0])

            # fft
            data = np.copy(self.savedata)
            fftres = np.array(0.0)
            if len(data) > 10:
                N = len(data)
                fft1 = data - np.mean(data)
                k = np.arange(N)
                T = N / Fs
                frq = k / T  # two sides frequency range
                frq1 = frq[range(int(N / 2))]  # one side frequency range

                # YY = np.fft.fft(y)  # 未归一化
                Y = np.fft.fft(fft1) / N  # fft computing and normalization 归一化

                Y1 = Y[range(int(N / 2))]
                if max(abs(Y1)) > 1:
                    fftres = np.around(np.argmax(abs(Y1)) * frq1[-1] / len(Y1), decimals=2)
                else:
                    fftres = np.around(0, decimals=2)
            self.fft.set(str(fftres.tolist()))
            self.savetime.append(time1)
            file.write(self.savetime[-1])
            file.write(" 火焰亮度：" + str(brightness) + "% 火焰面积：" + str(sizetemp) + "% 火焰频率：" + str(
                fftres.tolist()) + "Hz 着火状态：" + pan + "\n")
            ser.write('$' + " " + str(brightness) + " " + str(sizetemp) + " " + str(fftres.tolist()) + " " + str(
                brightness + sizetemp + fftres.tolist()) + " %\n")

        # 设置参数

    def callback(self):
        messagebox.showwarning('警告', '没有权限')

    def setup_config(self):
        # 接收弹窗的数据
        res = self.ask_userinfo()
        # print(res)
        if res is None:
            return

        # 更改参数
        self.delayH, self.delayL = res

        # 更新界面
        self.l1.config(text=self.delayH)
        self.l2.config(text=self.delayL)

    # 弹窗
    def ask_userinfo(self):
        inputDialog = MyDialog()

        self.wait_window(inputDialog)  # 这一句很重要！！！

        return inputDialog.userinfo


if __name__ == '__main__':
    f = open('init')
    str1 = f.readline()
    sern = f.readline()
    sern = sern.strip('\n')
    str1 = f.readline()
    btl = int(f.readline())
    str1 = f.readline()
    pw = f.readline()
    pw = pw.strip('\n')
    f.close()

    ser = serial.Serial(sern, btl)
    ser.isOpen()

    DAC = DAC8532.DAC8532()
    DAC.DAC8532_Out_Voltage(DAC8532.channel_A, 0)
    DAC.DAC8532_Out_Voltage(DAC8532.channel_B, 0)

    t = time.localtime()
    year = str(t.tm_year)
    mon = str('%02d' % t.tm_mon)
    day = str('%02d' % t.tm_mday)
    filetime = year + "_" + mon + "_" + day
    file = open(filetime + ".txt", "a+")

    GPIO.setmode(GPIO.BCM)
    GPIO.setup(21, GPIO.OUT)

    camera = cv2.VideoCapture(0)  # 摄像头

    model_dir = "model/model.ckpt"
    saver = tf.train.import_meta_graph(model_dir + ".meta")
    with tf.Session() as sess:
        saver.restore(sess, model_dir)
        x = tf.get_default_graph().get_tensor_by_name("images:0")
        keep_prob = tf.get_default_graph().get_tensor_by_name("keep_prob:0")
        y = tf.get_default_graph().get_tensor_by_name("fc2/output:0")
        app = MyApp()
        app.mainloop()
