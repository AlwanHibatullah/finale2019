# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'GUITesting.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from Preprocessing import *
from CNN import *
from LoadData import *
import matplotlib.pyplot as plt
from PIL import Image
from PIL.ImageQt import ImageQt
import os
import errno
import cv2 as cv
import numpy as np
from keras.models import load_model
import pandas as pd

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1024, 650)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("logo.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.tabTraining = QtWidgets.QTabWidget(self.centralwidget)
        self.tabTraining.setGeometry(QtCore.QRect(0, 30, 1024, 601))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(8)
        self.tabTraining.setFont(font)
        self.tabTraining.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.tabTraining.setObjectName("tabTraining")
        self.tabInputCitra = QtWidgets.QWidget()
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(8)
        self.tabInputCitra.setFont(font)
        self.tabInputCitra.setObjectName("tabInputCitra")
        self.txtPath = QtWidgets.QLineEdit(self.tabInputCitra)
        self.txtPath.setGeometry(QtCore.QRect(220, 10, 671, 22))
        self.txtPath.setObjectName("txtPath")
        self.btnInputCitra = QtWidgets.QPushButton(self.tabInputCitra)
        self.btnInputCitra.setGeometry(QtCore.QRect(900, 10, 101, 24))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(8)
        self.btnInputCitra.setFont(font)
        self.btnInputCitra.setObjectName("btnInputCitra")
        self.btnPreprocessing = QtWidgets.QPushButton(self.tabInputCitra)
        self.btnPreprocessing.setGeometry(QtCore.QRect(10, 40, 121, 24))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(8)
        self.btnPreprocessing.setFont(font)
        self.btnPreprocessing.setObjectName("btnPreprocessing")
        self.imageScrollArea = QtWidgets.QScrollArea(self.tabInputCitra)
        self.imageScrollArea.setGeometry(QtCore.QRect(220, 40, 781, 501))
        self.imageScrollArea.setWidgetResizable(True)
        self.imageScrollArea.setObjectName("imageScrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 779, 499))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.scrollAreaWidgetContents)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(0, 0, 781, 511))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.vLay = QtWidgets.QVBoxLayout(self.scrollAreaWidgetContents)
        self.vLay.setContentsMargins(0, 0, 0, 0)
        self.vLay.setObjectName("vLay")
        self.imageLabel = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.imageLabel.setText("")
        self.imageLabel.setObjectName("imageLabel")
        self.vLay.addWidget(self.imageLabel)
        self.imageScrollArea.setWidget(self.scrollAreaWidgetContents)
        self.lblStatus = QtWidgets.QLabel(self.tabInputCitra)
        self.lblStatus.setGeometry(QtCore.QRect(220, 550, 781, 16))
        self.lblStatus.setText("")
        self.lblStatus.setObjectName("lblStatus")
        self.tabTraining.addTab(self.tabInputCitra, "")
        self.tabPelatihan = QtWidgets.QWidget()
        self.tabPelatihan.setObjectName("tabPelatihan")
        self.label_3 = QtWidgets.QLabel(self.tabPelatihan)
        self.label_3.setGeometry(QtCore.QRect(200, 115, 91, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.txtLR = QtWidgets.QLineEdit(self.tabPelatihan)
        self.txtLR.setGeometry(QtCore.QRect(310, 110, 421, 31))
        self.txtLR.setObjectName("txtLR")
        self.pushButton = QtWidgets.QPushButton(self.tabPelatihan)
        self.pushButton.setGeometry(QtCore.QRect(10, 30, 161, 23))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.progressBar = QtWidgets.QProgressBar(self.tabPelatihan)
        self.progressBar.setGeometry(QtCore.QRect(200, 30, 801, 23))
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.lblHasil = QtWidgets.QLabel(self.tabPelatihan)
        self.lblHasil.setGeometry(QtCore.QRect(200, 180, 791, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.lblHasil.setFont(font)
        self.lblHasil.setText("")
        self.lblHasil.setObjectName("lblHasil")
        self.tabTraining.addTab(self.tabPelatihan, "")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(730, 10, 281, 31))
        font = QtGui.QFont()
        font.setFamily("Segoe UI Semibold")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabTraining.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.btnInputCitra.clicked.connect(self.inputCitra)
        self.btnPreprocessing.clicked.connect(self.preprocess)
        self.pushButton.clicked.connect(self.test)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "CNN Sandi Rumput"))
        self.btnInputCitra.setText(_translate("MainWindow", "Input Citra"))
        self.btnPreprocessing.setText(_translate("MainWindow", "Preprocessing"))
        self.tabTraining.setTabText(self.tabTraining.indexOf(self.tabInputCitra), _translate("MainWindow", "Input Citra"))
        self.label_3.setText(_translate("MainWindow", "Hasil Pengujian"))
        self.pushButton.setText(_translate("MainWindow", "Uji Data"))
        self.tabTraining.setTabText(self.tabTraining.indexOf(self.tabPelatihan), _translate("MainWindow", "Pengujian"))
        self.label_2.setText(_translate("MainWindow", "CNN PENGUJIAN SANDI RUMPUT"))

    def inputCitra(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select Image", "",
                                                            "JPEG File (*.jpg)")

        try:
            if filename:
                self.txtPath.setText(filename)
                self.vLay.addWidget(self.imageLabel)
                pixmap = QtGui.QPixmap(filename)
                self.imageLabel.setPixmap(pixmap)
                self.imageScrollArea.setWidgetResizable(True)
                self.imageLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            else:
                nopath = ""
                self.txtPath.setText(nopath)
                self.vLay.addWidget(self.imageLabel)
                pixmap = QtGui.QPixmap(nopath)
                self.imageLabel.setPixmap(pixmap)
                self.imageScrollArea.setWidgetResizable(True)
                self.imageLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)

        except OSError as e:
            raise

    def preprocess(self):
        # pass
        msg = QtWidgets.QMessageBox()

        file = self.txtPath.text()

        if file == "":  # Check if File Path is Empty
            msg.setIcon(msg.Information)
            msg.setWindowTitle("Peringatan")
            msg.setText("File Masih Kosong, Silahkan Pilih File Terlebih Dahulu!")
            msg.exec_()
        else:
            path = './data_testing'

            try:
                p = Preprocessing()

                # -------------Thresholding Image--------------#

                print("\n........Program Initiated.......\n")
                src_img = cv.imread(file, 1)
                copy = src_img.copy()
                height = src_img.shape[0]
                width = src_img.shape[1]

                # print("\n Resizing Image........")

                if (height > 2000) or (width > 2000):
                    src_img = cv.resize(copy, dsize=(int(width / 2), int(int(width / 2) * height / width)),
                                        interpolation=cv.INTER_AREA)

                height = src_img.shape[0]
                width = src_img.shape[1]

                print("#---------Image Info:--------#")
                print("\tHeight =", height, "\n\tWidth =", width)
                print("#----------------------------#")

                grey_img = cv.cvtColor(src_img, cv.COLOR_BGR2GRAY)

                # print("Applying Adaptive Threshold with kernel :- 21 X 21")
                bin_img = cv.adaptiveThreshold(grey_img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 21, 20)
                bin_img1 = bin_img.copy()
                bin_img2 = bin_img.copy()

                kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
                kernel1 = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=np.uint8)
                # print("Noise Removal From Image.........")
                final_thr = cv.morphologyEx(bin_img, cv.MORPH_CLOSE, kernel)
                contr_retrival = final_thr.copy()

                # -------------Thresholding Image--------------#

                # -------------Line Detection------------------#

                print("Beginning Character Semenation..............")
                count_x = np.zeros(shape=(height))
                for y in range(height):
                    for x in range(width):
                        if bin_img[y][x] == 255:
                            count_x[y] = count_x[y] + 1

                upper_lines, lower_lines = p.line_array(count_x)

                upperlines, lowerlines = p.refine_array(upper_lines, lower_lines)

                print(upperlines, lowerlines)
                if len(upperlines) == len(lowerlines):
                    lines = []
                    for y in upperlines:
                        final_thr[y][:] = 0
                    for y in lowerlines:
                        final_thr[y][:] = 0
                    for y in range(len(upperlines)):
                        lines.append((upperlines[y], lowerlines[y]))

                else:
                    print(
                        "Too much noise in image, unable to process.\nPlease try with another image. Ctrl-C to exit:- ")
                    # showimages()
                    k = cv2.waitKey(0)
                    while 1:
                        k = cv2.waitKey(0)
                        if k & 0xFF == ord('q'):
                            cv2.destroyAllWindows()
                            exit()

                lines = np.array(lines)

                no_of_lines = len(lines)

                # print("\nGiven Text has   # ", no_of_lines, " #   no. of lines")

                lines_img = []

                for i in range(no_of_lines):
                    lines_img.append(bin_img2[lines[i][0]:lines[i][1], :])

                # -------------/Line Detection-----------------#

                # -------------Letter Width Calculation--------#

                contours, hierarchy = cv2.findContours(contr_retrival, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                final_contr = np.zeros((final_thr.shape[0], final_thr.shape[1], 3), dtype=np.uint8)

                mean_lttr_width = p.letter_width(contours)  # letter_width(contours)
                # print("\nAverage Width of Each Letter:- ", mean_lttr_width)

                # -------------/Letter Width Calculation-------#

                # --------------Word Detection-----------------#
                x_lines = []

                for i in range(len(lines_img)):
                    x_lines.append(p.end_wrd_dtct(width, lines, i, bin_img, mean_lttr_width))

                for i in range(len(x_lines)):
                    x_lines[i].append(width)

                # -------------/Word Detection-----------------#

                # -------------Letter Segmentation-------------#

                for i in range(len(lines)):
                    p.letter_seg(lines_img, x_lines, i)

                # ------------\Letter Segmentation-------------#

                # -------------Rename and Move Files-------------#

                p.renameTest()

                # -------------/Rename and Move Files-------------#
                # -------------Character segmenting------------#

                chr_img = bin_img1.copy()

                contours, hierarchy = cv.findContours(chr_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

                for cnt in contours:
                    if cv.contourArea(cnt) > 20:
                        x, y, w, h = cv.boundingRect(cnt)
                        cv.rectangle(src_img, (x, y), (x + w, y + h), (0, 0, 0), 1)

                # -------------/Character segmenting-----------#

                # -------------Show Segmentation Result-----------#
                p.showimages(src_img, grey_img, final_thr)

                # -------------Show Segmentation Result-----------#

                # ---------------------------------- Preprocessing Section ----------------------------------#
                print("Segmentation Ended.")
                # Show Success Message
                msg.setIcon(msg.Information)
                msg.setWindowTitle("Sukses")
                msg.setText("Citra Berhasil Di Preprocessing")
                msg.exec_()

            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

    def test(self):
        load_test = LoadData()
        self.folder_test = load_test.data_test
        self.train_generator, self.x_train, self.x_valid, self.y_train, self.y_valid = load_test.loadDataTrain()
        self.test_generator, self.x_test = load_test.loadDataTest(self.folder_test)

        model = load_model('./model.h5')

        self.test_generator.reset()
        pred = model.predict_generator(self.test_generator, verbose=1, steps=600/1)

        predicted_class_indices = np.argmax(pred, axis=1)

        labels = (self.train_generator.class_indices)
        labels = dict((v,k) for k, v in labels.items())
        prediksi = [labels[k] for k in predicted_class_indices]
        path = self.test_generator.filenames

        filenames = []
        for x in range(len(path)):
            filenames.append(path[x][12:len(path[x]) - 8])

        true_pred = 0
        compare = []
        for x in range(len(filenames)):
            if filenames[x] == prediksi[x]:
                # true_pred = true_pred + 1
                compare.append("False")
            else:
                true_pred = true_pred + 1
                compare.append("True")

        row = len(self.test_generator)

        list_prediksi = []
        for i in range(row):
            list_prediksi.append([filenames[i], prediksi[i], compare[i]])

        # print result to console
        # s = ""
        # for i in range(row):
        #     print(i, list_prediksi[i])
        s = ''.join(prediksi[0:row])
        # print(s)

        self.progressBar.setValue(100)

        self.txtLR.setText(s)


        persentase = (true_pred / len(filenames)) * 100
        # print(persentase)

        self.lblHasil.setText("Tingkat Akurasi : %.2f%%" % (persentase))

        # results = pd.DataFrame({"Filename": path,
        #                         "Predictions": prediksi,
        #                         "Compare": compare})
        # results.to_csv("./Prediksi.csv")



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

