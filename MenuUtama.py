# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MenuUtama.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
import GUITraining
import GUITesting


class Ui_MenuUtama(object):
    def __init__(self):
        super(Ui_MenuUtama, self).__init__()
        # self.setFixedSize(400,300)

    def setupUi(self, MenuUtama):
        MenuUtama.setObjectName("MenuUtama")
        MenuUtama.resize(400, 300)
        self.centralwidget = QtWidgets.QWidget(MenuUtama)
        self.centralwidget.setObjectName("centralwidget")
        self.btnMenuTrain = QtWidgets.QPushButton(self.centralwidget)
        self.btnMenuTrain.setGeometry(QtCore.QRect(90, 150, 111, 41))
        self.btnMenuTrain.setObjectName("btnMenuTrain")
        self.btnMenuTest = QtWidgets.QPushButton(self.centralwidget)
        self.btnMenuTest.setGeometry(QtCore.QRect(210, 150, 111, 41))
        self.btnMenuTest.setObjectName("btnMenuTest")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(80, 30, 251, 21))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(12)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(100, 60, 211, 21))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(12)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(90, 110, 231, 21))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(12)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(230, 260, 151, 21))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(8)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        MenuUtama.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MenuUtama)
        self.statusbar.setObjectName("statusbar")
        MenuUtama.setStatusBar(self.statusbar)

        self.retranslateUi(MenuUtama)
        QtCore.QMetaObject.connectSlotsByName(MenuUtama)

        self.btnMenuTrain.clicked.connect(self.launchTraining)
        self.btnMenuTest.clicked.connect(self.launchTesting)

    def retranslateUi(self, MenuUtama):
        _translate = QtCore.QCoreApplication.translate
        MenuUtama.setWindowTitle(_translate("MenuUtama", "CNN Sandi Rumput"))
        self.btnMenuTrain.setText(_translate("MenuUtama", "Pelatihan"))
        self.btnMenuTest.setText(_translate("MenuUtama", "Pengujian"))
        self.label.setText(_translate("MenuUtama", "Program Pengenalan Sandi Rumput"))
        self.label_2.setText(_translate("MenuUtama", "Convolutional Neural Network"))
        self.label_3.setText(_translate("MenuUtama", "Silahkan Pilih Menu Dibawah Ini:"))
        self.label_4.setText(_translate("MenuUtama", "10115425 - Alwan Hibatullah"))

    def launchTraining(self):
        # pass
        self.window = QtWidgets.QMainWindow()
        self.ui = GUITraining.Ui_MainWindow()
        self.ui.setupUi(self.window)
        self.window.setWindowFlags(QtCore.Qt.WindowMinimizeButtonHint | QtCore.Qt.WindowCloseButtonHint)
        self.window.show()


    def launchTesting(self):
        # pass
        self.window = QtWidgets.QMainWindow()
        self.ui = GUITesting.Ui_MainWindow()
        self.ui.setupUi(self.window)
        self.window.setWindowFlags(QtCore.Qt.WindowMinimizeButtonHint | QtCore.Qt.WindowCloseButtonHint)
        self.window.show()

# if __name__ == "__main__":
#     import sys
#     app = QtWidgets.QApplication(sys.argv)
#     MenuUtama = QtWidgets.QMainWindow()
#     ui = Ui_MenuUtama()
#     ui.setupUi(MenuUtama)
#     MenuUtama.show()
#     sys.exit(app.exec_())

