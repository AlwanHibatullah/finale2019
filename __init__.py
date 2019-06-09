from PyQt5 import QtWidgets, QtCore
from MenuUtama import *

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MenuUtama = QtWidgets.QMainWindow()
    ui = Ui_MenuUtama()
    ui.setupUi(MenuUtama)
    MenuUtama.setWindowFlags(QtCore.Qt.WindowMinimizeButtonHint | QtCore.Qt.WindowCloseButtonHint)
    MenuUtama.show()
    sys.exit(app.exec_())
