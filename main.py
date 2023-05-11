from MyWindow import MyWindow
from MainWindow import Ui_MainWindow
from PyQt5.QtWidgets import QApplication, QMainWindow
import sys

app = QApplication(sys.argv)
MainWindow = QMainWindow()
ui = Ui_MainWindow()
My_Ui = MyWindow(ui)
sys.exit(app.exec_())





