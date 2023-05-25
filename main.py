from MyWindow import MyWindow
from MainWindow import Ui_MainWindow
from PyQt5.QtGui import QIcon
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow
import sys

QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
app = QApplication(sys.argv)
app.setStyleSheet(
    """
    QPushButton
    {
         font: 25 10pt '微软雅黑 Light';
         color: rgb(255,255,255);
         background-color: rgb(70, 84, 99);
         border:none;
         border-radius:9px;
         padding:2px 4px;
    }
    
    QPushButton:hover
    {
        background-color: rgb(95, 122, 139);
    }
    
    QPushButton:pressed
    {
        background-color: rgb(90, 100, 110);
    }
    
    QLabel
    {
        font: 25 10pt '微软雅黑 Light';
        color: rgb(255,255,255);
    }
    
    QMainWindow
    {
        background-color: rgb(25, 34, 45)
    }
    
    """
)
app.setWindowIcon(QIcon(r'./pic/icon.jpg'))
MainWindow = QMainWindow()
ui = Ui_MainWindow()
My_Ui = MyWindow(ui)

sys.exit(app.exec_())





