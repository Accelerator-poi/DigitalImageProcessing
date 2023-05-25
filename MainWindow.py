# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1246, 826)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("pic/icon.jpg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.splitter = QtWidgets.QSplitter(self.centralwidget)
        self.splitter.setGeometry(QtCore.QRect(10, 10, 180, 772))
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.setObjectName("splitter")
        self.layoutWidget = QtWidgets.QWidget(self.splitter)
        self.layoutWidget.setObjectName("layoutWidget")
        self.horizontalLayout_14 = QtWidgets.QHBoxLayout(self.layoutWidget)
        self.horizontalLayout_14.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_14.setObjectName("horizontalLayout_14")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.line = QtWidgets.QFrame(self.layoutWidget)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.horizontalLayout.addWidget(self.line)
        self.label = QtWidgets.QLabel(self.layoutWidget)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.line_2 = QtWidgets.QFrame(self.layoutWidget)
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.horizontalLayout.addWidget(self.line_2)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.gridLayout_3 = QtWidgets.QGridLayout()
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.BinarizationBtn = QtWidgets.QPushButton(self.layoutWidget)
        self.BinarizationBtn.setObjectName("BinarizationBtn")
        self.gridLayout_3.addWidget(self.BinarizationBtn, 0, 1, 1, 1)
        self.geometryBtn = QtWidgets.QPushButton(self.layoutWidget)
        self.geometryBtn.setObjectName("geometryBtn")
        self.gridLayout_3.addWidget(self.geometryBtn, 1, 0, 1, 2)
        self.GrayscaleBtn = QtWidgets.QPushButton(self.layoutWidget)
        self.GrayscaleBtn.setObjectName("GrayscaleBtn")
        self.gridLayout_3.addWidget(self.GrayscaleBtn, 0, 0, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout_3)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.line_3 = QtWidgets.QFrame(self.layoutWidget)
        self.line_3.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.horizontalLayout_2.addWidget(self.line_3)
        self.label_2 = QtWidgets.QLabel(self.layoutWidget)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_2.addWidget(self.label_2)
        self.line_4 = QtWidgets.QFrame(self.layoutWidget)
        self.line_4.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_4.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_4.setObjectName("line_4")
        self.horizontalLayout_2.addWidget(self.line_4)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.SobelBtn = QtWidgets.QPushButton(self.layoutWidget)
        self.SobelBtn.setObjectName("SobelBtn")
        self.gridLayout.addWidget(self.SobelBtn, 3, 1, 1, 1)
        self.LowpassBtn = QtWidgets.QPushButton(self.layoutWidget)
        self.LowpassBtn.setObjectName("LowpassBtn")
        self.gridLayout.addWidget(self.LowpassBtn, 4, 0, 1, 1)
        self.RobertsBtn = QtWidgets.QPushButton(self.layoutWidget)
        self.RobertsBtn.setObjectName("RobertsBtn")
        self.gridLayout.addWidget(self.RobertsBtn, 2, 1, 1, 1)
        self.PrewittBtn = QtWidgets.QPushButton(self.layoutWidget)
        self.PrewittBtn.setObjectName("PrewittBtn")
        self.gridLayout.addWidget(self.PrewittBtn, 3, 0, 1, 1)
        self.histogramBtn = QtWidgets.QPushButton(self.layoutWidget)
        self.histogramBtn.setObjectName("histogramBtn")
        self.gridLayout.addWidget(self.histogramBtn, 0, 0, 1, 1)
        self.HighpassBtn = QtWidgets.QPushButton(self.layoutWidget)
        self.HighpassBtn.setObjectName("HighpassBtn")
        self.gridLayout.addWidget(self.HighpassBtn, 4, 1, 1, 1)
        self.medianBtn = QtWidgets.QPushButton(self.layoutWidget)
        self.medianBtn.setObjectName("medianBtn")
        self.gridLayout.addWidget(self.medianBtn, 2, 0, 1, 1)
        self.EqualizeBtn = QtWidgets.QPushButton(self.layoutWidget)
        self.EqualizeBtn.setObjectName("EqualizeBtn")
        self.gridLayout.addWidget(self.EqualizeBtn, 0, 1, 1, 1)
        self.MeanBtn = QtWidgets.QPushButton(self.layoutWidget)
        self.MeanBtn.setObjectName("MeanBtn")
        self.gridLayout.addWidget(self.MeanBtn, 1, 0, 1, 1)
        self.vagueBtn = QtWidgets.QPushButton(self.layoutWidget)
        self.vagueBtn.setObjectName("vagueBtn")
        self.gridLayout.addWidget(self.vagueBtn, 1, 1, 1, 1)
        self.BoxBtn = QtWidgets.QPushButton(self.layoutWidget)
        self.BoxBtn.setObjectName("BoxBtn")
        self.gridLayout.addWidget(self.BoxBtn, 5, 0, 1, 2)
        self.verticalLayout.addLayout(self.gridLayout)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem1)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.line_5 = QtWidgets.QFrame(self.layoutWidget)
        self.line_5.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_5.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_5.setObjectName("line_5")
        self.horizontalLayout_3.addWidget(self.line_5)
        self.label_3 = QtWidgets.QLabel(self.layoutWidget)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_3.addWidget(self.label_3)
        self.line_6 = QtWidgets.QFrame(self.layoutWidget)
        self.line_6.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_6.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_6.setObjectName("line_6")
        self.horizontalLayout_3.addWidget(self.line_6)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.corrosionBtn = QtWidgets.QPushButton(self.layoutWidget)
        self.corrosionBtn.setObjectName("corrosionBtn")
        self.gridLayout_2.addWidget(self.corrosionBtn, 0, 0, 1, 1)
        self.CloseBtn = QtWidgets.QPushButton(self.layoutWidget)
        self.CloseBtn.setObjectName("CloseBtn")
        self.gridLayout_2.addWidget(self.CloseBtn, 1, 1, 1, 1)
        self.OpenBtn = QtWidgets.QPushButton(self.layoutWidget)
        self.OpenBtn.setObjectName("OpenBtn")
        self.gridLayout_2.addWidget(self.OpenBtn, 1, 0, 1, 1)
        self.expansionBtn = QtWidgets.QPushButton(self.layoutWidget)
        self.expansionBtn.setObjectName("expansionBtn")
        self.gridLayout_2.addWidget(self.expansionBtn, 0, 1, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout_2)
        spacerItem2 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem2)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.line_7 = QtWidgets.QFrame(self.layoutWidget)
        self.line_7.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_7.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_7.setObjectName("line_7")
        self.horizontalLayout_4.addWidget(self.line_7)
        self.label_4 = QtWidgets.QLabel(self.layoutWidget)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_4.addWidget(self.label_4)
        self.line_8 = QtWidgets.QFrame(self.layoutWidget)
        self.line_8.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_8.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_8.setObjectName("line_8")
        self.horizontalLayout_4.addWidget(self.line_8)
        self.verticalLayout.addLayout(self.horizontalLayout_4)
        self.gridLayout_4 = QtWidgets.QGridLayout()
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.CannyBtn = QtWidgets.QPushButton(self.layoutWidget)
        self.CannyBtn.setObjectName("CannyBtn")
        self.gridLayout_4.addWidget(self.CannyBtn, 1, 0, 1, 2)
        self.ScharrBtn = QtWidgets.QPushButton(self.layoutWidget)
        self.ScharrBtn.setObjectName("ScharrBtn")
        self.gridLayout_4.addWidget(self.ScharrBtn, 0, 1, 1, 1)
        self.LOGBtn = QtWidgets.QPushButton(self.layoutWidget)
        self.LOGBtn.setObjectName("LOGBtn")
        self.gridLayout_4.addWidget(self.LOGBtn, 0, 0, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout_4)
        self.horizontalLayout_5.addLayout(self.verticalLayout)
        self.verticalLayout_3.addLayout(self.horizontalLayout_5)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        spacerItem3 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_2.addItem(spacerItem3)
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.line_14 = QtWidgets.QFrame(self.layoutWidget)
        self.line_14.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_14.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_14.setObjectName("line_14")
        self.horizontalLayout_8.addWidget(self.line_14)
        self.label_9 = QtWidgets.QLabel(self.layoutWidget)
        self.label_9.setObjectName("label_9")
        self.horizontalLayout_8.addWidget(self.label_9)
        self.line_15 = QtWidgets.QFrame(self.layoutWidget)
        self.line_15.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_15.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_15.setObjectName("line_15")
        self.horizontalLayout_8.addWidget(self.line_15)
        self.verticalLayout_2.addLayout(self.horizontalLayout_8)
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.ImportBtn = QtWidgets.QPushButton(self.layoutWidget)
        self.ImportBtn.setObjectName("ImportBtn")
        self.horizontalLayout_9.addWidget(self.ImportBtn)
        self.SaveBtn = QtWidgets.QPushButton(self.layoutWidget)
        self.SaveBtn.setObjectName("SaveBtn")
        self.horizontalLayout_9.addWidget(self.SaveBtn)
        self.verticalLayout_2.addLayout(self.horizontalLayout_9)
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.label_10 = QtWidgets.QLabel(self.layoutWidget)
        self.label_10.setObjectName("label_10")
        self.horizontalLayout_10.addWidget(self.label_10)
        self.Label_H = QtWidgets.QLabel(self.layoutWidget)
        self.Label_H.setObjectName("Label_H")
        self.horizontalLayout_10.addWidget(self.Label_H)
        self.verticalLayout_2.addLayout(self.horizontalLayout_10)
        self.horizontalLayout_11 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_11.setObjectName("horizontalLayout_11")
        self.label_12 = QtWidgets.QLabel(self.layoutWidget)
        self.label_12.setObjectName("label_12")
        self.horizontalLayout_11.addWidget(self.label_12)
        self.Label_W = QtWidgets.QLabel(self.layoutWidget)
        self.Label_W.setObjectName("Label_W")
        self.horizontalLayout_11.addWidget(self.Label_W)
        self.verticalLayout_2.addLayout(self.horizontalLayout_11)
        self.horizontalLayout_12 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_12.setObjectName("horizontalLayout_12")
        self.label_14 = QtWidgets.QLabel(self.layoutWidget)
        self.label_14.setObjectName("label_14")
        self.horizontalLayout_12.addWidget(self.label_14)
        self.Label_T = QtWidgets.QLabel(self.layoutWidget)
        self.Label_T.setObjectName("Label_T")
        self.horizontalLayout_12.addWidget(self.Label_T)
        self.verticalLayout_2.addLayout(self.horizontalLayout_12)
        self.horizontalLayout_13 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_13.setObjectName("horizontalLayout_13")
        self.label_16 = QtWidgets.QLabel(self.layoutWidget)
        self.label_16.setObjectName("label_16")
        self.horizontalLayout_13.addWidget(self.label_16)
        self.Label_Type = QtWidgets.QLabel(self.layoutWidget)
        self.Label_Type.setObjectName("Label_Type")
        self.horizontalLayout_13.addWidget(self.Label_Type)
        self.verticalLayout_2.addLayout(self.horizontalLayout_13)
        spacerItem4 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_2.addItem(spacerItem4)
        self.verticalLayout_3.addLayout(self.verticalLayout_2)
        self.horizontalLayout_14.addLayout(self.verticalLayout_3)
        self.line_9 = QtWidgets.QFrame(self.layoutWidget)
        self.line_9.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_9.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_9.setObjectName("line_9")
        self.horizontalLayout_14.addWidget(self.line_9)
        self.layoutWidget1 = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget1.setGeometry(QtCore.QRect(230, 20, 471, 16))
        self.layoutWidget1.setObjectName("layoutWidget1")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout(self.layoutWidget1)
        self.horizontalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.line_11 = QtWidgets.QFrame(self.layoutWidget1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.line_11.sizePolicy().hasHeightForWidth())
        self.line_11.setSizePolicy(sizePolicy)
        self.line_11.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_11.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_11.setObjectName("line_11")
        self.horizontalLayout_6.addWidget(self.line_11)
        self.label_6 = QtWidgets.QLabel(self.layoutWidget1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_6.sizePolicy().hasHeightForWidth())
        self.label_6.setSizePolicy(sizePolicy)
        self.label_6.setObjectName("label_6")
        self.horizontalLayout_6.addWidget(self.label_6)
        self.line_10 = QtWidgets.QFrame(self.layoutWidget1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.line_10.sizePolicy().hasHeightForWidth())
        self.line_10.setSizePolicy(sizePolicy)
        self.line_10.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_10.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_10.setObjectName("line_10")
        self.horizontalLayout_6.addWidget(self.line_10)
        self.PicBefore = QtWidgets.QLabel(self.centralwidget)
        self.PicBefore.setGeometry(QtCore.QRect(230, 50, 471, 331))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.PicBefore.sizePolicy().hasHeightForWidth())
        self.PicBefore.setSizePolicy(sizePolicy)
        self.PicBefore.setText("")
        self.PicBefore.setObjectName("PicBefore")
        self.PicAfter = QtWidgets.QLabel(self.centralwidget)
        self.PicAfter.setGeometry(QtCore.QRect(230, 450, 471, 331))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.PicAfter.sizePolicy().hasHeightForWidth())
        self.PicAfter.setSizePolicy(sizePolicy)
        self.PicAfter.setText("")
        self.PicAfter.setObjectName("PicAfter")
        self.layoutWidget2 = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget2.setGeometry(QtCore.QRect(230, 410, 471, 16))
        self.layoutWidget2.setObjectName("layoutWidget2")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout(self.layoutWidget2)
        self.horizontalLayout_7.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.line_12 = QtWidgets.QFrame(self.layoutWidget2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.line_12.sizePolicy().hasHeightForWidth())
        self.line_12.setSizePolicy(sizePolicy)
        self.line_12.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_12.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_12.setObjectName("line_12")
        self.horizontalLayout_7.addWidget(self.line_12)
        self.label_7 = QtWidgets.QLabel(self.layoutWidget2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_7.sizePolicy().hasHeightForWidth())
        self.label_7.setSizePolicy(sizePolicy)
        self.label_7.setObjectName("label_7")
        self.horizontalLayout_7.addWidget(self.label_7)
        self.line_13 = QtWidgets.QFrame(self.layoutWidget2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.line_13.sizePolicy().hasHeightForWidth())
        self.line_13.setSizePolicy(sizePolicy)
        self.line_13.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_13.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_13.setObjectName("line_13")
        self.horizontalLayout_7.addWidget(self.line_13)
        self.layoutWidget_2 = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget_2.setGeometry(QtCore.QRect(740, 20, 471, 16))
        self.layoutWidget_2.setObjectName("layoutWidget_2")
        self.horizontalLayout_15 = QtWidgets.QHBoxLayout(self.layoutWidget_2)
        self.horizontalLayout_15.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_15.setObjectName("horizontalLayout_15")
        self.line_16 = QtWidgets.QFrame(self.layoutWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.line_16.sizePolicy().hasHeightForWidth())
        self.line_16.setSizePolicy(sizePolicy)
        self.line_16.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_16.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_16.setObjectName("line_16")
        self.horizontalLayout_15.addWidget(self.line_16)
        self.label_8 = QtWidgets.QLabel(self.layoutWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_8.sizePolicy().hasHeightForWidth())
        self.label_8.setSizePolicy(sizePolicy)
        self.label_8.setObjectName("label_8")
        self.horizontalLayout_15.addWidget(self.label_8)
        self.line_17 = QtWidgets.QFrame(self.layoutWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.line_17.sizePolicy().hasHeightForWidth())
        self.line_17.setSizePolicy(sizePolicy)
        self.line_17.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_17.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_17.setObjectName("line_17")
        self.horizontalLayout_15.addWidget(self.line_17)
        self.FFTBefore = QtWidgets.QLabel(self.centralwidget)
        self.FFTBefore.setGeometry(QtCore.QRect(740, 50, 471, 331))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.FFTBefore.sizePolicy().hasHeightForWidth())
        self.FFTBefore.setSizePolicy(sizePolicy)
        self.FFTBefore.setText("")
        self.FFTBefore.setObjectName("FFTBefore")
        self.layoutWidget_3 = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget_3.setGeometry(QtCore.QRect(740, 410, 471, 16))
        self.layoutWidget_3.setObjectName("layoutWidget_3")
        self.horizontalLayout_16 = QtWidgets.QHBoxLayout(self.layoutWidget_3)
        self.horizontalLayout_16.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_16.setObjectName("horizontalLayout_16")
        self.line_18 = QtWidgets.QFrame(self.layoutWidget_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.line_18.sizePolicy().hasHeightForWidth())
        self.line_18.setSizePolicy(sizePolicy)
        self.line_18.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_18.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_18.setObjectName("line_18")
        self.horizontalLayout_16.addWidget(self.line_18)
        self.label_11 = QtWidgets.QLabel(self.layoutWidget_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_11.sizePolicy().hasHeightForWidth())
        self.label_11.setSizePolicy(sizePolicy)
        self.label_11.setObjectName("label_11")
        self.horizontalLayout_16.addWidget(self.label_11)
        self.line_19 = QtWidgets.QFrame(self.layoutWidget_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.line_19.sizePolicy().hasHeightForWidth())
        self.line_19.setSizePolicy(sizePolicy)
        self.line_19.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_19.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_19.setObjectName("line_19")
        self.horizontalLayout_16.addWidget(self.line_19)
        self.FFTAfter = QtWidgets.QLabel(self.centralwidget)
        self.FFTAfter.setGeometry(QtCore.QRect(740, 450, 471, 331))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.FFTAfter.sizePolicy().hasHeightForWidth())
        self.FFTAfter.setSizePolicy(sizePolicy)
        self.FFTAfter.setText("")
        self.FFTAfter.setObjectName("FFTAfter")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "基本处理"))
        self.BinarizationBtn.setText(_translate("MainWindow", "二值化处理"))
        self.geometryBtn.setText(_translate("MainWindow", "几何变换"))
        self.GrayscaleBtn.setText(_translate("MainWindow", "灰度处理"))
        self.label_2.setText(_translate("MainWindow", "图像增强"))
        self.SobelBtn.setText(_translate("MainWindow", "Sobel算子"))
        self.LowpassBtn.setText(_translate("MainWindow", "低通滤波"))
        self.RobertsBtn.setText(_translate("MainWindow", "Roberts算子"))
        self.PrewittBtn.setText(_translate("MainWindow", "Prewitt算子"))
        self.histogramBtn.setText(_translate("MainWindow", "灰度直方图"))
        self.HighpassBtn.setText(_translate("MainWindow", "高通滤波"))
        self.medianBtn.setText(_translate("MainWindow", "中值滤波"))
        self.EqualizeBtn.setText(_translate("MainWindow", "均衡化"))
        self.MeanBtn.setText(_translate("MainWindow", "均值滤波"))
        self.vagueBtn.setText(_translate("MainWindow", "高斯模糊滤波"))
        self.BoxBtn.setText(_translate("MainWindow", "方框滤波"))
        self.label_3.setText(_translate("MainWindow", "形态学处理"))
        self.corrosionBtn.setText(_translate("MainWindow", "腐蚀运算"))
        self.CloseBtn.setText(_translate("MainWindow", "闭运算"))
        self.OpenBtn.setText(_translate("MainWindow", "开运算"))
        self.expansionBtn.setText(_translate("MainWindow", "膨胀运算"))
        self.label_4.setText(_translate("MainWindow", "边缘检测"))
        self.CannyBtn.setText(_translate("MainWindow", "Canny算子"))
        self.ScharrBtn.setText(_translate("MainWindow", "Scharr算子"))
        self.LOGBtn.setText(_translate("MainWindow", "LoG检测"))
        self.label_9.setText(_translate("MainWindow", "图像信息"))
        self.ImportBtn.setText(_translate("MainWindow", "导入"))
        self.SaveBtn.setText(_translate("MainWindow", "保存"))
        self.label_10.setText(_translate("MainWindow", "高度："))
        self.Label_H.setText(_translate("MainWindow", "Label_H"))
        self.label_12.setText(_translate("MainWindow", "宽度："))
        self.Label_W.setText(_translate("MainWindow", "Label_W"))
        self.label_14.setText(_translate("MainWindow", "通道数："))
        self.Label_T.setText(_translate("MainWindow", "Label_T"))
        self.label_16.setText(_translate("MainWindow", "图片类型："))
        self.Label_Type.setText(_translate("MainWindow", "Label_Type"))
        self.label_6.setText(_translate("MainWindow", " 处理前"))
        self.label_7.setText(_translate("MainWindow", " 处理后"))
        self.label_8.setText(_translate("MainWindow", "频谱图"))
        self.label_11.setText(_translate("MainWindow", "频谱图"))
