# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'c:\Users\Cabrite\Documents\GitHub\Image-Label\Labelling.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1323, 739)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_Image = ImageLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_Image.sizePolicy().hasHeightForWidth())
        self.label_Image.setSizePolicy(sizePolicy)
        self.label_Image.setMinimumSize(QtCore.QSize(960, 540))
        self.label_Image.setMaximumSize(QtCore.QSize(960, 540))
        self.label_Image.setStyleSheet("")
        self.label_Image.setText("")
        self.label_Image.setObjectName("label_Image")
        self.verticalLayout.addWidget(self.label_Image)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.checkBox_Undone = QtWidgets.QCheckBox(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.checkBox_Undone.setFont(font)
        self.checkBox_Undone.setObjectName("checkBox_Undone")
        self.horizontalLayout.addWidget(self.checkBox_Undone)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem2)
        self.pushButton_prior = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_prior.sizePolicy().hasHeightForWidth())
        self.pushButton_prior.setSizePolicy(sizePolicy)
        self.pushButton_prior.setMinimumSize(QtCore.QSize(0, 20))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.pushButton_prior.setFont(font)
        self.pushButton_prior.setFocusPolicy(QtCore.Qt.NoFocus)
        self.pushButton_prior.setStyleSheet("border-image: url(:/Icons/Resources/fast_backward_170px_1201057_easyicon.net.png);")
        self.pushButton_prior.setText("")
        self.pushButton_prior.setObjectName("pushButton_prior")
        self.horizontalLayout.addWidget(self.pushButton_prior)
        spacerItem3 = QtWidgets.QSpacerItem(10, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem3)
        self.lineEdit_present_page = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_present_page.setMaximumSize(QtCore.QSize(60, 16777215))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.lineEdit_present_page.setFont(font)
        self.lineEdit_present_page.setStyleSheet("border:none;\n"
"background:transparent;\n"
"border-style:solid;\n"
"border-bottom-width:1px;\n"
"border-top-width:0px;\n"
"border-right-width:0px;\n"
"border-left-width:0px;")
        self.lineEdit_present_page.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_present_page.setObjectName("lineEdit_present_page")
        self.horizontalLayout.addWidget(self.lineEdit_present_page)
        self.label_page = QtWidgets.QLabel(self.centralwidget)
        self.label_page.setMinimumSize(QtCore.QSize(80, 0))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_page.setFont(font)
        self.label_page.setText("")
        self.label_page.setAlignment(QtCore.Qt.AlignCenter)
        self.label_page.setObjectName("label_page")
        self.horizontalLayout.addWidget(self.label_page)
        spacerItem4 = QtWidgets.QSpacerItem(10, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem4)
        self.pushButton_next = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_next.sizePolicy().hasHeightForWidth())
        self.pushButton_next.setSizePolicy(sizePolicy)
        self.pushButton_next.setMinimumSize(QtCore.QSize(0, 20))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.pushButton_next.setFont(font)
        self.pushButton_next.setFocusPolicy(QtCore.Qt.NoFocus)
        self.pushButton_next.setStyleSheet("border-image: url(:/Icons/Resources/fast_forward_170px_1201057_easyicon.net.png);")
        self.pushButton_next.setText("")
        self.pushButton_next.setObjectName("pushButton_next")
        self.horizontalLayout.addWidget(self.pushButton_next)
        spacerItem5 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem5)
        self.pushButton_Checked = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_Checked.sizePolicy().hasHeightForWidth())
        self.pushButton_Checked.setSizePolicy(sizePolicy)
        self.pushButton_Checked.setMinimumSize(QtCore.QSize(24, 24))
        self.pushButton_Checked.setMaximumSize(QtCore.QSize(24, 24))
        self.pushButton_Checked.setStyleSheet("border-image: url(:/Icons/Resources/ok_291px_1211043_easyicon.net.png);\n"
"")
        self.pushButton_Checked.setText("")
        self.pushButton_Checked.setObjectName("pushButton_Checked")
        self.horizontalLayout.addWidget(self.pushButton_Checked)
        spacerItem6 = QtWidgets.QSpacerItem(200, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem6)
        self.pushButton_refresh = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_refresh.sizePolicy().hasHeightForWidth())
        self.pushButton_refresh.setSizePolicy(sizePolicy)
        self.pushButton_refresh.setMinimumSize(QtCore.QSize(24, 24))
        self.pushButton_refresh.setMaximumSize(QtCore.QSize(24, 24))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.pushButton_refresh.setFont(font)
        self.pushButton_refresh.setFocusPolicy(QtCore.Qt.NoFocus)
        self.pushButton_refresh.setStyleSheet("border-image: url(:/Icons/Resources/refresh_556px_1236470_easyicon.net.png);")
        self.pushButton_refresh.setText("")
        self.pushButton_refresh.setObjectName("pushButton_refresh")
        self.horizontalLayout.addWidget(self.pushButton_refresh)
        self.label_isCheckImage = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_isCheckImage.sizePolicy().hasHeightForWidth())
        self.label_isCheckImage.setSizePolicy(sizePolicy)
        self.label_isCheckImage.setMinimumSize(QtCore.QSize(20, 20))
        self.label_isCheckImage.setMaximumSize(QtCore.QSize(20, 20))
        self.label_isCheckImage.setStyleSheet("border-image: url(:/Icons/Resources/close_600px_1181428_easyicon.net.png);")
        self.label_isCheckImage.setText("")
        self.label_isCheckImage.setObjectName("label_isCheckImage")
        self.horizontalLayout.addWidget(self.label_isCheckImage)
        self.label_isCheck = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_isCheck.setFont(font)
        self.label_isCheck.setObjectName("label_isCheck")
        self.horizontalLayout.addWidget(self.label_isCheck)
        spacerItem7 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem7)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.label_tl_2 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_tl_2.setFont(font)
        self.label_tl_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_tl_2.setObjectName("label_tl_2")
        self.gridLayout.addWidget(self.label_tl_2, 1, 6, 1, 1)
        self.spinBox_br_x = QtWidgets.QSpinBox(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.spinBox_br_x.setFont(font)
        self.spinBox_br_x.setStyleSheet("border:none;\n"
"background:transparent;")
        self.spinBox_br_x.setAlignment(QtCore.Qt.AlignCenter)
        self.spinBox_br_x.setMinimum(0)
        self.spinBox_br_x.setMaximum(1920)
        self.spinBox_br_x.setProperty("value", 0)
        self.spinBox_br_x.setObjectName("spinBox_br_x")
        self.gridLayout.addWidget(self.spinBox_br_x, 1, 3, 1, 1)
        self.label_tl = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_tl.setFont(font)
        self.label_tl.setAlignment(QtCore.Qt.AlignCenter)
        self.label_tl.setObjectName("label_tl")
        self.gridLayout.addWidget(self.label_tl, 0, 1, 1, 1)
        self.label_path_image = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_path_image.sizePolicy().hasHeightForWidth())
        self.label_path_image.setSizePolicy(sizePolicy)
        self.label_path_image.setMinimumSize(QtCore.QSize(300, 0))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.label_path_image.setFont(font)
        self.label_path_image.setStyleSheet("background:transparent;\n"
"border-style:solid;\n"
"border-bottom-width:1px;\n"
"border-top-width:0px;\n"
"border-right-width:0px;\n"
"border-left-width:0px;")
        self.label_path_image.setText("")
        self.label_path_image.setObjectName("label_path_image")
        self.gridLayout.addWidget(self.label_path_image, 0, 12, 1, 1)
        self.spinBox_tl_y = QtWidgets.QSpinBox(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.spinBox_tl_y.setFont(font)
        self.spinBox_tl_y.setStyleSheet("border:none;\n"
"background:transparent;")
        self.spinBox_tl_y.setAlignment(QtCore.Qt.AlignCenter)
        self.spinBox_tl_y.setMinimum(0)
        self.spinBox_tl_y.setMaximum(1080)
        self.spinBox_tl_y.setProperty("value", 0)
        self.spinBox_tl_y.setObjectName("spinBox_tl_y")
        self.gridLayout.addWidget(self.spinBox_tl_y, 0, 7, 1, 1)
        self.label_tl_3 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_tl_3.setFont(font)
        self.label_tl_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_tl_3.setObjectName("label_tl_3")
        self.gridLayout.addWidget(self.label_tl_3, 0, 2, 1, 1)
        spacerItem8 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem8, 1, 15, 1, 1)
        spacerItem9 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem9, 1, 0, 1, 1)
        self.label_tl_4 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_tl_4.setFont(font)
        self.label_tl_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_tl_4.setObjectName("label_tl_4")
        self.gridLayout.addWidget(self.label_tl_4, 0, 6, 1, 1)
        self.label_tl_9 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_tl_9.setFont(font)
        self.label_tl_9.setWhatsThis("")
        self.label_tl_9.setAlignment(QtCore.Qt.AlignCenter)
        self.label_tl_9.setObjectName("label_tl_9")
        self.gridLayout.addWidget(self.label_tl_9, 1, 11, 1, 1)
        self.label_br = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_br.setFont(font)
        self.label_br.setAlignment(QtCore.Qt.AlignCenter)
        self.label_br.setObjectName("label_br")
        self.gridLayout.addWidget(self.label_br, 1, 1, 1, 1)
        self.spinBox_tl_x = QtWidgets.QSpinBox(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.spinBox_tl_x.setFont(font)
        self.spinBox_tl_x.setStyleSheet("border:none;\n"
"background:transparent;")
        self.spinBox_tl_x.setAlignment(QtCore.Qt.AlignCenter)
        self.spinBox_tl_x.setMinimum(0)
        self.spinBox_tl_x.setMaximum(1920)
        self.spinBox_tl_x.setProperty("value", 0)
        self.spinBox_tl_x.setObjectName("spinBox_tl_x")
        self.gridLayout.addWidget(self.spinBox_tl_x, 0, 3, 1, 1)
        self.label_img = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_img.sizePolicy().hasHeightForWidth())
        self.label_img.setSizePolicy(sizePolicy)
        self.label_img.setMinimumSize(QtCore.QSize(20, 20))
        self.label_img.setMaximumSize(QtCore.QSize(20, 20))
        self.label_img.setStyleSheet("border-image: url(:/Icons/Resources/hourglass_155px_1201105_easyicon.net.png);")
        self.label_img.setText("")
        self.label_img.setObjectName("label_img")
        self.gridLayout.addWidget(self.label_img, 0, 9, 1, 1)
        self.label_path_annotation = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_path_annotation.sizePolicy().hasHeightForWidth())
        self.label_path_annotation.setSizePolicy(sizePolicy)
        self.label_path_annotation.setMinimumSize(QtCore.QSize(300, 0))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.label_path_annotation.setFont(font)
        self.label_path_annotation.setStyleSheet("background:transparent;\n"
"border-style:solid;\n"
"border-bottom-width:1px;\n"
"border-top-width:0px;\n"
"border-right-width:0px;\n"
"border-left-width:0px;")
        self.label_path_annotation.setText("")
        self.label_path_annotation.setObjectName("label_path_annotation")
        self.gridLayout.addWidget(self.label_path_annotation, 1, 12, 1, 1)
        spacerItem10 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem10, 1, 8, 1, 1)
        self.pushButton_Image = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_Image.sizePolicy().hasHeightForWidth())
        self.pushButton_Image.setSizePolicy(sizePolicy)
        self.pushButton_Image.setMinimumSize(QtCore.QSize(24, 24))
        self.pushButton_Image.setMaximumSize(QtCore.QSize(24, 24))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.pushButton_Image.setFont(font)
        self.pushButton_Image.setFocusPolicy(QtCore.Qt.NoFocus)
        self.pushButton_Image.setStyleSheet("border-image: url(:/Icons/Resources/file_587px_1293445_easyicon.net.png);")
        self.pushButton_Image.setText("")
        self.pushButton_Image.setObjectName("pushButton_Image")
        self.gridLayout.addWidget(self.pushButton_Image, 0, 13, 1, 1)
        self.label_ant = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_ant.sizePolicy().hasHeightForWidth())
        self.label_ant.setSizePolicy(sizePolicy)
        self.label_ant.setMinimumSize(QtCore.QSize(20, 20))
        self.label_ant.setMaximumSize(QtCore.QSize(20, 20))
        self.label_ant.setStyleSheet("border-image: url(:/Icons/Resources/hourglass_155px_1201105_easyicon.net.png);")
        self.label_ant.setText("")
        self.label_ant.setObjectName("label_ant")
        self.gridLayout.addWidget(self.label_ant, 1, 9, 1, 1)
        self.label_tl_5 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_tl_5.setFont(font)
        self.label_tl_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_tl_5.setObjectName("label_tl_5")
        self.gridLayout.addWidget(self.label_tl_5, 1, 2, 1, 1)
        self.pushButton_Labelling = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_Labelling.sizePolicy().hasHeightForWidth())
        self.pushButton_Labelling.setSizePolicy(sizePolicy)
        self.pushButton_Labelling.setMinimumSize(QtCore.QSize(24, 24))
        self.pushButton_Labelling.setMaximumSize(QtCore.QSize(24, 24))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.pushButton_Labelling.setFont(font)
        self.pushButton_Labelling.setFocusPolicy(QtCore.Qt.NoFocus)
        self.pushButton_Labelling.setStyleSheet("border-image: url(:/Icons/Resources/file_587px_1293445_easyicon.net.png);")
        self.pushButton_Labelling.setText("")
        self.pushButton_Labelling.setObjectName("pushButton_Labelling")
        self.gridLayout.addWidget(self.pushButton_Labelling, 1, 13, 1, 1)
        self.spinBox_br_y = QtWidgets.QSpinBox(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.spinBox_br_y.setFont(font)
        self.spinBox_br_y.setStyleSheet("border:none;\n"
"background:transparent;")
        self.spinBox_br_y.setAlignment(QtCore.Qt.AlignCenter)
        self.spinBox_br_y.setMinimum(0)
        self.spinBox_br_y.setMaximum(1080)
        self.spinBox_br_y.setProperty("value", 0)
        self.spinBox_br_y.setObjectName("spinBox_br_y")
        self.gridLayout.addWidget(self.spinBox_br_y, 1, 7, 1, 1)
        self.label_tl_8 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_tl_8.setFont(font)
        self.label_tl_8.setWhatsThis("")
        self.label_tl_8.setAlignment(QtCore.Qt.AlignCenter)
        self.label_tl_8.setObjectName("label_tl_8")
        self.gridLayout.addWidget(self.label_tl_8, 0, 11, 1, 1)
        spacerItem11 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem11, 1, 5, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout)
        self.horizontalLayout_3.addLayout(self.verticalLayout)
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.horizontalLayout_3.addWidget(self.line)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label_Label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_Label.setFont(font)
        self.label_Label.setWhatsThis("")
        self.label_Label.setAlignment(QtCore.Qt.AlignCenter)
        self.label_Label.setObjectName("label_Label")
        self.verticalLayout_2.addWidget(self.label_Label)
        self.tableWidget = QtWidgets.QTableWidget(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.tableWidget.setFont(font)
        self.tableWidget.setStyleSheet("")
        self.tableWidget.setRowCount(0)
        self.tableWidget.setColumnCount(3)
        self.tableWidget.setObjectName("tableWidget")
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        self.tableWidget.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        self.tableWidget.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        self.tableWidget.setHorizontalHeaderItem(2, item)
        self.verticalLayout_2.addWidget(self.tableWidget)
        self.label_progressbar = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_progressbar.setFont(font)
        self.label_progressbar.setObjectName("label_progressbar")
        self.verticalLayout_2.addWidget(self.label_progressbar)
        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.verticalLayout_2.addWidget(self.progressBar)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        spacerItem12 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem12)
        self.pushButton_addLabel = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.pushButton_addLabel.setFont(font)
        self.pushButton_addLabel.setObjectName("pushButton_addLabel")
        self.horizontalLayout_2.addWidget(self.pushButton_addLabel)
        spacerItem13 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem13)
        self.pushButton_delLabel = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.pushButton_delLabel.setFont(font)
        self.pushButton_delLabel.setObjectName("pushButton_delLabel")
        self.horizontalLayout_2.addWidget(self.pushButton_delLabel)
        spacerItem14 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem14)
        self.pushButton_ClearLabel = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.pushButton_ClearLabel.setFont(font)
        self.pushButton_ClearLabel.setObjectName("pushButton_ClearLabel")
        self.horizontalLayout_2.addWidget(self.pushButton_ClearLabel)
        spacerItem15 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem15)
        self.verticalLayout_2.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_3.addLayout(self.verticalLayout_2)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1323, 23))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.toolBar = QtWidgets.QToolBar(MainWindow)
        self.toolBar.setObjectName("toolBar")
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)
        self.action_A = QtWidgets.QAction(MainWindow)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/Icons/Resources/backward_page_200px_1189181_easyicon.net.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_A.setIcon(icon)
        self.action_A.setObjectName("action_A")
        self.action_D = QtWidgets.QAction(MainWindow)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/Icons/Resources/forward_page_200px_1189511_easyicon.net.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_D.setIcon(icon1)
        self.action_D.setObjectName("action_D")
        self.action_E = QtWidgets.QAction(MainWindow)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/Icons/Resources/delete_201px_1189401_easyicon.net.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_E.setIcon(icon2)
        self.action_E.setObjectName("action_E")
        self.action_Q = QtWidgets.QAction(MainWindow)
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(":/Icons/Resources/close_256px_1159854_easyicon.net.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_Q.setIcon(icon3)
        self.action_Q.setObjectName("action_Q")
        self.toolBar.addAction(self.action_A)
        self.toolBar.addAction(self.action_D)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.action_E)
        self.toolBar.addAction(self.action_Q)

        self.retranslateUi(MainWindow)
        self.action_Q.triggered.connect(MainWindow.close)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        MainWindow.setTabOrder(self.lineEdit_present_page, self.pushButton_next)
        MainWindow.setTabOrder(self.pushButton_next, self.pushButton_prior)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.checkBox_Undone.setText(_translate("MainWindow", "仅显示未完成"))
        self.label_isCheck.setText(_translate("MainWindow", "未标注"))
        self.label_tl_2.setText(_translate("MainWindow", "y"))
        self.label_tl.setText(_translate("MainWindow", "左上角点："))
        self.label_path_image.setToolTip(_translate("MainWindow", "选择待查验图像 .jpg 目录"))
        self.label_tl_3.setText(_translate("MainWindow", "x"))
        self.label_tl_4.setText(_translate("MainWindow", "y"))
        self.label_tl_9.setToolTip(_translate("MainWindow", "选择待查验图像标注 .xml 目录"))
        self.label_tl_9.setText(_translate("MainWindow", "标注目录："))
        self.label_br.setText(_translate("MainWindow", "右下角点："))
        self.label_path_annotation.setToolTip(_translate("MainWindow", "选择待查验图像标注 .xml 目录"))
        self.label_tl_5.setText(_translate("MainWindow", "x"))
        self.label_tl_8.setToolTip(_translate("MainWindow", "选择待查验图像 .jpg 目录"))
        self.label_tl_8.setText(_translate("MainWindow", "图像目录："))
        self.label_Label.setToolTip(_translate("MainWindow", "选择待查验图像 .jpg 目录"))
        self.label_Label.setText(_translate("MainWindow", "标签及统计信息"))
        item = self.tableWidget.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "标签"))
        item = self.tableWidget.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "计数"))
        item = self.tableWidget.horizontalHeaderItem(2)
        item.setText(_translate("MainWindow", "颜色"))
        self.label_progressbar.setText(_translate("MainWindow", "载入状态："))
        self.pushButton_addLabel.setText(_translate("MainWindow", "添加标签"))
        self.pushButton_delLabel.setText(_translate("MainWindow", "删除标签"))
        self.pushButton_ClearLabel.setText(_translate("MainWindow", "清空标签"))
        self.toolBar.setWindowTitle(_translate("MainWindow", "toolBar"))
        self.action_A.setText(_translate("MainWindow", "上一张(←)"))
        self.action_A.setToolTip(_translate("MainWindow", "上一张(←)"))
        self.action_A.setShortcut(_translate("MainWindow", "Left"))
        self.action_D.setText(_translate("MainWindow", "下一张(→)"))
        self.action_D.setToolTip(_translate("MainWindow", "下一张(→)"))
        self.action_D.setShortcut(_translate("MainWindow", "Right"))
        self.action_E.setText(_translate("MainWindow", "删除(E)"))
        self.action_E.setToolTip(_translate("MainWindow", "删除(E)"))
        self.action_E.setShortcut(_translate("MainWindow", "Ctrl+E"))
        self.action_Q.setText(_translate("MainWindow", "退出(Q)"))
        self.action_Q.setShortcut(_translate("MainWindow", "Ctrl+Q"))
from imagelabel import ImageLabel
import Labelling_rc
