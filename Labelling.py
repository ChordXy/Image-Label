# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'e:\Github\Image-Labelling\Labelling.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(978, 767)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_Image = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_Image.sizePolicy().hasHeightForWidth())
        self.label_Image.setSizePolicy(sizePolicy)
        self.label_Image.setMinimumSize(QtCore.QSize(960, 540))
        self.label_Image.setMaximumSize(QtCore.QSize(960, 540))
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
        self.label_tl_7 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_tl_7.setFont(font)
        self.label_tl_7.setAlignment(QtCore.Qt.AlignCenter)
        self.label_tl_7.setObjectName("label_tl_7")
        self.gridLayout.addWidget(self.label_tl_7, 2, 6, 1, 1)
        self.label_br = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_br.setFont(font)
        self.label_br.setAlignment(QtCore.Qt.AlignCenter)
        self.label_br.setObjectName("label_br")
        self.gridLayout.addWidget(self.label_br, 1, 1, 1, 1)
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
        spacerItem8 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem8, 1, 15, 1, 1)
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
        self.label_w = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_w.setFont(font)
        self.label_w.setText("")
        self.label_w.setAlignment(QtCore.Qt.AlignCenter)
        self.label_w.setObjectName("label_w")
        self.gridLayout.addWidget(self.label_w, 2, 3, 1, 1)
        self.label_tl_9 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_tl_9.setFont(font)
        self.label_tl_9.setWhatsThis("")
        self.label_tl_9.setAlignment(QtCore.Qt.AlignCenter)
        self.label_tl_9.setObjectName("label_tl_9")
        self.gridLayout.addWidget(self.label_tl_9, 1, 11, 1, 1)
        self.label_tl_6 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_tl_6.setFont(font)
        self.label_tl_6.setAlignment(QtCore.Qt.AlignCenter)
        self.label_tl_6.setObjectName("label_tl_6")
        self.gridLayout.addWidget(self.label_tl_6, 2, 2, 1, 1)
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
        self.label_tl_3 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_tl_3.setFont(font)
        self.label_tl_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_tl_3.setObjectName("label_tl_3")
        self.gridLayout.addWidget(self.label_tl_3, 0, 2, 1, 1)
        self.label_tl_4 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_tl_4.setFont(font)
        self.label_tl_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_tl_4.setObjectName("label_tl_4")
        self.gridLayout.addWidget(self.label_tl_4, 0, 6, 1, 1)
        self.pushButton_color = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_color.sizePolicy().hasHeightForWidth())
        self.pushButton_color.setSizePolicy(sizePolicy)
        self.pushButton_color.setMinimumSize(QtCore.QSize(24, 24))
        self.pushButton_color.setMaximumSize(QtCore.QSize(24, 24))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.pushButton_color.setFont(font)
        self.pushButton_color.setFocusPolicy(QtCore.Qt.NoFocus)
        self.pushButton_color.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.pushButton_color.setStyleSheet("border-image: url(:/Icons/Resources/color_palette_1876px_1202956_easyicon.net.png);")
        self.pushButton_color.setText("")
        self.pushButton_color.setObjectName("pushButton_color")
        self.gridLayout.addWidget(self.pushButton_color, 2, 0, 1, 1)
        self.label_tl_2 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_tl_2.setFont(font)
        self.label_tl_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_tl_2.setObjectName("label_tl_2")
        self.gridLayout.addWidget(self.label_tl_2, 1, 6, 1, 1)
        self.label_size = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_size.setFont(font)
        self.label_size.setAlignment(QtCore.Qt.AlignCenter)
        self.label_size.setObjectName("label_size")
        self.gridLayout.addWidget(self.label_size, 2, 1, 1, 1)
        spacerItem9 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem9, 1, 5, 1, 1)
        spacerItem10 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem10, 1, 0, 1, 1)
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
        self.label_tl_8 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_tl_8.setFont(font)
        self.label_tl_8.setWhatsThis("")
        self.label_tl_8.setAlignment(QtCore.Qt.AlignCenter)
        self.label_tl_8.setObjectName("label_tl_8")
        self.gridLayout.addWidget(self.label_tl_8, 0, 11, 1, 1)
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
        self.label_h = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_h.setFont(font)
        self.label_h.setText("")
        self.label_h.setAlignment(QtCore.Qt.AlignCenter)
        self.label_h.setObjectName("label_h")
        self.gridLayout.addWidget(self.label_h, 2, 7, 1, 1)
        spacerItem11 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem11, 1, 8, 1, 1)
        self.label_tl = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_tl.setFont(font)
        self.label_tl.setAlignment(QtCore.Qt.AlignCenter)
        self.label_tl.setObjectName("label_tl")
        self.gridLayout.addWidget(self.label_tl, 0, 1, 1, 1)
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
        self.gridLayout.addWidget(self.pushButton_refresh, 0, 15, 1, 1)
        self.pushButton_ClearDirectory = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_ClearDirectory.sizePolicy().hasHeightForWidth())
        self.pushButton_ClearDirectory.setSizePolicy(sizePolicy)
        self.pushButton_ClearDirectory.setMinimumSize(QtCore.QSize(20, 24))
        self.pushButton_ClearDirectory.setMaximumSize(QtCore.QSize(20, 24))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.pushButton_ClearDirectory.setFont(font)
        self.pushButton_ClearDirectory.setFocusPolicy(QtCore.Qt.NoFocus)
        self.pushButton_ClearDirectory.setStyleSheet("border-image: url(:/Icons/Resources/waste_bin_432px_1284190_easyicon.net.png);")
        self.pushButton_ClearDirectory.setText("")
        self.pushButton_ClearDirectory.setObjectName("pushButton_ClearDirectory")
        self.gridLayout.addWidget(self.pushButton_ClearDirectory, 2, 15, 1, 1)
        self.pushButton_Generate = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_Generate.sizePolicy().hasHeightForWidth())
        self.pushButton_Generate.setSizePolicy(sizePolicy)
        self.pushButton_Generate.setMinimumSize(QtCore.QSize(24, 24))
        self.pushButton_Generate.setMaximumSize(QtCore.QSize(24, 24))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.pushButton_Generate.setFont(font)
        self.pushButton_Generate.setFocusPolicy(QtCore.Qt.NoFocus)
        self.pushButton_Generate.setStyleSheet("border-image: url(:/Icons/Resources/file_587px_1293445_easyicon.net.png);")
        self.pushButton_Generate.setText("")
        self.pushButton_Generate.setObjectName("pushButton_Generate")
        self.gridLayout.addWidget(self.pushButton_Generate, 2, 13, 1, 1)
        self.label_generate = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_generate.sizePolicy().hasHeightForWidth())
        self.label_generate.setSizePolicy(sizePolicy)
        self.label_generate.setMinimumSize(QtCore.QSize(300, 0))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.label_generate.setFont(font)
        self.label_generate.setStyleSheet("background:transparent;\n"
"border-style:solid;\n"
"border-bottom-width:1px;\n"
"border-top-width:0px;\n"
"border-right-width:0px;\n"
"border-left-width:0px;")
        self.label_generate.setText("")
        self.label_generate.setObjectName("label_generate")
        self.gridLayout.addWidget(self.label_generate, 2, 12, 1, 1)
        self.label_tl_10 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_tl_10.setFont(font)
        self.label_tl_10.setWhatsThis("")
        self.label_tl_10.setAlignment(QtCore.Qt.AlignCenter)
        self.label_tl_10.setObjectName("label_tl_10")
        self.gridLayout.addWidget(self.label_tl_10, 2, 11, 1, 1)
        self.label_gnt = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_gnt.sizePolicy().hasHeightForWidth())
        self.label_gnt.setSizePolicy(sizePolicy)
        self.label_gnt.setMinimumSize(QtCore.QSize(20, 20))
        self.label_gnt.setMaximumSize(QtCore.QSize(20, 20))
        self.label_gnt.setStyleSheet("border-image: url(:/Icons/Resources/hourglass_155px_1201105_easyicon.net.png);")
        self.label_gnt.setText("")
        self.label_gnt.setObjectName("label_gnt")
        self.gridLayout.addWidget(self.label_gnt, 2, 9, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout)
        self.verticalLayout.setStretch(0, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 978, 23))
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
        self.label_isCheck.setText(_translate("MainWindow", "未查验"))
        self.label_tl_7.setText(_translate("MainWindow", "h"))
        self.label_br.setText(_translate("MainWindow", "右下角点："))
        self.label_tl_9.setToolTip(_translate("MainWindow", "选择待查验图像标注 .xml 目录"))
        self.label_tl_9.setText(_translate("MainWindow", "标注目录："))
        self.label_tl_6.setText(_translate("MainWindow", "w"))
        self.label_path_image.setToolTip(_translate("MainWindow", "选择待查验图像 .jpg 目录"))
        self.label_tl_3.setText(_translate("MainWindow", "x"))
        self.label_tl_4.setText(_translate("MainWindow", "y"))
        self.label_tl_2.setText(_translate("MainWindow", "y"))
        self.label_size.setText(_translate("MainWindow", "边框尺寸："))
        self.label_path_annotation.setToolTip(_translate("MainWindow", "选择待查验图像标注 .xml 目录"))
        self.label_tl_8.setToolTip(_translate("MainWindow", "选择待查验图像 .jpg 目录"))
        self.label_tl_8.setText(_translate("MainWindow", "图像目录："))
        self.label_tl_5.setText(_translate("MainWindow", "x"))
        self.label_tl.setText(_translate("MainWindow", "左上角点："))
        self.label_generate.setToolTip(_translate("MainWindow", "选择待查验图像转换 .txt 目录"))
        self.label_tl_10.setToolTip(_translate("MainWindow", "选择待查验图像转换 .txt 目录"))
        self.label_tl_10.setText(_translate("MainWindow", "生成目录："))
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
import Labelling_rc
