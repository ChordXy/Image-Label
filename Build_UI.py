'''
Author: Cabrite
Date: 2020-10-28 21:36:51
LastEditors: Cabrite
LastEditTime: 2021-01-01 10:28:52
Description: Do not edit
'''
#-*-coding:utf-8-*-


'''
快捷键
    ←        : 上一张
    →        : 下一张
    ctrl + d : 保存并下一张
    ctrl + e : 删除当前图像
    ctrl + q : 退出程序
'''


import sys
import platform
import datetime
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from Labelling import Ui_MainWindow
import Functions
import Labelling_rc


class MyWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyWindow, self).__init__(parent)
        self.setupUi(self)
        self.setFixedSize(self.width(), self.height())
        self.setWindowFlags(Qt.WindowMinimizeButtonHint)
        self.setWindowIcon(QIcon(':/Icons/Resources/inkscape_128px_1174969_easyicon.net.ico'))
        Functions.setupUIFunctions(self)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())

