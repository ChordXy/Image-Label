'''
Author: Cabrite
Date: 2021-01-02 00:19:29
LastEditors: Cabrite
LastEditTime: 2021-01-02 23:56:48
Description: Do not edit
'''

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import sys


class ImageLabel(QLabel):
    # function, x, y
    MidRoll = pyqtSignal(int, int, int)
    LeftClick = pyqtSignal(int, int)
    RightClick = pyqtSignal(int, int)
    LeftControlClick = pyqtSignal(int, int, int, int)
    MouseMove = pyqtSignal(int, int, int, int)

    def __init__(self, parent=None):
        super(ImageLabel, self).__init__(parent)
        self.ClickedPosition = None
        self.state = False

    def mousePressEvent(self, event):
        if event.buttons() == QtCore.Qt.LeftButton:
            if QApplication.keyboardModifiers () == QtCore.Qt.ControlModifier:
                self.ClickedPosition = (event.x(), event.y())
                self.state = True
                # print("({}, {})".format(event.x(), event.y()))
            else:
                self.LeftClick.emit(event.x(), event.y())
                # print("({}, {})".format(event.x(), event.y()))
        elif event.buttons() == QtCore.Qt.RightButton:
            self.RightClick.emit(event.x(), event.y())
            # print("({}, {})".format(event.x(), event.y()))

    def wheelEvent(self, event):
        self.MidRoll.emit((event.angleDelta() / 8).y(), event.x(), event.y())
        self.ClickedPosition = None
        self.state = False
        # print("{}, ({}, {})".format((event.angleDelta() / 8).y(), event.x(), event.y()))

    def mouseReleaseEvent(self, event):
        if self.state:
            x0, y0 = self.ClickedPosition
            if x0 < event.x():
                x1 = event.x()
                y1 = event.y()
            else:
                x1 = x0
                y1 = y0
                x0 = event.x()
                y0 = event.y()
            self.LeftControlClick.emit(x0, y0, x1, y1)
            self.ClickedPosition = None
            self.state = False
            # print("({}, {}) \t({}, {})".format(*self.ClickedPosition, event.x(), event.y()))

    def mouseMoveEvent(self, event):
        if self.state:
            x0, y0 = self.ClickedPosition
            if x0 < event.x():
                x1 = event.x()
                y1 = event.y()
            else:
                x1 = x0
                y1 = y0
                x0 = event.x()
                y0 = event.y()
            self.MouseMove.emit(x0, y0, x1, y1)
            # print("({}, {}) \t({}, {})".format(*self.ClickedPosition, event.x(), event.y()))
