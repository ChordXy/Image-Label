from xml.dom.minidom import parse
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import numpy as np
import threading
import datetime
import platform
import time
import cv2
import os
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

class BoundingBox():
    def __init__(self, cls=0):
        self.__tlx = 0
        self.__tly = 0
        self.__brx = 0
        self.__bry = 0
        self.__width = 0
        self.__height = 0
        self.__cls = cls

    def setPoints(self, tlx, tly, brx, bry):
        self.__tlx = tlx
        self.__tly = tly
        self.__brx = brx
        self.__bry = bry
        self.__width = self.__brx - self.__tlx
        self.__height = self.__bry - self.__tly

    def setSize(self, tlx, tly, width, height):
        self.__tlx = tlx
        self.__tly = tly
        self.__width = width
        self.__height = height
        self.__brx = self.__tlx + self.__width
        self.__bry = self.__tly + self.__height

    ########################  左上角点 - x  ########################
    @property
    def tlx(self):
        return self.__tlx

    @tlx.setter
    def tlx(self, value):
        self.__tlx = value
        self.__width = self.__brx - self.__tlx

    ########################  左上角点 - y  ########################
    @property
    def tly(self):
        return self.__tly

    @tly.setter
    def tly(self, value):
        self.__tly = value
        self.__height = self.__bry - self.__tly

    ########################  右下角点 - x  ########################
    @property
    def brx(self):
        return self.__brx

    @brx.setter
    def brx(self, value):
        self.__brx = value
        self.__width = self.__brx - self.__tlx

    ########################  右下角点 - y  ########################
    @property
    def bry(self):
        return self.__bry

    @bry.setter
    def bry(self, value):
        self.__bry = value
        self.__height = self.__bry - self.__tly

    ########################  左上角 与 右下角  ########################
    @property
    def topLeft(self):
        return (self.__tlx, self.__tly)

    @property
    def bottomRight(self):
        return (self.__brx, self.__bry)

    @property
    def size(self):
        return (self.__width, self.__height)

    ########################  宽度 - w  ########################
    @property
    def width(self):
        return self.__width

    @width.setter
    def width(self, value):
        self.__width = value
        self.__brx = self.__tlx + self.__width

    ########################  高度 - h  ########################
    @property
    def height(self):
        return self.__height

    @height.setter
    def height(self, value):
        self.__height = value
        self.__bry = self.__tly + self.__height

    ########################  类别 - cls  ########################
    @property
    def cls(self):
        return self.__cls

    @cls.setter
    def cls(self, value):
        self.__cls = value

    ########################  是否合法  ########################
    @property
    def isValid(self):
        return self.__tlx > 0

class StatisticThread(QThread):
    StatisticState = pyqtSignal(int)
    StatisticResult = pyqtSignal(list)
    
    def __init__(self, parent=None):
        super(StatisticThread, self).__init__(parent)
        self.Statistics = []

    def setParameters(self, TxtsPath, totalClass, emitCount=1):
        self.TxtsPath = TxtsPath
        self.totalClass = totalClass
        self.emitCount = emitCount
        for i in range(self.totalClass):
            self.Statistics.append(0)

    def run(self):
        for index, txtfile in enumerate(self.TxtsPath):
            if not os.path.exists(txtfile):
                continue
            result = self.getTxt(txtfile)
            for res in result:
                self.Statistics[res] += 1
            if index % self.emitCount == 0:
                self.StatisticState.emit(index)
        self.StatisticResult.emit(self.Statistics)

    def getTxt(self, txtFile):
        result = []
        with open(txtFile, 'r') as file:
            lines = file.readlines()
            for line in lines:
                line = line.strip()
                if not line:
                    return 
                data = line.split(',')
                result.append(int(data[0]))
        return result

class CheckThread(QThread):
    CheckState = pyqtSignal(int)
    CheckResult = pyqtSignal(list)
    
    def __init__(self, parent=None):
        super(CheckThread, self).__init__(parent)

    def setParameters(self, Images, pathAnnotation):
        self.Images = Images
        self.pathAnnotation = pathAnnotation

    def run(self):
        UnLabelledImages = []
        for index, elem in enumerate(self.Images):
            self.CheckState.emit(index)
            if not os.path.exists(os.path.join(elem.replace('.jpg', '.txt'))):
                UnLabelledImages.append(elem)
        self.CheckResult.emit(UnLabelledImages)


class setupUIFunctions():
    def __init__(self, Window):
        self.Window = Window
        self.pathReady = [ 0, 0 ]
        self.pathImage = None
        self.pathAnnotation = None
        self.PresentPage = 1
        self.presentImage = None
        self.PresentBox = 0
        self.BBColor = [(0, 255, 255), (84, 46, 8), (18, 153, 255), (225, 105, 65), (0, 97, 255), (179, 222, 245), (42, 42, 128), (0, 252, 124), (87, 139, 46), (143, 143, 188)]
        self.Statistics = []
        self.replaceDict_N2E = dict()
        self.replaceDict_E2N = []
        self.isWordAnnotation = True
        self.ResizeRatio = []
        self.bbox = []
        self.totalClass = 2
        self.version = "图像标注 v1.2"
        self.Window.setWindowTitle(self.version)

        self.pathRemenber = 'C:'
        if platform.system() == "Linux":
            self.pathRemenber = '~/'
            
        self.setupUIFunctions()

    def setupUIFunctions(self):
        self.connectSignals2Slots()
        self.InitTable()

    def connectSignals2Slots(self):
        self.Window.pushButton_Image.clicked.connect(lambda: self.getDirectory(0))
        self.Window.pushButton_Labelling.clicked.connect(lambda: self.getDirectory(1))
        self.Window.pushButton_refresh.clicked.connect(self.refreshDirectory)

        self.Window.pushButton_prior.clicked.connect(self.priorImage)
        self.Window.pushButton_next.clicked.connect(self.nextImage)
        self.Window.pushButton_Checked.clicked.connect(self.passImage)
        self.Window.action_A.triggered.connect(self.priorImage)
        self.Window.action_D.triggered.connect(self.nextImage)
        self.Window.action_E.triggered.connect(self.deleteImage)

        self.Window.spinBox_tl_x.valueChanged.connect(self.changetlx)
        self.Window.spinBox_tl_y.valueChanged.connect(self.changetly)
        self.Window.spinBox_br_x.valueChanged.connect(self.changebrx)
        self.Window.spinBox_br_y.valueChanged.connect(self.changebry)

        self.Window.lineEdit_present_page.editingFinished.connect(self.JumpPages)
        self.Window.checkBox_Undone.stateChanged.connect(self.DisplaySwitch)

        QShortcut(QKeySequence(self.Window.tr("Ctrl+d")), self.Window, self.passImage)

        self.Window.pushButton_addLabel.clicked.connect(self.addNewLabel)
        self.Window.pushButton_delLabel.clicked.connect(self.delCurrentLabel)
        self.Window.pushButton_ClearLabel.clicked.connect(self.delAllLabels)

    ####################################################################################
    #                                   路径相关                                       #
    ####################################################################################

    def getDirectory(self, rtype):
        info = ["请选择标注图像 jpg 路径", "请选择标注文件 txt 路径"]
        dirt = QFileDialog.getExistingDirectory(None, info[rtype], self.pathRemenber)
        if dirt == "":
            return
        if rtype == 0:
            self.pathImage = dirt
            self.checkImageDirectory()
        if rtype == 1:
            self.pathAnnotation = dirt
            self.checkAnnotationDirectory()
            
        self.pathRemenber = os.path.dirname(dirt)
        if sum(self.pathReady) == 2:
            self.analyseData()
            #- 图像操作
            self.Window.label_Image.MidRoll.connect(self.ChangeCurrentLabel)
            self.Window.label_Image.LeftClick.connect(self.SelectAnnotation)
            self.Window.label_Image.RightClick.connect(self.DeleteAnnotation)
            self.Window.label_Image.LeftControlClick.connect(self.Annotate)
            self.Window.label_Image.MouseMove.connect(self.DynamicAnnotate)

    def checkImageDirectory(self):
        if os.listdir(self.pathImage):
            self.setRightPath(0)
            self.pathReady[0] = 1
        else:
            self.pathReady[0] = 0
            self.setWrongPath(0)
            self.pathImage = ""
            return
        
        self.Window.label_path_image.setText(self.pathImage)
        self.Images = [elem for elem in os.listdir(self.pathImage) if elem.endswith('.jpg')]
        self.ImageOrders = [elem.replace('.jpg', '') for elem in self.Images]

    def checkAnnotationDirectory(self):
        self.setRightPath(1)
        self.pathReady[1] = 1
        self.Window.label_path_annotation.setText(self.pathAnnotation)
        self.Annotations = [elem for elem in os.listdir(self.pathAnnotation) if elem.endswith('.txt')]

    def refreshDirectory(self):
        if sum(self.pathReady) == 2:
            self.Images = [elem for elem in os.listdir(self.pathImage) if elem.endswith('.jpg')]
            self.analyseData()
        else:
            return

    def setWrongPath(self, pos):
        if pos == 0:
            icon = self.Window.label_img
            label = self.Window.label_path_image
        if pos == 1:
            icon = self.Window.label_ant
            label = self.Window.label_path_annotation

        icon.setStyleSheet("border-image: url(:/Icons/Resources/close_600px_1181428_easyicon.net.png);")
        label.setStyleSheet(    "background-color: rgba(255, 0, 0, 100);\n"
                                "border-style:solid;\n"
                                "border-bottom-width:1px;\n"
                                "border-top-width:0px;\n"
                                "border-right-width:0px;\n"
                                "border-left-width:0px;") 

    def setRightPath(self, pos):
        if pos == 0:
            icon = self.Window.label_img
            label = self.Window.label_path_image
        if pos == 1:
            icon = self.Window.label_ant
            label = self.Window.label_path_annotation
            
        icon.setStyleSheet("border-image: url(:/Icons/Resources/yes_600px_1181432_easyicon.net.png);")
        label.setStyleSheet("background:transparent;\n"
                                "border-style:solid;\n"
                                "border-bottom-width:1px;\n"
                                "border-top-width:0px;\n"
                                "border-right-width:0px;\n"
                                "border-left-width:0px;") 

    def setWaitingPath(self):
        self.Window.label_img.setStyleSheet("border-image: url(:/Icons/Resources/hourglass_155px_1201105_easyicon.net.png);")
        self.Window.label_ant.setStyleSheet("border-image: url(:/Icons/Resources/hourglass_155px_1201105_easyicon.net.png);")
        self.Window.label_path_image.setStyleSheet("background:transparent;\n"
                                "border-style:solid;\n"
                                "border-bottom-width:1px;\n"
                                "border-top-width:0px;\n"
                                "border-right-width:0px;\n"
                                "border-left-width:0px;") 
        self.Window.label_path_annotation.setStyleSheet("background:transparent;\n"
                                "border-style:solid;\n"
                                "border-bottom-width:1px;\n"
                                "border-top-width:0px;\n"
                                "border-right-width:0px;\n"
                                "border-left-width:0px;") 

    def CheckIsAllDone(self):
        txt_orders = [elem.replace('.txt', '') for elem in os.listdir(self.pathAnnotation) if elem.endswith('.txt')]
        if txt_orders == self.ImageOrders:
            return True
        else:
            return False

    ####################################################################################
    #                                   读入图像                                       #
    ####################################################################################

    def statisticLabels(self):
        if not self.TxtsPath:
            self.Window.progressBar.setValue(100)
        else:
            self.Window.label_progressbar.setText("载入状态：读取已标记标签信息...")
            self.MyDetectThread = StatisticThread()
            self.MyDetectThread.setParameters(self.TxtsPath, self.totalClass, 1000)
            self.MyDetectThread.StatisticState.connect(self.displayProgress)
            self.MyDetectThread.StatisticResult.connect(self.displayStatisticResult)
            self.MyDetectThread.start()

    def displayProgress(self, index):
        progress = int(100 * (index / len(self.TxtsPath)))
        self.Window.progressBar.setValue(progress)
    
    def displayStatisticResult(self, stat):
        self.Window.progressBar.setValue(100)
        self.Statistics = stat
        for i in range(self.totalClass):
            Count = QTableWidgetItem(str(self.Statistics[i]))
            Count.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
            self.Window.tableWidget.setItem(i, 1, Count)
        self.locateFirstUnlabelled()

    def locateFirstUnlabelled(self):
        if not self.TxtsPath:
            self.Window.progressBar.setValue(100)
        else:
            self.Window.label_progressbar.setText("载入状态：读取未标记标签信息...")
            self.MyCheckThread = CheckThread()
            self.MyCheckThread.setParameters(self.Images, self.pathAnnotation)
            self.MyCheckThread.CheckResult.connect(self.CheckUnlabelledResult)
            self.MyCheckThread.CheckState.connect(self.CheckUnlabelledState)
            self.MyCheckThread.start()

    def CheckUnlabelledState(self, index):
        progress = int(100 * (index / len(self.Images)))
        self.Window.progressBar.setValue(progress)

    def CheckUnlabelledResult(self, UnLabelledImages):
        self.Window.progressBar.setValue(100)
        if UnLabelledImages:
            self.PresentPage = self.Images.index(UnLabelledImages[0].replace('.txt', '.jpg')) + 1
        else:
            self.PresentPage = 1
        
        self.Window.label_progressbar.setText("载入状态：完成！")
        self.setTitle()
        self.refreshPages()
        self.Window.lineEdit_present_page.setText("{}".format(self.PresentPage))
        self.showImage()

    def analyseData(self):
        self.Txts = [elem.replace('.jpg', '.txt') for elem in self.Images]
        self.ImagesPath = [os.path.join(self.pathImage, elem) for elem in self.Images]
        self.TxtsPath = [os.path.join(self.pathAnnotation, elem) for elem in self.Txts]
        self.TotalImages = len(self.Images)

        self.statisticLabels()

        if self.CheckIsAllDone():
            self.Window.checkBox_Undone.setCheckable(False)
        else:
            self.Window.checkBox_Undone.setCheckable(True)

    def setTitle(self):
        self.Window.setWindowTitle(self.version + "  -  " + self.Images[self.PresentPage - 1])  

    def refreshPages(self):
        self.Window.label_page.setText("({}/{})".format(self.PresentPage, self.TotalImages))

    ####################################################################################
    #                                   图像显示                                       #
    ####################################################################################

    def getXml(self, xmlFile):
        result = []
        domTree = ET.ElementTree(file=xmlFile)
        rootNode = domTree.getroot()
        objects = rootNode.findall('object')
        if not objects:
            return result

        for obj in objects:
            BBox = obj.find('bndbox')
            target = ['xmin', 'ymin', 'xmax', 'ymax']
            data = [BBox.find(elem).text for elem in target]
            if 'NaN' in data:
                continue
            int_data = [int(elem) for elem in data]

            temp = BoundingBox(self.replaceDict_N2E[obj.find('name').text])
            temp.setPoints(*int_data)
            result.append(temp)
        return result

    def getTxt(self, txtFile):
        result = []
        with open(txtFile, 'r') as file:
            lines = file.readlines()
            for line in lines:
                line = line.strip()
                if not line:
                    return 
                data = line.split(',')
                temp = BoundingBox(int(data[0]))
                temp.setSize(int(data[1]), int(data[2]), int(data[3]), int(data[4]))
                result.append(temp)
        return result

    def annotateImage(self, image, refine=False, ExtraBndBox=None):
        if not refine and os.path.exists(self.TxtsPath[self.PresentPage - 1]):
            self.bbox = self.getTxt(self.TxtsPath[self.PresentPage - 1])

        if ExtraBndBox:
            cv2.rectangle(image, ExtraBndBox.topLeft, ExtraBndBox.bottomRight, self.BBColor[ExtraBndBox.cls], 1)
            if self.isWordAnnotation:
                image = cv2.putText(image, self.replaceDict_E2N[ExtraBndBox.cls], ExtraBndBox.topLeft, cv2.FONT_HERSHEY_TRIPLEX, 0.5, self.BBColor[ExtraBndBox.cls], 1)

        if not self.bbox:
            return image

        for bndbox in self.bbox:
            if not bndbox.isValid:
                return image
        for index, bndbox in enumerate(self.bbox):
            if index == self.PresentBox:
                thickness = 4
            else:
                thickness = 2
            cv2.rectangle(image, bndbox.topLeft, bndbox.bottomRight, self.BBColor[bndbox.cls], thickness)
            if self.isWordAnnotation:
                image = cv2.putText(image, self.replaceDict_E2N[bndbox.cls], bndbox.topLeft, cv2.FONT_HERSHEY_TRIPLEX, 0.5, self.BBColor[bndbox.cls], thickness)

        return image

    def ProcessImage(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_cvt = QImage(image_rgb[:], image_rgb.shape[1], image_rgb.shape[0], image_rgb.shape[1] * 3, QImage.Format_RGB888)
        image_show = QPixmap(image_cvt).scaled(960, 540)
        return image_show

    def refreshInfo(self):
        if self.bbox:
            self.Window.spinBox_tl_x.setValue(self.bbox[self.PresentBox].tlx)
            self.Window.spinBox_tl_y.setValue(self.bbox[self.PresentBox].tly)
            self.Window.spinBox_br_x.setValue(self.bbox[self.PresentBox].brx)
            self.Window.spinBox_br_y.setValue(self.bbox[self.PresentBox].bry)
        else:
            self.Window.spinBox_tl_x.valueChanged.disconnect(self.changetlx)
            self.Window.spinBox_tl_y.valueChanged.disconnect(self.changetly)
            self.Window.spinBox_br_x.valueChanged.disconnect(self.changebrx)
            self.Window.spinBox_br_y.valueChanged.disconnect(self.changebry)
            self.Window.spinBox_tl_x.setValue(0)
            self.Window.spinBox_tl_y.setValue(0)
            self.Window.spinBox_br_x.setValue(0)
            self.Window.spinBox_br_y.setValue(0)
            self.Window.spinBox_tl_x.valueChanged.connect(self.changetlx)
            self.Window.spinBox_tl_y.valueChanged.connect(self.changetly)
            self.Window.spinBox_br_x.valueChanged.connect(self.changebrx)
            self.Window.spinBox_br_y.valueChanged.connect(self.changebry)

    def isCheckedImage(self):
        if os.path.exists(self.TxtsPath[self.PresentPage - 1]):
            return True
        else:
            return False

    def changeLabelState(self):
        if self.isCheckedImage():
            self.Window.label_isCheck.setText('已标注')
            self.Window.label_isCheckImage.setStyleSheet("border-image: url(:/Icons/Resources/yes_600px_1181432_easyicon.net.png);")
        else:
            self.Window.label_isCheck.setText('未标注')
            self.Window.label_isCheckImage.setStyleSheet("border-image: url(:/Icons/Resources/close_600px_1181428_easyicon.net.png);")

    def showImage(self):
        self.presentImage = cv2.imdecode(np.fromfile(self.ImagesPath[self.PresentPage - 1], dtype=np.uint8), cv2.IMREAD_COLOR)
        self.ResizeRatio = [self.presentImage.shape[1] / 960, self.presentImage.shape[0] / 540]
        image_show = self.ProcessImage(self.annotateImage(self.presentImage.copy()))
        self.Window.label_Image.setPixmap(QPixmap(image_show))
        self.refreshInfo()
        self.changeLabelState()

    ####################################################################################
    #                                   功能操作                                       #
    ####################################################################################

    def priorImage(self):
        if self.PresentPage == 1:
            return
        else:
            self.PresentBox = 0
            self.bbox = []
            self.PresentPage -= 1
            self.setTitle()
            self.refreshPages()
            self.Window.lineEdit_present_page.setText("{}".format(self.PresentPage))
            self.showImage()

    def nextImage(self):
        if self.PresentPage == self.TotalImages:
            return
        else:
            self.PresentBox = 0
            self.bbox = []
            self.PresentPage += 1
            self.setTitle()
            self.refreshPages()
            self.Window.lineEdit_present_page.setText("{}".format(self.PresentPage))
            self.showImage()

    def passImage(self):
        # 保存
        self.saveFile()
        # 统计信息
        for i in range(self.totalClass):
            Count = QTableWidgetItem(str(self.Statistics[i]))
            Count.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
            self.Window.tableWidget.setItem(i, 1, Count)

        if self.PresentPage == self.TotalImages:
            self.changeLabelState()
            return
        else:
            self.PresentBox = 0
            self.bbox = []
            self.PresentPage += 1
            self.setTitle()
            self.refreshPages()
            self.Window.lineEdit_present_page.setText("{}".format(self.PresentPage))
            self.showImage()

    def saveFile(self):
        sentence = ""
        for bndbox in self.bbox:
            if bndbox.isValid:
                sentence += "{},{},{},{},{}\n".format(bndbox.cls, bndbox.tlx, bndbox.tly, bndbox.width, bndbox.height)
            else:
                sentence += ""
        with open(self.TxtsPath[self.PresentPage - 1], 'w+') as file:
            file.write(sentence)

    def deleteImage(self):
        del self.Images[self.PresentPage - 1]
        del self.Txts[self.PresentPage - 1]
        self.TotalImages = len(self.Images)
        self.PresentBox = 0
        delImage = self.ImagesPath.pop(self.PresentPage - 1)
        delTxt = self.TxtsPath.pop(self.PresentPage - 1)

        os.remove(delImage)
        if os.path.exists(delTxt):
            os.remove(delTxt)
        self.refreshPages()
        self.setTitle()
        self.Window.lineEdit_present_page.setText("{}".format(self.PresentPage))
        self.showImage()

    def DisplaySwitch(self, state):
        if self.CheckIsAllDone():
            return
        if state == Qt.Checked:
            self.ChangeToUnlabelled()
        if state == Qt.Unchecked:
            self.ChangeToAll()
        self.analyseData()
        self.PresentBox = 0
        
    def JumpPages(self):
        page = int(self.Window.lineEdit_present_page.text())
        if page > self.TotalImages:
            page = self.TotalImages
        elif page < 1:
            page = 1
        self.PresentPage = page
        self.refreshPages()
        self.Window.lineEdit_present_page.setText("{}".format(self.PresentPage))
        self.showImage()
        self.setTitle()
        self.PresentBox = 0

    def ChangeToAll(self):
        self.Images = self.BackupImages

    def ChangeToUnlabelled(self):
        labelledImages = os.listdir(self.pathAnnotation)
        UnLabelledImages = [elem for elem in self.Images if elem.replace('.jpg', '.txt') not in labelledImages]
        self.BackupImages = self.Images
        self.Images = UnLabelledImages

    ####################################################################################
    #                             Bounding Box Refine                                  #
    ####################################################################################

    def changetlx(self):
        if self.Window.spinBox_tl_x.value() >= self.bbox[self.PresentBox].brx:
            self.Window.spinBox_tl_x.setValue(self.bbox[self.PresentBox].brx - 1)
            return
        else:
            self.bbox[self.PresentBox].tlx = self.Window.spinBox_tl_x.value()
            self.refreshBoundingBox()

    def changetly(self):
        if self.Window.spinBox_tl_y.value() >= self.bbox[self.PresentBox].bry:
            self.Window.spinBox_tl_y.setValue(self.bbox[self.PresentBox].bry - 1)
            return
        else:
            self.bbox[self.PresentBox].tly = self.Window.spinBox_tl_y.value()
            self.refreshBoundingBox()

    def changebrx(self):
        if self.Window.spinBox_br_x.value() <= self.bbox[self.PresentBox].tlx:
            self.Window.spinBox_br_x.setValue(self.bbox[self.PresentBox].tlx - 1)
            return
        else:
            self.bbox[self.PresentBox].brx = self.Window.spinBox_br_x.value()
            self.refreshBoundingBox()

    def changebry(self):
        if self.Window.spinBox_br_y.value() <= self.bbox[self.PresentBox].tly:
            self.Window.spinBox_br_y.setValue(self.bbox[self.PresentBox].tly - 1)
            return
        else:
            self.bbox[self.PresentBox].bry = self.Window.spinBox_br_y.value()
            self.refreshBoundingBox()

    def refreshBoundingBox(self):
        image_show = self.ProcessImage(self.annotateImage(self.presentImage.copy(), True))
        self.Window.label_Image.setPixmap(QPixmap(image_show))

    ####################################################################################
    #                                      图像标注                                    #
    ####################################################################################
    def ChangeCurrentLabel(self, value, x, y):
        dx = x * self.ResizeRatio[0]
        dy = y * self.ResizeRatio[1]
        if dx >= self.bbox[self.PresentBox].tlx and dx <= self.bbox[self.PresentBox].brx and dy >= self.bbox[self.PresentBox].tly and dy <= self.bbox[self.PresentBox].bry:
            if value > 0:
                self.Statistics[self.bbox[self.PresentBox].cls] -= 1
                self.bbox[self.PresentBox].cls += 1
                if self.bbox[self.PresentBox].cls >= self.totalClass:
                    self.bbox[self.PresentBox].cls = 0
                self.Statistics[self.bbox[self.PresentBox].cls] += 1
            if value < 0:
                self.Statistics[self.bbox[self.PresentBox].cls] -= 1
                self.bbox[self.PresentBox].cls -= 1
                if self.bbox[self.PresentBox].cls < 0:
                    self.bbox[self.PresentBox].cls = self.totalClass - 1
                self.Statistics[self.bbox[self.PresentBox].cls] += 1
            image_show = self.ProcessImage(self.annotateImage(self.presentImage.copy(), True))
            self.Window.label_Image.setPixmap(QPixmap(image_show))

    def SelectAnnotation(self, x, y):
        for index, bndbox in enumerate(self.bbox):
            dx = x * self.ResizeRatio[0]
            dy = y * self.ResizeRatio[1]
            if dx >= bndbox.tlx and dx <= bndbox.brx and dy >= bndbox.tly and dy <= bndbox.bry:
                self.PresentBox = index
                break
        self.refreshInfo()
        image_show = self.ProcessImage(self.annotateImage(self.presentImage.copy(), True))
        self.Window.label_Image.setPixmap(QPixmap(image_show))

    def DeleteAnnotation(self, x, y):
        isInside = False
        for index, bndbox in enumerate(self.bbox):
            dx = x * self.ResizeRatio[0]
            dy = y * self.ResizeRatio[1]
            if dx >= bndbox.tlx and dx <= bndbox.brx and dy >= bndbox.tly and dy <= bndbox.bry:
                self.PresentBox = index
                isInside = True
                break
        if not isInside:
            return

        self.Statistics[self.bbox[self.PresentBox].cls] -= 1
        del self.bbox[self.PresentBox]
        self.PresentBox -= 1
        if self.PresentBox <= 0:
            self.PresentBox = 0

        self.refreshInfo()
        image_show = self.ProcessImage(self.annotateImage(self.presentImage.copy(), True))
        self.Window.label_Image.setPixmap(QPixmap(image_show))

    def Annotate(self, x0, y0, x1, y1):
        dx0 = int(x0 * self.ResizeRatio[0])
        dy0 = int(y0 * self.ResizeRatio[1])
        dx1 = int(x1 * self.ResizeRatio[0])
        dy1 = int(y1 * self.ResizeRatio[1])

        temp = BoundingBox(0)
        temp.setPoints(dx0, dy0, dx1, dy1)
        self.bbox.append(temp)


        self.PresentBox = len(self.bbox) - 1
        self.Statistics[0] += 1
        self.refreshInfo()
        image_show = self.ProcessImage(self.annotateImage(self.presentImage.copy(), True))
        self.Window.label_Image.setPixmap(QPixmap(image_show))

    def DynamicAnnotate(self, x0, y0, x1, y1):
        dx0 = int(x0 * self.ResizeRatio[0])
        dy0 = int(y0 * self.ResizeRatio[1])
        dx1 = int(x1 * self.ResizeRatio[0])
        dy1 = int(y1 * self.ResizeRatio[1])

        temp = BoundingBox(0)
        temp.setPoints(dx0, dy0, dx1, dy1)

        image_show = self.ProcessImage(self.annotateImage(self.presentImage.copy(), True, temp))
        self.Window.label_Image.setPixmap(QPixmap(image_show))


    ####################################################################################
    #                                      数据表格                                    #
    ####################################################################################
    def selectColor(self, item):
        row = self.Window.tableWidget.row(item)
        if self.Window.tableWidget.column(item) == 2:
            color = QColorDialog.getColor().name()
            color = color.replace("#", '')
            blue = eval('0x' + color[0:2])
            green = eval('0x' + color[2:4])
            red = eval('0x' + color[4:6])
            self.BBColor[row] = (red, green, blue)
            self.Window.tableWidget.item(row, 2).setBackground(QBrush(QColor(*reversed(self.BBColor[row]))))
            if self.bbox:
                image_show = self.ProcessImage(self.annotateImage(self.presentImage.copy(), True))
                self.Window.label_Image.setPixmap(QPixmap(image_show))

    def InitTable(self):
        self.Window.tableWidget.itemClicked.connect(self.selectColor)
        self.Window.tableWidget.setRowCount(0)
        self.Window.tableWidget.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.Window.tableWidget.horizontalHeader().setSectionResizeMode(1, QHeaderView.Fixed)
        self.Window.tableWidget.horizontalHeader().setSectionResizeMode(2, QHeaderView.Fixed)
        self.Window.tableWidget.setColumnWidth(1, 100)
        self.Window.tableWidget.setColumnWidth(2, 50)
        self.Window.tableWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)

    def addNewLabel(self):
        labels, isSuccess = QInputDialog.getText(self.Window, "标签信息录入", "请输入所有标签信息，以','作为分隔", text="")
        if isSuccess:
            labels = labels.split(',')
            rCount = self.Window.tableWidget.rowCount()
            for label in labels:
                self.Window.tableWidget.setRowCount(rCount + 1)
                Name = QTableWidgetItem(label)
                Count = QTableWidgetItem('0')
                clr = QTableWidgetItem('')
                Name.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
                Count.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
                clr.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
                self.Window.tableWidget.setItem(rCount, 0, Name)
                self.Window.tableWidget.setItem(rCount, 1, Count)
                self.Window.tableWidget.setItem(rCount, 2, clr)

                self.Window.tableWidget.item(rCount, 2).setBackground(QBrush(QColor(*reversed(self.BBColor[rCount]))))
                self.replaceDict_N2E[label] = rCount
                self.Statistics.append(0)
                self.replaceDict_E2N.append(label)
                rCount += 1
        self.totalClass = rCount

    def delCurrentLabel(self):
        self.Window.tableWidget.removeRow(self.Window.tableWidget.currentRow())
        # self.replaceDict_N2E[label] = rCount
        # self.Statistics[label] = 0
        # self.replaceDict_E2N.append(label)

    def delAllLabels(self):
        self.Window.tableWidget.setRowCount(0)

