from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QMainWindow, QLabel, QVBoxLayout, QWidget

import service as s


class AnotherWindow(QMainWindow):
    urla = ''

    def __init__(self, parent=None, url=None):
        super(AnotherWindow, self).__init__(parent)
        self.acceptDrops()
        self.urla = url
        self.setWindowTitle("Image")
        self.setGeometry(0, 0, 400, 300)
        self.label = QLabel(self)
        self.pixmap = QPixmap(self.urla)
        self.label.setPixmap(self.pixmap)
        self.label.resize(self.pixmap.width(),
                          self.pixmap.height())


class Ui_MainWindow(object):
    back = None
    inputs = []

    def setupUi(self, MainWindow):
        self.back = s
        self.mainWindow = MainWindow
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 609)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.label = QtWidgets.QLabel(self.centralwidget)

        self.label.setGeometry(QtCore.QRect(10, 10, 171, 16))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(10, 30, 171, 16))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(10, 50, 171, 16))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(10, 70, 171, 16))
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(10, 90, 171, 16))
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(10, 110, 171, 16))
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(10, 130, 171, 16))
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(10, 170, 171, 16))
        self.label_8.setObjectName("label_8")
        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        self.label_9.setGeometry(QtCore.QRect(10, 150, 171, 16))
        self.label_9.setObjectName("label_9")
        self.label_10 = QtWidgets.QLabel(self.centralwidget)
        self.label_10.setGeometry(QtCore.QRect(10, 190, 171, 16))
        self.label_10.setObjectName("label_10")
        self.label_11 = QtWidgets.QLabel(self.centralwidget)
        self.label_11.setGeometry(QtCore.QRect(10, 230, 171, 16))
        self.label_11.setObjectName("label_11")
        self.label_12 = QtWidgets.QLabel(self.centralwidget)
        self.label_12.setGeometry(QtCore.QRect(10, 210, 171, 16))
        self.label_12.setObjectName("label_12")
        self.label_13 = QtWidgets.QLabel(self.centralwidget)
        self.label_13.setGeometry(QtCore.QRect(10, 250, 171, 16))
        self.label_13.setObjectName("label_13")
        self.label_14 = QtWidgets.QLabel(self.centralwidget)
        self.label_14.setGeometry(QtCore.QRect(10, 270, 171, 16))
        self.label_14.setObjectName("label_14")
        self.label_15 = QtWidgets.QLabel(self.centralwidget)
        self.label_15.setGeometry(QtCore.QRect(10, 290, 171, 16))
        self.label_15.setObjectName("label_15")
        self.label_16 = QtWidgets.QLabel(self.centralwidget)
        self.label_16.setGeometry(QtCore.QRect(10, 310, 171, 16))
        self.label_16.setObjectName("label_16")
        self.label_17 = QtWidgets.QLabel(self.centralwidget)
        self.label_17.setGeometry(QtCore.QRect(10, 330, 171, 16))
        self.label_17.setObjectName("label_17")
        self.label_18 = QtWidgets.QLabel(self.centralwidget)
        self.label_18.setGeometry(QtCore.QRect(10, 350, 171, 16))
        self.label_18.setObjectName("label_18")
        self.label_19 = QtWidgets.QLabel(self.centralwidget)
        self.label_19.setGeometry(QtCore.QRect(10, 390, 171, 16))
        self.label_19.setObjectName("label_19")
        self.label_20 = QtWidgets.QLabel(self.centralwidget)
        self.label_20.setGeometry(QtCore.QRect(10, 370, 171, 16))
        self.label_20.setObjectName("label_20")
        self.label_21 = QtWidgets.QLabel(self.centralwidget)
        self.label_21.setGeometry(QtCore.QRect(10, 410, 171, 16))
        self.label_21.setObjectName("label_21")
        self.label_22 = QtWidgets.QLabel(self.centralwidget)
        self.label_22.setGeometry(QtCore.QRect(10, 430, 171, 16))
        self.label_22.setObjectName("label_22")
        self.label_23 = QtWidgets.QLabel(self.centralwidget)
        self.label_23.setGeometry(QtCore.QRect(10, 450, 171, 16))
        self.label_23.setObjectName("label_23")
        self.label_24 = QtWidgets.QLabel(self.centralwidget)
        self.label_24.setGeometry(QtCore.QRect(10, 470, 171, 16))
        self.label_24.setObjectName("label_24")
        self.label_25 = QtWidgets.QLabel(self.centralwidget)
        self.label_25.setGeometry(QtCore.QRect(10, 490, 171, 16))
        self.label_25.setObjectName("label_25")
        self.label_26 = QtWidgets.QLabel(self.centralwidget)
        self.label_26.setGeometry(QtCore.QRect(10, 530, 171, 16))
        self.label_26.setObjectName("label_26")
        self.label_27 = QtWidgets.QLabel(self.centralwidget)
        self.label_27.setGeometry(QtCore.QRect(10, 510, 171, 16))
        self.label_27.setObjectName("label_27")
        self.doubleSpinBox = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox.setGeometry(QtCore.QRect(170, 10, 68, 24))
        self.doubleSpinBox.setObjectName("doubleSpinBox")
        self.doubleSpinBox_2 = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox_2.setGeometry(QtCore.QRect(170, 30, 68, 24))
        self.doubleSpinBox_2.setObjectName("doubleSpinBox_2")
        self.doubleSpinBox_3 = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox_3.setGeometry(QtCore.QRect(170, 50, 68, 24))
        self.doubleSpinBox_3.setObjectName("doubleSpinBox_3")
        self.doubleSpinBox_4 = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox_4.setGeometry(QtCore.QRect(170, 70, 68, 24))
        self.doubleSpinBox_4.setObjectName("doubleSpinBox_4")
        self.doubleSpinBox_5 = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox_5.setGeometry(QtCore.QRect(170, 90, 68, 24))
        self.doubleSpinBox_5.setObjectName("doubleSpinBox_5")
        self.doubleSpinBox_6 = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox_6.setGeometry(QtCore.QRect(170, 110, 68, 24))
        self.doubleSpinBox_6.setObjectName("doubleSpinBox_6")
        self.doubleSpinBox_7 = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox_7.setGeometry(QtCore.QRect(170, 130, 68, 24))
        self.doubleSpinBox_7.setObjectName("doubleSpinBox_7")
        self.doubleSpinBox_8 = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox_8.setGeometry(QtCore.QRect(170, 150, 68, 24))
        self.doubleSpinBox_8.setObjectName("doubleSpinBox_8")
        self.doubleSpinBox_9 = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox_9.setGeometry(QtCore.QRect(170, 170, 68, 24))
        self.doubleSpinBox_9.setObjectName("doubleSpinBox_9")
        self.doubleSpinBox_10 = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox_10.setGeometry(QtCore.QRect(170, 190, 68, 24))
        self.doubleSpinBox_10.setObjectName("doubleSpinBox_10")
        self.doubleSpinBox_11 = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox_11.setGeometry(QtCore.QRect(170, 210, 68, 24))
        self.doubleSpinBox_11.setObjectName("doubleSpinBox_11")
        self.doubleSpinBox_12 = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox_12.setGeometry(QtCore.QRect(170, 230, 68, 24))
        self.doubleSpinBox_12.setObjectName("doubleSpinBox_12")
        self.doubleSpinBox_13 = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox_13.setGeometry(QtCore.QRect(170, 250, 68, 24))
        self.doubleSpinBox_13.setObjectName("doubleSpinBox_13")
        self.doubleSpinBox_14 = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox_14.setGeometry(QtCore.QRect(170, 270, 68, 24))
        self.doubleSpinBox_14.setObjectName("doubleSpinBox_14")
        self.doubleSpinBox_15 = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox_15.setGeometry(QtCore.QRect(170, 290, 68, 24))
        self.doubleSpinBox_15.setObjectName("doubleSpinBox_15")
        self.doubleSpinBox_16 = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox_16.setGeometry(QtCore.QRect(170, 310, 68, 24))
        self.doubleSpinBox_16.setObjectName("doubleSpinBox_16")
        self.doubleSpinBox_17 = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox_17.setGeometry(QtCore.QRect(170, 330, 68, 24))
        self.doubleSpinBox_17.setObjectName("doubleSpinBox_17")
        self.doubleSpinBox_18 = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox_18.setGeometry(QtCore.QRect(170, 350, 68, 24))
        self.doubleSpinBox_18.setObjectName("doubleSpinBox_18")
        self.doubleSpinBox_19 = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox_19.setGeometry(QtCore.QRect(170, 370, 68, 24))
        self.doubleSpinBox_19.setObjectName("doubleSpinBox_19")
        self.doubleSpinBox_20 = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox_20.setGeometry(QtCore.QRect(170, 390, 68, 24))
        self.doubleSpinBox_20.setObjectName("doubleSpinBox_20")
        self.doubleSpinBox_21 = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox_21.setGeometry(QtCore.QRect(170, 410, 68, 24))
        self.doubleSpinBox_21.setObjectName("doubleSpinBox_21")
        self.doubleSpinBox_22 = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox_22.setGeometry(QtCore.QRect(170, 430, 68, 24))
        self.doubleSpinBox_22.setObjectName("doubleSpinBox_22")
        self.doubleSpinBox_23 = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox_23.setGeometry(QtCore.QRect(170, 450, 68, 24))
        self.doubleSpinBox_23.setObjectName("doubleSpinBox_23")
        self.doubleSpinBox_24 = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox_24.setGeometry(QtCore.QRect(170, 470, 68, 24))
        self.doubleSpinBox_24.setObjectName("doubleSpinBox_24")
        self.doubleSpinBox_25 = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox_25.setGeometry(QtCore.QRect(170, 490, 68, 24))
        self.doubleSpinBox_25.setObjectName("doubleSpinBox_25")
        self.doubleSpinBox_26 = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox_26.setGeometry(QtCore.QRect(170, 510, 68, 24))
        self.doubleSpinBox_26.setObjectName("doubleSpinBox_26")
        self.doubleSpinBox_27 = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox_27.setGeometry(QtCore.QRect(170, 530, 68, 24))
        self.doubleSpinBox_27.setObjectName("doubleSpinBox_27")
        self.inputs.append(self.doubleSpinBox)
        self.inputs.append(self.doubleSpinBox_2)
        self.inputs.append(self.doubleSpinBox_3)
        self.inputs.append(self.doubleSpinBox_4)
        self.inputs.append(self.doubleSpinBox_5)
        self.inputs.append(self.doubleSpinBox_6)
        self.inputs.append(self.doubleSpinBox_7)
        self.inputs.append(self.doubleSpinBox_8)
        self.inputs.append(self.doubleSpinBox_9)
        self.inputs.append(self.doubleSpinBox_10)
        self.inputs.append(self.doubleSpinBox_11)
        self.inputs.append(self.doubleSpinBox_12)
        self.inputs.append(self.doubleSpinBox_13)
        self.inputs.append(self.doubleSpinBox_14)
        self.inputs.append(self.doubleSpinBox_15)
        self.inputs.append(self.doubleSpinBox_16)
        self.inputs.append(self.doubleSpinBox_17)
        self.inputs.append(self.doubleSpinBox_18)
        self.inputs.append(self.doubleSpinBox_19)
        self.inputs.append(self.doubleSpinBox_20)
        self.inputs.append(self.doubleSpinBox_21)
        self.inputs.append(self.doubleSpinBox_22)
        self.inputs.append(self.doubleSpinBox_23)
        self.inputs.append(self.doubleSpinBox_24)
        self.inputs.append(self.doubleSpinBox_25)
        self.inputs.append(self.doubleSpinBox_26)
        self.inputs.append(self.doubleSpinBox_27)
        self.label_28 = QtWidgets.QLabel(self.centralwidget)
        self.label_28.setGeometry(QtCore.QRect(480, 10, 101, 16))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.label_28.setFont(font)
        self.label_28.setObjectName("label_28")
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(310, 0, 431, 51))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.line_2 = QtWidgets.QFrame(self.centralwidget)
        self.line_2.setGeometry(QtCore.QRect(310, 200, 431, 51))
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")

        self.comboBox = QtWidgets.QSpinBox(self.centralwidget)
        self.comboBox.setMinimum(0)
        self.comboBox.setMaximum(1000)
        self.comboBox.setGeometry(QtCore.QRect(250, 60, 104, 26))
        self.comboBox.setObjectName("comboBox")

        self.comboBox_2 = QtWidgets.QSpinBox(self.centralwidget)
        self.comboBox_2.setGeometry(QtCore.QRect(250, 90, 104, 26))
        self.comboBox_2.setObjectName("comboBox_2")
        self.comboBox_2.setMinimum(0)
        self.comboBox_2.setMaximum(1000)

        self.comboBox_3 = QtWidgets.QSpinBox(self.centralwidget)
        self.comboBox_3.setGeometry(QtCore.QRect(250, 120, 104, 26))
        self.comboBox_3.setObjectName("comboBox_3")
        self.comboBox_3.setMinimum(0)
        self.comboBox_3.setMaximum(1000)

        self.comboBox_4 = QtWidgets.QSpinBox(self.centralwidget)
        self.comboBox_4.setGeometry(QtCore.QRect(250, 150, 104, 26))
        self.comboBox_4.setObjectName("comboBox_4")
        self.comboBox_4.setMinimum(0)
        self.comboBox_4.setMaximum(1000)

        self.comboBox_5 = QtWidgets.QSpinBox(self.centralwidget)
        self.comboBox_5.setGeometry(QtCore.QRect(250, 180, 104, 26))
        self.comboBox_5.setObjectName("comboBox_5")
        self.comboBox_5.setMinimum(0)
        self.comboBox_5.setMaximum(1000)

        self.label_29 = QtWidgets.QLabel(self.centralwidget)
        self.label_29.setGeometry(QtCore.QRect(350, 60, 16, 21))
        self.label_29.setObjectName("label_29")

        self.label_F1 = QtWidgets.QLabel(self.centralwidget)
        self.label_F1.setGeometry(QtCore.QRect(240, 60, 20, 20))
        self.label_F1.setObjectName("labelF1")

        self.label_F2 = QtWidgets.QLabel(self.centralwidget)
        self.label_F2.setGeometry(QtCore.QRect(240, 90, 20, 20))
        self.label_F2.setObjectName("labelF2")

        self.label_F3 = QtWidgets.QLabel(self.centralwidget)
        self.label_F3.setGeometry(QtCore.QRect(240, 120, 20, 20))
        self.label_F3.setObjectName("labelF3")

        self.label_F5 = QtWidgets.QLabel(self.centralwidget)
        self.label_F5.setGeometry(QtCore.QRect(240, 150, 20, 20))
        self.label_F5.setObjectName("labelF5")

        self.label_F4 = QtWidgets.QLabel(self.centralwidget)
        self.label_F4.setGeometry(QtCore.QRect(240, 180, 20, 20))
        self.label_F4.setObjectName("labelF4")

        self.label_30 = QtWidgets.QLabel(self.centralwidget)
        self.label_30.setGeometry(QtCore.QRect(350, 90, 16, 21))
        self.label_30.setObjectName("label_30")
        self.label_31 = QtWidgets.QLabel(self.centralwidget)
        self.label_31.setGeometry(QtCore.QRect(350, 120, 16, 21))
        self.label_31.setObjectName("label_31")
        self.label_32 = QtWidgets.QLabel(self.centralwidget)
        self.label_32.setGeometry(QtCore.QRect(350, 150, 16, 21))
        self.label_32.setObjectName("label_32")
        self.label_33 = QtWidgets.QLabel(self.centralwidget)
        self.label_33.setGeometry(QtCore.QRect(350, 180, 16, 21))
        self.label_33.setObjectName("label_33")

        self.doubleSpinBox_28 = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox_28.setGeometry(QtCore.QRect(360, 60, 51, 21))
        self.doubleSpinBox_28.setObjectName("doubleSpinBox_28")
        self.doubleSpinBox_29 = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox_29.setGeometry(QtCore.QRect(430, 60, 51, 21))
        self.doubleSpinBox_29.setObjectName("doubleSpinBox_29")
        self.doubleSpinBox_33 = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox_33.setGeometry(QtCore.QRect(510, 60, 51, 21))
        self.doubleSpinBox_33.setObjectName("doubleSpinBox_33")
        self.doubleSpinBox_34 = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox_34.setGeometry(QtCore.QRect(610, 60, 51, 21))
        self.doubleSpinBox_34.setObjectName("doubleSpinBox_34")
        self.label_34 = QtWidgets.QLabel(self.centralwidget)
        self.label_34.setGeometry(QtCore.QRect(420, 60, 16, 21))
        self.label_34.setObjectName("label_34")
        self.label_35 = QtWidgets.QLabel(self.centralwidget)
        self.label_35.setGeometry(QtCore.QRect(480, 60, 41, 21))
        self.label_35.setObjectName("label_35")
        self.label_36 = QtWidgets.QLabel(self.centralwidget)
        self.label_36.setGeometry(QtCore.QRect(560, 60, 51, 21))
        self.label_36.setObjectName("label_36")
        self.label_38 = QtWidgets.QLabel(self.centralwidget)
        self.label_38.setGeometry(QtCore.QRect(660, 60, 41, 21))
        self.label_38.setObjectName("label_38")
        self.label_42 = QtWidgets.QLabel(self.centralwidget)
        self.label_42.setGeometry(QtCore.QRect(470, 240, 111, 16))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.label_42.setFont(font)
        self.label_42.setObjectName("label_42")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(320, 270, 113, 32))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(320, 310, 113, 32))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(320, 360, 113, 32))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(320, 410, 113, 32))
        self.pushButton_4.setObjectName("pushButton_4")
        self.doubleSpinBox_41 = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox_41.setGeometry(QtCore.QRect(460, 370, 51, 21))
        self.doubleSpinBox_41.setObjectName("doubleSpinBox_41")
        self.label_43 = QtWidgets.QLabel(self.centralwidget)
        self.label_43.setGeometry(QtCore.QRect(430, 370, 41, 21))
        self.label_43.setObjectName("label_43")
        self.label_37 = QtWidgets.QLabel(self.centralwidget)
        self.label_37.setGeometry(QtCore.QRect(480, 90, 41, 21))
        self.label_37.setObjectName("label_37")
        self.doubleSpinBox_35 = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox_35.setGeometry(QtCore.QRect(510, 90, 51, 21))
        self.doubleSpinBox_35.setObjectName("doubleSpinBox_35")
        self.label_39 = QtWidgets.QLabel(self.centralwidget)
        self.label_39.setGeometry(QtCore.QRect(560, 90, 51, 21))
        self.label_39.setObjectName("label_39")
        self.label_40 = QtWidgets.QLabel(self.centralwidget)
        self.label_40.setGeometry(QtCore.QRect(660, 90, 41, 21))
        self.label_40.setObjectName("label_40")
        self.doubleSpinBox_30 = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox_30.setGeometry(QtCore.QRect(430, 90, 51, 21))
        self.doubleSpinBox_30.setObjectName("doubleSpinBox_30")
        self.doubleSpinBox_31 = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox_31.setGeometry(QtCore.QRect(360, 90, 51, 21))
        self.doubleSpinBox_31.setObjectName("doubleSpinBox_31")
        self.doubleSpinBox_36 = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox_36.setGeometry(QtCore.QRect(610, 90, 51, 21))
        self.doubleSpinBox_36.setObjectName("doubleSpinBox_36")
        self.label_41 = QtWidgets.QLabel(self.centralwidget)
        self.label_41.setGeometry(QtCore.QRect(420, 90, 16, 21))
        self.label_41.setObjectName("label_41")
        self.label_44 = QtWidgets.QLabel(self.centralwidget)
        self.label_44.setGeometry(QtCore.QRect(480, 120, 41, 21))
        self.label_44.setObjectName("label_44")
        self.doubleSpinBox_37 = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox_37.setGeometry(QtCore.QRect(510, 120, 51, 21))
        self.doubleSpinBox_37.setObjectName("doubleSpinBox_37")
        self.label_45 = QtWidgets.QLabel(self.centralwidget)
        self.label_45.setGeometry(QtCore.QRect(560, 120, 51, 21))
        self.label_45.setObjectName("label_45")
        self.label_46 = QtWidgets.QLabel(self.centralwidget)
        self.label_46.setGeometry(QtCore.QRect(660, 120, 41, 21))
        self.label_46.setObjectName("label_46")
        self.doubleSpinBox_32 = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox_32.setGeometry(QtCore.QRect(430, 120, 51, 21))
        self.doubleSpinBox_32.setObjectName("doubleSpinBox_32")
        self.doubleSpinBox_38 = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox_38.setGeometry(QtCore.QRect(360, 120, 51, 21))
        self.doubleSpinBox_38.setObjectName("doubleSpinBox_38")
        self.doubleSpinBox_39 = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox_39.setGeometry(QtCore.QRect(610, 120, 51, 21))
        self.doubleSpinBox_39.setObjectName("doubleSpinBox_39")
        self.label_47 = QtWidgets.QLabel(self.centralwidget)
        self.label_47.setGeometry(QtCore.QRect(420, 120, 16, 21))
        self.label_47.setObjectName("label_47")
        self.label_48 = QtWidgets.QLabel(self.centralwidget)
        self.label_48.setGeometry(QtCore.QRect(480, 150, 41, 21))
        self.label_48.setObjectName("label_48")
        self.doubleSpinBox_40 = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox_40.setGeometry(QtCore.QRect(510, 150, 51, 21))
        self.doubleSpinBox_40.setObjectName("doubleSpinBox_40")
        self.label_49 = QtWidgets.QLabel(self.centralwidget)
        self.label_49.setGeometry(QtCore.QRect(560, 150, 51, 21))
        self.label_49.setObjectName("label_49")
        self.label_50 = QtWidgets.QLabel(self.centralwidget)
        self.label_50.setGeometry(QtCore.QRect(660, 150, 41, 21))
        self.label_50.setObjectName("label_50")
        self.doubleSpinBox_42 = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox_42.setGeometry(QtCore.QRect(430, 150, 51, 21))
        self.doubleSpinBox_42.setObjectName("doubleSpinBox_42")
        self.doubleSpinBox_43 = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox_43.setGeometry(QtCore.QRect(360, 150, 51, 21))
        self.doubleSpinBox_43.setObjectName("doubleSpinBox_43")
        self.doubleSpinBox_44 = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox_44.setGeometry(QtCore.QRect(610, 150, 51, 21))
        self.doubleSpinBox_44.setObjectName("doubleSpinBox_44")
        self.label_51 = QtWidgets.QLabel(self.centralwidget)
        self.label_51.setGeometry(QtCore.QRect(420, 150, 16, 21))
        self.label_51.setObjectName("label_51")
        self.label_52 = QtWidgets.QLabel(self.centralwidget)
        self.label_52.setGeometry(QtCore.QRect(480, 180, 41, 21))
        self.label_52.setObjectName("label_52")
        self.doubleSpinBox_45 = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox_45.setGeometry(QtCore.QRect(510, 180, 51, 21))
        self.doubleSpinBox_45.setObjectName("doubleSpinBox_45")
        self.label_53 = QtWidgets.QLabel(self.centralwidget)
        self.label_53.setGeometry(QtCore.QRect(560, 180, 51, 21))
        self.label_53.setObjectName("label_53")
        self.label_54 = QtWidgets.QLabel(self.centralwidget)
        self.label_54.setGeometry(QtCore.QRect(660, 180, 41, 21))
        self.label_54.setObjectName("label_54")
        self.doubleSpinBox_46 = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox_46.setGeometry(QtCore.QRect(430, 180, 51, 21))
        self.doubleSpinBox_46.setObjectName("doubleSpinBox_46")
        self.doubleSpinBox_47 = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox_47.setGeometry(QtCore.QRect(360, 180, 51, 21))
        self.doubleSpinBox_47.setObjectName("doubleSpinBox_47")
        self.doubleSpinBox_48 = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox_48.setGeometry(QtCore.QRect(610, 180, 51, 21))
        self.doubleSpinBox_48.setObjectName("doubleSpinBox_48")
        self.label_55 = QtWidgets.QLabel(self.centralwidget)
        self.label_55.setGeometry(QtCore.QRect(420, 180, 16, 21))
        self.label_55.setObjectName("label_55")
        self.label_56 = QtWidgets.QLabel(self.centralwidget)
        self.label_56.setGeometry(QtCore.QRect(260, 30, 400, 16))
        self.label_56.setObjectName("label_56")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 24))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.add_func()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "u1 - Функц Возможности"))
        self.label_2.setText(_translate("MainWindow", "u7 - Пригодность"))
        self.label_3.setText(_translate("MainWindow", "u8 - Правильность"))
        self.label_4.setText(_translate("MainWindow", "u9 - Способность к Взаи"))
        self.label_5.setText(_translate("MainWindow", "u10 - Защищенность"))
        self.label_6.setText(_translate("MainWindow", "u11 - Согласованность"))
        self.label_7.setText(_translate("MainWindow", "u57 - Ошибки Надежн"))
        self.label_8.setText(_translate("MainWindow", "u48 - Несовместимость"))
        self.label_9.setText(_translate("MainWindow", "u35 - Нессответствия"))
        self.label_10.setText(_translate("MainWindow", "u47 - Ошибки при испол"))
        self.label_11.setText(_translate("MainWindow", "u50 - Недостаток средс"))
        self.label_12.setText(_translate("MainWindow", "u46 - Ошибки Сети"))
        self.label_13.setText(_translate("MainWindow", "u51 - Недостатки Защиты"))
        self.label_14.setText(_translate("MainWindow", "u52 - Несовме Криптогр"))
        self.label_15.setText(_translate("MainWindow", "u53 -Защита Вирусов"))
        self.label_16.setText(_translate("MainWindow", "u34 - Невыполнение функ"))
        self.label_17.setText(_translate("MainWindow", "u2 - Надежность"))
        self.label_18.setText(_translate("MainWindow", "u36 - Недостки Докумен"))
        self.label_19.setText(_translate("MainWindow", "u38 - Отсутствие функц"))
        self.label_20.setText(_translate("MainWindow", "u37 - Отсут файла комп"))
        self.label_21.setText(_translate("MainWindow", "u39 - Отссутсвие интер"))
        self.label_22.setText(_translate("MainWindow", "u40 - Противоречия"))
        self.label_23.setText(_translate("MainWindow", "u41 - Проблемы Докумен"))
        self.label_24.setText(_translate("MainWindow", "u42 - Проблемы Алгорит"))
        self.label_25.setText(_translate("MainWindow", "u43 - Ошибки Вычислен"))
        self.label_26.setText(_translate("MainWindow", "u45 - Проблемы взаимо"))
        self.label_27.setText(_translate("MainWindow", "u44 - Против настроек"))
        self.label_28.setText(_translate("MainWindow", "Уравнения"))
        self.label_29.setText(_translate("MainWindow", "="))
        self.label_F1.setText('F')
        self.label_F2.setText('F')
        self.label_F3.setText('F')
        self.label_F4.setText('F')
        self.label_F5.setText('F')

        self.label_30.setText(_translate("MainWindow", "="))
        self.label_31.setText(_translate("MainWindow", "="))
        self.label_32.setText(_translate("MainWindow", "="))
        self.label_33.setText(_translate("MainWindow", "="))
        self.label_34.setText(_translate("MainWindow", "+"))
        self.label_35.setText(_translate("MainWindow", "*X +"))
        self.label_36.setText(_translate("MainWindow", "*X^2 +"))
        self.label_38.setText(_translate("MainWindow", "*X^3"))
        self.label_42.setText(_translate("MainWindow", "Управление"))
        self.pushButton.setText(_translate("MainWindow", "Посчитать"))
        self.pushButton_2.setText(_translate("MainWindow", "График"))
        self.pushButton_3.setText(_translate("MainWindow", "Диаграмма"))
        self.pushButton_4.setText(_translate("MainWindow", "Возмущения"))
        self.label_43.setText(_translate("MainWindow", " T = "))
        self.label_37.setText(_translate("MainWindow", "*X +"))
        self.label_39.setText(_translate("MainWindow", "*X^2 +"))
        self.label_40.setText(_translate("MainWindow", "*X^3"))
        self.label_41.setText(_translate("MainWindow", "+"))
        self.label_44.setText(_translate("MainWindow", "*X +"))
        self.label_45.setText(_translate("MainWindow", "*X^2 +"))
        self.label_46.setText(_translate("MainWindow", "*X^3"))
        self.label_47.setText(_translate("MainWindow", "+"))
        self.label_48.setText(_translate("MainWindow", "*X +"))
        self.label_49.setText(_translate("MainWindow", "*X^2 +"))
        self.label_50.setText(_translate("MainWindow", "*X^3"))
        self.label_51.setText(_translate("MainWindow", "+"))
        self.label_52.setText(_translate("MainWindow", "*X +"))
        self.label_53.setText(_translate("MainWindow", "*X^2 +"))
        self.label_54.setText(_translate("MainWindow", "*X^3"))
        self.label_55.setText(_translate("MainWindow", "+"))
        self.label_56.setText(
            _translate("MainWindow", "    d          +     c       *X  +    b         *X^2 +     a        *X^3"))
        self.fill_def_values()

    def fill_def_values(self):
        for i, value in enumerate(s.chars.start_values):
            # print(i,value)
            self.inputs[i].setValue(value)
        self.label_56.setText(
            "F = [0 : " + str(self.back.chars.func_m.__len__()) + "]           d          +     c       *X  +  "
                                                                  "  b         *X^2 +     a        "
                                                                  "*X^3")

    res = None
    dialogs = list()

    def calc(self):
        Q1 = self.comboBox.value()
        Q2 = self.comboBox_2.value()
        Q3 = self.comboBox_3.value()
        Q4 = self.comboBox_4.value()
        Q5 = self.comboBox_5.value()

        Q1k = list(range(4))
        Q2k = list(range(4))
        Q3k = list(range(4))
        Q4k = list(range(4))
        Q5k = list(range(4))
        Q1k[0] = self.doubleSpinBox_34.value()
        Q1k[1] = self.doubleSpinBox_33.value()
        Q1k[2] = self.doubleSpinBox_29.value()
        Q1k[3] = self.doubleSpinBox_28.value()

        Q2k[0] = self.doubleSpinBox_36.value()
        Q2k[1] = self.doubleSpinBox_35.value()
        Q2k[2] = self.doubleSpinBox_30.value()
        Q2k[3] = self.doubleSpinBox_31.value()

        Q3k[0] = self.doubleSpinBox_39.value()
        Q3k[1] = self.doubleSpinBox_37.value()
        Q3k[2] = self.doubleSpinBox_32.value()
        Q3k[3] = self.doubleSpinBox_38.value()

        Q4k[0] = self.doubleSpinBox_44.value()
        Q4k[1] = self.doubleSpinBox_40.value()
        Q4k[2] = self.doubleSpinBox_42.value()
        Q4k[3] = self.doubleSpinBox_43.value()

        Q5k[0] = self.doubleSpinBox_48.value()
        Q5k[1] = self.doubleSpinBox_45.value()
        Q5k[2] = self.doubleSpinBox_46.value()
        Q5k[3] = self.doubleSpinBox_47.value()
        self.back.chars.init_par(Q1, Q2, Q3, Q4, Q5, Q1k, Q2k, Q3k, Q4k, Q5k)

        values = [n.value() for n in self.inputs]

        self.res = s.chars.calculate(values)
        s.chars.get_diag(self.doubleSpinBox_41.value(), 'diag.png')
        s.chars.get_diag(0.0, 'diag0.png')
        s.chars.get_graphics()
        s.get_faks_image()

    def show_diag(self):
        win = AnotherWindow(self.centralwidget, 'diag.png')
        self.dialogs.append(win)
        win.show()
        win2 = AnotherWindow(self.centralwidget, 'diag0.png')
        self.dialogs.append(win)
        win2.show()

    def show_fak(self):
        win = AnotherWindow(self.centralwidget, 'fak.png')
        self.dialogs.append(win)
        win.show()

    def show_funcs(self):
        win = AnotherWindow(self.centralwidget, 'funcs.png')
        self.dialogs.append(win)
        win.show()

    def add_func(self):
        self.pushButton.clicked.connect(self.calc)
        self.pushButton_3.clicked.connect(self.show_diag)
        self.pushButton_2.clicked.connect(self.show_funcs)
        self.pushButton_4.clicked.connect(self.show_fak)


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
