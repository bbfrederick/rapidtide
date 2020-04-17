# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'graderTemplate.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(988, 525)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.timecourse_graphicsView = GraphicsLayoutWidget(self.centralwidget)
        self.timecourse_graphicsView.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.timecourse_graphicsView.sizePolicy().hasHeightForWidth())
        self.timecourse_graphicsView.setSizePolicy(sizePolicy)
        self.timecourse_graphicsView.setMinimumSize(QtCore.QSize(610, 100))
        self.timecourse_graphicsView.setMaximumSize(QtCore.QSize(3000, 1000))
        self.timecourse_graphicsView.setSizeIncrement(QtCore.QSize(1, 1))
        self.timecourse_graphicsView.setObjectName("timecourse_graphicsView")
        self.gridLayout.addWidget(self.timecourse_graphicsView, 0, 0, 1, 1)
        self.spectrum_graphicsView = GraphicsLayoutWidget(self.centralwidget)
        self.spectrum_graphicsView.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.spectrum_graphicsView.sizePolicy().hasHeightForWidth())
        self.spectrum_graphicsView.setSizePolicy(sizePolicy)
        self.spectrum_graphicsView.setMinimumSize(QtCore.QSize(610, 100))
        self.spectrum_graphicsView.setMaximumSize(QtCore.QSize(3000, 1000))
        self.spectrum_graphicsView.setSizeIncrement(QtCore.QSize(1, 1))
        self.spectrum_graphicsView.setObjectName("spectrum_graphicsView")
        self.gridLayout.addWidget(self.spectrum_graphicsView, 1, 0, 1, 1)
        self.gridLayout_2.addLayout(self.gridLayout, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 988, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
from pyqtgraph import GraphicsLayoutWidget
