##################################################
### Created by :Rehapriadarsini Manikandasamy
### Project Name : HR Analytics:Job Change
### Date 05/03/2021
### Data Mining
##################################################


#################################
## Load the packages 
#################################
import sys

from PyQt5.QtWidgets import (QMainWindow, QApplication, QWidget, QPushButton, QAction, QComboBox, QLabel,
                             QGridLayout, QCheckBox, QGroupBox, QVBoxLayout, QHBoxLayout, QLineEdit, QPlainTextEdit)
from PyQt5 import QtCore
from PyQt5.QtGui import QIcon, QImage , QPalette , QBrush,QPixmap
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtCore import Qt


from scipy import interp
from itertools import cycle

from PyQt5 import QtGui
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QSizePolicy, QMessageBox, QMenu

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import pandas as pd
import numpy as np
from numpy.polynomial.polynomial import polyfit

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
# from scikitplot.metrics import plot_roc_curve
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# Libraries to display decision tree
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
import webbrowser

import warnings

warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

import random
import seaborn as sns



#::--------------------------------
# Deafault font size for all the windows
#::--------------------------------
font_size_window = 'font-size:15px'




class SupportVector(QMainWindow):
    #::--------------------------------------------------------------------------------
    # Implementation of Support Vector Classifier using the HR dataset
    # the methods in this class are
    #       _init_ : initialize the class
    #       initUi : creates the canvas and add all the elements in the canvas
    #       update : populates the elements of the canvas based on the parameters
    #               chosen by the user
    #::---------------------------------------------------------------------------------
    send_fig = pyqtSignal(str)

    def __init__(self):
        super(SupportVector, self).__init__()
        self.Title = "Support Vector Classifier"
        self.initUi()

    def initUi(self):
        #::-----------------------------------------------------------------
        #  Create the canvas and all the element to create a dashboard with
        #  all the necessary elements to present the results from the algorithm
        #  The canvas is divided using a  grid layout to facilitate the drawing
        #  of the elements
        #::-----------------------------------------------------------------

        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)

        self.main_widget = QWidget(self)

        self.layout = QGridLayout(self.main_widget)

        self.groupBox1 = QGroupBox('Support Vector Features')
        self.groupBox1Layout = QGridLayout()
        self.groupBox1.setLayout(self.groupBox1Layout)

        # We create a checkbox of each Features
        self.feature = []

        for i in range(12):
            self.feature.append(QCheckBox(features_list[i], self))

        for i in self.feature:
            i.setChecked(True)

        self.lblPercentTest = QLabel('Percentage for Test :')
        self.lblPercentTest.adjustSize()

        self.txtPercentTest = QLineEdit(self)
        self.txtPercentTest.setText("30")

        self.lblKernelType = QLabel('kernel :')
        self.lblKernelType.adjustSize()

        self.txtKernelType = QLineEdit(self)
        self.txtKernelType.setText("rbf")

        self.btnExecute = QPushButton("Build Model")
        self.btnExecute.clicked.connect(self.update1)

        self.btnRoc_Execute = QPushButton("Plot ROC")
        self.btnRoc_Execute.clicked.connect(self.roc_update)

        self.groupBox1Layout.addWidget(self.feature[0], 0, 0)
        self.groupBox1Layout.addWidget(self.feature[1], 0, 1)
        self.groupBox1Layout.addWidget(self.feature[2], 1, 0)
        self.groupBox1Layout.addWidget(self.feature[3], 1, 1)
        self.groupBox1Layout.addWidget(self.feature[4], 2, 0)
        self.groupBox1Layout.addWidget(self.feature[5], 2, 1)
        self.groupBox1Layout.addWidget(self.feature[6], 3, 0)
        self.groupBox1Layout.addWidget(self.feature[7], 3, 1)
        self.groupBox1Layout.addWidget(self.feature[8], 4, 0)
        self.groupBox1Layout.addWidget(self.feature[9], 4, 1)
        self.groupBox1Layout.addWidget(self.feature[10], 5, 0)
        self.groupBox1Layout.addWidget(self.feature[11], 5, 1)
        self.groupBox1Layout.addWidget(self.lblPercentTest, 7, 0)
        self.groupBox1Layout.addWidget(self.txtPercentTest, 7, 1)
        self.groupBox1Layout.addWidget(self.lblKernelType, 8, 0)
        self.groupBox1Layout.addWidget(self.txtKernelType, 8, 1)
        self.groupBox1Layout.addWidget(self.btnExecute, 9, 0)
        self.groupBox1Layout.addWidget(self.btnRoc_Execute, 10, 0)

        self.groupBox2 = QGroupBox('Results from the SVC model')
        self.groupBox2Layout = QVBoxLayout()
        self.groupBox2.setLayout(self.groupBox2Layout)

        self.lblResults1 = QLabel('Results :')
        self.lblResults1.adjustSize()
        self.txtResults1 = QPlainTextEdit()
        self.lblAccuracy1 = QLabel('Accuracy :')
        self.txtAccuracy1 = QLineEdit()
        self.lblRoc_auc1 = QLabel('ROC_AUC :')
        self.txtRoc_auc1 = QLineEdit()

        self.groupBox2Layout.addWidget(self.lblResults1)
        self.groupBox2Layout.addWidget(self.txtResults1)
        self.groupBox2Layout.addWidget(self.lblAccuracy1)
        self.groupBox2Layout.addWidget(self.txtAccuracy1)
        self.groupBox2Layout.addWidget(self.lblRoc_auc1)
        self.groupBox2Layout.addWidget(self.txtRoc_auc1)

        #::--------------------------------------
        # Graphic 1 : Confusion Matrix - SVC model
        #::--------------------------------------

        self.fig1 = Figure()
        self.ax1 = self.fig1.add_subplot(111)
        self.axes = [self.ax1]
        self.canvas1 = FigureCanvas(self.fig1)

        self.canvas1.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas1.updateGeometry()

        self.groupBoxG1 = QGroupBox('Confusion Matrix :')
        self.groupBoxG1Layout = QVBoxLayout()
        self.groupBoxG1.setLayout(self.groupBoxG1Layout)

        self.groupBoxG1Layout.addWidget(self.canvas1)

        #::-------------------------------------------------
        # End of graphs
        #::-------------------------------------------------

        self.layout.addWidget(self.groupBox1, 0, 0)
        self.layout.addWidget(self.groupBoxG1, 0, 1)
        self.layout.addWidget(self.groupBox2, 1, 1)

        self.setCentralWidget(self.main_widget)
        self.resize(1100, 700)
        self.show()
    def Message(self):
        QMessageBox.about(self, "Warning", " You have not selected any features")
    def update1(self):
        # processing the parameters
        self.current_features = pd.DataFrame([])
        self.notchecked=0
        for i in range(12):
            if self.feature[i].isChecked():
                if len(self.current_features) == 0:
                    self.current_features = data[features_list[i]]
                else:
                    self.current_features = pd.concat([self.current_features, data[features_list[i]]], axis=1)
            else:
                self.notchecked+=1

        if self.notchecked==12:
            self.Message()
        else:
            self.update()

    def update(self):
        '''
        Support Vector Classifier
        We pupulate the dashboard using the parameters chosen by the user
        The parameters are processed to execute in the skit-learn Support Vector algorithm
          then the results are presented in graphics and reports in the canvas
        '''

        vtest_per = float(self.txtPercentTest.text())
        kernel1 = self.txtKernelType.text()
        # Clear the graphs to populate them with the new information

        self.ax1.clear()

        self.txtResults1.clear()
        self.txtResults1.setUndoRedoEnabled(False)

        vtest_per = vtest_per / 100

        # Assign the X and y to run the support vector classifier
        class_le1 = LabelEncoder()
        if self.notchecked==11:
            self.current_features=class_le1.fit_transform(self.current_features)
            X=self.current_features
            X=X.reshape(-1,1)
        else:
            features_list1 = self.current_features.loc[:, self.current_features.dtypes == 'object'].columns
            for i in features_list1:
                self.current_features[i] = class_le1.fit_transform(self.current_features[i])
            X = self.current_features.values

        class_le1 = LabelEncoder()
        features_list1 = self.current_features.loc[:, self.current_features.dtypes == 'object'].columns
        for i in features_list1:
            self.current_features[i] = class_le1.fit_transform(self.current_features[i])

        X = self.current_features.values

        y = data.iloc[:, -1]

        class_le = LabelEncoder()

        # fit and transform the class

        y = class_le.fit_transform(y)

        # split the dataset into train and test

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=vtest_per, random_state=100)

       

        #::------------------------------------
        ##  Model

        #::------------------------------------

        # specify support vector classifier
        self.clf_svc = SVC(kernel=kernel1)

        # perform training
        self.clf_svc.fit(X_train, y_train)

        # -----------------------------------------------------------------------

        # predicton on test using all features
        y_pred = self.clf_svc.predict(X_test)
        y_pred_score = self.clf_svc.decision_function(X_test)

        # confusion matrix for RandomForest
        conf_matrix = confusion_matrix(y_test, y_pred)

        # clasification report

        self.class_rep = classification_report(y_test, y_pred)
        self.txtResults1.appendPlainText(self.class_rep)

        # accuracy score

        self.accuracy_score = accuracy_score(y_test, y_pred) * 100
        self.txtAccuracy1.setText(str(self.accuracy_score))

        # ROC-AUC
        self.rocauc_score = roc_auc_score(y_test, y_pred_score) * 100
        self.txtRoc_auc1.setText(str(self.rocauc_score))

        self.fpr, self.tpr, _ = roc_curve(y_test, y_pred_score)
        self.auc = roc_auc_score(y_test, y_pred_score)


        #::------------------------------------
        ##  Graph1 :
        ##  Confusion Matrix
        #::------------------------------------

        df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)

        hm1 = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 15},
                          yticklabels=df_cm.columns, xticklabels=df_cm.columns, ax=self.ax1)

        hm1.yaxis.set_ticklabels(hm1.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
        hm1.xaxis.set_ticklabels(hm1.xaxis.get_ticklabels(), rotation=90, ha='right', fontsize=10)
        self.ax1.set_xlabel('Predicted label')
        self.ax1.set_ylabel('True label')
        self.fig1.tight_layout()
        self.fig1.canvas.draw_idle()

    def roc_update(self):
        dialog = ROC_Main(self)
        dialog.roc.plot()
        dialog.roc.ax.plot(self.fpr, self.tpr, color='#90EE90', lw=3, label='ROC curve (area = %0.2f)' % self.auc)
        dialog.roc.ax.plot([0, 1], [0, 1], color='blue', lw=3, linestyle='--')
        dialog.roc.ax.set_title('ROC of SVC model')
        dialog.roc.ax.set_xlim([0.0, 1.0])
        dialog.roc.ax.set_ylim([0.0, 1.0])
        dialog.roc.ax.set_xlabel("False Positive Rate")
        dialog.roc.ax.set_ylabel("True Positive Rate")
        dialog.roc.ax.legend(loc="lower right")
        dialog.roc.draw()
        dialog.show()
#---------------------------------------------------------
# Class to Plot any Graph, used from ROC_Main()
# Imp_Main() classes
#----------------------------------------------------------
class Plotter(FigureCanvas):
    #::----------------------------------------------------------
    # creates a figure on the canvas
    # later on this element will be used to draw a ROC curve
    #::----------------------------------------------------------
    def __init__(self, parent=None, width=7, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)

        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def plot(self):
        self.ax = self.figure.add_subplot(111)
#
#---------------------------------------------------------
# Class to plot ROC curves
#----------------------------------------------------------
class ROC_Main(QMainWindow):
    #::----------------------------------
    # Creates a canvas containing the plot for the ROC curve
    # ;;----------------------------------
    def __init__(self, parent=None):
        super(ROC_Main, self).__init__(parent)

        self.left = 250
        self.top = 250
        self.Title = 'ROC curve'
        self.width = 700
        self.height = 600
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)

        self.setGeometry(self.left, self.top, self.width, self.height)

        self.roc = Plotter(self, width=7, height=6)


class App(QMainWindow):
    #::-------------------------------------------------------
    # This class creates all the elements of the application
    #::-------------------------------------------------------

    def __init__(self):
        super().__init__()
        self.left = 100
        self.top = 100
        self.Title = 'The HR decision prediction'
        self.width = 800
        self.height = 500
        self.initUI()

    def initUI(self):
        #::-------------------------------------------------
        # Creates the menu and the items
        #::-------------------------------------------------
        self.setWindowTitle(self.Title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.label5 = QLabel(self)
        self.label5.setPixmap(QPixmap("backg.jpg"))
        self.label5.setGeometry(0,-10,900,550)

        #::-----------------------------
        # Create the menu bar
        # and setsup UI with buttons to load dataset, EDA Analysis, ML models
        #::-----------------------------
        mainMenu = self.menuBar()
        mainMenu.setStyleSheet("color: white;"
                               "background-color: black;"
                               "selection-color: black;"
                               "selection-background-color: white;")


        label1 = QLabel(self)
        label1.setText("<font color = white>HR Analytics Application</font>")
        label1.setFont(QtGui.QFont("Times", 16, QtGui.QFont.Bold))
        label1.move(200, 5)
        label1.resize(400, 350)

        fileMenu = mainMenu.addMenu('File')

        #::--------------------------------------
        # Exit application
        # Creates the actions for the fileMenu item
        #::--------------------------------------

        exitButton = QAction(QIcon('enter.jpg'), 'Exit', self)
        exitButton.setShortcut('Ctrl+Q')
        exitButton.setStatusTip('Exit application')
        exitButton.triggered.connect(self.close)

        fileMenu.addAction(exitButton)

        #::----------------------------------------
        # Upload dataset

        #::----------------------------------------
        l1=QLabel(self)
        l1.setText("<font color = white>Load Dataset</font>")
        l1.move(60,250)
        l1.setAlignment(QtCore.Qt.AlignCenter)
        l1.setStyleSheet("background-color:black")

        btn = QPushButton("Upload Data", self)
        btn.setStyleSheet('''font-size:15px;''')
        btn.clicked.connect(self.find_dataset)
        btn.resize(140, 40)
        btn.move(40, 290)

        #::----------------------------------------
        # EDA analysis
        # create a histogram of choice
        # Create a scatterplot of choice
        # ML models
        # Create DT, RF, SVC of choice
        #::----------------------------------------
        l2 = QLabel(self)
        l2.setText("<font color = white>EDA Analysis</font>")
        l2.move(300,250)
        l2.setStyleSheet("background-color:black")
        l2.setAlignment(QtCore.Qt.AlignCenter)
        analysis_menu = ['Histogram', 'Scatter Plot']

        self.comboBox = QComboBox(self)
        self.comboBox.setGeometry(255, 290, 200, 40)
        self.comboBox.addItems(analysis_menu)

        self.comboBox.setEditable(True)

        # getting the line edit of combo box
        line_edit = self.comboBox.lineEdit()

        # setting line edit alignment to the center
        line_edit.setAlignment(Qt.AlignCenter)

        # setting line edit to read only
        line_edit.setReadOnly(True)
        self.btn2 = QPushButton('Open', self)
        self.btn2.setGeometry(300, 340, 100, 35)
        self.btn2.clicked.connect(self.getComboValue)

        l3 = QLabel(self)
        l3.setText("<font color = white>ML Models</font>")
        l3.move(600, 250)
        l3.setStyleSheet("background-color:black")
        l3.setAlignment(QtCore.Qt.AlignCenter)
        ml_menu = ['Decision Tree Classifier', 'Random Forest Classifier', 'Support Vector Machine']

        self.comboBox1 = QComboBox(self)
        self.comboBox1.setGeometry(540, 290, 210, 40)
        self.comboBox1.addItems(ml_menu)
        self.comboBox1.setEditable(True)

        # getting the line edit of combo box
        line_edit1 = self.comboBox1.lineEdit()

        # setting line edit alignment to the center
        line_edit1.setAlignment(Qt.AlignCenter)

        # setting line edit to read only
        line_edit1.setReadOnly(True)
        self.btn3 = QPushButton('Open', self)
        self.btn3.setGeometry(600, 340, 100, 35)
        self.btn3.clicked.connect(self.getComboValue1)

        self.dialogs = list()

    def getComboValue(self):
        if upload1==0:
            self.Message_up()
        else:
            if self.comboBox.currentText() == 'Histogram':
                self.EDA1()
            else:
                self.EDA2()

    def getComboValue1(self):
        if upload1==0:
            self.Message_up()
        else:
            if self.comboBox1.currentText() == 'Decision Tree Classifier':
                self.MLDT()
            elif self.comboBox1.currentText() == 'Random Forest Classifier':
                self.MLRF()
            else:
                self.MLSVM()

    def find_dataset(self):
        dialog = Data_find()
        self.dialogs.append(dialog)
        dialog.show()

    def EDA1(self):
        #::------------------------------------------------------
        # Creates the histogram
        #::------------------------------------------------------
        dialog = Histogram_plots()
        self.dialogs.append(dialog)
        dialog.show()

    def EDA2(self):
        #::------------------------------------------------------
        # Creates the scatter plot
        #::------------------------------------------------------
        dialog = Scatter_plots()
        self.dialogs.append(dialog)
        dialog.show()

    def MLDT(self):
        #::-----------------------------------------------------------
        # This function creates an instance of the DecisionTree class
        # using the HR Analytics dataset 
        #::-----------------------------------------------------------
        dialog = DecisionTree()
        self.dialogs.append(dialog)
        dialog.show()

    def MLRF(self):
        #::-------------------------------------------------------------
        # This function creates an instance of the Random Forest Classifier Algorithm
        # using the HR Analytics dataset
        #::-------------------------------------------------------------
        dialog = RandomForest()
        self.dialogs.append(dialog)
        dialog.show()

    def MLSVM(self):
        #::-------------------------------------------------------------
        # This function creates an instance of the Support Vector Classifier Algorithm
        # using the HR Analytics dataset
        #::-------------------------------------------------------------
        dialog = SupportVector()
        self.dialogs.append(dialog)
        dialog.show()
    def Message_up(self):
        QMessageBox.about(self, "Warning", " You have not Uploaded the data")


def main():
    #::-------------------------------------------------
    # Initiates the application
    #::-------------------------------------------------
    global upload1
    upload1=0
    app = QApplication(sys.argv)
    app.setStyle('Breeze')
    ex = App()
    ex.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    #::------------------------------------
    # First reads the data then calls for the application
    #::------------------------------------
    main()
