import sys

from PyQt5.QtWidgets import (QMainWindow, QApplication, QWidget, QPushButton, QAction, QComboBox, QLabel,
                             QGridLayout, QCheckBox, QGroupBox, QVBoxLayout, QHBoxLayout, QLineEdit, QPlainTextEdit)
from PyQt5 import QtCore
from PyQt5.QtGui import QIcon, QImage, QPalette, QBrush, QPixmap
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

class RandomForest(QMainWindow):
    #::--------------------------------------------------------------------------------
    # Implementation of Random Forest Classifier using the HR dataset
    # the methods in this class are
    #       _init_ : initialize the class
    #       initUi : creates the canvas and add all the elements in the canvas
    #       update : populates the elements of the canvas based on the parameters
    #                chosen by the user
    #::---------------------------------------------------------------------------------
    send_fig = pyqtSignal(str)

    def __init__(self):
        super(RandomForest, self).__init__()
        self.Title = "Random Forest Classifier"
        self.initUi()

    def initUi(self):
        #::-----------------------------------------------------------------
        #  Create the canvas and all the element to create a dashboard with
        #  all the necessary elements to present the results from the algorithm
        #  The canvas is divided using a  grid loyout to facilitate the drawing
        #  of the elements
        #::-----------------------------------------------------------------

        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)

        self.main_widget = QWidget(self)

        self.layout = QGridLayout(self.main_widget)

        self.groupBox1 = QGroupBox('Random Forest Features')
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

        self.lblNumberesti = QLabel('Number of estimators :')
        self.lblNumberesti.adjustSize()

        self.txtNumberesti = QLineEdit(self)
        self.txtNumberesti.setText("10")

        self.btnExecute = QPushButton("Build Model")
        self.btnExecute.clicked.connect(self.update1)

        self.btnRoc_Execute = QPushButton("Plot ROC")
        self.btnRoc_Execute.clicked.connect(self.roc_update)

        self.btnImp_Execute = QPushButton("Imp_Features")
        self.btnImp_Execute.clicked.connect(self.imp_update)

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
        self.groupBox1Layout.addWidget(self.lblNumberesti, 8, 0)
        self.groupBox1Layout.addWidget(self.txtNumberesti, 8, 1)
        self.groupBox1Layout.addWidget(self.btnExecute, 9, 0)
        self.groupBox1Layout.addWidget(self.btnRoc_Execute, 9, 1)
        self.groupBox1Layout.addWidget(self.btnImp_Execute, 10, 0)

        self.groupBox2 = QGroupBox('Results from the Gini model')
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

        self.groupBox3 = QGroupBox('Results from the Entropy model')
        self.groupBox3Layout = QVBoxLayout()
        self.groupBox3.setLayout(self.groupBox3Layout)

        self.lblResults2 = QLabel('Results :')
        self.lblResults2.adjustSize()
        self.txtResults2 = QPlainTextEdit()
        self.lblAccuracy2 = QLabel('Accuracy :')
        self.txtAccuracy2 = QLineEdit()
        self.lblRoc_auc2 = QLabel('ROC_AUC :')
        self.txtRoc_auc2 = QLineEdit()

        self.groupBox3Layout.addWidget(self.lblResults2)
        self.groupBox3Layout.addWidget(self.txtResults2)
        self.groupBox3Layout.addWidget(self.lblAccuracy2)
        self.groupBox3Layout.addWidget(self.txtAccuracy2)
        self.groupBox3Layout.addWidget(self.lblRoc_auc2)
        self.groupBox3Layout.addWidget(self.txtRoc_auc2)

        #::--------------------------------------
        # Graphic 1 : Confusion Matrix - Gini model
        #::--------------------------------------

        self.fig1 = Figure()
        self.ax1 = self.fig1.add_subplot(111)
        self.axes = [self.ax1]
        self.canvas1 = FigureCanvas(self.fig1)

        self.canvas1.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas1.updateGeometry()

        self.groupBoxG1 = QGroupBox('Confusion Matrix (Gini model):')
        self.groupBoxG1Layout = QVBoxLayout()
        self.groupBoxG1.setLayout(self.groupBoxG1Layout)

        self.groupBoxG1Layout.addWidget(self.canvas1)

        #::--------------------------------------
        # Graphic 2 : Confusion Matrix - Entropy model
        #::--------------------------------------

        self.fig2 = Figure()
        self.ax2 = self.fig2.add_subplot(111)
        self.axes1 = [self.ax2]
        self.canvas2 = FigureCanvas(self.fig2)

        self.canvas2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas2.updateGeometry()

        self.groupBoxG2 = QGroupBox('Confusion Matrix (Entropy model):')
        self.groupBoxG2Layout = QVBoxLayout()
        self.groupBoxG2.setLayout(self.groupBoxG2Layout)

        self.groupBoxG2Layout.addWidget(self.canvas2)

        #::-------------------------------------------------
        # End of graphs
        #::-------------------------------------------------

        self.layout.addWidget(self.groupBox1, 0, 0)
        self.layout.addWidget(self.groupBoxG1, 0, 1)
        self.layout.addWidget(self.groupBox2, 1, 1)
        self.layout.addWidget(self.groupBoxG2, 0, 2)
        self.layout.addWidget(self.groupBox3, 1, 2)

        self.setCentralWidget(self.main_widget)
        self.resize(1100, 700)
        self.show()

    def Message(self):
        QMessageBox.about(self, "Warning", " You have not selected any features")

    def update1(self):
        self.current_features = pd.DataFrame([])
        self.notchecked = 0
        for i in range(12):
            if self.feature[i].isChecked():
                if len(self.current_features) == 0:
                    self.current_features = data[features_list[i]]
                else:
                    self.current_features = pd.concat([self.current_features, data[features_list[i]]], axis=1)
            else:
                self.notchecked += 1
        if self.notchecked == 12:
            self.Message()
        else:
            self.update()

    def update(self):
        '''
        Random Forest Classifier
        We populate the dashboard using the parameters chosen by the user
        The parameters are processed to execute in the skit-learn Random Forest algorithm
        then the results are presented in graphics and reports in the canvas

        '''

        vtest_per = float(self.txtPercentTest.text())
        n_esti = int(self.txtNumberesti.text())
        # Clear the graphs to populate them with the new information

        self.ax1.clear()
        self.ax2.clear()

        self.txtResults1.clear()
        self.txtResults1.setUndoRedoEnabled(False)

        self.txtResults2.clear()
        self.txtResults2.setUndoRedoEnabled(False)

        vtest_per = vtest_per / 100

        # # label encoding the categorical data
        class_le1 = LabelEncoder()

        if self.notchecked == 11:
            self.current_features = class_le1.fit_transform(self.current_features)
            X = self.current_features
            X = X.reshape(-1, 1)
        else:
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
        ##  Model 1 - gini model:

        #::------------------------------------

        # specify random forest classifier
        self.clf_rf_gini = RandomForestClassifier(n_estimators=n_esti, criterion='gini', random_state=100)

        # perform training
        self.clf_rf_gini.fit(X_train, y_train)

        # -----------------------------------------------------------------------

        # predicton on test using all features
        y_pred_gini = self.clf_rf_gini.predict(X_test)
        y_pred_score_gini = self.clf_rf_gini.predict_proba(X_test)

        # confusion matrix for RandomForest
        conf_matrix_gini = confusion_matrix(y_test, y_pred_gini)

        # clasification report

        self.class_rep_gini = classification_report(y_test, y_pred_gini)
        self.txtResults1.appendPlainText(self.class_rep_gini)

        # accuracy score

        self.accuracy_score_gini = accuracy_score(y_test, y_pred_gini) * 100
        self.txtAccuracy1.setText(str(self.accuracy_score_gini))

        # ROC-AUC
        self.rocauc_score_gini = roc_auc_score(y_test, y_pred_score_gini[:, 1]) * 100
        self.txtRoc_auc1.setText(str(self.rocauc_score_gini))

        self.fpr_gini, self.tpr_gini, _ = roc_curve(y_test, y_pred_score_gini[:, 1])
        self.auc_gini = roc_auc_score(y_test, y_pred_score_gini[:, 1])

        # important features

        importances_gini = self.clf_rf_gini.feature_importances_

        # convert the importances into one-dimensional 1darray with corresponding df column names as axis labels
        f_importances_gini = pd.Series(importances_gini, self.current_features.columns)

        # sort the array in descending order of the importances
        f_importances_gini.sort_values(ascending=True, inplace=True)

        self.X_Features_gini = f_importances_gini.index
        self.y_Importance_gini = list(f_importances_gini)

        #::------------------------------------
        ##  Ghaph1 :
        ##  Confusion Matrix
        #::------------------------------------

        df_cm_gini = pd.DataFrame(conf_matrix_gini, index=class_names, columns=class_names)

        hm1 = sns.heatmap(df_cm_gini, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 15},
                          yticklabels=df_cm_gini.columns, xticklabels=df_cm_gini.columns, ax=self.ax1)

        hm1.yaxis.set_ticklabels(hm1.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
        hm1.xaxis.set_ticklabels(hm1.xaxis.get_ticklabels(), rotation=90, ha='right', fontsize=10)
        self.ax1.set_xlabel('Predicted label')
        self.ax1.set_ylabel('True label')
        self.fig1.tight_layout()
        self.fig1.canvas.draw_idle()

        #::------------------------------------
        ##  Model 2 - entropy model:

        #::------------------------------------

        # specify random forest classifier
        self.clf_rf_entropy = RandomForestClassifier(n_estimators=n_esti, criterion='entropy', random_state=100)

        # perform training
        self.clf_rf_entropy.fit(X_train, y_train)

        # predicton on test using all features
        y_pred_entropy = self.clf_rf_entropy.predict(X_test)
        y_pred_score_entropy = self.clf_rf_entropy.predict_proba(X_test)

        # confusion matrix for RandomForest
        conf_matrix_entropy = confusion_matrix(y_test, y_pred_entropy)

        # clasification report

        self.class_rep_entropy = classification_report(y_test, y_pred_entropy)
        self.txtResults2.appendPlainText(self.class_rep_entropy)

        # accuracy score

        self.accuracy_score_entropy = accuracy_score(y_test, y_pred_entropy) * 100
        self.txtAccuracy2.setText(str(self.accuracy_score_entropy))

        # ROC-AUC
        self.rocauc_score_entropy = roc_auc_score(y_test, y_pred_score_entropy[:, 1]) * 100
        self.txtRoc_auc2.setText(str(self.rocauc_score_entropy))

        self.fpr_entropy, self.tpr_entropy, _ = roc_curve(y_test, y_pred_score_entropy[:, 1])
        self.auc_entropy = roc_auc_score(y_test, y_pred_score_entropy[:, 1])

        # important features

        importances_entropy = self.clf_rf_entropy.feature_importances_

        # convert the importances into one-dimensional 1darray with corresponding df column names as axis labels
        f_importances_entropy = pd.Series(importances_entropy, self.current_features.columns)

        # sort the array in descending order of the importances
        f_importances_entropy.sort_values(ascending=True, inplace=True)

        self.X_Features_entropy = f_importances_entropy.index
        self.y_Importance_entropy = list(f_importances_entropy)

        #::------------------------------------
        ##  Ghaph2 :
        ##  Confusion Matrix - entropy model
        #::------------------------------------

        df_cm_entropy = pd.DataFrame(conf_matrix_entropy, index=class_names, columns=class_names)

        hm2 = sns.heatmap(df_cm_entropy, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 15},
                          yticklabels=df_cm_entropy.columns, xticklabels=df_cm_entropy.columns, ax=self.ax2)

        hm2.yaxis.set_ticklabels(hm2.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
        hm2.xaxis.set_ticklabels(hm2.xaxis.get_ticklabels(), rotation=90, ha='right', fontsize=10)
        self.ax2.set_xlabel('Predicted label')
        self.ax2.set_ylabel('True label')
        self.fig2.tight_layout()
        self.fig2.canvas.draw_idle()

    def roc_update(self):
        # gini model
        dialog = ROC_Main(self)

        dialog.roc.plot()
        dialog.roc.ax.plot(self.fpr_gini, self.tpr_gini, color='#90EE90', lw=3,
                           label='ROC curve (area = %0.2f)' % self.auc_gini)
        dialog.roc.ax.plot([0, 1], [0, 1], color='blue', lw=3, linestyle='--')
        dialog.roc.ax.set_title('ROC of Gini model')
        dialog.roc.ax.set_xlim([0.0, 1.0])
        dialog.roc.ax.set_ylim([0.0, 1.0])
        dialog.roc.ax.set_xlabel("False Positive Rate")
        dialog.roc.ax.set_ylabel("True Positive Rate")
        dialog.roc.ax.legend(loc="lower right")
        dialog.roc.draw()
        dialog.show()

        # entropy model
        dialog = ROC_Main(self)
        dialog.roc.plot()
        dialog.roc.ax.plot(self.fpr_entropy, self.tpr_entropy, color='#90EE90', lw=3,
                           label='ROC curve (area = %0.2f)' % self.auc_entropy)
        dialog.roc.ax.plot([0, 1], [0, 1], color='blue', lw=3, linestyle='--')
        dialog.roc.ax.set_title('ROC of Entropy model')
        dialog.roc.ax.set_xlim([0.0, 1.0])
        dialog.roc.ax.set_ylim([0.0, 1.0])
        dialog.roc.ax.set_xlabel("False Positive Rate")
        dialog.roc.ax.set_ylabel("True Positive Rate")
        dialog.roc.ax.legend(loc="lower right")
        dialog.roc.draw()
        dialog.show()

    def imp_update(self):
        # gini model
        dialog = Imp_Main(self)

        dialog.imp.plot()
        dialog.imp.ax.barh(self.X_Features_gini, self.y_Importance_gini)
        dialog.imp.ax.set_title('Important features - Gini model')
        dialog.imp.ax.set_xlabel("Importance")
        dialog.imp.ax.set_ylabel("Features")
        dialog.imp.fig.tight_layout()
        dialog.imp.draw()
        dialog.show()

        # entropy model
        dialog = Imp_Main(self)

        dialog.imp.plot()
        dialog.imp.ax.barh(self.X_Features_entropy, self.y_Importance_entropy)
        dialog.imp.ax.set_title('Important features - Entropy model')
        dialog.imp.ax.set_xlabel("Importance")
        dialog.imp.ax.set_ylabel("Features")
        dialog.imp.fig.tight_layout()
        dialog.imp.draw()
        dialog.show()

class Scatter_plots(QMainWindow):
    send_fig = pyqtSignal(str)

    def __init__(self):
        super(Scatter_plots, self).__init__()
        self.Title = "Scatter Plots"
        self.initUi()

    def initUi(self):

        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)

        self.main_widget = QWidget(self)

        self.layout = QGridLayout(self.main_widget)

        self.groupBox1 = QGroupBox('Select X-variable here')
        self.groupBox1Layout = QGridLayout()  # Grid
        self.groupBox1.setLayout(self.groupBox1Layout)
        self.work_x = 0
        self.work_y = 0
        self.feature1 = []

        for i in range(13):
            self.feature1.append(QCheckBox(features_list_hist[i], self))

        for i in self.feature1:
            i.setChecked(False)

        self.btnSelectx = QPushButton("Select X-variable")
        self.btnSelectx.clicked.connect(self.select_x)

        self.groupBox1Layout.addWidget(self.feature1[0], 0, 0)
        self.groupBox1Layout.addWidget(self.feature1[1], 0, 1)
        self.groupBox1Layout.addWidget(self.feature1[2], 0, 2)
        self.groupBox1Layout.addWidget(self.feature1[3], 0, 3)
        self.groupBox1Layout.addWidget(self.feature1[4], 0, 4)
        self.groupBox1Layout.addWidget(self.feature1[5], 0, 5)
        self.groupBox1Layout.addWidget(self.feature1[6], 0, 6)
        self.groupBox1Layout.addWidget(self.feature1[7], 1, 0)
        self.groupBox1Layout.addWidget(self.feature1[8], 1, 1)
        self.groupBox1Layout.addWidget(self.feature1[9], 1, 2)
        self.groupBox1Layout.addWidget(self.feature1[10], 1, 3)
        self.groupBox1Layout.addWidget(self.feature1[11], 1, 4)
        self.groupBox1Layout.addWidget(self.feature1[12], 1, 5)
        self.groupBox1Layout.addWidget(self.btnSelectx, 2, 0)

        self.groupBox2 = QGroupBox('Select Y-variable here')
        self.groupBox2Layout = QGridLayout()
        self.groupBox2.setLayout(self.groupBox2Layout)

        self.feature2 = []

        for i in range(13):
            self.feature2.append(QCheckBox(features_list_hist[i], self))

        for i in self.feature2:
            i.setChecked(False)

        self.btnSelecty = QPushButton("Select Y-variable")
        self.btnSelecty.clicked.connect(self.select_y)

        self.btnExecute = QPushButton("Create Plot")
        self.btnExecute.clicked.connect(self.update)

        self.groupBox2Layout.addWidget(self.feature2[0], 0, 0)
        self.groupBox2Layout.addWidget(self.feature2[1], 0, 1)
        self.groupBox2Layout.addWidget(self.feature2[2], 1, 0)
        self.groupBox2Layout.addWidget(self.feature2[3], 1, 1)
        self.groupBox2Layout.addWidget(self.feature2[4], 2, 0)
        self.groupBox2Layout.addWidget(self.feature2[5], 2, 1)
        self.groupBox2Layout.addWidget(self.feature2[6], 3, 0)
        self.groupBox2Layout.addWidget(self.feature2[7], 3, 1)
        self.groupBox2Layout.addWidget(self.feature2[8], 4, 0)
        self.groupBox2Layout.addWidget(self.feature2[9], 4, 1)
        self.groupBox2Layout.addWidget(self.feature2[10], 5, 0)
        self.groupBox2Layout.addWidget(self.feature2[11], 5, 1)
        self.groupBox2Layout.addWidget(self.feature2[12], 6, 0)
        self.groupBox2Layout.addWidget(self.btnSelecty, 6, 1)
        self.groupBox2Layout.addWidget(self.btnExecute, 7, 1)

        self.fig2, self.ax2 = plt.subplots()
        self.axes = [self.ax2]
        self.canvas2 = FigureCanvas(self.fig2)

        self.canvas2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas2.updateGeometry()

        self.groupBoxG1 = QGroupBox('Scatter Plot :')
        self.groupBoxG1Layout = QVBoxLayout()
        self.groupBoxG1.setLayout(self.groupBoxG1Layout)

        self.groupBoxG1Layout.addWidget(self.canvas2)

        self.layout.addWidget(self.groupBox1, 0, 1)
        self.layout.addWidget(self.groupBox2, 1, 0)
        self.layout.addWidget(self.groupBoxG1, 1, 1)

        self.setCentralWidget(self.main_widget)
        self.resize(1200, 900)
        self.show()

    def Message_x(self):
        QMessageBox.about(self, "Warning", " You selected more than 1 X-variable")

    def Message_y(self):
        QMessageBox.about(self, "Warning", " You selected more than 1 Y-variable")

    def select_x(self):
        self.current_x = pd.DataFrame([])
        for i in range(13):
            if self.feature1[i].isChecked():
                if len(self.current_x) > 1:
                    self.work_x = 1
                    self.Message_x()
                    break
                elif len(self.current_x) == 0:
                    self.current_x = data[features_list_hist[i]]
                    self.x_a = features_list_hist[i]
                    self.work_x = 0

    def select_y(self):
        self.current_y = pd.DataFrame([])
        for i in range(13):
            if self.feature2[i].isChecked():
                if len(self.current_y) > 1:
                    self.work_y = 1
                    self.Message_y()
                    break

                elif len(self.current_y) == 0:
                    self.current_y = data[features_list_hist[i]]
                    self.y_a = features_list_hist[i]
                    self.work_y = 0

    def update(self):
        if self.work_x == 0 and self.work_y == 0:
            self.ax2.clear()
            self.ax2.scatter(self.current_x, self.current_y)
            self.ax2.set_title('Scatter plot : ' + self.y_a + ' vs ' + self.x_a)
            self.ax2.set_xlabel(self.x_a)
            self.ax2.set_ylabel(self.y_a)
            self.fig2.tight_layout()
            self.fig2.canvas.draw_idle()
        elif self.work_x == 1 and self.work_y == 0:
            self.Message_x()
        elif self.work_x == 0 and self.work_y == 1:
            self.Message_y()
        elif self.work_x == 1 and self.work_y == 1:
            self.Message_x()
            self.Message_y()

