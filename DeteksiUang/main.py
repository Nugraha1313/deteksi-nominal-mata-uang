import glob
import sys
import cv2
import numpy as np
from PyQt5 import QtWidgets,QtGui
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QFileDialog
import utils
from DeteksiUang import Ui_MainWindow

#global variable
mataUangValue = ""
dataset = []

class window(QtWidgets.QMainWindow):

    def __init__(self):
        super(window, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowIcon(QtGui.QIcon('./logo/deteksi-uang-logo.png'))
        self.loadTemplate()
        self.ui.PilihUangButton.clicked.connect(self.tampilkanUang)
        self.ui.DeteksiButton.clicked.connect(self.deteksi_uang)

    def tampilkanUang(self):
        global mataUangValue
        fname = QFileDialog.getOpenFileName(self, 'Open file',
                                            '', "Image files (*.jpg)")
        mataUangValue = fname[0]
        print(mataUangValue)
        self.ui.HasilTextEdit.setPlainText("")
        self.ui.GambarMataUang.clear()
        pixmap = QPixmap(mataUangValue)
        pixmap3 = pixmap.scaledToHeight(275)
        self.ui.GambarMataUang.setPixmap(pixmap3)

    def loadTemplate(self):
        files_dataset = glob.glob('dataset_template/*/*.jpg', recursive=True)
        print("Dataset Loaded :", files_dataset)

        for file_dataset in files_dataset:
            temp = cv2.imread(file_dataset)
            temp = utils.resize(temp, width=int(temp.shape[1] * 0.5))
            temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
            temp = cv2.Canny(temp, 50, 200)
            nominal = file_dataset.replace('dataset_template\\', '').replace('\\1.jpg', '').replace('\\2.jpg', '').replace('\\3.jpg', '').replace('\\4.jpg', '').replace('\\5.jpg', '').replace('\\6.jpg', '').replace('\\7.jpg', '').replace('\\8.jpg', '').replace('\\9.jpg', '').replace('\\10.jpg', '').replace('\\11.jpg', '').replace('\\12.jpg', '').replace('\\13.jpg', '')
            dataset.append({"glob": temp, "nominal": nominal})

    def deteksi_uang(self):
        match = []
        foundCount = False
        for data in dataset:
            tempLocationImage = str(mataUangValue)
            test_image = cv2.imread(tempLocationImage)
            pixmap = QPixmap(mataUangValue)
            pixmap = pixmap.scaledToHeight(150)
            self.ui.GambarMataUang.setPixmap(pixmap)
            test_image_p = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
            test_image_p = cv2.Canny(test_image_p, 50, 200)

            (temp_height, temp_width) = data['glob'].shape[:2]
            found = None
            threshold = 0.5

            for scale in np.linspace(0.2, 1.0, 20)[::-1]:
                resized = utils.resize(test_image_p, width=int(test_image_p.shape[1] * scale))
                r = test_image_p.shape[1] / float(resized.shape[1])
                if resized.shape[0] < temp_height or resized.shape[1] < temp_width:
                    break

                result = cv2.matchTemplate(resized, data['glob'], cv2.TM_CCOEFF_NORMED)
                (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
                if found is None or maxVal > found[0]:
                    found = (maxVal, maxLoc, r)
                    if maxVal > threshold:
                        foundCount = True
                        match.append(
                            {"maxVal": maxVal, "nominal": data['nominal'], "maxLoc0": maxLoc[0], "maxLoc1": maxLoc[1],
                             "r": r, "image": test_image, 'width': temp_width, 'height': temp_height})
                        break

        if (foundCount):
            hasil = max(match[:1])
            (startX, startY) = (int(hasil['maxLoc0'] * hasil['r']), int(hasil['maxLoc1'] * hasil['r']))
            (endX, endY) = (
                int((hasil['maxLoc0'] + hasil['width']) * hasil['r']),
                int((hasil['maxLoc1'] + hasil['height']) * hasil['r']))
            cv2.rectangle(hasil['image'], (startX, startY), (endX, endY), (0, 0, 255), 2)
            img = hasil['image']
            cv2.imwrite('hasil.jpg', img)

            pixmap = QPixmap("hasil.jpg")
            pixmap3 = pixmap.scaledToHeight(275)
            self.ui.GambarMataUang.setPixmap(pixmap3)
            self.ui.HasilTextEdit.setPlainText(hasil['nominal'] + " Terdeteksi")

        else:
            self.ui.HasilTextEdit.setPlainText("Tidak Terdeteksi")

def app():
    app = QtWidgets.QApplication(sys.argv)
    win = window()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    app()