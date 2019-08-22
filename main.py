from PyQt5 import QtGui
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QImage, QPalette, QBrush
import time
import cv2
from detect_iris import Detection_Iris as ds
from detect_iridologi import *
from houghcircle import hough as hg


class HalAwal(QMainWindow):
    def __init__(self):
        super(HalAwal, self).__init__()
        loadUi('view\\awal.ui', self)
        oImage = QImage("icon\eyes.jpg")
        sImage = oImage.scaled(QSize(1065, 673))  # resize Image to widgets size
        palette = QPalette()
        palette.setBrush(10, QBrush(sImage))  # 10 = Windowrole
        self.setPalette(palette)
        self.setWindowFlags(QtCore.Qt.WindowCloseButtonHint | QtCore.Qt.WindowMinimizeButtonHint)
        self.setFixedSize(1065, 673)
        self.exit.clicked.connect(self.exitClicked)
        self.takephoto.setIcon(QIcon(QPixmap("icon\icon_ambil_gambar.png")))
        self.exit.setIcon(QIcon(QPixmap("icon\icon_keluar.png")))
        self.openfile.setIcon(QIcon(QPixmap("icon\icon_open_file.png")))

    def takeClicked(self):
        self.next1.show()
        window.close()

    def loadClicked(self):
        self.next.show()
        window.close()

    def exitClicked(self):
        QApplication.quit()
        sys.exit()


class PilihanMata(QMainWindow):
    def __init__(self):
        super(PilihanMata, self).__init__()
        loadUi('view\pilihan_mata.ui', self)
        self.matakanan.setIcon(QIcon(QPixmap("icon\eye_kiri.png")))
        self.matakiri.setIcon(QIcon(QPixmap("icon\eye_kanan.png")))
        self.back.setIcon(QIcon(QPixmap("icon\\back.png")))
        oImage = QImage("icon\eyes.jpg")
        sImage = oImage.scaled(QSize(1065, 673))  # resize Image to widgets size
        palette = QPalette()
        palette.setBrush(10, QBrush(sImage))  # 10 = Windowrole
        self.setPalette(palette)
        self.setWindowFlags(QtCore.Qt.WindowCloseButtonHint | QtCore.Qt.WindowMinimizeButtonHint)
        self.setFixedSize(1065, 673)


class PilihanMata_Ambil(QMainWindow):
    def __init__(self):
        super(PilihanMata_Ambil, self).__init__()
        loadUi('view\pilihan_mata.ui', self)
        self.back.setIcon(QIcon(QPixmap("icon\\back.png")))
        self.matakanan.setIcon(QIcon(QPixmap("icon\eye_kiri.png")))
        self.matakiri.setIcon(QIcon(QPixmap("icon\eye_kanan.png")))
        oImage = QImage("icon\eyes.jpg")
        sImage = oImage.scaled(QSize(1065, 673))  # resize Image to widgets size
        palette = QPalette()
        palette.setBrush(10, QBrush(sImage))  # 10 = Windowrole
        self.setPalette(palette)
        self.setWindowFlags(QtCore.Qt.WindowCloseButtonHint | QtCore.Qt.WindowMinimizeButtonHint)
        self.setFixedSize(1065, 673)


class take_kanan(QMainWindow):
    def __init__(self):
        super(take_kanan, self).__init__()
        loadUi('view\\take_gambar.ui', self)
        oImage = QImage("icon\eyes.jpg")
        sImage = oImage.scaled(QSize(800, 700))  # resize Image to widgets size
        palette = QPalette()
        palette.setBrush(10, QBrush(sImage))  # 10 = Windowrole
        self.setPalette(palette)
        self.setWindowFlags(QtCore.Qt.WindowCloseButtonHint | QtCore.Qt.WindowMinimizeButtonHint)
        self.setFixedSize(800, 700)
        self.startButton.clicked.connect(self.start_webcam)
        self.startButton.setIcon(QIcon(QPixmap("icon\play.png")))
        self.cap.setIcon(QIcon(QPixmap("icon\capture.png")))
        self.back.setIcon(QIcon(QPixmap("icon\\back.png")))
        self.imgLabel.setScaledContents(True)
        self.capture = None
        self.timer = QtCore.QTimer(self, interval=5)
        self.timer.timeout.connect(self.update_frame)
        self._image_counter = 0


    def start_webcam(self):
        if self.capture is None:
            self.capture = cv2.VideoCapture(1)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
        self.timer.start()

    def update_frame(self):
        ret, image = self.capture.read()
        self.displayImage(image, True)


    def displayImage(self, img, window=True):
        qformat = QtGui.QImage.Format_Indexed8
        if len(img.shape) == 3:
            if img.shape[2] == 4:
                qformat = QtGui.QImage.Format_RGBA8888
            else:
                qformat = QtGui.QImage.Format_RGB888
        outImage = QtGui.QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
        outImage = outImage.rgbSwapped()
        if window:
            self.imgLabel.setPixmap(QtGui.QPixmap.fromImage(outImage))


class take_kiri(QMainWindow):
    def __init__(self):
        super(take_kiri, self).__init__()
        loadUi('view\\take_gambar.ui', self)
        oImage = QImage("icon\eyes.jpg")
        sImage = oImage.scaled(QSize(800, 700))  # resize Image to widgets size
        palette = QPalette()
        palette.setBrush(10, QBrush(sImage))  # 10 = Windowrole
        self.setPalette(palette)
        self.setWindowFlags(QtCore.Qt.WindowCloseButtonHint | QtCore.Qt.WindowMinimizeButtonHint)
        self.setFixedSize(800, 700)
        self.startButton.clicked.connect(self.start_webcam)
        self.startButton.setIcon(QIcon(QPixmap("icon\play.png")))
        self.cap.setIcon(QIcon(QPixmap("icon\capture.png")))
        self.back.setIcon(QIcon(QPixmap("icon\\back.png")))
        self.imgLabel.setScaledContents(True)
        self.capture = None
        self.timer = QtCore.QTimer(self, interval=5)
        self.timer.timeout.connect(self.update_frame)
        self._image_counter = 0

    def start_webcam(self):
        if self.capture is None:
            self.capture = cv2.VideoCapture(0)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
        self.timer.start()

    def update_frame(self):
        ret, image = self.capture.read()
        simage = cv2.flip(image, 1)
        self.displayImage(image, True)

    def displayImage(self, img, window=True):
        qformat = QtGui.QImage.Format_Indexed8
        if len(img.shape) == 3:
            if img.shape[2] == 4:
                qformat = QtGui.QImage.Format_RGBA8888
            else:
                qformat = QtGui.QImage.Format_RGB888
        outImage = QtGui.QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
        outImage = outImage.rgbSwapped()
        if window:
            self.imgLabel.setPixmap(QtGui.QPixmap.fromImage(outImage))


class GuiPreprocessingDataKanan(QMainWindow):
    def __init__(self):
        super(GuiPreprocessingDataKanan, self).__init__()
        loadUi('view\GUI_preprocessing.ui', self)
        self.processedImage = None
        self.previewImage = None
        self.roiGinjal.setEnabled(False)
        self.roiParu.setEnabled(False)
        self.baru.setIcon(QIcon(QPixmap("icon\\new.png")))
        self.SAVE.setIcon(QIcon(QPixmap("icon\save.png")))
        self.roiGinjal.setIcon(QIcon(QPixmap("icon\ginjal.png")))
        self.roiParu.setIcon(QIcon(QPixmap("icon\paru_paru.png")))
        self.SAVE.clicked.connect(self.saveclick)
        self.crop.clicked.connect(self.cropclick)
        self.crop.clicked.connect(self.handleButton)
        self.roiGinjal.clicked.connect(self.roiGinjalclick)
        self.roiGinjal.clicked.connect(self.handleButton)
        self.roiParu.clicked.connect(self.roiParuclick)
        self.roiParu.clicked.connect(self.handleButton)
        self.setFixedSize(1065, 673)
        pixmap = QPixmap("icon\chart_matakanan.jpg")
        pixmap_resized = pixmap.scaled(460, 460)
        self.image2.setPixmap(pixmap_resized)
        oImage = QImage("icon\eyes.jpg")
        sImage = oImage.scaled(QSize(1065, 673))  # resize Image to widgets size
        palette = QPalette()
        palette.setBrush(10, QBrush(sImage))  # 10 = Windowrole
        self.setPalette(palette)
        self.progressBar.setValue(0)
        self._active = False

    def handleButton(self):
        if not self._active:
            self._active = True
            if self.progressBar.value() == self.progressBar.maximum():
                self.progressBar.reset()
            QtCore.QTimer.singleShot(0, self.startLoop)
        else:
            self._active = False

    def closeEvent(self, event):
        self._active = False

    def startLoop(self):
        while True:
            time.sleep(0.0008)
            value = self.progressBar.value() + 1
            self.progressBar.setValue(value)
            qApp.processEvents()
            if (not self._active or
                value >= self.progressBar.maximum()):
                break
        self._active = False

    def disable(self):
        self.roiGinjal.setEnabled(False)
        self.roiParu.setEnabled(False)
        self.progressBar.setValue(0)
        self.hsl.clear()

    def loadImage(self, fname):
        self.processedImage = fname.copy()
        self.previewImage = cv2.resize(self.processedImage, (450, 450))
        self.displayImage(self.previewImage)

    def autocrop(self, img):
        self.progressBar.setValue(0)
        self.roiGinjal.setEnabled(True)
        self.roiParu.setEnabled(True)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        result = ds.detection_iris(self,img)
        iris = cv2.resize(result, (450, 450))
        self.progressBar.setValue(50)
        medianImage = cv2.medianBlur(iris, 5, 5)
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpImage = cv2.filter2D(medianImage, -1, kernel)
        imggr = cv2.cvtColor(sharpImage, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        blackhat = cv2.morphologyEx(imggr, cv2.MORPH_BLACKHAT, kernel)
        bottom_hat_filtered = cv2.add(blackhat, imggr)
        self.retval, self.thresh_gray = cv2.threshold(imggr, thresh=125, maxval=255, type=cv2.THRESH_BINARY)
        bnrImage = cv2.medianBlur(bottom_hat_filtered, 3)
        self.cropimg = hg(bnrImage)
        self.displayImage(self.cropimg)

    def regionofinterestGinjal(self, img):
        self.progressBar.setValue(0)
        Detection_Iridologi.dc = None
        self.hsl.clear()
        img = cv2.resize(img, (450, 450))
        height, width = img.shape[:2]
        leftIndex = width * 16 / 32
        rightIndex = width * 21 / 32
        topIndex = height * 22 / 32
        bottomIndex = height * 32 / 32
        # roi = cv2.rectangle(img, (int(leftIndex), int(bottomIndex)), (int(rightIndex), int(topIndex)), (35,5,55), 2)
        x1 = int(leftIndex)
        x2 = int(rightIndex)
        y1 = int(topIndex)
        y2 = int(bottomIndex)
        imgku = img[y1:y2, x1:width]
        roi = Detection_Iridologi.detection_iridologi(Detection_Iridologi,img[y1: y2, x1:x2], imgku)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        #left = width - roi.shape[0] # 275
        #right = width #400
        #top = height - roi.shape[1]#20
        #bottom = height#400
        #print(Detection_Iridologi.x)
        x = Detection_Iridologi.dc
        if x == 1:
            self.hsl.setText("Ginjal Normal")
        elif x == 2:
            self.hsl.setText("Ginjal Tidak Normal")
        elif x == 0 or x == None:
            self.hsl.setText("Ginjal Tidak Terdeteksi")
        img[y1:y2, x1:width] = roi
        self.displayImage(img)

    def regionofinterestParu(self, img):
        self.progressBar.setValue(0)
        Detection_Iridologi.dc = None
        self.hsl.clear()
        img = cv2.resize(img, (450, 450))
        height, width = img.shape[:2]
        leftIndex = width * 2 / 32
        rightIndex = width * 9 / 32
        topIndex = height * 12 / 32
        bottomIndex = height * 17 / 32
        x1 = int(leftIndex)
        x2 = int(rightIndex)
        y1 = int(topIndex)
        y2 = int(bottomIndex)
        imgku = img[y1:y2, x1:width]
        roi = Detection_Iridologi.detection_iridologi(Detection_Iridologi,img[y1: y2, x1:x2], imgku)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        #left = width - roi.shape[0]  # 275
        #right = width  # 400
        #top = height - roi.shape[1]  # 200
        #bottom = height  # 400
        x = Detection_Iridologi.dc
        if x == 1:
            self.hsl.setText("Paru - Paru Normal")
        elif x == 2:
            self.hsl.setText("Paru - Paru Tidak Normal")
        elif x == 0 or x == None:
            self.hsl.setText("Paru - Paru Tidak Terdeteksi")
        img[y1:y2, x1:width] = roi
        self.displayImage(img)


    def displayImage(self,output):
        self.imgs = output
        qFormat = QImage.Format_Indexed8
        if len(output.shape) == 3:
            if (output.shape[2]) == 4:
                qFormat = QImage.Format_RGBA8888
            else:
                qFormat = QImage.Format_RGB888
        img = QImage(output, output.shape[1], output.shape[0],
                     output.strides[0], qFormat)
        img = img.rgbSwapped()
        self.image1.setPixmap(QPixmap.fromImage(img))
        self.image1.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)


    def cropclick(self):
        self.autocrop(self.previewImage)

    def sobelclick(self):
        self.sobeledge(self.cropimg)

    def roiGinjalclick(self):
        self.regionofinterestGinjal(self.cropimg)

    def roiParuclick(self):
        self.regionofinterestParu(self.cropimg)

    def saveclick(self):
        fname, filter = QFileDialog.getSaveFileName(self, 'Save File', 'E:\Dataset\Training\\Normal',
                                                    ("Images (*.jpg)"))
        if fname:
            cv2.imwrite(str(fname), self.imgs)
            # file.close()
        else:
            print('Invalid Image')


class GuiPreprocessingDataKiri(QMainWindow):
    def __init__(self):
        super(GuiPreprocessingDataKiri, self).__init__()
        loadUi('view\GUI_preprocessing.ui', self)
        self.baru.setIcon(QIcon(QPixmap("icon\\new.png")))
        self.SAVE.setIcon(QIcon(QPixmap("icon\save.png")))
        self.roiGinjal.setIcon(QIcon(QPixmap("icon\ginjal.png")))
        self.roiParu.setIcon(QIcon(QPixmap("icon\paru_paru.png")))
        self.roiGinjal.setEnabled(False)
        self.roiParu.setEnabled(False)
        self.SAVE.clicked.connect(self.saveclick)
        self.crop.clicked.connect(self.cropclick)
        self.crop.clicked.connect(self.handleButton)
        self.roiGinjal.clicked.connect(self.roiGinjalclick)
        self.roiGinjal.clicked.connect(self.handleButton)
        self.roiParu.clicked.connect(self.roiParuclick)
        self.roiParu.clicked.connect(self.handleButton)
        pixmap = QPixmap("icon\chart_matakiri.jpg")
        pixmap_resized = pixmap.scaled(460, 460)
        self.image2.setPixmap(pixmap_resized)
        oImage = QImage("icon\eyes.jpg")
        sImage = oImage.scaled(QSize(1065, 673))  # resize Image to widgets size
        palette = QPalette()
        palette.setBrush(10, QBrush(sImage))  # 10 = Windowrole
        self.setPalette(palette)
        self.progressBar.setValue(0)
        self._active = False

    def handleButton(self):
        if not self._active:
            self._active = True
            if self.progressBar.value() == self.progressBar.maximum():
                self.progressBar.reset()
            QtCore.QTimer.singleShot(0, self.startLoop)
        else:
            self._active = False

    def closeEvent(self, event):
        self._active = False

    def startLoop(self):
        while True:
            time.sleep(0.0008)
            value = self.progressBar.value() + 1
            self.progressBar.setValue(value)
            qApp.processEvents()
            if (not self._active or
                value >= self.progressBar.maximum()):
                break
        self._active = False

    def disable(self):
        self.roiGinjal.setEnabled(False)
        self.roiParu.setEnabled(False)
        self.progressBar.setValue(0)

    def loadImage(self, fname):
        self.processedImage = fname.copy()
        self.previewImage = cv2.resize(self.processedImage, (450, 450))
        self.displayImage(self.previewImage)

    def autocrop(self, img):
        self.progressBar.setValue(0)
        self.roiGinjal.setEnabled(True)
        self.roiParu.setEnabled(True)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        result = ds.detection_iris(self,img)
        iris = cv2.resize(result, (450, 450))
        self.progressBar.setValue(50)
        medianImage = cv2.medianBlur(iris, 5, 5)
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpImage = cv2.filter2D(medianImage, -1, kernel)
        imggr = cv2.cvtColor(sharpImage, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        blackhat = cv2.morphologyEx(imggr, cv2.MORPH_BLACKHAT, kernel)
        bottom_hat_filtered = cv2.add(blackhat, imggr)
        self.retval, self.thresh_gray = cv2.threshold(imggr, thresh=125, maxval=255, type=cv2.THRESH_BINARY)
        bnrImage = cv2.medianBlur(bottom_hat_filtered, 3)
        self.cropimg = hg(bnrImage)
        self.displayImage(self.cropimg)


    def regionofinterestGinjal(self, img):
        self.progressBar.setValue(0)
        Detection_Iridologi.dc = None
        self.hsl.clear()
        img = cv2.resize(img, (450, 450))
        height, width = img.shape[:2]
        leftIndex = width * 11 / 32
        rightIndex = width * 16 / 32
        topIndex = height * 22 / 32
        bottomIndex = height * 32 / 32
        x1 = int(leftIndex)
        x2 = int(rightIndex)
        y1 = int(topIndex)
        y2 = int(bottomIndex)
        imgku = img[y1:y2, x1:width]
        roi = Detection_Iridologi.detection_iridologi(Detection_Iridologi,img[y1: y2, x1:x2], imgku)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        #left = width - roi.shape[0]  # 275
        #right = width  # 400
        #top = height - roi.shape[1]  # 200
        #bottom = height  # 400
        x = Detection_Iridologi.dc
        if x == 1:
            self.hsl.setText("Ginjal Normal")
        elif x == 2:
            self.hsl.setText("Ginjal Tidak Normal")
        elif x == 0 or x == None:
            self.hsl.setText("Ginjal Tidak Terdeteksi")
        img[y1:y2, x1:width] = roi
        print(img.shape)
        self.displayImage(img)

    def regionofinterestParu(self, img):
        self.progressBar.setValue(0)
        Detection_Iridologi.dc = None
        self.hsl.clear()
        img = cv2.resize(img, (450, 450))
        height, width = img.shape[:2]
        leftIndex = width * 22 / 32
        rightIndex = width * 28 / 32
        topIndex = height * 12 / 32
        bottomIndex = height * 17 / 32
        x1 = int(leftIndex)
        x2 = int(rightIndex)
        y1 = int(topIndex)
        y2 = int(bottomIndex)
        imgku = img[y1:y2, x1:width]
        roi = Detection_Iridologi.detection_iridologi(Detection_Iridologi,img[y1: y2, x1:x2], imgku)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        #left = width - roi.shape[0]  # 275
        #right = width  # 400
        #top = height - roi.shape[1]  # 200
        #bottom = height  # 400
        x = Detection_Iridologi.dc
        if x == 1:
            self.hsl.setText("Paru - Paru Normal")
        elif x == 2:
            self.hsl.setText("Paru - Paru Tidak Normal")
        elif x == 0 or x == None:
            self.hsl.setText("Paru - Paru Tidak Terdeteksi")
        img[y1:y2, x1:width] = roi
        print(img.shape)
        self.displayImage(img)

    def displayImage(self, output):
        self.imgs = output
        qFormat = QImage.Format_Indexed8
        if len(output.shape) == 3:
            if (output.shape[2]) == 4:
                qFormat = QImage.Format_RGBA8888
            else:
                qFormat = QImage.Format_RGB888
        img = QImage(output, output.shape[1], output.shape[0],
                     output.strides[0], qFormat)
        img = img.rgbSwapped()
        self.image1.setPixmap(QPixmap.fromImage(img))
        self.image1.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)

    def cropclick(self):
        self.autocrop(self.previewImage)

    def roiGinjalclick(self):
        self.regionofinterestGinjal(self.cropimg)

    def roiParuclick(self):
        self.regionofinterestParu(self.cropimg)

    def saveclick(self):
        fname, filter = QFileDialog.getSaveFileName(self, 'Save File', 'E:\Dataset\Training\\Normal',
                                                    ("Images (*.jpg)"))
        if fname:
            cv2.imwrite(str(fname), self.imgs)
            # file.close()
        else:
            print('Invalid Image')


class Manager:
    def __init__(self):
        self.first = HalAwal()
        self.second = PilihanMata_Ambil()
        self.second2 = PilihanMata()
        self.third_kanan = take_kanan()
        self.third_kiri = take_kiri()
        self.nextkiri = GuiPreprocessingDataKiri()
        self.nextkanan = GuiPreprocessingDataKanan()
        self.first.takephoto.clicked.connect(self.second.show)
        self.first.takephoto.clicked.connect(self.first.hide)
        self.first.openfile.clicked.connect(self.second2.show)
        self.first.openfile.clicked.connect(self.first.hide)
        self.second.back.clicked.connect(self.first.show)
        self.second.back.clicked.connect(self.second.hide)
        self.second2.back.clicked.connect(self.first.show)
        self.second2.back.clicked.connect(self.second2.hide)
        self.second.matakiri.clicked.connect(self.third_kiri.show)
        self.second.matakiri.clicked.connect(self.second.hide)
        self.second.matakanan.clicked.connect(self.third_kanan.show)
        self.second.matakanan.clicked.connect(self.second.hide)
        self.second2.matakanan.clicked.connect(self.second2.hide)
        self.second2.matakiri.clicked.connect(self.second2.hide)
        self.third_kanan.cap.clicked.connect(self.capture_image_kanan)
        self.third_kanan.cap.clicked.connect(self.third_kanan.hide)
        self.third_kanan.back.clicked.connect(self.second.show)
        self.third_kanan.back.clicked.connect(self.third_kanan.hide)
        self.third_kiri.cap.clicked.connect(self.capture_image_kiri)
        self.third_kiri.cap.clicked.connect(self.third_kiri.hide)
        self.third_kiri.back.clicked.connect(self.second.show)
        self.third_kiri.back.clicked.connect(self.third_kiri.hide)
        self.second2.matakiri.clicked.connect(self.kiriClicked)
        self.second2.matakanan.clicked.connect(self.kananClicked)
        self.nextkiri.baru.clicked.connect(self.first.show)
        self.nextkiri.baru.clicked.connect(self.nextkiri.disable)
        self.nextkiri.baru.clicked.connect(self.nextkiri.hide)
        self.nextkanan.baru.clicked.connect(self.first.show)
        self.nextkanan.baru.clicked.connect(self.nextkanan.hide)
        self.nextkanan.baru.clicked.connect(self.nextkanan.disable)
        self.first.show()


    def kananClicked(self):
        fname, filter = QFileDialog.getOpenFileName(self.second2, 'Open File', 'D:\Dataset\pengujian', "Image File (*.jpg)")

        if fname:
            fname = cv2.imread(fname, cv2.IMREAD_ANYCOLOR)
            self.nextkanan.loadImage(fname)
            self.nextkanan.show()
            self.third_kanan.hide()
        else:
            self.second2.show()
            print('Invalid Image')

    def kiriClicked(self):
        fname, filter = QFileDialog.getOpenFileName(self.second2, 'Open File', 'D:\Dataset\pengujian', "Image File (*.jpg)")

        if fname:
            fname = cv2.imread(fname, cv2.IMREAD_ANYCOLOR)
            self.nextkiri.loadImage(fname)
            self.nextkiri.show()
            self.third_kiri.hide()
        else:
            self.second2.show()
            print('Invalid Image')

    def capture_image_kiri(self):
        flag, frame = self.third_kiri.capture.read()
        if flag:
            self.nextkiri.loadImage(frame)
            self.nextkiri.show()
            self.third_kiri.hide()
        else:
            print('Invalid Image')

    def capture_image_kanan(self):
        flag, frame = self.third_kanan.capture.read()
        if flag:
            self.nextkanan.loadImage(frame)
            self.nextkanan.show()
            self.third_kanan.hide()
        else:
            print('Invalid Image')


app = QApplication(sys.argv)
window = Manager()
sys.exit(app.exec())
