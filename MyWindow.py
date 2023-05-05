import os
import sys
import cv2
from PyQt5.QtGui import QPixmap, QImage, qRed, qGreen, qBlue
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox
import matplotlib.pyplot as plt
import numpy as np
from MainWindow import Ui_MainWindow


def cvImgtoQtImg(cvImg):  # 定义opencv图像转PyQt图像的函数
    QtImgBuf = cv2.cvtColor(cvImg, cv2.COLOR_BGR2BGRA)
    QtImg = QImage(QtImgBuf.data, QtImgBuf.shape[1], QtImgBuf.shape[0], QImage.Format_RGB32)
    return QtImg


def QImage2CV(qimg):
    tmp = qimg

    # 使用numpy创建空的图象
    cv_image = np.zeros((tmp.height(), tmp.width(), 3), dtype=np.uint8)

    for row in range(0, tmp.height()):
        for col in range(0, tmp.width()):
            r = qRed(tmp.pixel(col, row))
            g = qGreen(tmp.pixel(col, row))
            b = qBlue(tmp.pixel(col, row))
            cv_image[row, col, 0] = r
            cv_image[row, col, 1] = g
            cv_image[row, col, 2] = b

    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)

    return cv_image


def QPixmap2cv(qtpixmap):
    qimg = qtpixmap.toImage()
    temp_shape = (qimg.height(), qimg.bytesPerLine() * 8 // qimg.depth())
    temp_shape += (4,)
    ptr = qimg.bits()
    ptr.setsize(qimg.byteCount())
    result = np.array(ptr, dtype=np.uint8).reshape(temp_shape)
    result = result[..., :3]
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

    return result

def FFT2(img):
    # 傅里叶变换
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    res = np.log(np.abs(fshift))

    # plt.plot(res)
    # plt.savefig('img.jpg')
    # result = cv2.imread('img.jpg', 1)
    plt.imshow(res, 'gray')
    plt.axis('off')
    plt.savefig('Img.png', bbox_inches='tight', pad_inches=0.0)
    plt.close()
    result = cv2.imread('Img.png', 0)
    os.remove('Img.png')

    return result


class MyWindow(QMainWindow):
    def __init__(self, Ui_MainWindow):
        super().__init__()
        self.ui = Ui_MainWindow
        app = QApplication(sys.argv)
        MainWindow = QMainWindow()
        self.ui.setupUi(MainWindow)
        self.picpath = ' '
        self.openfile_name = ' '

        self.ui.ImportBtn.clicked.connect(lambda: self.Import())
        self.ui.GrayscaleBtn.clicked.connect(lambda: self.Grayscale())
        self.ui.BinarizationBtn.clicked.connect(lambda: self.Binarization())
        self.ui.geometryBtn.clicked.connect(lambda: self.Geometry())
        self.ui.histogramBtn.clicked.connect(lambda: self.Histogram())
        self.ui.EqualizeBtn.clicked.connect(lambda: self.Equalize())
        self.ui.MeanBtn.clicked.connect(lambda: self.Mean())
        self.ui.BoxBtn.clicked.connect(lambda: self.Box())
        self.ui.vagueBtn.clicked.connect(lambda: self.Vague())
        self.ui.medianBtn.clicked.connect(lambda: self.Median())
        self.ui.RobertsBtn.clicked.connect(lambda: self.Roberts())
        self.ui.PrewittBtn.clicked.connect(lambda: self.Prewitt())
        self.ui.SobelBtn.clicked.connect(lambda: self.Sobel())
        self.ui.LowpassBtn.clicked.connect(lambda: self.Lowpass())
        self.ui.HighpassBtn.clicked.connect(lambda: self.Highpass())
        self.ui.corrosionBtn.clicked.connect(lambda: self.Corrosion())
        self.ui.expansionBtn.clicked.connect(lambda: self.Expansion())
        self.ui.OpenBtn.clicked.connect(lambda: self.Open())
        self.ui.CloseBtn.clicked.connect(lambda: self.Close())
        self.ui.LOGBtn.clicked.connect(lambda: self.LOG())
        self.ui.ScharrBtn.clicked.connect(lambda: self.Scharr())
        self.ui.CannyBtn.clicked.connect(lambda: self.Canny())
        self.ui.SaveBtn.clicked.connect(lambda: self.Save())

        MainWindow.show()
        sys.exit(app.exec_())

    def Import(self):
        # 导入图片
        self.openfile_name = QFileDialog.getOpenFileName(self, '选择文件', '', "Image Files (*.png *.jpg *.bmp)")[0]
        if self.openfile_name != " ":
            self.ui.PicBefore.setPixmap(QPixmap(self.openfile_name))
            self.picpath = self.openfile_name
            self.ui.Label_H.setText(str(cv2.imread(self.picpath).shape[0]))
            self.ui.Label_W.setText(str(cv2.imread(self.picpath).shape[1]))
            self.ui.Label_T.setText(str(cv2.imread(self.picpath).shape[2]))
            self.ui.Label_Type.setText(str(cv2.imread(self.picpath).dtype))
            self.ui.PicBefore.setScaledContents(True)
            self.fftbefore = FFT2(cv2.imread(self.picpath, 0))
            self.fftbefore = cvImgtoQtImg(self.fftbefore)
            self.ui.FFTBefore.setPixmap(QPixmap(self.fftbefore))
            self.ui.FFTBefore.setScaledContents(True)

    def Grayscale(self):
        # 灰度处理
        if self.picpath == ' ':
            QMessageBox.critical(self, '操作失败', '请先导入图片',
                                 QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            return
        img = cv2.imread(self.picpath)

        self.fftbefore = FFT2(cv2.imread(self.picpath, 0))
        self.fftbefore = cvImgtoQtImg(self.fftbefore)
        self.ui.FFTBefore.setPixmap(QPixmap(self.fftbefore))
        self.ui.FFTBefore.setScaledContents(True)

        self.PicAfter = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        self.fftafter = self.PicAfter
        self.PicAfter = cvImgtoQtImg(self.PicAfter)
        self.ui.PicAfter.setPixmap(QPixmap(self.PicAfter))
        self.ui.PicAfter.setScaledContents(True)
        self.fftafter = FFT2(self.fftafter)
        self.fftafter = cvImgtoQtImg(self.fftafter)
        self.ui.FFTAfter.setPixmap(QPixmap(self.fftafter))
        self.ui.FFTAfter.setScaledContents(True)

    def Binarization(self):
        # 二值化处理
        if self.picpath == ' ':
            QMessageBox.critical(self, '操作失败', '请先导入图片',
                                 QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            return
        img = cv2.imread(self.picpath)

        self.fftbefore = FFT2(cv2.imread(self.picpath, 0))
        self.fftbefore = cvImgtoQtImg(self.fftbefore)
        self.ui.FFTBefore.setPixmap(QPixmap(self.fftbefore))
        self.ui.FFTBefore.setScaledContents(True)

        self.PicAfter = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret, self.PicAfter = cv2.threshold(self.PicAfter, 127, 255, cv2.THRESH_BINARY)
        self.fftafter = self.PicAfter
        self.PicAfter = cvImgtoQtImg(self.PicAfter)
        self.ui.PicAfter.setPixmap(QPixmap(self.PicAfter))
        self.ui.PicAfter.setScaledContents(True)
        self.fftafter = FFT2(self.fftafter)
        self.fftafter = cvImgtoQtImg(self.fftafter)
        self.ui.FFTAfter.setPixmap(QPixmap(self.fftafter))
        self.ui.FFTAfter.setScaledContents(True)

    def Geometry(self):
        # 几何变换
        if self.picpath == ' ':
            QMessageBox.critical(self, '操作失败', '请先导入图片',
                                 QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            return
        img = cv2.imread(self.picpath)

        self.fftbefore = FFT2(cv2.imread(self.picpath, 0))
        self.fftbefore = cvImgtoQtImg(self.fftbefore)
        self.ui.FFTBefore.setPixmap(QPixmap(self.fftbefore))
        self.ui.FFTBefore.setScaledContents(True)

        self.PicAfter = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        self.fftafter = self.PicAfter
        self.PicAfter = cvImgtoQtImg(self.PicAfter)
        self.ui.PicAfter.setPixmap(QPixmap(self.PicAfter))
        self.ui.PicAfter.setScaledContents(False)
        self.fftafter = FFT2(self.fftafter)
        self.fftafter = cvImgtoQtImg(self.fftafter)
        self.ui.FFTAfter.setPixmap(QPixmap(self.fftafter))
        self.ui.FFTAfter.setScaledContents(True)

    def Histogram(self):
        # 灰度直方图
        if self.picpath == ' ':
            QMessageBox.critical(self, '操作失败', '请先导入图片',
                                 QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            return
        img = cv2.imread(self.picpath)

        self.fftbefore = FFT2(cv2.imread(self.picpath, 0))
        self.fftbefore = cvImgtoQtImg(self.fftbefore)
        self.ui.FFTBefore.setPixmap(QPixmap(self.fftbefore))
        self.ui.FFTBefore.setScaledContents(True)

        self.PicAfter = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        self.PicAfter = cv2.calcHist([self.PicAfter], [0], None, [256], [0, 255])
        plt.plot(self.PicAfter)
        plt.savefig('img.jpg')
        plt.close()
        self.ui.PicAfter.setPixmap(QPixmap('img.jpg'))
        self.ui.PicAfter.setScaledContents(True)
        os.remove('img.jpg')


    def Equalize(self):
        # 均衡化
        if self.picpath == ' ':
            QMessageBox.critical(self, '操作失败', '请先导入图片',
                                 QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            return
        img = cv2.imread(self.picpath)
        self.PicAfter = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        PicBefore = cv2.calcHist([self.PicAfter], [0], None, [256], [0, 255])
        plt.plot(PicBefore)
        plt.savefig('img.jpg')
        plt.close()
        self.ui.FFTBefore.setPixmap(QPixmap('img.jpg'))
        self.ui.FFTBefore.setScaledContents(True)
        os.remove('img.jpg')

        self.PicAfter = cv2.equalizeHist(self.PicAfter)
        self.PicAfter = cvImgtoQtImg(self.PicAfter)
        self.ui.PicAfter.setPixmap(QPixmap(self.PicAfter))
        self.ui.PicAfter.setScaledContents(True)
        self.fftafter = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        self.fftafter = cv2.equalizeHist(self.fftafter)
        self.fftafter = cv2.calcHist([self.fftafter], [0], None, [256], [0, 255])
        plt.plot(self.fftafter)
        plt.savefig('img.jpg')
        plt.close()
        self.ui.FFTAfter.setPixmap(QPixmap('img.jpg'))
        os.remove('img.jpg')
        self.ui.FFTAfter.setScaledContents(True)

    def Mean(self):
        # 均值滤波
        if self.picpath == ' ':
            QMessageBox.critical(self, '操作失败', '请先导入图片',
                                 QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            return
        img = cv2.imread(self.picpath)

        self.fftbefore = FFT2(cv2.imread(self.picpath, 0))
        self.fftbefore = cvImgtoQtImg(self.fftbefore)
        self.ui.FFTBefore.setPixmap(QPixmap(self.fftbefore))
        self.ui.FFTBefore.setScaledContents(True)

        self.PicAfter = cv2.blur(img, (3, 5))
        self.fftafter = self.PicAfter
        self.PicAfter = cvImgtoQtImg(self.PicAfter)
        self.ui.PicAfter.setPixmap(QPixmap(self.PicAfter))
        self.ui.PicAfter.setScaledContents(True)
        self.fftafter = FFT2(self.fftafter)
        self.fftafter = cvImgtoQtImg(self.fftafter)
        self.ui.FFTAfter.setPixmap(QPixmap(self.fftafter))
        self.ui.FFTAfter.setScaledContents(True)

    def Box(self):
        # 方框滤波
        if self.picpath == ' ':
            QMessageBox.critical(self, '操作失败', '请先导入图片',
                                 QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            return
        img = cv2.imread(self.picpath)

        self.fftbefore = FFT2(cv2.imread(self.picpath, 0))
        self.fftbefore = cvImgtoQtImg(self.fftbefore)
        self.ui.FFTBefore.setPixmap(QPixmap(self.fftbefore))
        self.ui.FFTBefore.setScaledContents(True)

        self.PicAfter = cv2.boxFilter(img, -1, (3, 5))
        self.fftafter = self.PicAfter
        self.PicAfter = cvImgtoQtImg(self.PicAfter)
        self.ui.PicAfter.setPixmap(QPixmap(self.PicAfter))
        self.ui.PicAfter.setScaledContents(True)
        self.fftafter = FFT2(self.fftafter)
        self.fftafter = cvImgtoQtImg(self.fftafter)
        self.ui.FFTAfter.setPixmap(QPixmap(self.fftafter))
        self.ui.FFTAfter.setScaledContents(True)

    def Vague(self):
        # 高斯模糊滤波
        if self.picpath == ' ':
            QMessageBox.critical(self, '操作失败', '请先导入图片',
                                 QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            return
        img = cv2.imread(self.picpath)

        self.fftbefore = FFT2(cv2.imread(self.picpath, 0))
        self.fftbefore = cvImgtoQtImg(self.fftbefore)
        self.ui.FFTBefore.setPixmap(QPixmap(self.fftbefore))
        self.ui.FFTBefore.setScaledContents(True)

        self.PicAfter = cv2.GaussianBlur(img, (5, 5), 0)
        self.fftafter = self.PicAfter
        self.PicAfter = cvImgtoQtImg(self.PicAfter)
        self.ui.PicAfter.setPixmap(QPixmap(self.PicAfter))
        self.ui.PicAfter.setScaledContents(True)
        self.fftafter = FFT2(self.fftafter)
        self.fftafter = cvImgtoQtImg(self.fftafter)
        self.ui.FFTAfter.setPixmap(QPixmap(self.fftafter))
        self.ui.FFTAfter.setScaledContents(True)

    def Median(self):
        # 中值滤波
        if self.picpath == ' ':
            QMessageBox.critical(self, '操作失败', '请先导入图片',
                                 QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            return
        img = cv2.imread(self.picpath)

        self.fftbefore = FFT2(cv2.imread(self.picpath, 0))
        self.fftbefore = cvImgtoQtImg(self.fftbefore)
        self.ui.FFTBefore.setPixmap(QPixmap(self.fftbefore))
        self.ui.FFTBefore.setScaledContents(True)

        self.PicAfter = cv2.medianBlur(img, 5)
        self.fftafter = self.PicAfter
        self.PicAfter = cvImgtoQtImg(self.PicAfter)
        self.ui.PicAfter.setPixmap(QPixmap(self.PicAfter))
        self.ui.PicAfter.setScaledContents(True)
        self.fftafter = FFT2(self.fftafter)
        self.fftafter = cvImgtoQtImg(self.fftafter)
        self.ui.FFTAfter.setPixmap(QPixmap(self.fftafter))
        self.ui.FFTAfter.setScaledContents(True)

    def Roberts(self):
        # Roberts算子
        if self.picpath == ' ':
            QMessageBox.critical(self, '操作失败', '请先导入图片',
                                 QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            return
        img = cv2.imread(self.picpath)
        self.fftbefore = FFT2(cv2.imread(self.picpath, 0))
        self.fftbefore = cvImgtoQtImg(self.fftbefore)
        self.ui.FFTBefore.setPixmap(QPixmap(self.fftbefore))
        self.ui.FFTBefore.setScaledContents(True)

        self.PicAfter = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret, img_binary = cv2.threshold(self.PicAfter, 127, 255, cv2.THRESH_BINARY)
        kernelx_Robert = np.array([[-1, 0], [0, 1]], dtype=int)
        kernely_Robert = np.array([[0, -1], [1, 0]], dtype=int)
        x_Robert = cv2.filter2D(img_binary, cv2.CV_16S, kernelx_Robert)
        y_Robert = cv2.filter2D(img_binary, cv2.CV_16S, kernely_Robert)
        absX_Robert = cv2.convertScaleAbs(x_Robert)
        absY_Robert = cv2.convertScaleAbs(y_Robert)
        self.PicAfter = cv2.addWeighted(absX_Robert, 0.5, absY_Robert, 0.5, 0)
        self.fftafter = self.PicAfter
        self.PicAfter = cvImgtoQtImg(self.PicAfter)
        self.ui.PicAfter.setPixmap(QPixmap(self.PicAfter))
        self.ui.PicAfter.setScaledContents(True)
        self.fftafter = FFT2(self.fftafter)
        self.fftafter = cvImgtoQtImg(self.fftafter)
        self.ui.FFTAfter.setPixmap(QPixmap(self.fftafter))
        self.ui.FFTAfter.setScaledContents(True)

    def Prewitt(self):
        # Prewitt算子
        if self.picpath == ' ':
            QMessageBox.critical(self, '操作失败', '请先导入图片',
                                 QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            return
        img = cv2.imread(self.picpath)

        self.fftbefore = FFT2(cv2.imread(self.picpath, 0))
        self.fftbefore = cvImgtoQtImg(self.fftbefore)
        self.ui.FFTBefore.setPixmap(QPixmap(self.fftbefore))
        self.ui.FFTBefore.setScaledContents(True)

        self.PicAfter = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret, img_binary = cv2.threshold(self.PicAfter, 127, 255, cv2.THRESH_BINARY)
        kernelx_Prewitt = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
        kernely_Prewitt = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)
        x_Prewitt = cv2.filter2D(img_binary, -1, kernelx_Prewitt)
        y_Prewitt = cv2.filter2D(img_binary, -1, kernely_Prewitt)
        absX_Prewitt = cv2.convertScaleAbs(x_Prewitt)
        absY_Prewitt = cv2.convertScaleAbs(y_Prewitt)
        self.PicAfter = cv2.addWeighted(absX_Prewitt, 0.5, absY_Prewitt, 0.5, 0)
        self.fftafter = self.PicAfter
        self.PicAfter = cvImgtoQtImg(self.PicAfter)
        self.ui.PicAfter.setPixmap(QPixmap(self.PicAfter))
        self.ui.PicAfter.setScaledContents(True)
        self.fftafter = FFT2(self.fftafter)
        self.fftafter = cvImgtoQtImg(self.fftafter)
        self.ui.FFTAfter.setPixmap(QPixmap(self.fftafter))
        self.ui.FFTAfter.setScaledContents(True)

    def Sobel(self):
        # Sobel算子
        if self.picpath == ' ':
            QMessageBox.critical(self, '操作失败', '请先导入图片',
                                 QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            return
        img = cv2.imread(self.picpath)

        self.fftbefore = FFT2(cv2.imread(self.picpath, 0))
        self.fftbefore = cvImgtoQtImg(self.fftbefore)
        self.ui.FFTBefore.setPixmap(QPixmap(self.fftbefore))
        self.ui.FFTBefore.setScaledContents(True)

        self.PicAfter = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret, img_binary = cv2.threshold(self.PicAfter, 127, 255, cv2.THRESH_BINARY)
        x_Sobel = cv2.Sobel(img_binary, cv2.CV_16S, 1, 0)
        y_Sobel = cv2.Sobel(img_binary, cv2.CV_16S, 0, 1)
        absX_Sobel = cv2.convertScaleAbs(x_Sobel)
        absY_Sobel = cv2.convertScaleAbs(y_Sobel)
        self.PicAfter = cv2.addWeighted(absX_Sobel, 0.5, absY_Sobel, 0.5, 0)
        self.fftafter = self.PicAfter
        self.PicAfter = cvImgtoQtImg(self.PicAfter)
        self.ui.PicAfter.setPixmap(QPixmap(self.PicAfter))
        self.ui.PicAfter.setScaledContents(True)
        self.fftafter = FFT2(self.fftafter)
        self.fftafter = cvImgtoQtImg(self.fftafter)
        self.ui.FFTAfter.setPixmap(QPixmap(self.fftafter))
        self.ui.FFTAfter.setScaledContents(True)

    def Lowpass(self):
        # 低通滤波
        if self.picpath == ' ':
            QMessageBox.critical(self, '操作失败', '请先导入图片',
                                 QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            return
        img = cv2.imread(self.picpath)

        self.fftbefore = FFT2(cv2.imread(self.picpath, 0))
        self.fftbefore = cvImgtoQtImg(self.fftbefore)
        self.ui.FFTBefore.setPixmap(QPixmap(self.fftbefore))
        self.ui.FFTBefore.setScaledContents(True)

        self.PicAfter = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # self.PicAfter = cv2.cvtColor(self.PicAfter, cv2.COLOR_GRAY2RGB)
        img_dft = np.fft.fft2(self.PicAfter)
        dft_shift_low = np.fft.fftshift(img_dft)
        h, w = dft_shift_low.shape[0:2]
        h_center, w_center = int(h / 2), int(w / 2)
        img_black = np.zeros((h, w), np.uint8)
        img_black[h_center - int(100 / 2):h_center + int(100 / 2), w_center - int(100 / 2):w_center + int(100 / 2)] = 1
        dft_shift_low = dft_shift_low * img_black
        res = np.log(np.abs(dft_shift_low))
        idft_shift = np.fft.ifftshift(dft_shift_low)
        ifimg = np.fft.ifft2(idft_shift)
        ifimg = np.abs(ifimg)
        ifimg = np.int8(ifimg)
        cv2.imwrite('img.jpg', ifimg)
        self.PicAfter = QPixmap('img.jpg')
        self.ui.PicAfter.setPixmap(self.PicAfter)
        self.ui.PicAfter.setScaledContents(True)
        self.fftafter = cv2.imread('img.jpg')
        self.fftafter = FFT2(self.fftafter)
        self.fftafter = cvImgtoQtImg(self.fftafter)
        self.ui.FFTAfter.setPixmap(QPixmap(self.fftafter))
        self.ui.FFTAfter.setScaledContents(True)
        os.remove('img.jpg')

    def Highpass(self):
        # 高通滤波
        if self.picpath == ' ':
            QMessageBox.critical(self, '操作失败', '请先导入图片',
                                 QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            return
        img = cv2.imread(self.picpath)

        self.fftbefore = FFT2(cv2.imread(self.picpath, 0))
        self.fftbefore = cvImgtoQtImg(self.fftbefore)
        self.ui.FFTBefore.setPixmap(QPixmap(self.fftbefore))
        self.ui.FFTBefore.setScaledContents(True)

        self.PicAfter = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_dft = np.fft.fft2(self.PicAfter)
        dft_shift = np.fft.fftshift(img_dft)
        h, w = dft_shift.shape[0:2]
        h_center, w_center = int(h / 2), int(w / 2)
        dft_shift[h_center - int(50 / 2):h_center + int(50 / 2),
        w_center - int(50 / 2):w_center + int(50 / 2)] = 0
        res = np.log(np.abs(dft_shift))
        idft_shift = np.fft.ifftshift(dft_shift)
        img_idft = np.fft.ifft2(idft_shift)
        img_idft = np.abs(img_idft)
        self.PicAfter = np.int8(img_idft)
        cv2.imwrite('img.jpg', self.PicAfter)
        self.PicAfter = QPixmap('img.jpg')
        self.ui.PicAfter.setPixmap(self.PicAfter)
        self.ui.PicAfter.setScaledContents(True)
        self.fftafter = cv2.imread('img.jpg')
        self.fftafter = FFT2(self.fftafter)
        self.fftafter = cvImgtoQtImg(self.fftafter)
        self.ui.FFTAfter.setPixmap(QPixmap(self.fftafter))
        self.ui.FFTAfter.setScaledContents(True)
        os.remove('img.jpg')

    def Corrosion(self):
        # 腐蚀运算
        if self.picpath == ' ':
            QMessageBox.critical(self, '操作失败', '请先导入图片',
                                 QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            return
        img = cv2.imread(self.picpath)
        self.fftbefore = FFT2(cv2.imread(self.picpath, 0))
        self.fftbefore = cvImgtoQtImg(self.fftbefore)
        self.ui.FFTBefore.setPixmap(QPixmap(self.fftbefore))
        self.ui.FFTBefore.setScaledContents(True)

        self.PicAfter = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret, img_binary = cv2.threshold(self.PicAfter, 55, 255, cv2.THRESH_BINARY)
        img_binary = np.ones(img_binary.shape, np.uint8) * 255 - img_binary
        kernel = np.ones((3, 3), np.uint8)
        self.PicAfter = cv2.erode(img_binary, kernel)
        self.fftafter = self.PicAfter
        self.PicAfter = cvImgtoQtImg(self.PicAfter)
        self.ui.PicAfter.setPixmap(QPixmap(self.PicAfter))
        self.ui.PicAfter.setScaledContents(True)
        self.fftafter = FFT2(self.fftafter)
        self.fftafter = cvImgtoQtImg(self.fftafter)
        self.ui.FFTAfter.setPixmap(QPixmap(self.fftafter))
        self.ui.FFTAfter.setScaledContents(True)

    def Expansion(self):
        # 膨胀运算
        if self.picpath == ' ':
            QMessageBox.critical(self, '操作失败', '请先导入图片',
                                 QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            return
        img = cv2.imread(self.picpath)

        self.fftbefore = FFT2(cv2.imread(self.picpath, 0))
        self.fftbefore = cvImgtoQtImg(self.fftbefore)
        self.ui.FFTBefore.setPixmap(QPixmap(self.fftbefore))
        self.ui.FFTBefore.setScaledContents(True)

        self.PicAfter = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret, img_binary = cv2.threshold(self.PicAfter, 55, 255, cv2.THRESH_BINARY)
        img_binary = np.ones(img_binary.shape, np.uint8) * 255 - img_binary
        kernel = np.ones((3, 3), np.uint8)
        self.PicAfter = cv2.dilate(img_binary, kernel, iterations=1)
        self.fftafter = self.PicAfter
        self.PicAfter = cvImgtoQtImg(self.PicAfter)
        self.ui.PicAfter.setPixmap(QPixmap(self.PicAfter))
        self.ui.PicAfter.setScaledContents(True)
        self.fftafter = FFT2(self.fftafter)
        self.fftafter = cvImgtoQtImg(self.fftafter)
        self.ui.FFTAfter.setPixmap(QPixmap(self.fftafter))
        self.ui.FFTAfter.setScaledContents(True)

    def Open(self):
        # 开运算
        if self.picpath == ' ':
            QMessageBox.critical(self, '操作失败', '请先导入图片',
                                 QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            return
        img = cv2.imread(self.picpath)

        self.fftbefore = FFT2(cv2.imread(self.picpath, 0))
        self.fftbefore = cvImgtoQtImg(self.fftbefore)
        self.ui.FFTBefore.setPixmap(QPixmap(self.fftbefore))
        self.ui.FFTBefore.setScaledContents(True)

        self.PicAfter = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret, img_binary = cv2.threshold(self.PicAfter, 55, 255, cv2.THRESH_BINARY)
        img_binary = np.ones(img_binary.shape, np.uint8) * 255 - img_binary
        kernel = np.ones((3, 3), np.uint8)
        self.PicAfter = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, kernel)
        self.fftafter = self.PicAfter
        self.PicAfter = cvImgtoQtImg(self.PicAfter)
        self.ui.PicAfter.setPixmap(QPixmap(self.PicAfter))
        self.ui.PicAfter.setScaledContents(True)
        self.fftafter = FFT2(self.fftafter)
        self.fftafter = cvImgtoQtImg(self.fftafter)
        self.ui.FFTAfter.setPixmap(QPixmap(self.fftafter))
        self.ui.FFTAfter.setScaledContents(True)

    def Close(self):
        # 闭运算
        if self.picpath == ' ':
            QMessageBox.critical(self, '操作失败', '请先导入图片',
                                 QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            return
        img = cv2.imread(self.picpath)

        self.fftbefore = FFT2(cv2.imread(self.picpath, 0))
        self.fftbefore = cvImgtoQtImg(self.fftbefore)
        self.ui.FFTBefore.setPixmap(QPixmap(self.fftbefore))
        self.ui.FFTBefore.setScaledContents(True)

        self.PicAfter = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret, img_binary = cv2.threshold(self.PicAfter, 55, 255, cv2.THRESH_BINARY)
        img_binary = np.ones(img_binary.shape, np.uint8) * 255 - img_binary
        kernel = np.ones((3, 3), np.uint8)
        self.PicAfter = cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, kernel)
        self.fftafter = self.PicAfter
        self.PicAfter = cvImgtoQtImg(self.PicAfter)
        self.ui.PicAfter.setPixmap(QPixmap(self.PicAfter))
        self.ui.PicAfter.setScaledContents(True)
        self.fftafter = FFT2(self.fftafter)
        self.fftafter = cvImgtoQtImg(self.fftafter)
        self.ui.FFTAfter.setPixmap(QPixmap(self.fftafter))
        self.ui.FFTAfter.setScaledContents(True)

    def LOG(self):
        # LOG检测
        if self.picpath == ' ':
            QMessageBox.critical(self, '操作失败', '请先导入图片',
                                 QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            return
        img = cv2.imread(self.picpath)

        self.fftbefore = FFT2(cv2.imread(self.picpath, 0))
        self.fftbefore = cvImgtoQtImg(self.fftbefore)
        self.ui.FFTBefore.setPixmap(QPixmap(self.fftbefore))
        self.ui.FFTBefore.setScaledContents(True)

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(img_gray, (3, 3), 1, 1)
        LOG_result = cv2.Laplacian(img_blur, cv2.CV_16S, ksize=1)
        self.PicAfter = cv2.convertScaleAbs(LOG_result)
        self.fftafter = self.PicAfter
        self.PicAfter = cvImgtoQtImg(self.PicAfter)
        self.ui.PicAfter.setPixmap(QPixmap(self.PicAfter))
        self.ui.PicAfter.setScaledContents(True)
        self.fftafter = FFT2(self.fftafter)
        self.fftafter = cvImgtoQtImg(self.fftafter)
        self.ui.FFTAfter.setPixmap(QPixmap(self.fftafter))
        self.ui.FFTAfter.setScaledContents(True)

    def Scharr(self):
        # Scharr算子
        if self.picpath == ' ':
            QMessageBox.critical(self, '操作失败', '请先导入图片',
                                 QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            return
        img = cv2.imread(self.picpath)

        self.fftbefore = FFT2(cv2.imread(self.picpath, 0))
        self.fftbefore = cvImgtoQtImg(self.fftbefore)
        self.ui.FFTBefore.setPixmap(QPixmap(self.fftbefore))
        self.ui.FFTBefore.setScaledContents(True)

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        Scharr_result = cv2.Scharr(img_gray, cv2.CV_16S, dx=1, dy=0)
        self.PicAfter = cv2.convertScaleAbs(Scharr_result)
        self.fftafter = self.PicAfter
        self.PicAfter = cvImgtoQtImg(self.PicAfter)
        self.ui.PicAfter.setPixmap(QPixmap(self.PicAfter))
        self.ui.PicAfter.setScaledContents(True)
        self.fftafter = FFT2(self.fftafter)
        self.fftafter = cvImgtoQtImg(self.fftafter)
        self.ui.FFTAfter.setPixmap(QPixmap(self.fftafter))
        self.ui.FFTAfter.setScaledContents(True)

    def Canny(self):
        # Canny算子
        if self.picpath == ' ':
            QMessageBox.critical(self, '操作失败', '请先导入图片',
                                 QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            return
        img = cv2.imread(self.picpath)

        self.fftbefore = FFT2(cv2.imread(self.picpath, 0))
        self.fftbefore = cvImgtoQtImg(self.fftbefore)
        self.ui.FFTBefore.setPixmap(QPixmap(self.fftbefore))
        self.ui.FFTBefore.setScaledContents(True)
        
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_blur_canny = cv2.GaussianBlur(img_gray, (7, 7), 1, 1)
        self.PicAfter = cv2.Canny(img_blur_canny, 50, 150)
        self.fftafter = self.PicAfter
        self.PicAfter = cvImgtoQtImg(self.PicAfter)
        self.ui.PicAfter.setPixmap(QPixmap(self.PicAfter))
        self.ui.PicAfter.setScaledContents(True)
        self.fftafter = FFT2(self.fftafter)
        self.fftafter = cvImgtoQtImg(self.fftafter)
        self.ui.FFTAfter.setPixmap(QPixmap(self.fftafter))
        self.ui.FFTAfter.setScaledContents(True)

    def Save(self):
        # 保存
        self.SaveName = QFileDialog.getSaveFileName(self, '选择文件', '', "Image Files (*.png *.jpg *.bmp)")[0]

        if self.SaveName != '':
            if type(self.PicAfter) == QImage:
                result = cv2.imwrite(self.SaveName, QImage2CV(self.PicAfter))
            else:
                result = cv2.imwrite(self.SaveName, QPixmap2cv(self.PicAfter))
            if result:
                QMessageBox.about(self, '保存成功', '保存成功')
            else:
                QMessageBox.critical(self, '保存失败', '路径中不能含有中文和空格',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
