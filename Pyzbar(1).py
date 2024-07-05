import sys
import cv2 as cv
import numpy as np
import os
import time
from pyzbar.pyzbar import decode
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import threading
import socket

def cv_imread(file_path):
    return cv.imdecode(np.fromfile(file_path, dtype=np.uint8), cv.IMREAD_COLOR)

class QRCodeProcessor:
    def __init__(self):
        self.clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    def preprocess_image(self, image):
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        enhanced = self.clahe.apply(gray)
        denoised = cv.fastNlMeansDenoising(enhanced)
        return denoised

    def detect_and_decode(self, image):
        preprocessed = self.preprocess_image(image)
        results = decode(preprocessed)
        valid_results = []
        for barcode in results:
            if self.validate_qr_code(barcode):
                valid_results.append(barcode)
        return valid_results

    def validate_qr_code(self, barcode):
        return len(barcode.data) > 0

class DataTransmitter:
    def __init__(self, host='localhost', port=12345):
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def connect(self):
        try:
            self.socket.connect((self.host, self.port))
            return True
        except Exception as e:
            print(f"连接失败: {e}")
            return False

    def send_data(self, data):
        try:
            self.socket.sendall(data.encode())
            return True
        except Exception as e:
            print(f"发送数据失败: {e}")
            return False

    def close(self):
        self.socket.close()

class AppMainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(AppMainWindow, self).__init__(parent)
        self.init_ui()
        self.setWindowTitle("全景拼接与二维码识别工具")

        self.started = False
        self.currentPath = 'C:\\'
        self.cameraid = 0
        self.cap = cv.VideoCapture()
        self.timer_cam = QTimer(self)
        self.timer_cam.timeout.connect(self.capture_image)
        self.dir_index = 1

        self.btnStartStop.clicked.connect(self.start_recognize)
        self.btnFileRecognize.clicked.connect(self.onFileRecognize)
        self.spinBoxCamID.valueChanged.connect(self.camid_select)
        self.btnStop.clicked.connect(self.stop_recognize)

        self.main_save_dir = r'D:\QR Code\pyzbar'
        if not os.path.exists(self.main_save_dir):
            os.makedirs(self.main_save_dir)

        self.qr_processor = QRCodeProcessor()
        self.data_transmitter = DataTransmitter()
        self.processing_thread = None

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.labelView = QLabel()
        self.labelView.setFixedSize(640, 480)
        self.btnStartStop = QPushButton('开始识别')
        self.btnFileRecognize = QPushButton('文件识别')
        self.btnStop = QPushButton('停止')
        self.spinBoxCamID = QSpinBox()
        self.textEdit = QTextEdit()

        layout.addWidget(self.labelView)
        layout.addWidget(self.btnStartStop)
        layout.addWidget(self.btnFileRecognize)
        layout.addWidget(self.btnStop)
        layout.addWidget(QLabel('摄像头ID:'))
        layout.addWidget(self.spinBoxCamID)
        layout.addWidget(self.textEdit)

    def camid_select(self):
        self.cameraid = self.spinBoxCamID.value()
        print("当前摄像头ID:" + str(self.cameraid))

    def start_recognize(self):
        if not self.started:
            state = self.cap.open(self.cameraid)
            if state:
                self.started = True
                self.btnStartStop.setText('识别中...')
                self.capture_and_process_loop()
                if self.data_transmitter.connect():
                    print("已连接到数据传输服务器")
                else:
                    print("无法连接到数据传输服务器")
            else:
                QMessageBox.warning(self, '警告', '摄像头打开失败',
                                    buttons=QMessageBox.Ok,
                                    defaultButton=QMessageBox.Ok)
        else:
            self.stop_recognize()

    def stop_recognize(self):
        if self.started:
            self.cap.release()
            self.timer_cam.stop()
            self.started = False
            self.btnStartStop.setText('开始识别')
            self.data_transmitter.close()

    def capture_and_process_loop(self):
        if self.started:
            self.captured_images = []
            self.timer_cam.start(500)
            QTimer.singleShot(5000, self.process_camera_images)
        else:
            self.stop_recognize()

    def capture_image(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv.flip(frame, 1)
            self.captured_images.append(frame)
            save_dir = os.path.join(self.main_save_dir, f'folder_{self.dir_index}')

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            capture_save_path = os.path.join(save_dir, f'camera_capture_{len(self.captured_images)}.jpg')
            cv.imwrite(capture_save_path, frame)
            print(f"摄像头捕获的图片已保存至: {capture_save_path}")

        if len(self.captured_images) >= 10:
            self.timer_cam.stop()

    def process_camera_images(self):
        if self.captured_images:
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join()
            self.processing_thread = threading.Thread(target=self.process_images_thread)
            self.processing_thread.start()

    def process_images_thread(self):
        pano = self.stitch_images(self.captured_images)
        if pano is not None:
            self.recognize_qr(pano)

        self.dir_index += 1
        QMetaObject.invokeMethod(self, "capture_and_process_loop", Qt.QueuedConnection)

    def onFileRecognize(self):
        dataset_path = r'D:\QR Code\QR Code\dataset'
        image_files = [f for f in os.listdir(dataset_path) if f.endswith(('.jpg', '.png', '.bmp'))]
        image_files.sort()  # 确保文件按顺序处理

        total_qr_codes = 0
        total_time = 0
        group_count = 0

        for i in range(0, len(image_files), 5):
            group = image_files[i:i+5]
            group_images = [cv_imread(os.path.join(dataset_path, img)) for img in group]

            start_time = time.time()

            pano = self.stitch_images(group_images)
            if pano is not None:
                qr_codes = self.recognize_qr(pano)
                total_qr_codes += len(qr_codes)

            end_time = time.time()
            group_time = end_time - start_time
            total_time += group_time
            group_count += 1

            print(f"组 {group_count} 处理完成，耗时: {group_time:.2f} 秒")

        avg_time = total_time / group_count if group_count > 0 else 0
        print(f"\n总共识别出 {total_qr_codes} 个QR码")
        print(f"平均每组拼接加识别的时间: {avg_time:.2f} 秒")

        self.textEdit.append(f"\n总共识别出 {total_qr_codes} 个QR码")
        self.textEdit.append(f"平均每组拼接加识别的时间: {avg_time:.2f} 秒")

    def stitch_images(self, images):
        starttime = time.time()
        stitcher = cv.Stitcher_create(cv.Stitcher_PANORAMA)

        stitcher.setPanoConfidenceThresh(0.5)
        stitcher.setSeamEstimationResol(0.1)
        stitcher.setCompositingResol(0.5)
        stitcher.setPanoConfidenceThresh(1.0)
        stitcher.setWaveCorrection(False)

        status, pano = stitcher.stitch(images)
        if status != cv.Stitcher_OK:
            print(f"拼接失败, 错误代码 = {status}")
            return None
        else:
            endtime = time.time()
            print(f'图像拼接成功! 耗时: {endtime - starttime:.2f} 秒')

            timestamp = time.strftime('%Y%m%d_%H%M%S')
            save_dir = os.path.join(self.main_save_dir, 'result', timestamp)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            save_path = os.path.join(save_dir, 'pano_image.jpg')
            cv.imwrite(save_path, pano)
            print(f'全景图像已保存至: {save_path}')
            return pano

    def recognize_qr(self, image):
        results = self.qr_processor.detect_and_decode(image)
        self.textEdit.clear()
        if results:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            save_dir = os.path.join(self.main_save_dir, 'result', timestamp)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            for barcode in results:
                text = barcode.data.decode('utf-8')
                self.textEdit.append(text)
                pts = np.array(barcode.polygon, np.int32)
                cv.polylines(image, [pts], True, (0, 0, 255), 3)

                if self.data_transmitter.send_data(text):
                    print(f"数据已发送: {text}")
                else:
                    print(f"发送数据失败: {text}")

            result_file = os.path.join(save_dir, 'result.txt')
            with open(result_file, 'w', encoding='utf-8') as f:
                for barcode in results:
                    text = barcode.data.decode('utf-8')
                    f.write(text + '\n')
            print(f'识别结果已保存至: {result_file}')

            result_image_path = os.path.join(save_dir, 'result_image.jpg')
            cv.imwrite(result_image_path, image)
            print(f'识别结果图片已保存至: {result_image_path}')

        self.update_image_view(image)
        return results

    def update_image_view(self, image):
        img_height, img_width, _ = image.shape
        aspect_ratio = img_width / img_height
        if aspect_ratio > 1:
            image = cv.resize(image, (self.labelView.width(), int(self.labelView.width() / aspect_ratio)))
        else:
            image = cv.resize(image, (int(self.labelView.height() * aspect_ratio), self.labelView.height()))
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        qimage = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888)
        self.labelView.setPixmap(QPixmap.fromImage(qimage))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = AppMainWindow()
    win.show()
    sys.exit(app.exec_())