import cv2 as cv
import numpy as np
import os
import time
from dbr import *

# 禁用 OpenCL
cv.ocl.setUseOpenCL(False)

# 初始化 BarcodeReader
reader = BarcodeReader()
# 获取免费的试用许可证
license_key = "DLS2eyJoYW5kc2V5SGFuZHNkZUlqbzIwMDAwMS0xNjQ5ODI5NzkyNjM1Iiwib3JnYW5pemF0aW9uSURJZCI6InNlc3Npb25QYXNzd29yZCIsImlkIjoiVDM2c29UQVMyVG5uQk1SUjZaSy9Nb2c9PSJ9=="
reader.init_license(license_key)

def cv_imread(file_path):
    return cv.imdecode(np.fromfile(file_path, dtype=np.uint8), cv.IMREAD_COLOR)

def denoise_image(image):
    return cv.medianBlur(image, 5)

# 调整图像大小
def resize_image(image, scale=0.5):
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    return cv.resize(image, (width, height))

def read_qr_code(image):
    qr_count = 0
    results = reader.decode_buffer(image)
    if results:
        for result in results:
            print(f"QR码内容: {result.barcode_text}")
            print(f"QR码类型: {result.barcode_format_string}")

            points = result.localization_result.localization_points
            print(f"QR码位置: {points}")

            pts = np.array(points, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv.polylines(image, [pts], True, (0, 255, 0), 2)
            cv.putText(image, result.barcode_text, (points[0][0], points[0][1] - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            qr_count += 1
    else:
        print("未检测到QR码")
    return qr_count

def stitch_and_recognize_images(image_folder):
    total_qr_count = 0
    total_time = 0
    successful_groups = 0
    image_groups = [os.listdir(image_folder)[i:i + 5] for i in range(0, len(os.listdir(image_folder)), 5)]

    for idx, group in enumerate(image_groups):
        # 跳过第261组
        if idx + 1 == 261:
            print(f"Skipping group {idx + 1}")
            continue

        starttime = time.time()
        imgs = []

        for img_name in group:
            img_path = os.path.join(image_folder, img_name)
            if os.path.isfile(img_path):
                image = cv_imread(img_path)
                denoised_image = denoise_image(image)
                denoised_image = resize_image(denoised_image, scale=0.5)  # 调整图像大小
                imgs.append(denoised_image)

        if len(imgs) == 0:
            continue

        stitcher = cv.Stitcher.create()
        (status, pano) = stitcher.stitch(imgs)

        if status != cv.Stitcher_OK:
            print(f"Cannot stitch images in group {idx + 1}, error code = {status}")
            continue

        # 检查拼接图像的尺寸，如果超过限制则跳过该组
        if pano is not None:
            pano_height, pano_width = pano.shape[:2]
            print(f"Stitched image size: {pano_width}x{pano_height}")
            if pano_width >= 32767 or pano_height >= 32767:
                print(f"Skipping group {idx + 1} due to large image size: {pano_width}x{pano_height}")
                continue  # 跳过当前组

        successful_groups += 1
        endtime = time.time()
        elapsed_time = endtime - starttime
        total_time += elapsed_time

        # 识别二维码
        qr_count = read_qr_code(pano)
        total_qr_count += qr_count

        print(f"Group {idx + 1} processed. QR codes found: {qr_count}. Time taken: {elapsed_time} seconds")

        # 手动释放内存
        del pano
        del imgs

    avg_time_per_group = total_time / len(image_groups) if image_groups else 0
    print(f"Total QR codes detected: {total_qr_count}")
    print(f"Average time per group: {avg_time_per_group} seconds")
    print(f"Successfully stitched {successful_groups} groups of images.")

def main():
    image_folder = r"D:\QR Code\dataset\F\T"
    if not os.path.exists(image_folder):
        print(f"Folder {image_folder} does not exist.")
        return

    stitch_and_recognize_images(image_folder)

if __name__ == "__main__":
    main()