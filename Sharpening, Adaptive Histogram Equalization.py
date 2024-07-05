import cv2 as cv
import numpy as np
import os
import time
from dbr import *
# 初始化 BarcodeReader
reader = BarcodeReader()
# 获取免费的试用许可证
license_key = "DLS2eyJoYW5kc2RlIjoiMjAwMDAxLTE2NDk4Mjk3OTI2MzUiLCJvcmdhbml6YXRpb25JRCI6InNlc3Npb25QYXNzd29yZCI6indTcGR6Vm05WDJrcEQ5YUoifQ=="
reader.init_license(license_key)


def cv_imread(file_path):
    return cv.imdecode(np.fromfile(file_path, dtype=np.uint8), cv.IMREAD_COLOR)


def sharpen_image(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv.filter2D(image, -1, kernel)
    return sharpened


def apply_clahe(image):
    lab = cv.cvtColor(image, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab)
    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv.merge((cl, a, b))
    final = cv.cvtColor(limg, cv.COLOR_LAB2BGR)
    return final


def read_qr_code(image):
    qr_count = 0
    results = reader.decode_buffer(image)
    if results:
        for result in results:
            print(f"QR码内容: {result.barcode_text}")
            print(f"QR码类型: {result.barcode_format_string}")

            # Corrected attribute access
            points = result.localization_result.localization_points
            print(f"QR码位置: {points}")

            # 在图像上绘制 QR 码位置
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
        starttime = time.time()
        imgs = []

        for img_name in group:
            img_path = os.path.join(image_folder, img_name)
            if os.path.isfile(img_path):
                image = cv_imread(img_path)
                imgs.append(image)

        if len(imgs) == 0:
            continue

        stitcher = cv.Stitcher.create()
        (status, pano) = stitcher.stitch(imgs)

        if status != cv.Stitcher_OK:
            print(f"Cannot stitch images in group, error code = {status}")
            continue

        # 锐化和自适应直方图均衡化
        pano = sharpen_image(pano)
        pano = apply_clahe(pano)

        successful_groups += 1
        endtime = time.time()
        elapsed_time = endtime - starttime
        total_time += elapsed_time

        # 识别二维码
        qr_count = read_qr_code(pano)
        total_qr_count += qr_count

        print(f"Group {idx + 1} processed. QR codes found: {qr_count}. Time taken: {elapsed_time} seconds")

        # 保存拼接图像
        pano_filename = os.path.join(image_folder, f"pano_group_{idx + 1}.jpg")
        cv.imwrite(pano_filename, pano)
        print(f"Panorama image saved as {pano_filename}")

    avg_time_per_group = total_time / len(image_groups) if image_groups else 0
    print(f"Total QR codes detected: {total_qr_count}")
    print(f"Average time per group: {avg_time_per_group} seconds")
    print(f"Successfully stitched {successful_groups} groups of images.")


def main():
    image_folder = r"D:\QR Code\QR Code\dataset"
    if not os.path.exists(image_folder):
        print(f"Folder {image_folder} does not exist.")
        return

    stitch_and_recognize_images(image_folder)


if __name__ == "__main__":
    main()