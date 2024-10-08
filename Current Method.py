import cv2 as cv
import numpy as np
import os
import time
from dbr import *

# 初始化 BarcodeReader
reader = BarcodeReader()
# 获取免费的试用许可证
license_key = "DLS2eyJoYW5kc2RlIjoiMjAwMDAxLTE2NDk4Mjk3OTI2MzUiLCJvcmdhbml6YXRpb25JRCI6InNlc3Npb25QYXNzd29yZCI6IndTcGR6Vm05WDJrcEQ5YUoifQ=="
reader.init_license(license_key)


def cv_imread(file_path):
    return cv.imdecode(np.fromfile(file_path, dtype=np.uint8), cv.IMREAD_COLOR)


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
    successful_stitch_count = 0  # 新增计数器
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

        # 如果拼接成功，增加计数器
        successful_stitch_count += 1

        endtime = time.time()
        elapsed_time = endtime - starttime
        total_time += elapsed_time

        # 识别二维码
        qr_count = read_qr_code(pano)
        total_qr_count += qr_count

        print(f"Group {idx + 1} processed. QR codes found: {qr_count}. Time taken: {elapsed_time} seconds")

    avg_time_per_group = total_time / len(image_groups) if image_groups else 0
    print(f"Total QR codes detected: {total_qr_count}")
    print(f"Average time per group: {avg_time_per_group} seconds")

    # 输出拼接成功的组数
    print(f"Total groups stitched successfully: {successful_stitch_count}")


def main():
    image_folder = r"D:\QR Code\dataset\F\T"
    if not os.path.exists(image_folder):
        print(f"Folder {image_folder} does not exist.")
        return

    stitch_and_recognize_images(image_folder)


if __name__ == "__main__":
    main()