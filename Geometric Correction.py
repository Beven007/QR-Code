import cv2 as cv
import numpy as np
import os
import time
from dbr import *

# Initialize BarcodeReader
reader = BarcodeReader()
# Get free trial license
license_key = "DLS2eyJoYW5kc2RlIjoiMjAwMDAxLTE2NDk4Mjk3OTI2MzUiLCJvcmdhbml6YXRpb25JRCI6InNlc3Npb25QYXNzd29yZCI6IndTcGR6Vm05WDJrcEQ5YUoifQ=="
reader.init_license(license_key)


def cv_imread(file_path):
    return cv.imdecode(np.fromfile(file_path, dtype=np.uint8), cv.IMREAD_COLOR)


def correct_perspective(image, points):
    width = max(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3]))
    height = max(np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2]))

    dst_points = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype='float32')
    matrix = cv.getPerspectiveTransform(points.astype('float32'), dst_points)
    corrected_image = cv.warpPerspective(image, matrix, (int(width), int(height)))
    return corrected_image


def read_qr_code(image):
    qr_count = 0
    results = reader.decode_buffer(image)
    if results:
        for result in results:
            print(f"QR code content: {result.barcode_text}")
            print(f"QR code type: {result.barcode_format_string}")

            points = result.localization_result.localization_points
            print(f"QR code position: {points}")

            pts = np.array(points, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv.polylines(image, [pts], True, (0, 255, 0), 2)
            cv.putText(image, result.barcode_text, (points[0][0], points[0][1] - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Correct perspective and read the QR code again if necessary
            corrected_image = correct_perspective(image, np.array(points))
            additional_results = reader.decode_buffer(corrected_image)
            if additional_results:
                for additional_result in additional_results:
                    print(f"Corrected QR code content: {additional_result.barcode_text}")
                    print(f"Corrected QR code type: {additional_result.barcode_format_string}")
                    qr_count += 1
            else:
                qr_count += 1
    else:
        print("No QR code detected")
    return qr_count


def stitch_and_recognize_images(image_folder):
    total_qr_count = 0
    total_time = 0
    successful_groups = 0  # 统计成功拼接的组数
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

        successful_groups += 1  # 拼接成功的组计数器自增

        endtime = time.time()
        elapsed_time = endtime - starttime
        total_time += elapsed_time

        # Recognize QR codes
        qr_count = read_qr_code(pano)
        total_qr_count += qr_count

        print(f"Group {idx + 1} processed. QR codes found: {qr_count}. Time taken: {elapsed_time} seconds")

        # Save stitched image
       # pano_filename = os.path.join(image_folder, f"pano_group_{idx + 1}.jpg")
      #  cv.imwrite(pano_filename, pano)
      #  print(f"Panorama image saved as {pano_filename}")

    avg_time_per_group = total_time / len(image_groups) if image_groups else 0
    print(f"Total QR codes detected: {total_qr_count}")
    print(f"Average time per group: {avg_time_per_group} seconds")
    print(f"Successfully stitched {successful_groups} groups of images.")  # 输出拼接成功的组数


def main():
    image_folder = r"D:\QR Code\dataset\F\T"
    if not os.path.exists(image_folder):
        print(f"Folder {image_folder} does not exist.")
        return

    stitch_and_recognize_images(image_folder)


if __name__ == "__main__":
    main()