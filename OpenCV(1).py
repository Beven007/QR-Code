import cv2 as cv
import numpy as np
import os
import time
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import ImageTk, Image

# 禁用 OpenCL 以避免 GPU 内存不足的问题
cv.ocl.setUseOpenCL(False)

def cv_imread(file_path):
    return cv.imdecode(np.fromfile(file_path, dtype=np.uint8), cv.IMREAD_COLOR)

def stitch_images_auto():
    starttime_total = time.time()
    dataset_folder = r"D:\QR Code\dataset\F\4(2B)"
    files = sorted([os.path.join(dataset_folder, f) for f in os.listdir(dataset_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
    num_qr_codes = 0
    total_time = 0
    groups = [files[i:i + 5] for i in range(0, len(files), 5)]

    for i, group in enumerate(groups):
        if len(group) < 5:
            continue

        starttime = time.time()
        imgs = [cv_imread(img_path) for img_path in group]

        stitcher = cv.Stitcher.create()
        (status, pano) = stitcher.stitch(imgs)

        if status != cv.Stitcher_OK:
            messagebox.showerror("错误", f"无法拼接第 {i+1} 组图像，错误代码 = {status}")
            continue

        num_detected = recognize_qr(pano)
        num_qr_codes += num_detected

        endtime = time.time()
        total_time += (endtime - starttime)
        print(f"第 {i+1} 组拼接和识别耗时：{endtime - starttime} 秒")

    endtime_total = time.time()
    avg_time_per_group = total_time / len(groups)
    messagebox.showinfo("统计", f"总共识别了 {num_qr_codes} 个 QR 码。\n平均每组拼接和识别的时间：{avg_time_per_group:.2f} 秒。\n总时间：{endtime_total - starttime_total:.2f} 秒。")

def recognize_qr(image):
    # 创建 QRCodeDetector 对象
    qr_detector = cv.QRCodeDetector()

    # 检测并解码多个 QR 码
    retval, decoded_info, points, _ = qr_detector.detectAndDecodeMulti(image)

    if retval:
        qr_contents = "\n".join(decoded_info)
        num_qrcodes = len(decoded_info)

        for i in range(num_qrcodes):
            qr_code_points = points[i]
            num_points = len(qr_code_points)
            qr_code_points_int = np.int32(qr_code_points)
            for j in range(num_points):
                next_point_index = (j + 1) % num_points
                cv.line(image, tuple(qr_code_points_int[j]), tuple(qr_code_points_int[next_point_index]),
                        (0, 0, 255), 4)

        pil_image = Image.fromarray(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        pil_image.thumbnail((400, 400))

        result_label.config(text="QR 码内容：" + qr_contents)
        image_tk = ImageTk.PhotoImage(pil_image)
        image_label.config(image=image_tk)
        image_label.image = image_tk
    else:
        result_label.config(text="未检测到 QR 码")
        image_label.config(image='')

    return len(decoded_info)

def main():
    root = tk.Tk()
    root.title("图像拼接与 QR 码识别")

    label = tk.Label(root, text="图像拼接与 QR 码识别")
    label.pack(pady=10)

    auto_button = tk.Button(root, text="自动拼接并识别 QR 码", command=stitch_images_auto)
    auto_button.pack(pady=10)

    global result_label
    result_label = tk.Label(root, text="QR 码内容：")
    result_label.pack(pady=10)

    global image_label
    image_label = tk.Label(root)
    image_label.pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    main()