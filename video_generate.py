import os
import cv2

def generate_animated_video(sentence, model, label_map, data, animegan, output_video="animated_output.avi", fps=15):
    words = sentence.lower().split()
    frame_list = []
    target_resolution = (256, 256)

    for word in words:
        if word in data:
            for frame in data[word]:
                img_path = frame["image_path"]
                if os.path.exists(img_path):
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.resize(img, target_resolution)
                        anime_img = animegan(img)
                        frame_list.append(anime_img)

    if not frame_list:
        print("No frames found.")
        return

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video, fourcc, fps, target_resolution)
    for frame in frame_list:
        out.write(frame)
    out.release()
    print(f"Video saved: {output_video}")