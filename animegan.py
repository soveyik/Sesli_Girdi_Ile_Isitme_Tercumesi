import os
import numpy as np
import onnxruntime as ort
import cv2

class AnimeGAN:
    def _init_(self, model_path, downsize_ratio=1.0):
        if not os.path.exists(model_path):
            raise Exception(f"Model doesn't exist in {model_path}")

        self.downsize_ratio = downsize_ratio
        providers = ['CUDAExecutionProvider'] if ort.get_device() == "GPU" else ['CPUExecutionProvider']
        self.ort_sess = ort.InferenceSession(model_path, providers=providers)

    def to_32s(self, x):
        return 256 if x < 256 else x - x % 32

    def process_frame(self, frame, x32=True):
        h, w = frame.shape[:2]
        if x32:
            frame = cv2.resize(frame, (self.to_32s(int(w * self.downsize_ratio)), self.to_32s(int(h * self.downsize_ratio))))
        return frame.astype(np.float32) / 127.5 - 1.0

    def post_process(self, frame, wh):
        frame = (frame.squeeze() + 1.) / 2 * 255
        frame = frame.astype(np.uint8)
        return cv2.resize(frame, (wh[0], wh[1]))

    def _call_(self, frame):
        image = self.process_frame(frame)
        outputs = self.ort_sess.run(None, {self.ort_sess.get_inputs()[0].name: np.expand_dims(image, axis=0)})
        return self.post_process(outputs[0], frame.shape[:2][::-1])