import os
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import typing
import onnxruntime as ort
import speech_recognition as sr

# =================== AnimeGAN Sınıfı ===================
class AnimeGAN:
    def __init__(self, model_path: str = '', downsize_ratio: float = 1.0):
        # Model yolunun var olup olmadığını kontrol ederiz
        if not os.path.exists(model_path):
            raise Exception(f"Model doesn't exist in {model_path}")

        self.downsize_ratio = downsize_ratio
        # GPU kullanılıp kullanılmadığını kontrol ederiz
        providers = ['CUDAExecutionProvider'] if ort.get_device() == "GPU" else ['CPUExecutionProvider']
        # ONNX modelini yükleriz
        self.ort_sess = ort.InferenceSession(model_path, providers=providers)

    def to_32s(self, x):
        # Görüntü boyutlarını 32'nin katı olacak şekilde ayarlarız
        return 256 if x < 256 else x - x % 32

    def process_frame(self, frame: np.ndarray, x32: bool = True) -> np.ndarray:
        # Görüntü boyutlarını yeniden boyutlandırırız
        h, w = frame.shape[:2]
        if x32:
            frame = cv2.resize(frame, (self.to_32s(int(w * self.downsize_ratio)), self.to_32s(int(h * self.downsize_ratio))))
        # Görüntüyü işleme için normalize ederiz
        frame = frame.astype(np.float32) / 127.5 - 1.0
        return frame

    def post_process(self, frame: np.ndarray, wh: typing.Tuple[int, int]) -> np.ndarray:
        # Çıkışı geri dönüştürürüz ve boyutlandırırız
        frame = (frame.squeeze() + 1.) / 2 * 255
        frame = frame.astype(np.uint8)
        frame = cv2.resize(frame, (wh[0], wh[1]))
        return frame

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        # İşlem ve post işleme adımlarını uygularız
        image = self.process_frame(frame)
        outputs = self.ort_sess.run(None, {self.ort_sess._inputs_meta[0].name: np.expand_dims(image, axis=0)})
        frame = self.post_process(outputs[0], frame.shape[:2][::-1])
        return frame


# =================== Dataset Sınıfı ===================
class SignLanguageDataset(Dataset):
    def __init__(self, json_path, image_folder, image_size=(64, 64)):
        # JSON verilerini yükleriz
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.image_paths = []
        self.labels = []
        self.label_map = {word: idx for idx, word in enumerate(self.data.keys())}

        # Her kelime için görüntü dosyalarını okuruz
        for word, frames in self.data.items():
            for frame in frames:
                img_path = os.path.normpath(os.path.join(image_folder, os.path.basename(frame["image_path"])))
                if os.path.exists(img_path):
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.resize(img, image_size)
                        img = img / 255.0
                        self.image_paths.append(img)
                        self.labels.append(self.label_map[word])

        self.image_paths = np.array(self.image_paths, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)

    def _len_(self):
        # Dataset uzunluğunu döner
        return len(self.image_paths)

    def _getitem_(self, idx):
        # Belirtilen indeksteki görüntü ve etiketi döner
        img = self.image_paths[idx]
        label = self.labels[idx]
        img = np.transpose(img, (2, 0, 1))
        img = torch.tensor(img, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        return img, label


# =================== ANN Model Sınıfı ===================
class ANNModel(nn.Module):
    def __init__(self, num_classes):
        super(ANNModel, self).__init__()
        # Yapay sinir ağı (ANN) modelini oluşturuyoruz
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 64 * 3, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # İleri geçiş fonksiyonu
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


# =================== Konuşmadan Metne Dönüşüm (Real-time) ===================
def record_and_recognize():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        print("Bir şeyler söyleyin...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        print("Tanımlanıyor...")
        # Google API ile Türkçe dilinde ses tanıma
        text = recognizer.recognize_google(audio, language="tr-TR")
        print(f"Tanınan metin: {text}")  # Debug çıktısı
        
        # Baş harfi küçük yapmak için:
        text = text[0].lower() + text[1:] if text else text
        return text
    except sr.UnknownValueError:
        print("Üzgünüm, sesi anlayamadım.")
        return ""
    except sr.RequestError:
        print("İstek başarısız oldu. İnternet bağlantınızı kontrol edin.")
        return ""


# =================== Video Üretici ===================
def generate_animated_video(sentence, model, label_map, data, animegan, image_folder, output_video="animated_output.avi", fps=40):
    words = sentence.split()
    frame_list = []
    target_resolution = (256, 256)

    # Tanımlanan kelimeleri yazdırıyoruz
    print(f"Sentence words: {words}")

    # Her bir kelimeyi işleyip ilgili resimleri alıyoruz
    for word in words:
        if word in data:
            for frame in data[word]:
                img_path = os.path.normpath(os.path.join(image_folder, os.path.basename(frame["image_path"])))
                if os.path.exists(img_path):
                    img = cv2.imread(img_path)
                    if img is not None:
                        print(f"Found image: {img_path}")
                        img = cv2.resize(img, target_resolution)
                        anime_img = animegan(img)  # AnimeGAN ile animasyonlu hale getiriyoruz
                        frame_list.append(anime_img)
                    else:
                        print(f"Failed to read image: {img_path}")
                else:
                    print(f"Image path does not exist: {img_path}")
        else:
            print(f"Warning: '{word}' not found in dataset!")

    if not frame_list:
        print("No frames generated!")
        return

    # Video oluşturma işlemi
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video, fourcc, fps, target_resolution)

    for frame in frame_list:
        out.write(frame)

    out.release()
    print(f"Animated video saved: {output_video}")


# =================== Ana Fonksiyon ===================
if __name__ == '__main__':
    # Dosya yollarını belirliyoruz
    json_path = r"output_data.json"
    image_folder = r"OutputImages"
    model_path = r"ann_sign_language_model.pth"
    onnx_path = r"Hayao_64.onnx"
    
    # Dataset ve model initialization
    dataset = SignLanguageDataset(json_path, image_folder)
    num_classes = len(dataset.label_map)
    model = ANNModel(num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # AnimeGAN modelini yükliyoruz
    animegan = AnimeGAN(onnx_path)

    # Konuşmayı kaydediyoruz ve animasyona dönüştürüyoruz
    print("Recording speech...")
    sentence = record_and_recognize()
    if sentence:
        print(f"Recognized sentence: {sentence}")
        generate_animated_video(sentence, model, dataset.label_map, dataset.data, animegan, image_folder)
    else:
        print("No speech recognized. Exiting...")