# Gerekli kütüphaneler import ediliyor
import os  # Dosya ve klasör işlemleri için
import json  # JSON formatındaki verileri okuyup yazmak için
import cv2  # OpenCV: Görüntü işleme işlemleri için
import numpy as np  # Sayısal işlemler ve dizi işlemleri için
import torch  # PyTorch: Derin öğrenme işlemleri için
import torch.nn as nn  # Yapay sinir ağı katmanları için
import torch.optim as optim  # Model eğitimi için optimizasyon algoritmaları
from torch.utils.data import DataLoader, Dataset  # Veri yükleyici ve özel veri seti sınıfı için
import typing  # Tür tanımlamaları için (örn. Tuple[int, int])
import onnxruntime as ort  # ONNX formatındaki modelleri çalıştırmak için
import speech_recognition as sr  # Sesli konuşmayı yazıya çeviren kütüphane

# Anime stiline dönüştürme sınıfı tanımlanıyor
class AnimeGAN:
    def __init__(self, model_path: str = '', downsize_ratio: float = 1.0):  # Yapıcı metod, model yolu ve boyut oranı alır
        self.model = ort.InferenceSession(model_path)  # ONNX model dosyası yüklenir
        self.downsize_ratio = downsize_ratio  # Boyut küçültme oranı kaydedilir

    def to_32s(self, x):  # 32’nin katı en yakın değeri döndüren fonksiyon
        return 256 if x < 256 else x - x % 32  # 32’nin katı boyut ayarı yapılır

    def process_frame(self, frame: np.ndarray, x32: bool = True) -> np.ndarray:  # Görseli modele uygun hale getirir
        h, w = frame.shape[:2]  # Yükseklik ve genişlik alınır
        if x32:
            h, w = self.to_32s(h), self.to_32s(w)  # Boyutlar 32’nin katı yapılır
        frame = cv2.resize(frame, (w, h))  # Görüntü yeniden boyutlandırılır
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0  # RGB’ye çevirilip normalize edilir
        img = np.transpose(img, (2, 0, 1))  # Kanal boyutu en başa alınır (HWC -> CHW)
        img = np.expand_dims(img, axis=0)  # Batch boyutu eklenir
        return img  # İşlenmiş tensör döndürülür

    def post_process(self, frame: np.ndarray, wh: typing.Tuple[int, int]) -> np.ndarray:  # Model çıktısını işleyip orijinal boyuta getirir
        frame = np.squeeze(frame)  # Gereksiz boyut çıkarılır (1,3,H,W -> 3,H,W)
        frame = np.transpose(frame, (1, 2, 0))  # Kanal en sona alınır (CHW -> HWC)
        frame = ((frame + 1.0) * 127.5).clip(0, 255).astype(np.uint8)  # Normalize geri alınır ve uint8 yapılır
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # BGR formatına çevrilir
        return cv2.resize(frame, wh)  # Orijinal boyuta döndürülür

    def __call__(self, frame: np.ndarray) -> np.ndarray:  # Sınıf çağrıldığında otomatik olarak stilize işlemini yapar
        h, w = frame.shape[:2]  # Görsel boyutları alınır
        input_tensor = self.process_frame(frame)  # Giriş tensörü hazırlanır
        output = self.model.run(None, {'input': input_tensor})[0]  # Model tahmini alınır
        return self.post_process(output, (w, h))  # Çıktı işlenip döndürülür

# İşaret dili veri seti sınıfı tanımlanıyor
class SignLanguageDataset(Dataset):  # PyTorch Dataset sınıfından türetiliyor
    def __init__(self, json_path, image_folder, image_size=(64, 64)):  # Yapıcı metod
        with open(json_path, 'r') as f:  # JSON dosyası açılır
            self.data = json.load(f)  # JSON verisi yüklenir
        self.image_paths = []  # Görsel yolları tutulacak
        self.labels = []  # Etiketler tutulacak
        self.label_map = {word: idx for idx, word in enumerate(self.data.keys())}  # Kelime -> sayı eşleşmesi
        self.image_folder = image_folder  # Görsel klasörü kaydedilir
        self.image_size = image_size  # Görsel boyutu kaydedilir

        for word, filenames in self.data.items():  # Tüm kelime ve görseller dolaşılır
            for filename in filenames:
                self.image_paths.append(os.path.join(image_folder, filename))  # Görsel yolu eklenir
                self.labels.append(self.label_map[word])  # Etiketi eklenir

    def __len__(self):  # Veri seti uzunluğu
        return len(self.image_paths)

    def __getitem__(self, idx):  # Belirli bir indeks için veri döndürülür
        image_path = self.image_paths[idx]  # Görsel yolu alınır
        image = cv2.imread(image_path)  # Görsel okunur
        image = cv2.resize(image, self.image_size)  # Yeniden boyutlandırılır
        image = image.transpose((2, 0, 1))  # Kanal en başa alınır (CHW)
        image = torch.tensor(image, dtype=torch.float32) / 255.0  # Normalize edilip tensöre çevrilir
        label = torch.tensor(self.labels[idx], dtype=torch.long)  # Etiket tensörü alınır
        return image, label  # Görsel ve etiketi döndürülür

# Basit bir yapay sinir ağı (ANN) tanımlanıyor
class ANNModel(nn.Module):  # PyTorch sinir ağı modülünden türetiliyor
    def __init__(self, num_classes):  # Yapıcı metod, sınıf sayısı alır
        super(ANNModel, self).__init__()  # Üst sınıf yapıcısı çağrılır
        self.flatten = nn.Flatten()  # 3D tensörü tek boyuta indirir
        self.fc1 = nn.Linear(64 * 64 * 3, 256)  # Girişten 256 nörona
        self.relu1 = nn.ReLU()  # Aktivasyon fonksiyonu
        self.fc2 = nn.Linear(256, 128)  # Orta katman
        self.relu2 = nn.ReLU()  # Aktivasyon
        self.fc3 = nn.Linear(128, num_classes)  # Çıkış katmanı

    def forward(self, x):  # İleri besleme işlemi
        x = self.flatten(x)  # Görseli düzleştir
        x = self.relu1(self.fc1(x))  # İlk katman
        x = self.relu2(self.fc2(x))  # İkinci katman
        x = self.fc3(x)  # Çıkış katmanı
        return x  # Tahmin döndürülür

# Mikrofonla konuşma kaydeder ve yazıya çevirir
def record_and_recognize():
    r = sr.Recognizer()  # Tanıyıcı nesne
    with sr.Microphone() as source:  # Mikrofonu kaynak olarak al
        print("Konuşmanızı bekliyorum...")
        audio = r.listen(source)  # Ses kaydı yapılır
    try:
        text = r.recognize_google(audio, language="tr-TR")  # Google ile Türkçe tanıma
        print("Tanınan Metin:", text)
        return text  # Tanınan metin döndürülür
    except sr.UnknownValueError:
        print("Konuşma anlaşılamadı.")  # Ses anlaşılamadıysa
        return ""
    except sr.RequestError as e:
        print(f"Google API hatası: {e}")  # API hatası varsa
        return ""

# Girilen cümleye göre işaret dili görsellerinden video oluşturur
def generate_animated_video(sentence, model, label_map, data, animegan, image_folder, output_video="animated_output.avi", fps=40):
    words = sentence.lower().split()  # Cümle küçük harfe çevrilip kelimelere ayrılır
    frame_list = []  # Kare listesi oluşturulur

    for word in words:  # Her kelime için
        if word in data:  # Eğer kelime JSON verisinde varsa
            for image_name in data[word]:  # Kelimeye karşılık gelen her görsel için
                image_path = os.path.join(image_folder, image_name)  # Görsel yolu oluşturulur
                frame = cv2.imread(image_path)  # Görsel okunur
                if frame is None:
                    print(f"HATA: {image_path} dosyası bulunamadı.")  # Dosya yoksa uyarı
                    continue
                stylized_frame = animegan(frame)  # Anime stiline çevrilir
                frame_list.append(stylized_frame)  # Listeye eklenir

    if not frame_list:  # Eğer hiç kare yoksa
        print("Uygun çerçeve bulunamadı.")
        return

    height, width, _ = frame_list[0].shape  # Kare boyutu alınır
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Video formatı belirlenir
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))  # Video dosyası hazırlanır

    for frame in frame_list:  # Her kare video dosyasına yazılır
        out.write(frame)

    out.release()  # Video kaydı bitirilir
    print(f"Video oluşturuldu: {output_video}")  # Bilgilendirme yapılır

# Ana program
if __name__ == '__main__':
    json_path = 'data/sign_language_data.json'  # JSON veri dosyası yolu
    image_folder = 'data/sign_language_images'  # Görsellerin bulunduğu klasör
    onnx_model_path = 'model/animegan.onnx'  # AnimeGAN ONNX modeli yolu

    with open(json_path, 'r') as f:  # JSON dosyası açılır
        data = json.load(f)  # JSON içeriği yüklenir

    label_map = {word: idx for idx, word in enumerate(data.keys())}  # Etiket haritası oluşturulur

    num_classes = len(label_map)  # Sınıf sayısı hesaplanır
    model = ANNModel(num_classes)  # ANN modeli oluşturulur
    model.load_state_dict(torch.load("model/sign_language_model.pth", map_location=torch.device('cpu')))  # Eğitimli model yüklenir
    model.eval()  # Model değerlendirme moduna alınır

    animegan = AnimeGAN(model_path=onnx_model_path)  # AnimeGAN modeli yüklenir

    sentence = record_and_recognize()  # Ses kaydı alınıp metne çevrilir

    generate_animated_video(sentence, model, label_map, data, animegan, image_folder)  # Video oluşturulur