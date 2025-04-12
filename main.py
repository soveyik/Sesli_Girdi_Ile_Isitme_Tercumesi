from model.ann_model import ANNModel
from data.dataset import SignLanguageDataset
from gan.animegan import AnimeGAN
from utils.video_generator import generate_animated_video
from utils.speech_to_text import record_and_recognize
from config import JSON_PATH, IMAGE_FOLDER, MODEL_PATH, ONNX_PATH

import torch

def main():
    dataset = SignLanguageDataset(JSON_PATH, IMAGE_FOLDER)
    model = ANNModel(num_classes=len(dataset.label_map))
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    animegan = AnimeGAN(ONNX_PATH)

    print("Konuşmaya başlayın...")
    sentence = record_and_recognize()
    print("Algılanan cümle:", sentence)

    generate_animated_video(sentence, model, dataset.label_map, dataset.data, animegan)

if __name__ == '__main__':
    main()