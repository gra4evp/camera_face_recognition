import cv2
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms import functional as F
from tqdm import tqdm


def load_model(device):
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
    model.to(device)
    model.eval()
    return model


def process_frame(model, frame, device):
    img = F.to_tensor(frame).unsqueeze(0).to(device)  # Делаем батч из одного кадра
    with torch.no_grad():
        predictions = model(img)

    boxes = predictions[0]['boxes']
    labels = predictions[0]['labels']
    scores = predictions[0]['scores']  # predictions
    for box, label, score in zip(boxes, labels, scores):
        if label == 1 and score > 0.5:  # Класс 1 это человек в COCO датасете
            box = box.to('cpu').numpy().astype(int)
            # Отрисуем на кадре зеленый box
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

    return frame


def detect_people_in_video(input_video_path, output_video_path):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Нужно для прогресс бара
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = load_model(device)

    with tqdm(total=total_frames, desc="Processing video", unit="frame") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:  # Файл закончился
                break

            processed_frame = process_frame(model, frame, device)
            out.write(processed_frame)
            pbar.update(1)

    cap.release()
    out.release()


if __name__ == '__main__':
    import sys
    if 'google.colab' in sys.modules:  # Для решения в гугл коллабе
        from google.colab import drive
        drive.mount('/content/drive')
        input_video_path = '/content/drive/MyDrive/Colab Notebooks/junior_test_task/crowd.mp4'
    else:
        input_video_path = 'crowd.mp4'

    output_video_path = 'output_crowd.mp4'
    detect_people_in_video(input_video_path, output_video_path)
