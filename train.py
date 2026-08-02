from ultralytics import YOLO

def main():
    model = YOLO("yolo11n.pt")

    results = model.train(
        data = "/home/swarnava2/cognipilot/cranium/NXPCUP_2026.v2-v1_a.yolov11/data.yaml",
        epochs = 128,
        imgsz = 512,
        batch = 32,
        workers = 8,
        device = 0,
        project = "NPX_SIGN_CLASSIFIER",
        name = "yolo11n_sign_classifier"
    )

if __name__ == "__main__":
    main()