from ultralytics import YOLO

model = YOLO("last_1.pt")

model.train(data="C:\\Users\\Kalleby\\Documents\\Projeto Libras\\Libras proprio.v1i.yolo26\\data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    device = -1)