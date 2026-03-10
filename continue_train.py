from ultralytics import YOLO

model = YOLO("C:\\Users\\Kalleby\\Documents\\Vscode\\runs\\detect\\train2\\weights\\last.pt")

model.train(resume = True)