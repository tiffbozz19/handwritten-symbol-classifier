import numpy as np
import torch
import torch.nn as nn
import tkinter as tk
from PIL import Image, ImageDraw

# ----------------- MODEL SETUP -----------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class MathSymbolCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 64 -> 32

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 32 -> 16

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 16 -> 8

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 8 -> 4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


checkpoint = torch.load("math_symbol_cnn.pt", map_location=DEVICE)
class_names = checkpoint["class_names"]   
image_size = checkpoint["image_size"]    
num_classes = len(class_names)

model = MathSymbolCNN(num_classes)
model.load_state_dict(checkpoint["model"])
model.to(DEVICE)
model.eval()

# ----------------- TKINTER APP -----------------

last_x, last_y = None, None


class DigitClassifierApp:      
    def __init__(self, root):
        self.root = root
        self.root.title("Symbol Classifier")

        self.width = 600
        self.height = 600

        self.canvas = tk.Canvas(root, width=self.width, height=self.height, bg="white")
        self.canvas.pack()

        self.classify_button = tk.Button(root, text="Classify", command=self.classify_digit)
        self.classify_button.pack()

        self.clear_button = tk.Button(root, text="Clear", command=self.clear_canvas)
        self.clear_button.pack()

        self.status_label = tk.Label(root, text="Draw a symbol and click 'Classify'")
        self.status_label.pack()

        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
        self.canvas.bind("<B1-Motion>", self.paint)

        self.image = Image.new("L", (self.width, self.height), 255)
        self.draw = ImageDraw.Draw(self.image)

    def on_button_press(self, event):
        global last_x, last_y
        # start a new stroke here
        last_x, last_y = event.x, event.y

    def on_button_release(self, event):
        global last_x, last_y
        # end the current stroke so the next stroke won't connect
        last_x, last_y = None, None

    def paint(self, event):
        global last_x, last_y
        x, y = event.x, event.y
        if last_x is not None:
            self.canvas.create_line(x, y, last_x, last_y, fill="black", width=15)
            self.draw.line([x, y, last_x, last_y], fill="black", width=15)
        last_x, last_y = x, y

    def clear_canvas(self):
        global last_x, last_y
        last_x, last_y = None, None
        self.canvas.delete("all")
        self.image = Image.new("L", (self.width, self.height), 255)
        self.draw = ImageDraw.Draw(self.image)
        self.status_label.config(text="Draw a symbol and click 'Classify'")

    def classify_digit(self):
        global last_x, last_y
        last_x, last_y = None, None

        img = self.image.resize((image_size, image_size))

        img_arr = np.array(img).astype("float32") / 255.0 
        img_arr = (img_arr - 0.5) / 0.5                  

        img_tensor = torch.from_numpy(img_arr).unsqueeze(0).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(img_tensor)
            probs = torch.softmax(logits, dim=1)
            idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0, idx].item()

        predicted_label = class_names[idx]

        print("Predicted index:", idx, "label:", predicted_label, "conf:", confidence)

        self.status_label.config(
            text=f"Prediction: {predicted_label} (conf: {confidence:.2f})"
        )

    
 


# ----------------- RUN THE APP -----------------

if __name__ == "__main__":
    root = tk.Tk()
    app = DigitClassifierApp(root)
    root.mainloop()
