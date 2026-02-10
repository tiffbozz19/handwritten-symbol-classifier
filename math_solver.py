import numpy as np
import torch
import torch.nn as nn
import tkinter as tk
from PIL import Image, ImageDraw
import ast
import operator as op

# =============== MODEL SETUP ===============

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


# =============== HELPER FUNCTIONS ===============

def crop_to_content(img, threshold=250, pad=20):
    arr = np.array(img)
    mask = arr < threshold  
    if not mask.any():
        return img
    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)

    y0 = max(0, y0 - pad)
    x0 = max(0, x0 - pad)
    y1 = min(img.height, y1 + pad)
    x1 = min(img.width, x1 + pad)

    return img.crop((x0, y0, x1, y1))


def segment_symbols(img, threshold=250, min_width=5):
    arr = np.array(img)
    mask = arr < threshold
    # project ink along x-axis
    proj = mask.sum(axis=0)  

    ranges = []
    in_symbol = False
    start = 0

    for x, val in enumerate(proj):
        if not in_symbol and val > 0:
            in_symbol = True
            start = x
        elif in_symbol and val == 0:
            end = x
            if end - start >= min_width:
                ranges.append((start, end))
            in_symbol = False

    if in_symbol:
        end = len(proj) - 1
        if end - start >= min_width:
            ranges.append((start, end))

    return ranges


def predict_symbol(img_crop_pil):
    img = img_crop_pil.resize((image_size, image_size))
    arr = np.array(img).astype("float32") / 255.0
    arr = (arr - 0.5) / 0.5  # normalize like training
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        idx = torch.argmax(probs, dim=1).item()
        conf = probs[0, idx].item()

    return idx, conf


# ---- safe expression evaluation ----

OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
}

def eval_ast(node):
    if isinstance(node, ast.Num):          
        return node.n
    if isinstance(node, ast.Constant):     
        return node.value
    if isinstance(node, ast.BinOp):
        left = eval_ast(node.left)
        right = eval_ast(node.right)
        return OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
        return OPS[type(node.op)](eval_ast(node.operand))
    raise ValueError("Unsupported expression")

def safe_eval(expr: str):
    tree = ast.parse(expr, mode="eval")
    return eval_ast(tree.body)


# =============== TKINTER APP ===============

last_x, last_y = None, None

class MathSolverApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Math Expression Solver")

        self.width = 1000
        self.height = 600

        self.canvas = tk.Canvas(root, width=self.width, height=self.height, bg="white")
        self.canvas.pack()

        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=5)

        self.solve_button = tk.Button(btn_frame, text="Solve", command=self.solve_expression)
        self.solve_button.pack(side=tk.LEFT, padx=5)

        self.clear_button = tk.Button(btn_frame, text="Clear", command=self.clear_canvas)
        self.clear_button.pack(side=tk.LEFT, padx=5)

        self.status_label = tk.Label(root, text="Draw an expression (e.g. 2+3*4) and click 'Solve'")
        self.status_label.pack(pady=5)

        self.image = Image.new("L", (self.width, self.height), 255)
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
        self.canvas.bind("<B1-Motion>", self.paint)

    def on_button_press(self, event):
        global last_x, last_y
        last_x, last_y = event.x, event.y

    def on_button_release(self, event):
        global last_x, last_y
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
        self.status_label.config(text="Draw an expression and click 'Solve'")

    def solve_expression(self):
        cropped = crop_to_content(self.image)
        ranges = segment_symbols(cropped)

        if not ranges:
            self.status_label.config(text="No symbols detected – draw darker or larger.")
            return

        symbols = []
        confidences = []

        for (x_start, x_end) in ranges:
            # add a bit of horizontal padding per symbol
            pad = 4
            left = max(0, x_start - pad)
            right = min(cropped.width, x_end + pad)
            crop_sym = cropped.crop((left, 0, right, cropped.height))

            idx, conf = predict_symbol(crop_sym)
            label = class_names[idx]
            symbols.append(label)
            confidences.append(conf)

        mapping = {
            "add": "+",
            "sub": "-",
            "mul": "*",
            "div": "/",
            "dec": ".",   
            "eq": "=",    
        }

        expr_parts = []
        for lab in symbols:
            expr_parts.append(mapping.get(lab, lab))

        expr = "".join(expr_parts)

        avg_conf = sum(confidences) / len(confidences)

        if "=" in expr:
            left_str, right_str = expr.split("=", 1)
            try:
                left_val = safe_eval(left_str)
                right_val = safe_eval(right_str)
                result_text = f"{left_str} = {left_val}, {right_str} = {right_val}"
                if abs(left_val - right_val) < 1e-6:
                    result_text += "  ✓ (true)"
                else:
                    result_text += "  ✗ (not equal)"
            except Exception as e:
                result_text = f"Could not evaluate '{expr}': {e}"
        else:
            try:
                value = safe_eval(expr)
                result_text = f"{expr} = {value}"
            except Exception as e:
                result_text = f"Could not evaluate '{expr}': {e}"

        self.status_label.config(
            text=f"{result_text}   (avg conf: {avg_conf:.2f})"
        )
        print("Symbols:", symbols)
        print("Expression:", expr)
        print("Avg confidence:", avg_conf)


# =============== RUN APP ===============

if __name__ == "__main__":
    root = tk.Tk()
    app = MathSolverApp(root)
    root.mainloop()
