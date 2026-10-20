import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import threading
import queue
import sys
import ast
import time
from pathlib import Path

import settings


ROOT = Path(__file__).resolve().parent
MODEL_FILE = ROOT / "model3.py"
SETTINGS_FILE = ROOT / "settings.py"
WEIGHTS_FILE = ROOT / "weights.txt"

SETTING_NAMES = [
    "answerCol", "inputCol", "rowCount",
    "finishThreshold", "maxTrainingSteps", "learningRate",
    "numberToStartWith", "numberOfDivisions",
    "numberToDivideWith", "trials",
    "optimizeLearningRate", "showInitialGuesses",
    "termCount", "termDivisor", "progressBarWidth",
    "FileName", "load", "showGraph"
]

process = None
output_queue = queue.Queue()
setting_vars = {}
model_data = {}
training_done = False


# -------------------- WINDOW --------------------

root = tk.Tk()
root.title("Polynomial Regression")
root.geometry("1100x720")
root.minsize(850, 550)

style = ttk.Style()
style.configure("Title.TLabel", font=("Arial", 17, "bold"))
style.configure("Metric.TLabel", font=("Arial", 19, "bold"))

root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=0, minsize=260)
root.grid_columnconfigure(1, weight=1)


# -------------------- SETTINGS PANEL --------------------

settings_frame = ttk.Frame(root, padding=10)
settings_frame.grid(row=0, column=0, sticky="nsew")
settings_frame.grid_rowconfigure(1, weight=1)
settings_frame.grid_columnconfigure(0, weight=1)

ttk.Label(
    settings_frame, text="Model Settings", style="Title.TLabel"
).grid(row=0, column=0, sticky="w", pady=(0, 10))

settings_canvas = tk.Canvas(settings_frame, highlightthickness=0)
settings_scrollbar = ttk.Scrollbar(
    settings_frame, orient="vertical",
    command=settings_canvas.yview
)
settings_canvas.configure(yscrollcommand=settings_scrollbar.set)

settings_canvas.grid(row=1, column=0, sticky="nsew")
settings_scrollbar.grid(row=1, column=1, sticky="ns")

settings_contents = ttk.Frame(settings_canvas, padding=(2, 2, 8, 8))
canvas_window = settings_canvas.create_window(
    (0, 0), window=settings_contents, anchor="nw"
)
settings_contents.columnconfigure(0, weight=1)


def update_scroll_region(event=None):
    settings_canvas.configure(scrollregion=settings_canvas.bbox("all"))


def resize_settings_contents(event):
    settings_canvas.itemconfigure(canvas_window, width=event.width)


settings_contents.bind("<Configure>", update_scroll_region)
settings_canvas.bind("<Configure>", resize_settings_contents)


def add_setting(name, row):
    value = getattr(settings, name)

    ttk.Label(settings_contents, text=name).grid(
        row=row, column=0, sticky="w", pady=(7, 2)
    )

    if isinstance(value, bool):
        variable = tk.BooleanVar(value=value)
        widget = ttk.Checkbutton(
            settings_contents, variable=variable
        )
    else:
        variable = tk.StringVar(value=str(value))
        widget = ttk.Entry(
            settings_contents, textvariable=variable
        )

    widget.grid(row=row + 1, column=0, sticky="ew")
    setting_vars[name] = variable


for i, name in enumerate(SETTING_NAMES):
    add_setting(name, i * 2)


# -------------------- DASHBOARD --------------------

dashboard = ttk.Frame(root, padding=15)
dashboard.grid(row=0, column=1, sticky="nsew")
dashboard.grid_columnconfigure(0, weight=1)
dashboard.grid_rowconfigure(4, weight=1)

ttk.Label(
    dashboard, text="Training Dashboard", style="Title.TLabel"
).grid(row=0, column=0, sticky="w", pady=(0, 12))

status_var = tk.StringVar(value="Ready")
ttk.Label(
    dashboard, textvariable=status_var
).grid(row=1, column=0, sticky="w", pady=(0, 10))


# Metrics

metrics = ttk.Frame(dashboard)
metrics.grid(row=2, column=0, sticky="ew", pady=(0, 12))
metrics.grid_columnconfigure((0, 1, 2), weight=1)


def metric_card(parent, column, title):
    card = ttk.LabelFrame(parent, text=title, padding=10)
    card.grid(row=0, column=column, sticky="ew", padx=4)
    value = tk.StringVar(value="—")
    ttk.Label(
        card, textvariable=value, style="Metric.TLabel"
    ).pack(anchor="w")
    return value


steps_var = metric_card(metrics, 0, "Training Steps")
time_var = metric_card(metrics, 1, "Training Time")
error_var = metric_card(metrics, 2, "Average Error")


# Prediction table

table_frame = ttk.LabelFrame(
    dashboard, text="Training Predictions", padding=8
)
table_frame.grid(row=4, column=0, sticky="nsew", pady=(0, 12))
table_frame.grid_rowconfigure(0, weight=1)
table_frame.grid_columnconfigure(0, weight=1)

columns = ("input", "actual", "predicted", "error", "status")
table = ttk.Treeview(
    table_frame, columns=columns, show="headings", height=12
)

for column, heading, width in [
    ("input", "Input", 100),
    ("actual", "Actual", 110),
    ("predicted", "Predicted", 110),
    ("error", "Absolute Error", 110),
    ("status", "Accuracy", 100),
]:
    table.heading(column, text=heading)
    table.column(column, width=width, anchor="center")

table.grid(row=0, column=0, sticky="nsew")

table_scrollbar = ttk.Scrollbar(
    table_frame, orient="vertical", command=table.yview
)
table_scrollbar.grid(row=0, column=1, sticky="ns")
table.configure(yscrollcommand=table_scrollbar.set)

table.tag_configure("good", background="#d9f2df", foreground="#145c2b")
table.tag_configure("bad", background="#f8d7da", foreground="#842029")


# New prediction

prediction_frame = ttk.LabelFrame(
    dashboard, text="Predict a New Input", padding=10
)
prediction_frame.grid(row=5, column=0, sticky="ew")
prediction_frame.grid_columnconfigure(0, weight=1)

prediction_entry = ttk.Entry(prediction_frame)
prediction_entry.grid(row=0, column=0, sticky="ew", padx=(0, 8))

prediction_result = tk.StringVar(value="Enter an input value.")
ttk.Label(
    prediction_frame, textvariable=prediction_result
).grid(row=1, column=0, columnspan=2, sticky="w", pady=(8, 0))


# -------------------- SETTINGS --------------------

def save_settings():
    lines = []

    for name in SETTING_NAMES:
        value = setting_vars[name].get()
        original = getattr(settings, name)

        try:
            if isinstance(original, bool):
                parsed = bool(value)
            elif isinstance(original, int):
                parsed = int(value)
            elif isinstance(original, float):
                parsed = float(value)
            else:
                parsed = str(value)
        except ValueError:
            messagebox.showerror(
                "Invalid Setting", f"Invalid value for {name}: {value}"
            )
            return False

        lines.append(f"{name} = {parsed!r}")

    try:
        SETTINGS_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except OSError as exc:
        messagebox.showerror("Save Error", str(exc))
        return False

    return True


# -------------------- MODEL OUTPUT --------------------

def handle_output(line):
    global model_data, training_done

    line = line.strip()
    if not line:
        return

    output_queue.put(("log", line))

    # Parse the machine-readable results emitted by model3.py.
    if line.startswith("UI_RESULTS:"):
        try:
            model_data = ast.literal_eval(line[len("UI_RESULTS:"):].strip())
            output_queue.put(("results", model_data))
        except (ValueError, SyntaxError):
            output_queue.put(("log", "Could not parse UI_RESULTS output."))

    elif line.startswith("UI_READY"):
        training_done = True
        output_queue.put(("ready", None))


def poll_output():
    while True:
        try:
            kind, value = output_queue.get_nowait()
        except queue.Empty:
            break

        if kind == "results":
            show_results(value)
        elif kind == "ready":
            status_var.set("Training complete")
            run_button.configure(state="normal")
        elif kind == "log":
            # Only report errors/status in the dashboard, not a huge terminal.
            low = value.lower()
            if "traceback" in low or low.startswith("error:"):
                status_var.set("Error")
                prediction_result.set(value)

    if process is not None and process.poll() is not None:
        run_button.configure(state="normal")
        if process.returncode != 0:
            status_var.set("Process exited with an error")

    root.after(100, poll_output)


def show_results(data):
    steps_var.set(str(data.get("steps", "—")))
    time_var.set(f'{data.get("time", 0):.3f} s')

    avg_error = data.get("average_error")
    error_var.set(f"{avg_error:.2f}%" if avg_error is not None else "—")

    for item in table.get_children():
        table.delete(item)

    threshold = float(data.get("threshold", settings.finishThreshold))

    for row in data.get("rows", []):
        absolute_error = abs(row["actual"] - row["predicted"])
        good = absolute_error < threshold

        table.insert(
            "", "end",
            values=(
                f'{row["input"]:.5g}',
                f'{row["actual"]:.5g}',
                f'{row["predicted"]:.5g}',
                f'{absolute_error:.5g}',
                "Within threshold" if good else "Outside threshold"
            ),
            tags=("good" if good else "bad",)
        )


# -------------------- RUN MODEL --------------------

def run_model():
    global process, model_data, training_done

    if process is not None and process.poll() is None:
        return

    if not save_settings():
        return

    if not MODEL_FILE.exists():
        messagebox.showerror(
            "Missing File",
            f"Could not find {MODEL_FILE.name} in {ROOT}"
        )
        return

    model_data = {}
    training_done = False

    for item in table.get_children():
        table.delete(item)

    steps_var.set("—")
    time_var.set("—")
    error_var.set("—")
    prediction_result.set("Train the model before predicting.")
    status_var.set("Training...")
    run_button.configure(state="disabled")

    def worker():
        global process
        try:
            process = subprocess.Popen(
                [sys.executable, "-u", str(MODEL_FILE)],
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            for line in process.stdout:
                handle_output(line)

            process.wait()

        except Exception as exc:
            output_queue.put(("log", f"Error: {exc}"))
            root.after(0, lambda: run_button.configure(state="normal"))

    threading.Thread(target=worker, daemon=True).start()


# -------------------- PREDICT NEW INPUT --------------------

def predict_new_input(event=None):
    if not model_data:
        prediction_result.set("Train the model first.")
        return

    try:
        value = float(prediction_entry.get())
        min_input = float(model_data["min_input"])
        max_input = float(model_data["max_input"])
        weights = model_data["weights"]
        divisor = float(model_data["term_divisor"])

        if max_input == min_input:
            normalized = 0
        else:
            normalized = -4 + (value - min_input) * 8 / (max_input - min_input)

        prediction = sum(
            weight / divisor**term * normalized**term
            for term, weight in enumerate(weights)
        )

        prediction_result.set(
            f"Predicted output for {value:g}: {prediction:.6g}"
        )

    except (ValueError, KeyError, TypeError, ZeroDivisionError) as exc:
        prediction_result.set(f"Could not calculate prediction: {exc}")


predict_button = ttk.Button(
    prediction_frame, text="Predict",
    command=predict_new_input
)
predict_button.grid(row=0, column=1)
prediction_entry.bind("<Return>", predict_new_input)


# -------------------- CLOSE --------------------

def on_close():
    if process is not None and process.poll() is None:
        process.terminate()
    root.destroy()


run_button = ttk.Button(
    settings_frame, text="Save Settings & Run", command=run_model
)
run_button.grid(row=2, column=0, sticky="ew", pady=(10, 0))

root.protocol("WM_DELETE_WINDOW", on_close)
root.after(100, poll_output)
root.mainloop()
