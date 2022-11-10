import torch
import pickle
import nmslib
import gradio as gr
from sentence_transformers import SentenceTransformer
from typing import Callable, Dict


K = 5


def create_demo(callback: Callable):
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                fn = gr.Textbox(label="Company name", placeholder="Enter company name here...")
        with gr.Row():
            with gr.Column():
                outs = [gr.Text(show_label=False) for _ in range(K)]
                outs[0].label = "Similar company names"
                outs[0].show_label = True
        btn = gr.Button("Find similar companies", variant="primary")
        btn.click(callback, inputs=fn, outputs=outs)
    return demo


class Callback:
    def __init__(self, model, data: Dict):
        self.index = nmslib.init(method='hnsw', space='cosinesimil')
        self.index.addDataPointBatch(data["emb"])
        self.index.createIndex({'post': 2}, print_progress=True)
        self.model = model
        self.data = data

    def __call__(self, input_name):
        emb = self.model.encode(input_name)
        ids, _ = self.index.knnQuery(emb, k=K)
        names = [self.data["names"][id] for id in ids]
        return names


def load_data(filename: str):
    with open(filename, "rb") as file:
        data = pickle.load(file)
    return data


def main():
    data = load_data("data.pickle")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("Vsevolod/company-names-similarity-sentence-transformer").to(device)
    callback = Callback(model, data)

    demo = create_demo(callback)
    demo.launch()


if __name__ == "__main__":
    main()
