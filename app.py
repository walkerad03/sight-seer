import gradio as gr
import plotly.graph_objects as go
import torch
from torchvision import transforms
from ml_model_kit import model_builder


def process_image(image):
    image = image.convert("RGB")

    image_tensor = transform(image).unsqueeze(0)
    image_tensor = image_tensor.to(device)

    model.eval()
    with torch.inference_mode():
        output = model(image_tensor)

    coords = output.cpu().numpy()

    lat, lon = coords[0, 0], coords[0, 1]

    lat = lat * 5.2881766013678755 + 38.337428876696954
    lon = lon * 13.681640981664552 - 91.16433196360929

    fig = go.Figure(
        go.Scattermapbox(
            lat=[lat],
            lon=[lon],
            mode="markers",
            marker=go.scattermapbox.Marker(size=14),
            text=["Prediction"],
        )
    )

    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=lat, lon=lon),
            zoom=2,
        ),
        margin=dict(l=0, r=0, t=0, b=0),
    )

    return fig


if __name__ == "__main__":
    MODEL_PATH = "checkpoints/sightseer_512_30.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model_builder.ResNet18(
        3, resblock=model_builder.ResBlock, outputs=951
    )

    transform = transforms.Compose(
        [
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[137.1102, 144.0311, 137.1939],
                std=[46.3285, 45.0499, 60.7456],
            ),
        ]
    )

    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.to(device)

    with gr.Blocks() as interface:
        gr.Markdown("# SightSeer Demo")

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload Image")
                submit_button = gr.Button("Run")
            with gr.Column():
                map_output = gr.Plot(label="Map")

        submit_button.click(
            process_image, inputs=image_input, outputs=map_output
        )

    interface.launch(share=True)
