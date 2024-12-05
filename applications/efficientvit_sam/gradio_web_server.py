import argparse

import gradio as gr
from gradio_box_promptable_image import BoxPromptableImage
from gradio_clickable_arrow_dropdown import ClickableArrowDropdown
from gradio_point_promptable_image import PointPromptableImage
from gradio_sbmp_promptable_image import SBMPPromptableImage  # single box multiple point

from applications.efficientvit_sam.helpers.utils import (
    BOX_NMS_THRESH,
    POINTS_PER_BATCH,
    PRED_IOU_THRESH,
    STABILITY_SCORE_THRESH,
    get_available_models,
)
from applications.efficientvit_sam.helpers.process_prompts import process_boxes, process_full_img, process_points, process_points_and_boxes
from applications.efficientvit_sam.helpers.measure_throughput_trt import process_throughput

examples = [
    [{"image": "assets/fig/bear.jpeg", "points": [[]]}],
    [{"image": "assets/fig/cat.jpg", "points": [[]]}],
    [{"image": "assets/fig/city.png", "points": [[]]}],
    [{"image": "assets/fig/girl.png", "points": [[]]}],
    [{"image": "assets/fig/indoor.jpg", "points": [[]]}],
]

example_names = [
    "assets/fig/bear.jpeg",
    "assets/fig/cat.jpg",
    "assets/fig/city.png",
    "assets/fig/girl.png",
    "assets/fig/indoor.jpg",
]


def build_model_dropdown(runtime):
    available_models = get_available_models(runtime)

    interactive = len(available_models) > 0
    if len(available_models) == 0:
        available_models.append(f"No valid models found for {runtime} runtime")

    default_model = available_models[0]

    return ClickableArrowDropdown(
        choices=available_models,
        value=default_model,
        interactive=interactive,
        filterable=False,
        label="Model",
        info="Listed in order of decreasing segmentation quality/increasing speed",
    )


def build_promptable_segmentation_tab(prompter, label, process, runtime):
    output_img = gr.Image()
    with gr.Blocks() as tab_layout:
        with gr.Row(equal_height=False):
            with gr.Column():
                with gr.Column(variant="panel"):
                    input_image = prompter(label=label)
                    model_dropdown = build_model_dropdown(runtime)

                    with gr.Row():
                        clear_btn = gr.ClearButton(components=[input_image])
                        submit_btn = gr.Button("Submit", variant="primary")
                        submit_btn.click(
                            lambda *args: process(*args, runtime=runtime), [input_image, model_dropdown], output_img
                        )

                with gr.Column():
                    gr.Examples(examples=examples, inputs=[input_image])

            with gr.Column():
                with gr.Column(variant="panel"):
                    output_img.render()

    return tab_layout


def build_automatic_segmentation_tab(runtime):
    output_image_all_masks = gr.Image(show_label=False, elem_id="output_image")
    model_dropdown_all_masks = build_model_dropdown(runtime)

    points_per_batch = gr.Slider(
        minimum=1,
        maximum=128,
        value=POINTS_PER_BATCH,
        step=1,
        interactive=True,
        label=f"Guiding points per batch (default={POINTS_PER_BATCH})",
    )

    pred_iou_thresh = gr.Slider(
        minimum=0.0,
        maximum=1.0,
        value=PRED_IOU_THRESH,
        step=0.02,
        interactive=True,
        label=f"Prediction IOU threshold (default={PRED_IOU_THRESH})",
    )

    stability_score_thresh = gr.Slider(
        minimum=0,
        maximum=1.00,
        value=STABILITY_SCORE_THRESH,
        step=0.01,
        interactive=True,
        label=f"Stability score threshold (default={STABILITY_SCORE_THRESH})",
    )

    box_nms_thresh = gr.Slider(
        minimum=0.0,
        maximum=1.0,
        value=BOX_NMS_THRESH,
        step=0.02,
        interactive=True,
        label=f"Box NMS threshold (default={BOX_NMS_THRESH})",
    )

    def reset_sliders_to_default():
        return POINTS_PER_BATCH, PRED_IOU_THRESH, STABILITY_SCORE_THRESH, BOX_NMS_THRESH

    slider_components = [points_per_batch, pred_iou_thresh, stability_score_thresh, box_nms_thresh]

    with gr.Blocks() as full_img_segmentation:
        with gr.Row(equal_height=False):
            with gr.Column():
                with gr.Column(variant="panel"):
                    input_image = gr.Image(
                        label="Play with parameters to the right to change the output segmentation!",
                        value=example_names[4],
                    )
                    model_dropdown_all_masks.render()

                    with gr.Row():
                        clear_btn = gr.ClearButton(components=[input_image])
                        submit_btn = gr.Button("Submit", variant="primary")
                        submit_btn.click(
                            lambda *args: process_full_img(*args, runtime=runtime),
                            [input_image, model_dropdown_all_masks, *slider_components],
                            output_image_all_masks,
                        )

                with gr.Column():
                    gr.Examples(examples=example_names, inputs=[input_image])

            with gr.Column():
                with gr.Column(variant="panel"):
                    output_image_all_masks.render()

                with gr.Column():
                    with gr.Accordion(label="Customizable segmentation parameters", open=True):
                        points_per_batch.render()
                        pred_iou_thresh.render()
                        stability_score_thresh.render()
                        box_nms_thresh.render()

                    reset_thresh_btn = gr.Button("Reset segmentation parameters", variant="secondary")
                    reset_thresh_btn.click(reset_sliders_to_default, inputs=[], outputs=slider_components)

    return full_img_segmentation


def build_image_dataset_dropdown():
    # Define available datasets
    datasets = {
        "COCO 2017 (5000 Images)": "dataset/coco/val2017",
    }
    
    dataset_choices = list(datasets.keys())
    default_dataset = "COCO 2017 (5000 Images)"

    dropdown = gr.Dropdown(
        choices=dataset_choices,
        value=default_dataset,
        label="Select Image Dataset",
        info="Choose a dataset to use for throughput measurement.",
    )

    return dropdown, datasets

def build_throughput_measurement_tab(runtime):
    image_dataset_dropdown, datasets = build_image_dataset_dropdown()

    output_text = gr.Textbox(label = "Throughput Measurement Results",
                             interactive=False,
                             lines = 10)

    with gr.Blocks() as throughput_tab:
        with gr.Row(equal_height = False):
            with gr.Column():
                with gr.Column(variant = "panel"):
                    model_dropdown = build_model_dropdown(runtime)

                    image_dataset_dropdown.render()


                    # Number of Images to Process
                    num_iterations = gr.Number(
                        value = 1000, 
                        label = "Number of Images to Process", 
                        interactive=True,
                        precision = 0
                    )
                    
                    # Batch Size (4 or 16)
                    batch_size_dropdown = gr.Dropdown(
                        choices=[4, 16],
                        value=4,
                        label="Batch Size",
                        info="Choose the batch size for throughput measurement",
                        interactive=True
                    )

                    # Weight Precision: FP16 or FP32
                    weight_precision_dropdown = gr.Dropdown(
                        choices=["FP32", "FP16"],
                        value = "FP16",
                        label="Weight Precision",
                        info="Choose the weight precision for throughput measurement",
                        interactive=True
                    )

                    prompt_type_dropdown = gr.Dropdown(
                        choices=["point", "box"],
                        value="point",
                        label="Prompt Type",
                        info="Choose the prompt type for throughput measurement"
                    )

                    measure_button = gr.Button("Measure Throughput")
                    
                    measure_button.click(
                        lambda model, dataset, iterations, batch_size, prompt_type, weight_precision: process_throughput(
                            model, datasets[dataset], iterations, batch_size, prompt_type, weight_precision, runtime
                        ),
                        inputs=[model_dropdown, 
                                image_dataset_dropdown, 
                                num_iterations, 
                                batch_size_dropdown, 
                                prompt_type_dropdown, 
                                weight_precision_dropdown],
                        outputs=output_text,
                    )

                
            with gr.Column():
                with gr.Column(variant = "panel"):
                    output_text.render()
       
    return throughput_tab

def build_demo(runtime):
    point_label = "Click to add point(s)"
    point_segmentation = build_promptable_segmentation_tab(PointPromptableImage, point_label, process_points, runtime)

    box_label = "Click to add box(es)"
    box_segmentation = build_promptable_segmentation_tab(BoxPromptableImage, box_label, process_boxes, runtime)

    point_and_box_label = "Add points and/or boxes"
    point_and_box_segmentation = build_promptable_segmentation_tab(
        SBMPPromptableImage, point_and_box_label, process_points_and_boxes, runtime
    )

    full_img_segmentation = build_automatic_segmentation_tab(runtime)

    throughput_tab = build_throughput_measurement_tab(runtime)

    with gr.Blocks(theme=gr.themes.Base()) as demo:
        title_formatting = "<center><strong><font size='8'>EfficientViT-SAM</font></strong></center>"
        description = """
            This demo of EfficientViT-SAM can be prompted using points, boxes, a mix of a box and multiple points, as well as automatic full image segmentation.
            
            ### Instructions ###
            1) Provide an input to the image segmentation model - either select an image from the examples displayed below, upload your own, or use the webcam!
            2) To add prompts, simply click to add points and/or click, hold, and release to add boxes.  Point mode only registers point prompts, box mode
            only registers box prompts.  Mixed segmentation mode allows for both prompt types.
            3) Experiment with efficiency differences between our different models by changing the model from the dropdown menu.
            4) When experimenting with full image segmentation, feel free to play with the segmentation parameters! Hit the "reset segmentation parameters" button
            at the bottom to return parameters to their default values.

            ### Notes ###
            1) The prompts you provide within point segmentation mode and mixed segmentation mode will go towards segmenting a single object, not one object per prompt.
            2) Point prompt options
                - Left click: add foreground/ positive point
                - Right click (two-finger click on Mac trackpad): add background/negative point to exclude objects     

            [[GitHub](https://github.com/mit-han-lab/efficientvit)] 
            [[Models](https://github.com/mit-han-lab/efficientvit/blob/master/applications/sam.md#pretrained-models)]
            [[Paper](https://arxiv.org/abs/2402.05008)]
            """

        gr.Markdown(title_formatting)
        gr.Markdown(description)

        gr.TabbedInterface(
            interface_list=[point_segmentation, 
                            box_segmentation, 
                            point_and_box_segmentation, 
                            full_img_segmentation,
                            throughput_tab],
            tab_names=[
                "Point segmentation mode",
                "Box segmentation mode",
                "Mixed point and box segmentation mode",
                "Full image automatic segmentation mode",
                "Throughput Measurement"
            ],
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=str, default="pytorch", choices=["pytorch", "onnx", "tensorrt"])
    args = parser.parse_args()

    print()
    print(f"Running demo with {args.runtime}")
    print()

    demo = build_demo(args.runtime)
    demo.launch(share=True)
