import argparse
import os

import cv2
import numpy as np
import torch
from tqdm import tqdm

from applications.efficientvit_sam.helpers.utils import ONNX, PYTORCH, TENSORRT
from applications.efficientvit_sam.run_efficientvit_sam_trt import preprocess
from applications.efficientvit_sam.helpers.predictors.effvit_sam_tensorrt import get_decoder_engine, get_encoder_engine

DIR = "assets/export_models/efficientvit_sam/tensorrt"
get_encoder_path = lambda model_name: f"{DIR}/{model_name}_encoder.engine"
get_decoder_path = lambda model_name: f"{DIR}/{model_name}_decoder.engine"

def measure_throughput_trt(trt_encoder, trt_decoder, image_paths, batch_size, model, prompt_type = "point"):
    device = "cuda"
    sam_images = []
    origin_image_sizes = []
    total_computation_time = 0
    total_encoder_time = 0
    total_decoder_time = 0
    total_batches = 0

    # Compute Stream
    compute_stream = torch.cuda.Stream()

    # Preprocess images and form batches
    with tqdm(total=len(image_paths), desc="Processing Images") as pbar:
        for image_path in image_paths:
            image_array = np.array(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
            raw_sam_img = image_array
            origin_image_size = raw_sam_img.shape[:2]
            origin_image_sizes.append(origin_image_size)

            print(model)
            if model in ["efficientvit-sam-l0", "efficientvit-sam-l1", "efficientvit-sam-l2"]:
                sam_image = preprocess(raw_sam_img, img_size=512, device="cpu")
            elif model in ["efficientvit-sam-xl0", "efficientvit-sam-xl1"]:
                sam_image = preprocess(raw_sam_img, img_size=1024, device="cpu")
            else:
                raise NotImplementedError
            sam_images.append(sam_image)

            # Process in batches
            if len(sam_images) == batch_size:
                # Allocate pinned memory for CPU tensors
                sam_images_batch = torch.cat(sam_images, dim=0).pin_memory()
                origin_image_sizes_batch = origin_image_sizes.copy()

                # Create CUDA events for timing
                start_event = torch.cuda.Event(enable_timing=True)
                end_of_encoder_event = torch.cuda.Event(enable_timing=True)
                end_of_computation_event = torch.cuda.Event(enable_timing=True)
                start_event.record()

                # Data transfer_stream
                with torch.cuda.stream(compute_stream):
                    sam_images_batch_gpu = sam_images_batch.to(device, non_blocking=True)
                    # Ensure data is available before computation
                    compute_stream.wait_stream(torch.cuda.current_stream())

                    # Run Encoder
                    image_embeddings = trt_encoder(sam_images_batch_gpu)
                    image_embeddings = image_embeddings.reshape(batch_size, 256, 64, 64)

                    end_of_encoder_event.record()

                    # Decoder Inputs with Default Point in the Middle
                    if prompt_type == "point":
                        point_coords = torch.tensor(
                            [[[origin_image_sizes_batch[idx][1] // 2, origin_image_sizes_batch[idx][0] // 2]] for idx in range(batch_size)],
                            dtype=torch.float32,
                            device=device,
                        )
                        point_labels = torch.tensor(
                            [[1] for _ in range(batch_size)],
                            dtype=torch.float32,
                            device=device,
                        )
                    elif prompt_type == "box":
                        B = origin_image_sizes_batch
                        boxes = torch.tensor(
                            [
                                [
                                    [B[idx][1] // 4, B[idx][0] // 4],
                                    [3 * B[idx][1] // 4, 3 * B[idx][1] // 4]
                                ]
                                for idx in range(batch_size)
                            ],
                            dtype=torch.float32,
                            device=device,
                        )

                        box_label = np.array([[2, 3] for _ in range(boxes.shape[0])], dtype=np.float32).reshape((-1, 2))

                        point_coords = boxes
                        point_labels = box_label

                    # Run Decoder
                    inputs = (image_embeddings, point_coords, point_labels)
                    low_res_masks, _ = trt_decoder(*inputs)
                    low_res_masks = low_res_masks.reshape(batch_size, -1, 256, 256)

                    # Transfer Data back to CPU
                    low_res_masks_cpu = low_res_masks.cpu()

                # Record end event
                end_of_computation_event.record(stream=compute_stream)

                # Wait for computation to finish
                end_of_computation_event.synchronize()

                # Calculate elapsed times
                computation_elapsed_time = start_event.elapsed_time(end_of_computation_event) / 1000.0  # Convert ms to seconds
                encoder_elapsed_time = start_event.elapsed_time(end_of_encoder_event) / 1000.0  # Convert ms to second
                decoder_elapsed_time = end_of_encoder_event.elapsed_time(end_of_computation_event) / 1000.0  # Convert ms to second

                total_computation_time += computation_elapsed_time
                total_encoder_time += encoder_elapsed_time
                total_decoder_time += decoder_elapsed_time
                total_batches += 1

                # Update progress bar
                pbar.update(batch_size)
                sam_images.clear()
                origin_image_sizes.clear()

                # Print average throughput
                if total_batches > 0:
                    avg_throughput = (batch_size * total_batches) / total_computation_time
                    encoder_throughput = (batch_size * total_batches) / total_encoder_time
                    decoder_throughput = (batch_size * total_batches) / total_decoder_time
                    avg_latency = total_computation_time / total_batches * 1000
                    print(f"Average Throughput (inluding data transfer): {avg_throughput:.2f} images/s")
                    print(f"Encoder Throughput: {encoder_throughput:.2f} images/s")
                    print(f"Decoder Throughput: {decoder_throughput:.2f} images/s")
                    print(f"Average Latency: {avg_latency:.2f} ms")
    
    total_throughput = f"Average Throughput (inluding data transfer): {avg_throughput:.2f} images/s"
    total_encoder_throughput = f"Encoder Throughput: {encoder_throughput:.2f} images/s"
    total_decoder_throughput = f"Decoder Throughput: {decoder_throughput:.2f} images/s"
    total_avg_latency = f"Average Latency: {avg_latency:.2f} ms"
    return total_throughput + " \n" + total_encoder_throughput + " \n" + total_decoder_throughput + " \n" + total_avg_latency

def process_throughput(model, dataset, iterations, batch_size, prompt_type, runtime):
    if runtime == TENSORRT:
        trt_encoder = get_encoder_engine(get_encoder_path(model))
        trt_decoder = get_decoder_engine(get_decoder_path(model))

        # List all image paths
        image_paths = []
        all_images = [img for img in os.listdir(dataset) if img.endswith(('.jpg', '.png'))]

        for i in range(min(iterations, len(all_images))):
            image_paths.append(os.path.join(dataset, all_images[i]))

        return measure_throughput_trt(trt_encoder, trt_decoder, image_paths, batch_size, model, prompt_type)
    else:
        return "Throughput measurement is only supported for TensorRT runtime."
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # l0, l1, l2, xl0, xl1
    parser.add_argument("--model", type=str, required=True, help="model type.")
    parser.add_argument("--prompt_type", type=str, default = "point", choices=["point", "box", "box_from_detector"]) 
    parser.add_argument("--encoder_engine", type=str, required=True, help="TRT encoder engine.")
    parser.add_argument("--decoder_engine", type=str, required=True, help="TRT decoder engine.")
    parser.add_argument("--img_dir", type=str, required=True, help="Directory with images for throughput measurement.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for throughput measurement.")
    args = parser.parse_args()

    # Create TRT Predictor
    trt_encoder = get_encoder_engine(args.encoder_engine)
    trt_decoder = get_decoder_engine(args.decoder_engine)
    
    # List all image paths
    image_paths = [os.path.join(args.img_dir, img) for img in os.listdir(args.img_dir) if img.endswith(('.jpg', '.png'))]

    # Measure throughput
    if (args.prompt_type in ["point", "box"]):
        measure_throughput_trt(trt_encoder, trt_decoder, image_paths, args.batch_size, args.model, args.prompt_type)
    else:
        raise NotImplementedError
