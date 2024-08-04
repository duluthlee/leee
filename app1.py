import gradio as gr
import requests
import torch
import numpy as np
from PIL import Image, ImageDraw
from torchvision import transforms
import segmentation_models_pytorch as smp
from skimage import measure
import google.generativeai as genai
import json
import matplotlib.pyplot as plt

# 配置Google API Key
api_key = 'AIzaSyBCFIOLQIqFmJbz0OP_lqnAQcJMm9h25zs'
genai.configure(api_key=api_key)

# 加载分割模型
def load_model(model_path):
    model = smp.Unet(
        encoder_name="resnet34",  # 确保编码器名称与训练时的一致
        encoder_weights=None,  # 使用已经保存的权重
        in_channels=3,
        classes=1,
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# 进行图像分割
def segment_image(model, image):
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),  # 调整大小以适应模型输入
        transforms.ToTensor(),
    ])
    input_tensor = preprocess(image).unsqueeze(0)  # 增加batch维度

    with torch.no_grad():
        output = model(input_tensor)
        output = torch.sigmoid(output)  # 根据模型输出处理
        output = output.squeeze().cpu().numpy()

    return output

# 选出最长最细的线条
def select_longest_thinnest(segmented_image):
    labels = measure.label(segmented_image)
    properties = measure.regionprops(labels, intensity_image=segmented_image)

    max_length = 0
    min_width = float('inf')
    best_region = None
    regions = []

    for prop in properties:
        length = prop.major_axis_length
        width = prop.minor_axis_length
        mean_intensity = prop.mean_intensity
        
        regions.append((prop, length, width, mean_intensity))

        if length > max_length or (length == max_length and width < min_width):
            max_length = length
            min_width = width
            best_region = prop

    if best_region is not None:
        best_mask = (labels == best_region.label).astype(np.uint8)
        return best_mask, best_region.bbox, max_length, min_width, regions
    else:
        return None, None, 0, float('inf'), regions

# 找出平均强度最弱的线条
def find_weakest_intensity_region(regions):
    min_intensity = float('inf')
    best_region = None

    for region, length, width, mean_intensity in regions:
        if mean_intensity < min_intensity:
            min_intensity = mean_intensity
            best_region = region

    if best_region is not None:
        best_mask = (regions[0] == best_region.label).astype(np.uint8)
        return best_mask, best_region.bbox, best_region.major_axis_length, best_region.minor_axis_length
    else:
        return None, None, 0, float('inf')

# 在图像上绘制边界框
def draw_bbox(image, bbox):
    draw = ImageDraw.Draw(image)
    draw.rectangle(bbox, outline="red", width=3)
    return image

# 计算像素到微米的转换关系
def calculate_pixel_to_micron_ratio(scale_10_image, scale_5_image):
    scale_10_length = np.sum(scale_10_image)
    scale_5_length = np.sum(scale_5_image)
    ratio_10 = 10.0 / scale_10_length
    ratio_5 = 5.0 / scale_5_length
    return (ratio_10 + ratio_5) / 2  # 平均值

# 使用Gemini API分析线条信息
def analyze_lines_with_gemini(lines_info):
    model = genai.GenerativeModel('gemini-1.5-flash', generation_config={"response_mime_type": "application/json"})

    prompt = f"""
    Analyze the following nanobelts and determine the best one based on the provided criteria:
    {json.dumps(lines_info)}
    The best nanobelt is the one that is the longest and thinnest. If two nanobelts have similar length and width, choose the one with the weakest average intensity.
    """
    
    result = model.generate_content(prompt)
    response = json.loads(result.text)
    return response

def segment_and_analyze(image, model, pixel_to_micron_ratio):
    segmented_image = segment_image(model, image)
    
    best_mask, bbox, max_length, min_width, regions = select_longest_thinnest(segmented_image)
    
    if best_mask is None:
        return image, segmented_image, "No segments found", "No analysis performed", []

    if min_width == 0:  # 避免除以零错误
        return image, segmented_image, "No valid segments found", "No analysis performed", []

    lines_info = [{
        'length': length,
        'width': width,
        'mean_intensity': mean_intensity
    } for _, length, width, mean_intensity in regions]

    gemini_analysis = analyze_lines_with_gemini(lines_info)

    if gemini_analysis.get('best_nanobelt') == 'inconclusive':
        # 使用平均强度最弱的线条
        best_mask, bbox, max_length, min_width = find_weakest_intensity_region(regions)
    
    annotated_image = draw_bbox(image, bbox)
    length_in_micron = max_length * pixel_to_micron_ratio
    width_in_micron = min_width * pixel_to_micron_ratio

    return annotated_image, segmented_image, {
        "message": "Analyzing longest and thinnest segment.",
        "length_micron": length_in_micron,
        "width_micron": width_in_micron,
        "length_pixels": max_length,
        "width_pixels": min_width
    }, gemini_analysis, lines_info

# 加载模型
model_path = r"C:\Users\dulut\Desktop\hackathon\best_model.pth"  # 替换为你的模型路径
model = load_model(model_path)

# 加载比例尺图像
scale_10_image_path = r'C:\Users\dulut\Desktop\hackathon\binary_10micro.png'  # 替换为10微米比例尺图像路径
scale_5_image_path = r"C:\Users\dulut\Desktop\hackathon\binary_5micro.png"  # 替换为5微米比例尺图像路径
scale_10_image = np.array(Image.open(scale_10_image_path).convert("L")) > 0
scale_5_image = np.array(Image.open(scale_5_image_path).convert("L")) > 0

# 计算像素到微米的转换关系
pixel_to_micron_ratio = calculate_pixel_to_micron_ratio(scale_10_image, scale_5_image)

# 生成线条描述
def generate_description(line_data):
    length, width, mean_intensity = line_data['length'], line_data['width'], line_data['mean_intensity']
    return f"Length: {length:.6f} pixels, Width: {width:.6f} pixels, Mean Intensity: {mean_intensity:.6f}"

# Gradio 接口
def gradio_interface(image):
    annotated_image, segmented_image, result, analysis, lines_info = segment_and_analyze(image, model, pixel_to_micron_ratio)
    
    # 生成每条线条的描述
    descriptions = [generate_description(line_data) for line_data in lines_info]
    description_text = "\n\n".join([f"Line {i+1}:\n{desc}" for i, desc in enumerate(descriptions)])

    return annotated_image, segmented_image, str(result), str(analysis), description_text

iface = gr.Interface(
    fn=gradio_interface,
    inputs=gr.Image(label="Upload Image", type="pil"),
    outputs=[
        gr.Image(label="Annotated Image"),
        gr.Image(label="Segmented Image"),
        gr.Textbox(label="Result"),
        gr.Textbox(label="Analysis"),
        gr.Textbox(label="Line Descriptions")
    ],
    title="Automated Image-Based Measurement Tool for Nano Devices",
    description="Upload images to segment and analyze them using Gemini API."
)

iface.launch(share=True)  # 启用内网穿透
