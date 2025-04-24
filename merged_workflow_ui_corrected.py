
import cv2
import numpy as np
from tensorflow import keras
import tensorflow as tf
from PIL import Image, ImageTk
from huggingface_hub import from_pretrained_keras
import tkinter as tk
from tkinter import filedialog, messagebox

# Load the pre-trained low-light enhancement model
model = from_pretrained_keras("keras-io/low-light-image-enhancement")

def post_process(image, output):
    residuals = tf.split(output, num_or_size_splits=8, axis=-1)
    x = image
    for r in residuals:
        x = x + r * (tf.square(x) - x)
    return x

def low_light_enhancement(img, blend_ratio=0.5):
    img_rgb = Image.fromarray(img).convert('RGB')
    image = keras.preprocessing.image.img_to_array(img_rgb)
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)

    output_image = model(image)
    enhanced_img = post_process(image, output_image)
    enhanced_img = tf.cast((enhanced_img[0] * 255), dtype=tf.uint8)
    enhanced_img_pil = Image.fromarray(enhanced_img.numpy())
    
    final_image = Image.blend(img_rgb, enhanced_img_pil, alpha=blend_ratio)
    return np.array(final_image)

def denoise_image(img):
    return cv2.fastNlMeansDenoising(img, None, h=10, templateWindowSize=7, searchWindowSize=21)

def histogram_equalization_rgb(img, gamma=1.5):
    # Split the image into its respective R, G, B channels
    channels = cv2.split(img)

    # Apply histogram equalization on each channel
    equalized_channels = []
    for channel in channels:
        # Apply gamma correction
        channel_normalized = channel / 255.0
        channel_gamma_corrected = np.power(channel_normalized, gamma)
        channel_gamma_corrected = np.uint8(channel_gamma_corrected * 255)
        
        # Apply histogram equalization to each channel
        equalized_channel = cv2.equalizeHist(channel_gamma_corrected)
        equalized_channels.append(equalized_channel)

    # Merge the equalized channels back into a color image
    equalized_img = cv2.merge(equalized_channels)
    return equalized_img

# Blending the two sequences
def blend_images(img1, img2, blend_ratio=0.5):
    blended_img = cv2.addWeighted(img1, blend_ratio, img2, 1 - blend_ratio, 0)
    return blended_img

# Handling image processing based on workflow
def process_workflow(img, gamma1, gamma2, blend_ratio1, blend_ratio2):
    # Sequence 1: Histogram Equalization -> Denoising
    seq1_img = histogram_equalization_rgb(img, gamma=gamma1)
    seq1_img = denoise_image(seq1_img)
    seq1_img = low_light_enhancement(img, blend_ratio=blend_ratio1)
    
   #sequence2
    seq2_img = low_light_enhancement(img, blend_ratio=blend_ratio1)
    seq2_img = denoise_image(seq2_img)
    seq2_img = histogram_equalization_rgb(seq2_img, gamma=gamma2)

    
    blended_img = blend_images(seq1_img, seq2_img, blend_ratio2)
    return blended_img

def upload_and_process_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        try:
            img = cv2.imread(file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            
            gamma_value1 = gamma_slider.get() / 10.0
            gamma_value2 = gamma_slider2.get() / 10.0
            blend_ratio1 = blend_ratio_slider.get() / 100.0
            blend_ratio2 = blend_ratio_slider2.get() / 100.0
            
           
            output_img = process_workflow(img, gamma_value1, gamma_value2, blend_ratio1, blend_ratio2)
            
            
            output_img_pil = Image.fromarray(output_img)

           
            img_tk = ImageTk.PhotoImage(output_img_pil)

            
            image_label.config(image=img_tk)
            image_label.image = img_tk

            
            output_img_pil.save("enhanced_image.png")
            messagebox.showinfo("Success", "Image saved as 'enhanced_image.png'")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to process image: {e}")
root = tk.Tk()
root.title("Image Processing Workflow")
root.geometry("600x600")
root.config(bg="light blue")

sidebar = tk.Frame(root, bg="#ebbf87", width=150, height=400)
sidebar.pack(side="left", fill="y")
sidebar_label = tk.Label(sidebar, text="Upload your image", bg="#ebbf87", font=("Arial", 12, "bold"))
sidebar_label.pack(pady=20, padx=10)
sidebar_label1 = tk.Label(sidebar,text="Developed by Space-Tech", bg="#ebbf87", font=("Arial", 12, "bold"))
sidebar_label1.pack(pady=10,padx=10)

# Sliders for Gamma and Blend Ratio
gamma_slider_label = tk.Label(root, text="Gamma Value for Seq1 (1.0-3.0):", bg="light blue")
gamma_slider_label.pack(pady=10)
gamma_slider = tk.Scale(root, from_=10, to_=30, orient="horizontal")
gamma_slider.set(15)  # Default gamma of 1.5
gamma_slider.pack(pady=10)

gamma_slider2_label = tk.Label(root, text="Gamma Value for Seq2 (1.0-3.0):", bg="light blue")
gamma_slider2_label.pack(pady=10)
gamma_slider2 = tk.Scale(root, from_=10, to_=30, orient="horizontal")
gamma_slider2.set(15)  # Default gamma of 1.5
gamma_slider2.pack(pady=10)

blend_ratio_slider_label = tk.Label(root, text="Blend Ratio for Seq2 (0-1):", bg="light blue")
blend_ratio_slider_label.pack(pady=10)
blend_ratio_slider = tk.Scale(root, from_=0, to_=100, orient="horizontal")
blend_ratio_slider.set(50)  # Default blend ratio of 0.5
blend_ratio_slider.pack(pady=10)

blend_ratio_slider2_label = tk.Label(root, text="Final Blend Ratio (0-1):", bg="light blue")
blend_ratio_slider2_label.pack(pady=10)
blend_ratio_slider2 = tk.Scale(root, from_=0, to_=100, orient="horizontal")
blend_ratio_slider2.set(50)  # Default blend ratio of 0.5
blend_ratio_slider2.pack(pady=10)

upload_button = tk.Button(root, text="Upload and Process Image", command=upload_and_process_image)
upload_button.pack(pady=10)

image_label = tk.Label(root, bg="light blue")
image_label.pack(pady=10)

root.mainloop()
