![](feature.png)

# The Trained Model and Annotation Feature Are Now Shipped in StarPilot Stable Release 6.7.3!

Vision-Adjacent Spot Monitoring (Vision-ASM) is now officially available in **StarPilot Release 6.7.3** as part of [Pull Request #75](https://github.com/firestar5683/StarPilot/pull/75), and is included in both the **development and stable Master branches**. Since the initial release, additional bug fixes and enhancements have also been included through [Pull Request #76](https://github.com/firestar5683/StarPilot/pull/76) and [Pull Request #77](https://github.com/firestar5683/StarPilot/pull/77).

Want to try it out? Install StarPilot using the [StarPilot installation guide](https://wiki.firestar.link/software/starpilot). Test out the new feature and share your feedback!

Community-submitted data will help expand and improve the model across different vehicles, interiors, cameras, and driving environments. Keep reading below to do learn how to do so. 

# What is Vision-ASM?

Vision-ASM uses the driver-facing wide-angle camera to detect vehicles in adjacent lanes through the side windows, improving blind-spot awareness across different vehicles and environments.

This is a community-data driven, vision-based blind-spot monitoring system. This repository provides a easy to use bulk utility for submitting training routes from both [Comma.ai Connect](https://connect.comma.ai/) and [Konik.ai Stable](https://stable.konik.ai/) for use in Vision-ASM model development.

Use cases:

* **Vehicles without BSM:** Provides a vision-only "adjacent-spot" alert system.
* **Vehicles with factory BSM:** Works alongside radar-based systems to provide **additional visual awareness**, including vehicles that may be too far ahead (adjacent to you) to trigger factory sensors.

The current model has been trained on routes from a single vehicle and performs well, but requires diverse data from different vehicle interiors, cameras, and driving environments.

## Submit Data

### Google Colab Upload Utility (Recommended)

The utility simplifies route submission by automatically handling route selection, file verification, and upload requests.

Requires:

* Google account (only for running Colab)
* Comma or Konik JWT token
* Dongle ID

[Run Colab Utility](https://colab.research.google.com/drive/1YmdxznB9cSIzj-ngIbW-rpv2C-QbJtAW)

The utility:

* Authenticates directly with Comma APIs using your JWT token.
* Lets you select routes from your device.
* Uploads only required training assets:
  * Driver camera footage
  * Log files
* Sets route visibility for training ingestion.
* Registers submitted routes for Vision-ASM dataset tracking.

Your JWT token is never sent anywhere except Comma APIs. The tool runs entirely inside your own Google Colab session.

### Manual Submission

[Submit Routes](https://forms.gle/NCC1eQ38B62icg5a8)

Useful routes:

* Vehicles beside your car
* Highway driving
* Lane changes
* Dense traffic
* Different weather and lighting conditions

Submitted data is used only for Vision-ASM development. The final model will be open-source.

## Privacy & Data Handling

Only the following information is shared:

* Route (dcam + logs which contain BSM radar data)
* Optional contact information provided during submission

Data is used for semi-automated fine-tuning of open-source computer vision models.

If your vehicle has factory BSM, those signals will be used in an automated process to help label adjacent vehicles.

## Vision-ASM Detection Preview

![Vision-ASM Demo 1](demo1.mp4)

![Vision-ASM Demo 2](demo2.mp4)

## Training Pipeline

The Vision-ASM training pipeline utilizes a workflow to train both an object detector and an boolean based classifier. 

### 1. Extraction & Labeling

* **Windshield Masking:** Isolates the passenger (left) and driver (right) side window areas using custom bounding polygons mapped onto the wide-angle camera frame. The bounding boxes are cropped and masked out using `cv2.bitwise_and` to ignore car interior details, reducing potential false positives.
* **Automated Labeling:** Frame intervals marked with active adjacent vehicles are downsampled to target frames. Positive frames pass through a helper network (`yolo26n`) to extract bounding-box annotations, while negative background frames are extracted at regular 30-frame intervals.

### 2. Dataset Balancing & Stratification

To ensure robust generalization across various driving scenarios, the pipeline enforces a stratified balancing protocol:
* **Binning:** Negative frames are sorted into strata categories based on vehicle profile, calculated time-of-day (derived from the route start time and segment offset), and road types (Highway vs. In-roads).
* **Parity Enforcement:** A round-robin drawing process limits negative samples to a 1.5x ratio relative to positive samples, preserving environmental diversity and preventing day/night bias.

### 3. Model Training

The pipeline trains models using a dual-track architecture to find the optimal deployment fit:

* **Track 1: YOLO26n (Object Detection - Shipped Model) and YOLO8n**
  * **Epochs:** 70
  * **Input Resolution:** Scaled to `[256, 352]` aspect ratio to fit the average crop dimensions.
  * **Augmentation:** Mosaic augmentation is disabled (`mosaic=0.0`) to keep the physical perspective of the window crops intact. Standard horizontal flipping is enabled (`fliplr=0.5`).
  * **Optimization:** Runs on FP16 Automatic Mixed Precision (AMP) to streamline computational resources.
* **Track 2: MobileNetV3 Small (Pure Binary Classification)**
  * **Epochs:** 15
  * **Architecture:** Uses a frozen pre-trained backbone with a custom binary linear classification head.
  * **Optimization:** Handled via the AdamW optimizer paired with a Cosine Annealing learning rate scheduler (starting at 0.001, weight decay 0.01).

### 4. Evaluation & Temporal Smoothing

* **Dual-Track Scorecard:** The pipeline evaluates and compares both architectures. It measures the binary classification accuracy of MobileNetV3 against the object detector's performance to establish relative parity.
* **Temporal Smoothing:** To prevent flickering alerts under real-world driving conditions, inference utilizes an evaluation scoring buffer. Predictions must persist over a customizable time window (e.g., 0.2 seconds) before confirming the presence or absence of a vehicle in the adjacent spot.

## Credits

Inspired by:

* StarPilot  
  https://github.com/firestar5683/StarPilot

* Vision Speed Limit Training
  https://github.com/firestar5683/StarPilot/blob/Dom/docs/speed-limit-vision-training.md

* JWT Authentication Concepts  
  https://github.com/nelsonjchen/op-dm-reading-tool

Thank you to these projects for providing examples of community-driven open-source vision model development.
