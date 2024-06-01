import fiftyone as fo 
import fiftyone.zoo as foz 

# Load Dataset
dataset = foz.load_zoo_dataset("coco-2017", split="validation")

# Randomly select 20 samples on which to generate predictions

view = dataset.take(20)

# Load zoo model

model = foz.load_zoo_model("keypoint-rcnn-resnet50-fpn-coco-torch")

# Run Inference
view.apply_model(model, label_field="predictions")

# Launch the FiftyOne App to visualize your dataset

session = fo.launch_app(dataset)
session.view = view
