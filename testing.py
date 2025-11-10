from hpsv3 import HPSv3RewardInferencer

# Initialize the model
inferencer = HPSv3RewardInferencer(device='cuda')

# Evaluate images
image_paths = ["assets/304_ideogram.jpg", "assets/304_nanobanana.jpg"]
prompts = [
  "photorealistic image of a stainless steel kitchen sink with water droplets",
  "photorealistic image of a stainless steel kitchen sink with water droplets"
]

# Get preference scores
rewards = inferencer.reward(prompts, image_paths=image_paths)
scores = [reward[0].item() for reward in rewards]  # Extract mu values
print(f"Image scores: {scores}")