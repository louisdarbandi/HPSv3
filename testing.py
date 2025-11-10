from hpsv3 import HPSv3RewardInferencer

# Initialize the model
inferencer = HPSv3RewardInferencer(device='cuda')

# Evaluate images
image_paths = ["assets/304_ideogram.jpg", "assets/304_nanobanana.jpg", "assets/3829_hunyuan.png", "assets/3829_ideogram_quality.jpg", "assets/3829_nanobanana.jpg", "assets/3829_seedream_2k.jpg"]
prompts = [
  "photorealistic image of a stainless steel kitchen sink with water droplets",
  "photorealistic image of a stainless steel kitchen sink with water droplets",
  "woman wearing glasses reflecting city lights, calm evening feel", 
  "woman wearing glasses reflecting city lights, calm evening feel",
  "woman wearing glasses reflecting city lights, calm evening feel",
  "woman wearing glasses reflecting city lights, calm evening feel"
]

# Get preference scores
rewards = inferencer.reward(prompts, image_paths=image_paths)
scores = [reward[0].item() for reward in rewards]  # Extract mu values
print([f"Image scores: {score}, {image_path}"] for score, image_path in zip(scores,image_paths))