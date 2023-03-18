import torch

model = torch.hub.load('../yolov5','custom', path='north.pt', source='local')
model.confg = 0.5
# Image
im = 'nike_flower.png'

# Inference
results = model(im)

results.pandas().xyxy[0]
# print(results)
print('-==-==-=')
print(results.pandas().xyxy[0].to_json(orient="records"))
print('-==-==-=')
print(results.pandas().xyxy[0])

