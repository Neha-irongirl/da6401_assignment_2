
import torch
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from models.multitask import MultiTaskPerceptionModel

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

TRANSFORM = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2(),
])

CLASS_NAMES = [   # 37 Oxford-IIIT Pet breeds (fill in full list)
    'Abyssinian','Bengal','Birman','Bombay','British_Shorthair',
    'Egyptian_Mau','Maine_Coon','Persian','Ragdoll','Russian_Blue',
    'Siamese','Sphynx','american_bulldog','american_pit_bull_terrier',
    'basset_hound','beagle','boxer','chihuahua','english_cocker_spaniel',
    'english_setter','german_shorthaired','great_pyrenees','havanese',
    'japanese_chin','keeshond','leonberger','miniature_pinscher',
    'newfoundland','pomeranian','pug','saint_bernard','samoyed',
    'scottish_terrier','shiba_inu','staffordshire_bull_terrier',
    'wheaten_terrier','yorkshire_terrier',
]


def predict(image_path,
            classifier_path='checkpoints/classifier.pth',
            localizer_path='checkpoints/localizer.pth',
            unet_path='checkpoints/unet.pth'):
    """
    Run full multi-task pipeline on a single image.
    Returns:
      breed      : predicted breed name (str)
      bbox       : [x_center, y_center, width, height] in pixels (list of 4 floats)
      mask       : (224, 224) numpy array with class indices 0/1/2
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model (downloads checkpoints from Drive via gdown)
    model = MultiTaskPerceptionModel(
        classifier_path=classifier_path,
        localizer_path=localizer_path,
        unet_path=unet_path,
    ).to(device)
    model.eval()

    # Preprocess
    img   = np.array(Image.open(image_path).convert('RGB'))
    img_t = TRANSFORM(image=img)['image'].unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        cls_out, loc_out, seg_out = model(img_t)

    breed  = CLASS_NAMES[cls_out.argmax(1).item()]
    bbox   = loc_out[0].cpu().tolist()   # [cx, cy, w, h] in pixels
    mask   = seg_out[0].argmax(0).cpu().numpy().astype(np.uint8)

    return breed, bbox, mask


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python inference.py <image_path>")
        sys.exit(1)

    breed, bbox, mask = predict(sys.argv[1])
    print(f"Breed : {breed}")
    print(f"BBox  : x_center={bbox[0]:.1f}, y_center={bbox[1]:.1f}, "
          f"w={bbox[2]:.1f}, h={bbox[3]:.1f}")
    print(f"Mask  : shape={mask.shape}, unique classes={np.unique(mask).tolist()}")
