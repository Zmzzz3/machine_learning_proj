import os
import os.path as path
from PIL import Image
from torchvision import datasets
from torchvision.transforms import v2

def load_images_from_folder(folder, transform=None):
    images = []
    for fname in sorted(os.listdir(folder)):
        img = Image.open(path.join(folder, fname))
        if transform: images.append(transform(img))
        else: images.append(img)
    return images

def delete_imgs(savepath, clear):
    images = 0
    for file in os.listdir(savepath):
        filepath = path.join(savepath, file)
        if clear: os.remove(filepath)
        else: images += 1
    return images

def modify_and_save(datapath, savepath, transform, clear):
    images = load_images_from_folder(datapath, transform)
    img_count = 0
    if not path.exists(savepath): os.makedirs(savepath)
    else: img_count = delete_imgs(savepath, clear)

    for idx, img in enumerate(images):
        if not isinstance(img, Image.Image):
            img = v2.ToPILImage()(img)
        img.save(path.join(savepath, f"img{img_count + idx}.png"))

def overlay_imgs(fg, bg):
    rgb = fg[:3]
    alpha = fg[3:]
    if alpha.max() > 1:
        alpha = alpha / 255.0
    alpha = alpha.expand_as(rgb)
    return alpha * rgb + (1 - alpha) * bg[:3]

def overlay_multiple(datapath1, datapath2, savepath, clear):
    imgs1 = load_images_from_folder(datapath1, transform=v2.ToTensor())
    imgs2 = load_images_from_folder(datapath2, transform=v2.ToTensor())
    img_count = 0
    if not path.exists(savepath): os.makedirs(savepath)
    else: img_count = delete_imgs(savepath, clear)

    for i in range(min(len(imgs1), len(imgs2))):
        img = overlay_imgs(imgs1[i], imgs2[i])
        img = v2.ToPILImage()(img).convert("RGB")
        img.save(path.join(savepath, f"img{img_count + i}.png"))

def rgb_transform(image):
    r, g, b, a = image.split()
    img = Image.merge("RGB", (r, g, b))
    transform = v2.ColorJitter(0.02, 0.02, 0.04, 0.02)
    img = transform(img)
    img.putalpha(a)
    return img

'''
transforms used:
all: gaussianblur(mix of 1 and 3), randomhorizontalflip
gem: colorjitter(0.1, 0.1, 0.1, 0.5)
coin: randomrotation(45), colorjitter(0.02, 0.02, 0.04, 0), randomverticalflip
exit, human, key, locked, shield: colorjitter(0.02, 0.02, 0.04, 0.05)
floor, lava: colorjitter(0.02, 0.02, 0.04, 0), randomadjustsharpness(0.1), randomverticalflip
boots, ghost, box: colorjitter(0.02, 0.02, 0.04, 0)
wall: colorjitter(0.02, 0.02, 0.04, 0.05), randomverticalflip

overlays used (with floor): 
boots, box, coin, exit, gem, ghost, human, key, locked, shield
'''
