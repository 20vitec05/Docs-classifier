import fitz, os
from PIL import Image


if not os.path.exists('ImageDataset'):
     os.mkdir('ImageDataset')

for i in os.listdir('Dataset'):
    if not os.path.exists(f'ImageDataset/{i}'):
            os.mkdir(f'ImageDataset/{i}')
    for j in os.listdir(f'Dataset/{i}'):
        if not os.path.exists(f'ImageDataset/{i}/{j}'):
            os.mkdir(f'ImageDataset/{i}/{j}')

for i in os.listdir('Dataset'):
    for j in os.listdir(f'Dataset/{i}'):
        for k in os.listdir(f'Dataset/{i}/{j}'):
            data = fitz.open(f'Dataset/{i}/{j}/{k}')
            for l in range(len(data)):
                page = data.load_page(l)
                pix = page.get_pixmap()
                image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                image.save(f'ImageDataset/{i}/{j}/{k}-{l}.jpg', 'JPEG')