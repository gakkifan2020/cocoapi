from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import pathlib
pylab.rcParams['figure.figsize'] = (8.0, 10.0)


dataDir='./coco'
dataType='val2017'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)

pic_dir = './coco/images/val2017/'

# initialize COCO api for instance annotations
coco = COCO(annFile)


# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))

# 找到符合'person','dog','skateboard'过滤条件的category_id
catIds = coco.getCatIds(catNms=['person','dog','skateboard'])
# 找出符合category_id过滤条件的image_id
imgIds = coco.getImgIds(catIds=catIds )
# # 找出imgIds中images_id为324158的image_id
# imgIds = coco.getImgIds(imgIds = [323799])
# 遍历imgIds中的images_id
pictures = pathlib.Path(pic_dir)
for path in list(pictures.glob('*.jpg')):
    img_num = str(path)[26:32]
    imgIds = coco.getImgIds(imgIds = [int(img_num)])
    # 加载图片，获取图片的数字矩阵
    img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
    # 显示图片
    I = io.imread(img['coco_url'])
    # load and display instance annotations
    plt.imshow(I); plt.axis('off')
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    coco.showAnns(anns)

    plt.axis('off')
    plt.imshow(I)
    plt.show()