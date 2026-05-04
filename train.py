import warnings, os
warnings.filterwarnings('ignore')
from ultralytics import RTDETR
if __name__ == '__main__':
    model = RTDETR('ultralytics/cfg/models/rt-detr/tld-rtdetr.yaml')
    # model.load('') # loading pretrain weights
    model.train(data='dataset/data.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=4, # batchsize 
                workers=4, # Windows
                # device='0,1', 
                # resume='', # last.pt path
                project='runs/train',
                name='exp',
                )
