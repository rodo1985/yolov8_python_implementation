from multiprocessing import freeze_support
import os
import shutil
from ultralytics import YOLO
import torch

def main():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # delete directory and content if exists
    if os.path.exists('runs'):
        shutil.rmtree('runs')

    # build a new model from scratch
    # model = YOLO('yolov8n.yaml')                

    # load a pretrained model (recommended for best training results)
    model = YOLO('yolov8n.pt')                
    model.to(device)
    
    # train the model
    _ = model.train(data='datasets/cells.yaml', epochs = 100, batch = 32)  
    
    # evaluate model performance on the validation set
    _ = model.val(data='datasets/cells.yaml')                       

    # export the model to ONNX format
    # success = model.export(format='onnx')       

if __name__ == '__main__':
    freeze_support()
    main()