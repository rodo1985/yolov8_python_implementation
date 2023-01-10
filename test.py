from ultralytics import YOLO
import torch
import cv2

def main():

    # image_path
    image_path = r'datasets\cells\valid\images\BloodImage_00000_jpg.rf.3aa7a653c80726cbb25447cb697ad7a4.jpg'

    # select device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # set threshold
    threshold = 0.1

    # load a pretrained model (recommended for best training results)
    model = YOLO('runs/detect/train/weights/best.pt')                
    model.to(device)

    # predict on an image
    results = model.predict(image_path)   

    # read image
    image = cv2.imread(image_path)

    # loop through results
    for result in results[0].cpu().numpy():
        x1, y1, x2, y2, score, c = result

        # if score is greater than threshold
        if score >= threshold:

            # draw bounding box
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (int(c == 0) * 255, int(c == 1) * 255, int(c ==2) * 255), 2)
    
    # plot image
    cv2.imshow('image',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # print results
    print(results)
    

if __name__ == '__main__':
    main()