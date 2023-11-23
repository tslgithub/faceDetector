import os,cv2,sys
from face_detector import preProcess,inference,postProcess,fixResult,CONFIG,loadNet

def detector(img_raw,device,net,cfg,args,file):
    img,scale,im_height, im_width   = preProcess(img_raw,device)
    conf,landms,priors,loc          = inference( net,img,im_height, im_width,cfg)
    dets                            = postProcess(args,scale,conf,landms,priors,cfg,img,device,loc)
    boxes = fixResult(args,dets,img_raw,file,im_height, im_width)
    return boxes

def runTest(dataset,data,threshold):
    args  = CONFIG()
    if(threshold>0):
        args.vis_thres = threshold

    net,device,cfg = loadNet(args)
    print("\n\nstart detector")
    print("--"*50)
    if(dataset is not None):
        for file in os.listdir(dataset):
            if(file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg")   ):
                image_path  = os.path.join(dataset,file)
                img_raw     = cv2.imread(image_path, cv2.IMREAD_COLOR)
                detector(img_raw,device,net,cfg,args,file)

    if(data is not  None):
        image_path  = data
        img_raw     = cv2.imread(image_path, cv2.IMREAD_COLOR)
        detector(img_raw,device,net,cfg,args,data)

def main():
    print("sys -> {}".format(sys.argv))
    argv = sys.argv
    assert len(argv) < 4 ,("{} must has at lest one paramter, no more than 3 \n"
                           "for example `python3 script 1.jpg` ").format(argv)

    threshold=-1
    if(len(argv)==3):
        threshold = float(argv[2])
        assert  threshold> 0 and threshold<1

    data = None
    dataset = None
    if(  os.path.isdir(str(argv[1]) ) ):
        dataset = str(argv[1])
    elif( os.path.isfile(str(argv[1]) ) ):
        data = str(argv[1])
    else:
        print("{} is not file or folder".format(argv[1]))

    if((data is not None) or (dataset is not None) ):
        runTest(dataset,data,threshold)

if __name__ == '__main__':
    main()