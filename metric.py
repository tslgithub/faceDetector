import os
import os,cv2,sys
from face_detector import preProcess,inference,postProcess,fixResult,CONFIG,loadNet
from script import runTest,detector

def calculate_f1_precision_recall(true_positives, false_positives, false_negatives):
    precision   = true_positives / (true_positives + false_positives + 0.0001)
    recall      = true_positives / (true_positives + false_negatives + 0.0001)
    f1_score    = 2 * (precision * recall) / (precision + recall + 0.0001)
    return f1_score, precision, recall

def get_iou(box_a, box_b):
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    boxBArea = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-10)

    return iou

def metric_calculate(pred_boxes, gt_boxes, threshold=0.5):
    true_positives  = 0
    false_positives = 0
    false_negatives = 0

    for pred_box in pred_boxes:
        max_iou = -1
        for gt_box in gt_boxes:
            iou = get_iou(pred_box, gt_box)
            if iou > max_iou:
                max_iou = iou

        if max_iou >= threshold:
            true_positives += 1
        else:
            false_positives += 1
    false_negatives         = len(gt_boxes) - true_positives  # 1
    f1, precision, recall   = calculate_f1_precision_recall(true_positives, false_positives, false_negatives)

    return f1, precision, recall

def loadImagePath(image_folder):
    image_dict = {}
    for folder in os.listdir(image_folder):
        for file in os.listdir(os.path.join(image_folder,folder)):
            path = os.path.join(folder,file)
            image_dict[path]=[]
    return  image_dict

def loadLabel(label_folder):
    annotation_dict = {}
    with open(label_folder,"r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip()
            if line.endswith(".jpg"):
                annotation_dict[line.rstrip()] = []
                name = line.rstrip()
            elif len(list(line.split(" "))) > 4:
                box = list(map(int,line.split(" ")[:4]))
                annotation_dict[name].append(box)
    return annotation_dict

def merge(image_dict,annotation_dict):
    for item in image_dict:
        image_dict[item] = annotation_dict[item]
    return image_dict


def prediction(image_folder):
    args = CONFIG()
    net,device,cfg = loadNet(args)
    result_dict = {}
    for folder in os.listdir(image_folder):
        for file in os.listdir(  os.path.join(image_folder,folder)  ):
            img_raw = cv2.imread( os.path.join(os.path.join(image_folder,folder),file) )
            item = "/".join(os.path.join(os.path.join(image_folder,folder),file).split("/")[2:])
            boxes = detector(img_raw,device,net,cfg,args,file)
            result_dict[item] = boxes
    return result_dict

def getMetric(gt_dict,prection_dict):
    f1_total=0
    precision_total=0
    recall_total = 0
    num=0
    for msg in gt_dict:
        predictions  = prection_dict[msg.key]
        ground_truths = gt_dict[msg.key]
        f1, precision, recall = metric_calculate(predictions, ground_truths)
        f1_total+=f1
        precision_total+=precision
        recall_total+=recall
        num+=1
    return f1_total/(num*1.0), precision_total/(num*1.0), recall_total/(num*1.0)

def main():
    data_root       = "WIDER_val"
    image_folder    = os.path.join(data_root,"images")
    label_folder    = os.path.join(data_root,"annotation/wider_face_val_bbx_gt.txt")
    image_dict      = loadImagePath(image_folder)
    annotation_dict = loadLabel(label_folder)
    gt_dict         = merge(image_dict,annotation_dict)
    prection_dict   = prediction(image_folder)
    f1, precision, recall = getMetric(gt_dict,prection_dict)

    # 示例数据（可替换为你的实际数据）
    # predictions   = [[100, 100, 200, 200], [300, 300, 400, 400]]  # 检测结果的矩形框坐标 [x1, y1, x2, y2]
    # ground_truths = [[100, 100, 200, 200], [200, 200, 300, 300]]  # 真实矩形框坐标 [x1, y1, x2, y2]
    #
    # f1, precision, recall = metric_calculate(predictions, ground_truths)

    print("F1 Score:  {:.4f}".format(f1))
    print("Precision: {:.4f}".format(precision))
    print("Recall:    {:.4f}".format(recall))

if __name__ == "__main__":
    main()
