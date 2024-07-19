from mmdet.apis import init_detector, inference_detector, inference_detector_batch
import cv2
import os
import argparse
import numpy as np

def inference_one_image(model, image, bound):
    results = inference_detector(model, image)
    pre_class_list = results.pre_class_ids
    pre_class_scores = results.pre_class_scores
    pre_class = pre_class_list[0]
    img = cv2.imread(image)
    # 按得分降序输出类别与得分
    for i in range(len(pre_class_list)):
        print(str(class_list[pre_class_list[i]])+" "+str(pre_class_scores[i]))
    
    if class_list[pre_class] in detection_list:
        pre_detected_bbox = results.pred_instances.bboxes
        pre_detected_scores = results.pred_instances.scores
        pre_detected_labels = results.pred_instances.labels
        for index, bbox in enumerate(pre_detected_bbox):
            if pre_detected_scores[index] >= bound:
                cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                cv2.putText(img, f'{category_list[pre_detected_labels[index]]}: {pre_detected_scores[index]:.2f}',
                            (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            palette_list[pre_detected_labels[index] - 5], 2)
    elif class_list[pre_class] in segmentation_list:
        pre_detected_scores = results.pred_instances.scores
        pre_segmentation = results.pred_instances.masks
        for index, mask in enumerate(pre_segmentation):
            if pre_detected_scores[index] >= bound:
                mask_with_alpha = np.zeros_like(img, dtype=np.uint8)
                mask = mask.cpu().numpy()
                mask_with_alpha[..., 2] = (mask * 255).astype(np.uint8)
                img = cv2.addWeighted(img, 0.8, mask_with_alpha, 0.2, 0)
    return img, pre_class


def inference_images(model, images, bound, batchsize:int):
    if batchsize == 1:
        results = inference_detector(model, images)
    else:
        results = inference_detector_batch(model, images, batchsize)
    images_class = list()
    images_vis = list()
    for result, image in zip(results, images):  # 遍历每个图像的检测结果
        pre_class = result.pre_class_ids[0]  # 整个图像的类别
        img = cv2.imread(image)
        images_class.append(pre_class)
        if class_list[pre_class] in detection_list:  # 需要标出检测框的
            pre_detected_bbox = result.pred_instances.bboxes  # 预测的检测框
            pre_detected_scores = result.pred_instances.scores  # 分数
            pre_detected_labels = result.pred_instances.labels  # 物体标签
            for index, bbox in enumerate(pre_detected_bbox):
                if pre_detected_scores[index] >= bound:
                    cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                                  palette_list[pre_detected_labels[index] - 5], 2)
                    cv2.putText(img, f'{category_list[pre_detected_labels[index]]}: {pre_detected_scores[index]:.2f}',
                                (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                palette_list[pre_detected_labels[index] - 5], 2)
        elif class_list[pre_class] in segmentation_list:  # 需要语义分割的
            pre_detected_scores = result.pred_instances.scores  # 分数
            pre_segmentation = result.pred_instances.masks  # 语义分割结果
            for index, mask in enumerate(pre_segmentation):
                if pre_detected_scores[index] >= bound:
                    mask_with_alpha = np.zeros_like(img, dtype=np.uint8)
                    mask = mask.cpu().numpy()
                    mask_with_alpha[..., 2] = (mask * 255).astype(np.uint8)
                    img = cv2.addWeighted(img, 0.8, mask_with_alpha, 0.2, 0)

        images_vis.append(img)

    return images_class, images_vis


def main():
    parser = argparse.ArgumentParser(description="图像预测")

    # 添加命令行参数
    parser.add_argument("--config", type=str, help="输入模型配置文件路径",
                        default="D:/GIE/mmdetection/configs/swin/mask-rcnn_swin-t-p4-w7_fpn_1x_coco.py")
    parser.add_argument("--checkpoint", type=str, help="输入权重文件路径",
                        default="D:\\GIE\mmdetection\\work_dirs\\mask-rcnn_swin-t-p4-w7_fpn_1x_coco\\epoch_301.pth")
    parser.add_argument("--image", type=str, help="输入图像路径", default="D:/GIE/test/train2017")
    parser.add_argument("--bound", type=float, help="置信度阈值", default=0.45)
    parser.add_argument("--output", type=str, help="图像输出路径", default='output')
    parser.add_argument("--batchsize", type=int, help="推理批处理大小，为1时不调用批处理", default=4)

    args = parser.parse_args()
    bound = args.bound
    config_file = args.config
    checkpoint_file = args.checkpoint
    output = args.output
    image_input = args.image
    batch_size = args.batchsize

    path = os.getcwd()  # 当前文件路径
    if os.path.isabs(args.output):
        output = os.path.relpath(output, path)  # 相对输出路径

    model = init_detector(config_file, checkpoint_file, device='cuda:0')  # 初始化模型

    global category_list
    global palette_list
    global class_list
    global detection_list
    global segmentation_list

    category_list = list(model.dataset_meta['classes'])# 获取小类别列表
    palette_list = model.dataset_meta['palette']# 获取颜色列表
    class_list = list(model.dataset_meta['base_classes'])# 获取大类别列表
    detection_list = list(model.dataset_meta['detection_classes'])# 获取需检测的类别列表
    segmentation_list = list(model.dataset_meta['segmentation_classes'])# 获取需分割的类别列表

    class_list.append('all')
    # 矩阵尺寸
    matrix_lenth = len(class_list)
    class_compute = np.zeros((matrix_lenth, 1))


    if os.path.isdir(image_input):
        image_list = os.listdir(image_input)
        imgs = list()
        for image in image_list:
            if image.endswith('jpg') or image.endswith('png') or image.endswith('bmp') or image.endswith('jpeg'):
                image = os.path.join(image_input, image)
                imgs.append(image)

        results = inference_images(model, imgs, bound, batch_size)

        for classid, image_vis, img_path in zip(results[0], results[1], imgs):
            class_compute[classid, 0] += 1  # 该类别数据量+1
            class_compute[matrix_lenth-1, 0] += 1  # 所有数据量+1
            output_path = os.path.join(path, output)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            save_name = os.path.basename(img_path)
            error_path = output_path + '/' + class_list[classid]
            if os.path.isdir(error_path):
                cv2.imwrite(os.path.join(error_path, save_name), image_vis)
            else:
                os.mkdir(error_path)
                cv2.imwrite(os.path.join(error_path, save_name), image_vis)

        for i, j in zip(class_list, class_compute):  # 输出预测值中每个类别的数量
            print(i + ':' + str(j))
    else:
        img = inference_one_image(model, image_input, bound)
        output_path = os.path.join(path, output)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        save_name = os.path.basename(image_input)
        cv2.imwrite(os.path.join(output_path, save_name), img)


if __name__ == '__main__':
    main()