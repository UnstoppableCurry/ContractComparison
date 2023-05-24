import os
from rec.RecModel import RecModel
import torch
from addict import Dict as AttrDict
import cv2
import numpy as np
import math
import time
import os
from det.DetModel import DetModel
import torch
from addict import Dict as AttrDict
import cv2
import numpy as np
import math
import time
import pyclipper
from shapely.geometry import Polygon


class DBPostProcess():
    def __init__(self, thresh=0.3, box_thresh=0.4, max_candidates=1000, unclip_ratio=2):
        self.min_size = 3
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio

    def __call__(self, pred, h_w_list, is_output_polygon=False):
        '''
        batch: (image, polygons, ignore_tags
        h_w_list: 包含[h,w]的数组
        pred:
            binary: text region segmentation map, with shape (N, 1,H, W)
        '''
        pred = pred[:, 0, :, :]
        segmentation = self.binarize(pred)
        boxes_batch = []
        scores_batch = []
        for batch_index in range(pred.shape[0]):
            height, width = h_w_list[batch_index]
            boxes, scores = self.post_p(pred[batch_index], segmentation[batch_index], width, height,
                                        is_output_polygon=is_output_polygon)
            boxes_batch.append(boxes)
            scores_batch.append(scores)
        return boxes_batch, scores_batch

    def binarize(self, pred):
        return pred > self.thresh

    def post_p(self, pred, bitmap, dest_width, dest_height, is_output_polygon=False):
        '''
        _bitmap: single map with shape (H, W),
            whose values are binarized as {0, 1}
        '''
        height, width = pred.shape
        boxes = []
        new_scores = []
        bitmap = bitmap.cpu().numpy()
        if cv2.__version__.startswith('3'):
            _, contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if cv2.__version__.startswith('4'):
            contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours[:self.max_candidates]:
            epsilon = 0.005 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = approx.reshape((-1, 2))
            if points.shape[0] < 4:
                continue
            score = self.box_score_fast(pred, contour.squeeze(1))
            if self.box_thresh > score:
                continue
            if points.shape[0] > 2:
                box = self.unclip(points, unclip_ratio=self.unclip_ratio)
                if len(box) > 1:
                    continue
            else:
                continue
            four_point_box, sside = self.get_mini_boxes(box.reshape((-1, 1, 2)))
            if sside < self.min_size + 2:
                continue
            if not isinstance(dest_width, int):
                dest_width = dest_width.item()
                dest_height = dest_height.item()
            if not is_output_polygon:
                box = np.array(four_point_box)
            else:
                box = box.reshape(-1, 2)
            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box)
            new_scores.append(score)
        return boxes, new_scores

    def unclip(self, box, unclip_ratio=1.5):
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [points[index_1], points[index_2], points[index_3], points[index_4]]
        return box, min(bounding_box[1])

    def box_score_fast(self, bitmap, _box):
        bitmap = bitmap.detach().cpu().numpy()
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]


def narrow(image, expected_size=(224, 224)):
    ih, iw = image.shape[0:2]
    ew, eh = expected_size
    # scale = eh / ih
    scale = min((eh / ih), (ew / iw))
    # scale = eh / max(iw,ih)
    nh = int(ih * scale)
    nw = int(iw * scale)
    image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
    top = 0
    bottom = eh - nh
    left = 0
    right = ew - nw
    new_img = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return new_img


def draw_bbox(img_path, result, color=(0, 0, 255), thickness=2):
    import cv2
    if isinstance(img_path, str):
        img_path = cv2.imread(img_path)
        # img_path = cv2.cvtColor(img_path, cv2.COLOR_BGR2RGB)
    img_path = img_path.copy()
    for point in result:
        point = point.astype(int)
        cv2.polylines(img_path, [point], True, color, thickness)
    return img_path


def img_nchw_det(img):
    mean = 0.5
    std = 0.5
    resize_ratio = min((640 / img.shape[0]), (640 / img.shape[1]))
    img = cv2.resize(img, (0, 0), fx=resize_ratio, fy=resize_ratio, interpolation=cv2.INTER_LINEAR)
    h, w = img.shape[:2]
    if h == 640:
        w = (math.ceil(w / 32) + 1) * 32
    elif w == 640:
        h = (math.ceil(h / 32) + 1) * 32
    img1 = narrow(img, (w, h))

    img_data = (img1.astype(np.float32) / 255 - mean) / std
    img_np = img_data.transpose(2, 0, 1)
    img_np = np.expand_dims(img_np, 0)
    return img1, img_np


def img_nchw_rec(img):
    mean = 0.5
    std = 0.5
    resize_ratio = 48 / img.shape[0]
    img = cv2.resize(img, (0, 0), fx=resize_ratio, fy=resize_ratio, interpolation=cv2.INTER_LINEAR)
    # img = cv2.resize(img,(img.shape[1],32))

    W = math.ceil(img.shape[1] / 32) + 1
    img = narrow_224_32(img, expected_size=(W * 32, 48))
    img_data = (img.astype(np.float32) / 255 - mean) / std
    img_np = img_data.transpose(2, 0, 1)
    img_np = np.expand_dims(img_np, 0)
    return img_np


def get_rotate_crop_image(img, points):
    '''
    img_height, img_width = img.shape[0:2]
    left = int(np.min(points[:, 0]))
    right = int(np.max(points[:, 0]))
    top = int(np.min(points[:, 1]))
    bottom = int(np.max(points[:, 1]))
    img_crop = img[top:bottom, left:right, :].copy()
    points[:, 0] = points[:, 0] - left
    points[:, 1] = points[:, 1] - top
    '''
    assert len(points) == 4, "shape of points must be 4*2"
    img_crop_width = int(
        max(
            np.linalg.norm(points[0] - points[1]),
            np.linalg.norm(points[2] - points[3])))
    img_crop_height = int(
        max(
            np.linalg.norm(points[0] - points[3]),
            np.linalg.norm(points[1] - points[2])))
    pts_std = np.float32([[0, 0], [img_crop_width, 0],
                          [img_crop_width, img_crop_height],
                          [0, img_crop_height]])
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img,
        M, (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC)
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img


class CTCLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character):
        # character (str): set of the possible characters.
        dict_character = []
        with open(character, "rb") as fin:
            lines = fin.readlines()
            for line in lines:
                line = line.decode('utf-8').strip("\n").strip("\r\n")
                dict_character += list(line)
        # dict_character = list(character)

        self.dict = {}
        for i, char in enumerate(dict_character):
            # NOTE: 0 is reserved for 'blank' token required by CTCLoss
            self.dict[char] = i + 1
        # TODO replace ‘ ’ with special symbol
        self.character = ['[blank]'] + dict_character + [' ']  # dummy '[blank]' token for CTCLoss (index 0)

    def encode(self, text, batch_max_length=None):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
        output:
            text: concatenated text index for CTCLoss.
                    [sum(text_lengths)] = [text_index_0 + text_index_1 + ... + text_index_(n - 1)]
            length: length of each text. [batch_size]
        """
        length = [len(s) for s in text]
        # text = ''.join(text)
        # text = [self.dict[char] for char in text]
        d = []
        batch_max_length = max(length)
        for s in text:
            t = [self.dict[char] for char in s]
            t.extend([0] * (batch_max_length - len(s)))
            d.append(t)
        return (torch.tensor(d, dtype=torch.long), torch.tensor(length, dtype=torch.long))

    def decode(self, preds, raw=False):
        """ convert text-index into text-label. """
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)

        result_list = []
        for word, prob in zip(preds_idx, preds_prob):
            if raw:
                result_list.append((''.join([self.character[int(i)] for i in word]), prob))
            else:
                result = []
                conf = []
                for i, index in enumerate(word):
                    if word[i] != 0 and (not (i > 0 and word[i - 1] == word[i])):
                        # if prob[i] < 0.3:           # --------------------------------------------------
                        #     continue
                        result.append(self.character[int(index)])
                        conf.append(prob[i])
                result_list.append((''.join(result), conf))
        return result_list


def narrow_224_32(image, expected_size=(280, 48)):
    ih, iw = image.shape[0:2]
    ew, eh = expected_size
    scale = eh / ih
    nh = int(ih * scale)
    nw = int(iw * scale)
    image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
    top = 0
    bottom = eh - nh - top
    left = 0
    right = ew - nw - left

    new_img = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return new_img


def img_nchw(img):
    mean = 0.5
    std = 0.5
    resize_ratio = 48 / img.shape[0]
    img = cv2.resize(img, (0, 0), fx=resize_ratio, fy=resize_ratio, interpolation=cv2.INTER_LINEAR)
    # img = cv2.resize(img,(img.shape[1],32))

    W = math.ceil(img.shape[1] / 32) + 1
    img = narrow_224_32(img, expected_size=(W * 32, 48))
    img_data = (img.astype(np.float32) / 255 - mean) / std
    img_np = img_data.transpose(2, 0, 1)
    img_np = np.expand_dims(img_np, 0)
    return img_np


def infer_det(image, img0, threshold=0.5):
    det_model, post_proess = get_det_model()
    # img0, img_np_nchw = img_nchw(image)
    # input_for_torch = torch.from_numpy(img_np_nchw)
    out = det_model(image)  # torch model infer
    # post_proess = DBPostProcess()

    box_list, score_list = post_proess(out, [img0.shape[:2]], is_output_polygon=False)
    box_list, score_list = box_list[0], score_list[0]
    if len(box_list) > 0:
        idx = [x.sum() > 0 for x in box_list]
        box_list = [box_list[i] for i, v in enumerate(idx) if v]
        score_list = [score_list[i] for i, v in enumerate(idx) if v]
    else:
        box_list, score_list = [], []
    img1 = draw_bbox(img0, box_list)
    cv2.imshow("draw", img1)
    cv2.waitKey()
    results = []
    index = 0
    for points in box_list:
        if score_list[index] < threshold:
            continue
        result_img = get_rotate_crop_image(img0, points)
        results.append(result_img)
        index += 1
    return results


def get_dict():
    dict_path = r"./weights/ppocr_keys_v1.txt"
    converter = CTCLabelConverter(dict_path)
    return converter


def get_rec_model():
    rec_model_path = "./weights/ppv3_rec.pth"

    rec_config = AttrDict(
        in_channels=3,
        backbone=AttrDict(type='MobileNetV1Enhance', scale=0.5, last_conv_stride=[1, 2], last_pool_type='avg'),
        neck=AttrDict(type='None'),
        head=AttrDict(type='Multi', head_list=AttrDict(
            CTC=AttrDict(Neck=AttrDict(name="svtr", dims=64, depth=2, hidden_dims=120, use_guide=True)),
            # SARHead=AttrDict(enc_dim=512,max_text_length=70)
        ),
                      n_class=6625)
    )

    rec_model = RecModel(rec_config)
    rec_model.load_state_dict(torch.load(rec_model_path))
    rec_model.eval()
    return rec_model


def get_det_model():
    det_model_path = './weights/ppv3_db.pth'
    test_img = "./det_images"

    post_proess = DBPostProcess()

    db_config = AttrDict(
        in_channels=3,
        backbone=AttrDict(type='MobileNetV3', model_name='large', scale=0.5, pretrained=True),
        neck=AttrDict(type='RSEFPN', out_channels=96),
        head=AttrDict(type='DBHead')
    )

    det_model = DetModel(db_config)
    det_model.load_state_dict(torch.load(det_model_path))
    det_model.eval()
    return det_model, post_proess


def imgs2tensors(imgs):
    result_tensor = []
    if not isinstance(imgs, list):
        img0, img_np_nchw = img_nchw_det(imgs)
        input_for_torch = torch.from_numpy(img_np_nchw)
        return input_for_torch, img0
    for img in imgs:
        img_np_nchw = img_nchw_rec(img)
        input_for_torch = torch.from_numpy(img_np_nchw)
        result_tensor.append(input_for_torch)
        # if result_tensor is None:
        #     result_tensor=input_for_torch
        # else:
        #     result_tensor=torch.cat((result_tensor,input_for_torch),0)

    return result_tensor


def infer_rec(img_tensor):
    import re
    pattern = re.compile(r'[\u4e00-\u9fa5]')

    rec_model = get_rec_model()
    converter = get_dict()
    time1 = time.time()
    result = ''
    results = []
    for i in img_tensor:
        feat_2 = rec_model(i).softmax(dim=2)
        feat_2 = feat_2.cpu().data
        txt = converter.decode(feat_2.detach().cpu().numpy())
        result += txt[0][0]\
            .replace(',','').replace('.','').replace('/','').replace(';','').replace('\\','').replace(']','').replace('[','')\
            .replace('，','').replace('。','').replace('、','').replace('；','').replace('’','').replace('】','').replace('【','')\
            .replace('《','').replace('》','').replace('？','').replace('：','').replace('”','').replace('|','').replace('}','').replace('{','') \
            .replace('<','').replace('>','').replace('?','').replace(':','').replace('"','').replace('|','').replace('{','').replace('}','')\
            .replace('!','').replace('@','').replace('#','').replace('$','').replace('%','').replace('^','').replace('&','').replace('*','')\
            .replace('(','').replace(')','').replace('_','').replace('+','').replace('-','').replace('=','')
        results.append(txt[0][0])
    time2 = time.time()
    time3 = time2 - time1

    print(" txt:{}  time:{}".format(result, time3))

    return result, results


def infer_det_rec(path):
    img = cv2.imread(path)
    imgs1, img0 = imgs2tensors(img)
    imgs2 = infer_det(imgs1, img0)
    imgs_tensor = imgs2tensors(imgs2)
    result, results = infer_rec(imgs_tensor)
    return result, results


def get_same_numbers(s1, s2):
    # 创建字典
    d = {}

    # 遍历第一个字符串中的每个字符
    for c in s1:
        d[c] = d.get(c, 0) + 1

    # 遍历第二个字符串中的每个字符
    for c in s2:
        if c in d:
            d[c] = d[c] + 1
    return len(d)


def get_like_score(s1, s2):
    from difflib import SequenceMatcher

    # 计算两个字符串之间的差异
    diff = SequenceMatcher(None, s1, s2).ratio()

    # 输出相似度得分
    return diff


def get_rouge_socre(text1, text2):
    rouge = Rouge()
    result = rouge.get_scores(text1, text2, avg=True, ignore_empty=True)
    print('rouge1: ', result['rouge-1'])
    print('rouge2: ', result['rouge-2'])
    print('rougeL: ', result['rouge-l'])


def get_rouge_socre2(text1, text2):
    import jieba
    rouge = Rouge()
    result = rouge.get_scores([x for x in jieba.cut(text1)], [x for x in jieba.cut(text2)], avg=True, ignore_empty=True)
    print('rouge1: ', result['rouge-1'])
    print('rouge2: ', result['rouge-2'])
    print('rougeL: ', result['rouge-l'])


if __name__ == "__main__":
    from rouge import Rouge

    path1 = 'test/pdf3.png'
    # path2 = 'test/pdf3.png'
    path2 = 'test/3.jpg'
    text1, text1_list = infer_det_rec(path1)
    text2, text2_list = infer_det_rec(path2)
    get_rouge_socre(text1_list, text2_list)
    get_rouge_socre2(text1, text2)
    print(get_same_numbers(text1, text2) * 2 / (len(text2) + len(text1)))
    diff = get_like_score(text1, text2)
    print("相似度得分：", diff)
