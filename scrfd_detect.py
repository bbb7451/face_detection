# scrfd_detect.py
import cv2
import numpy as np
import onnxruntime as ort
import os

def nms(dets, thresh=0.4):
    if len(dets) == 0:
        return []
    x1, y1, x2, y2, scores = dets[:,0], dets[:,1], dets[:,2], dets[:,3], dets[:,4]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep

def distance2bbox(points, distance, max_shape=None):
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = np.clip(x1, 0, max_shape[1]-1)
        y1 = np.clip(y1, 0, max_shape[0]-1)
        x2 = np.clip(x2, 0, max_shape[1]-1)
        y2 = np.clip(y2, 0, max_shape[0]-1)
    return np.stack([x1, y1, x2, y2], axis=-1)

def distance2kps(points, distance, max_shape=None):
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:,0] + distance[:,i]
        py = points[:,1] + distance[:,i+1]
        if max_shape is not None:
            px = np.clip(px, 0, max_shape[1]-1)
            py = np.clip(py, 0, max_shape[0]-1)
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)

class SCRFD:
    def __init__(self, model_path):
        assert os.path.exists(model_path), f"모델 파일이 없습니다: {model_path}"
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = 640
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.fmc = len(self.output_names)//3 if len(self.output_names) in [9,15] else len(self.output_names)//2
        self._feat_stride_fpn = [8,16,32] if self.fmc==3 else [8,16,32,64,128]
        self.use_kps = len(self.output_names) in [9,15]

    def detect(self, img, thresh=0.3):
        h0, w0, _ = img.shape
        det_img = cv2.resize(img, (self.input_size, self.input_size))
        blob = cv2.dnn.blobFromImage(det_img, 1/128, (self.input_size,self.input_size), (127.5,127.5,127.5), swapRB=True)
        outs = self.session.run(self.output_names, {self.input_name: blob})
        
        scores_list, bboxes_list, kps_list = [], [], []
        for idx, stride in enumerate(self._feat_stride_fpn):
            scores = outs[idx].reshape(-1)
            bbox_preds = outs[idx+self.fmc].reshape(-1,4)*stride
            if self.use_kps:
                kps_preds = outs[idx+self.fmc*2].reshape(-1,10)*stride
            h, w = self.input_size//stride, self.input_size//stride
            xv, yv = np.meshgrid(np.arange(w), np.arange(h))
            anchor_centers = np.stack([xv, yv], axis=-1).reshape(-1,2)*stride
            pos_inds = np.where(scores>thresh)[0]
            if len(pos_inds)==0:
                continue
            scores_list.append(scores[pos_inds])
            bboxes_list.append(distance2bbox(anchor_centers[pos_inds], bbox_preds[pos_inds], max_shape=(h0,w0)))
            if self.use_kps:
                kps_list.append(distance2kps(anchor_centers[pos_inds], kps_preds[pos_inds], max_shape=(h0,w0)))
        
        if len(bboxes_list)==0:
            return np.array([]), None
        dets = np.vstack([np.hstack([b,s[:,None]]) for b,s in zip(bboxes_list, scores_list)])
        keep = nms(dets, thresh=thresh)
        dets = dets[keep]
        kpss = np.vstack(kps_list)[keep] if self.use_kps else None
        return dets, kpss

if __name__=='__main__':
    model_path = './scrfd_2.5g_bnkps.onnx'
    img_path = 'testImage1.jpg'
    img = cv2.imread(img_path)
    if img is None:
        print("이미지를 찾을 수 없습니다.")
        exit(1)
    
    detector = SCRFD(model_path)
    faces, kpss = detector.detect(img, thresh=0.3)

    if len(faces)==0:
        print("얼굴을 찾지 못했습니다.")
    else:
        for i, box in enumerate(faces):
            x1,y1,x2,y2,score = box.astype(int)
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(img,f'{score:.2f}',(x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
            if kpss is not None:
                kps = kpss[i].reshape(-1,2)
                for (kx,ky) in kps:
                    cv2.circle(img,(int(kx),int(ky)),2,(0,0,255),-1)
        cv2.imwrite('detected_testImage1_kps.jpg', img)
        print("얼굴+랜드마크 감지 완료: detected_testImage1_kps.jpg")
