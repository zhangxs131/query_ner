import torch
import numpy as np

class BaffineEntityScore(object):
    def __init__(self, id2label):
        self.id2label=id2label
        self.reset()

    def reset(self):
        self.X=0
        self.Y=0
        self.Z=0
        self.class_label={k:[0,0,0] for k in self.id2label}

    def compute(self, X, Y, Z):
        if Y==0 or Z ==0:
            return 0.0,0.0,0.0

        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        return recall, precision, f1

    def result(self):
        class_info = {}

        for type_, count in self.class_label.items():
            recall, precision, f1 = self.compute(count[0], count[1], count[2])
            class_info[self.id2label[type_]] = {"acc": round(precision, 4), 'recall': round(recall, 4), 'f1': round(f1, 4)}

        recall, precision, f1 = self.compute(self.X,self.Y,self.Z)
        return {'acc': precision, 'recall': recall, 'f1': f1}, class_info

    def update(self, y_true, y_pred):
        '''
        labels_paths: [[],[],[],....]
        pred_paths: [[],[],[],.....]

        :param label_paths:
        :param pred_paths:
        :return:
        '''


        pred = []
        true = []
        p_result={}
        t_result={}

        for b, start, end in zip(*np.where(y_pred > 0)):
            l=y_pred[b,start,end]
            if l not in p_result:
                p_result[l]=[(b,start,end)]
            else:
                p_result[l].append((b,start,end))

            pred.append((b, l, start, end))
        for b, start, end in zip(*np.where(y_true > 0)):
            l=y_true[b,start,end]
            if l not in t_result:
                t_result[l]=[(b,start,end)]
            else:
                t_result[l].append((b,start,end))
            true.append((b, l, start, end))

        R = set(pred)
        T = set(true)
        self.X += len(R & T)
        self.Y += len(R)
        self.Z += len(T)

        for k,v in p_result.items():
            R = set(v)
            T = set(t_result[k]) if k in t_result else set()
            self.class_label[k][0] += len(R & T)
            self.class_label[k][1] += len(R)
            self.class_label[k][2] += len(T)

