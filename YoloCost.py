import torch
from torch.autograd import Function
import numpy as np


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
 
    interArea = (xB*100 - xA*100 ) * (yB*100 - yA*100)
 
    boxAArea = (boxA[2]*100 - boxA[0]*100 ) * (boxA[3]*100 - boxA[1]*100)
    boxBArea = (boxB[2]*100 - boxB[0]*100 ) * (boxB[3]*100 - boxB[1]*100)
 
    iou = interArea / float(boxAArea + boxBArea - interArea)
 
    return iou


class Criterion(Function):

    def __init__(self, S, B, l_coord, l_nobj):
        super(Criterion, self).__init__()
        self.S = S # Number of Cell
        self.B = B # Number of Bouning Box
        self.l_coord = l_coord
        self.l_nobj = l_nobj
        print("hello")
    
    def forward(self, pred_out, real_out):
        self.save_for_backward(pred_out, real_out)
        po = torch.LongTensor([2]).float()
        sum = torch.sum
        pow = torch.pow
        sqr = torch.sqrt
        rt = real_out # Real Tensor
        pt = pred_out # Predicted Tensor
        numObj = rt.size()[0]
        interval = np.linspace(0, 1, self.S + 1)
        cost = torch.FloatTensor([0])
        objIn = np.zeros((self.S,self.S))
        for index in range(numObj):
            cls = rt[index,0]
            x = rt[index,1]
            y = rt[index,2]
            w = rt[index,3]
            h = rt[index,4]
            # Original Ground Truth
            box1 = (x-(w/2), y-(h/2), x+(w/2), h+(h/2))
            # Select cell ???
            colS = self.indices(interval, lambda q: q > x)[0]-1
            rowS = self.indices(interval, lambda q: q > y)[0]-1
            objIn[rowS, colS] = 1
            # Select BBox
            IOU = np.ndarray(shape=(1,self.B))
            for ind in range(self.B):
                px = pt[0, 0 + (5*ind),rowS, colS]
                py = pt[0, 1 + (5*ind),rowS, colS]
                pw = pt[0, 2 + (5*ind),rowS, colS]
                ph = pt[0, 3 + (5*ind),rowS, colS]
                box2 = (px - (pw/2), py - (ph/2), px + (pw/2), py +(ph/2))

                IOU[0,ind] = bb_intersection_over_union(box1, box2)
            # Select Best BBoc
            sel = IOU.argmax()
            x_hat = pt[0, 0 + (5*sel),rowS, colS]
            y_hat = pt[0, 1 + (5*sel),rowS, colS]
            w_hat = pt[0, 2 + (5*sel),rowS, colS]
            h_hat = pt[0, 3 + (5*sel),rowS, colS]
            c_hat_obj = pt[0, 4 + (5*sel),rowS, colS]
            if sel == 0:
                c_hat_noobj = pt[0, 4 + (5),rowS, colS]
            else:
                c_hat_noobj = pt[0, 4 + (0),rowS, colS]
            p = torch.zeros(1,20).view(-1)
            p[int(cls)] = 1
            p_hat = pt[0,10:,rowS, colS]
            cost1 = self.l_coord*(pow(x-x_hat, po)) + self.l_coord*(pow(y-y_hat, po))
            cost2 = pow(1-c_hat_obj,po) + self.l_nobj*pow(0-c_hat_noobj,po)
            cost3 = self.l_coord*(pow(sqr(torch.FloatTensor([w]))-sqr(torch.FloatTensor([w_hat])),po)) + self.l_coord*(pow(sqr(torch.FloatTensor([h]))-sqr(torch.FloatTensor([h_hat])),po))
            cost4 = torch.sum(pow((p-p_hat), 2))
            cost += (cost1 + cost2 + cost3 + cost4)
            del cost1, cost2, cost3, cost4, p
        for index1 in range(self.S):
            for index2 in range(self.S):
                if objIn[index1, index2] == 0:
                    cost1 = self.l_nobj*pow(0-pt[0, 4,index1, index2],po)
                    cost2 = self.l_nobj*pow(0-pt[0, 9,index1, index2],po)
                    cost += (cost1 + cost2)
        return cost
    
    def backward(self, grad_cost):
        pt, rt = self.saved_tensors
        #pred_out is FloatTensor not a variable
        grad_pred_out = torch.zeros(pt.size())
        po = torch.FloatTensor([0.5])
        sum = torch.sum
        pow = torch.pow
        numObj = rt.size()[0]
        interval = np.linspace(0, 1, self.S + 1)
        objIn = np.zeros((self.S,self.S))
        for index in range(numObj):
            cls = rt[index,0]
            x = rt[index,1]
            y = rt[index,2]
            w = rt[index,3]
            h = rt[index,4]
            # Original Ground Truth
            box1 = (x-(w/2), y-(h/2), x+(w/2), h+(h/2))
            # Select cell
            colS = self.indices(interval, lambda q: q > x)[0]-1
            rowS = self.indices(interval, lambda q: q > y)[0]-1
            objIn[rowS, colS] = 1
            # Select BBox
            IOU = np.ndarray(shape=(1,self.B))
            for ind in range(self.B):
                px = pt[0, 0 + (5*ind),rowS, colS]
                py = pt[0, 1 + (5*ind),rowS, colS]
                pw = pt[0, 2 + (5*ind),rowS, colS]
                ph = pt[0, 3 + (5*ind),rowS, colS]
                box2 = (px - (pw/2), py - (ph/2), px + (pw/2), py +(ph/2))

                IOU[0,ind] = bb_intersection_over_union(box1, box2)
            # Select Best BBox
            sel = IOU.argmax()
            #print(x,y,w,h, box1, IOU)
            x_hat = pt[0, 0 + (5*sel),rowS, colS]
            y_hat = pt[0, 1 + (5*sel),rowS, colS]
            w_hat = pt[0, 2 + (5*sel),rowS, colS]
            h_hat = pt[0, 3 + (5*sel),rowS, colS]
            c_hat_obj = pt[0, 4 + (5*sel),rowS, colS]
            if sel == 0:
                nonsel = 1
                c_hat_noobj = pt[0, 4 + (5),rowS, colS]
            else:
                nonsel = 0
                c_hat_noobj = pt[0, 4 + (0),rowS, colS]
            p = torch.zeros(1,20).view(-1)
            p[int(cls)] = 1
            p_hat = pt[0,10:,rowS, colS]
            grad_pred_out[0,0 + (5*sel), rowS, colS] += -2*self.l_coord*(x - x_hat)
            grad_pred_out[0,1 + (5*sel), rowS, colS] += -2*self.l_coord*(y - y_hat)
            grad_pred_out[0,2 + (5*sel), rowS, colS] += ((-self.l_coord/pow(w_hat,po))*(pow(w,po) - pow(w_hat,po)))[0]
            grad_pred_out[0,3 + (5*sel), rowS, colS] += ((-self.l_coord/pow(h_hat,po))*(pow(h,po) - pow(h_hat,po)))[0]
            grad_pred_out[0,4 + (5*sel), rowS, colS] += -2*(1-c_hat_obj)
            grad_pred_out[0,4 + (5*nonsel), rowS, colS] += -2*self.l_nobj*(0-c_hat_noobj)
            grad_pred_out[0,10:, rowS, colS] += -2*(p - p_hat)
        
        for index1 in range(self.S):
            for index2 in range(self.S):
                if objIn[index1, index2] == 0:
                    grad_pred_out[0, 4, index1, index2] += -2*self.l_nobj*(0-pt[0, 4,index1, index2])
                    grad_pred_out[0, 9, index1, index2] += -2*self.l_nobj*(0-pt[0, 9,index1, index2])
        grad_real_out = None
        return grad_pred_out, grad_real_out

            
    def indices(self, a, func):
        return [i for (i, val) in enumerate(a) if func(val)]