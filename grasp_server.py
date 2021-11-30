import sys
import os
import argparse
import Pyro4
import random
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from shutil import rmtree
from pathlib import Path
from ggcnn_torch import GGCNNTorch
from utils.dataset_processing.grasp import detect_grasps

MODEL_PATH = Path.home().joinpath('Project/gmdata/datasets/models/gg2/shahao_model')#('/Project/gmdata/datasets/models/gg/shahao_data')#('Project/shahao_ggcnn/shahao_data')
#MODEL_PATH = GMDATA_PATH.joinpath('')
#TEST_PATH = GMDATA_PATH.joinpath('datasets/test/test_poses')
#TEST_OUTPUT = GMDATA_PATH.joinpath('ggtest')


class Grasp2D(object):
    """
    2D夹爪类型，夹爪投影到深度图像上的坐标.
    这里使用的都是图像坐标和numpy数组的轴顺序相反
    """

    def __init__(self, center, angle, depth, width=0.0, z_depth=0.0, quality=None, coll=None, gh=None):
        """ 一个带斜向因子z的2d抓取, 这里需要假定p1的深度比较大
        center : 夹爪中心坐标，像素坐标表示
        angle : 抓取方向和相机x坐标的夹角, 由深度较小的p0指向深度较大的p1, (-pi, pi)
        depth : 夹爪中心点的深度
        width : 夹爪的宽度像素坐标
        z_depth: 抓取端点到抓取中心点z轴的距离,单位为m而非像素
        quality: 抓取质量
        coll: 抓取是否碰撞
        """
        self.center = center
        self.angle = angle
        self.depth = depth
        self.width_px = width
        self.z_depth = z_depth
        self.quality = quality
        self.coll = coll
        self.gh = gh

    @property
    def norm_angle(self):
        """ 归一化到-pi/2到pi/2的角度 """
        a = self.angle
        while a >= np.pi/2:
            a = a-np.pi
        while a < -np.pi/2:
            a = a+np.pi
        return a

    @property
    def axis(self):
        """ Returns the grasp axis. """
        return np.array([np.cos(self.angle), np.sin(self.angle)])

    @property
    def endpoints(self):
        """ Returns the grasp endpoints """
        p0 = self.center - (float(self.width_px) / 2) * self.axis
        p1 = self.center + (float(self.width_px) / 2) * self.axis
        p0 = p0.astype(np.int)
        p1 = p1.astype(np.int)
        return p0, p1

    @classmethod
    def from_jaq(cls, jaq_string):
        jaq_string = jaq_string.strip()
        x, y, theta, w, h = [float(v) for v in jaq_string[:-1].split(';')]
        return cls(np.array([x, y]), theta/180.0*np.pi, 0, w, gh=h)


def plot_output(depth_img, grasp_q_img, grasp_angle_img, no_grasps=1, grasp_width_img=None):
    """
    Plot the output of a GG-CNN
    :param rgb_img: RGB Image
    :param depth_img: Depth Image
    :param grasp_q_img: Q output of GG-CNN
    :param grasp_angle_img: Angle output of GG-CNN
    :param no_grasps: Maximum number of grasps to plot
    :param grasp_width_img: (optional) Width output of GG-CNN
    :return:
    """
    gs = detect_grasps(grasp_q_img, grasp_angle_img,
                       width_img=grasp_width_img, no_grasps=1)

    fig = plt.figure(figsize=(10, 10))

    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(depth_img, cmap='gray')
    for g in gs:
        g.plot(ax)
    ax.set_title('Depth')
    ax.axis('off')

    return gs


def get_model(model_path):
    model_path = Path(model_path).resolve()
    max_fn = 0
    max_f = None
    for f in model_path.iterdir():
        fs = f.name.split('_')
        if len(fs) == 4:
            fn = int(fs[1])
            if fn > max_fn:
                max_fn = fn
                max_f = f#这里是想找到最后一次epoch训练的参数结果，也就是使用最新参数
    return max_f


@Pyro4.expose
class Planer(object):
    def __init__(self, model_path):
        self.ggcnn = GGCNNTorch(model_path)
        # self.model_name=model_path.split("/")[-1]#从路径中剪切出模型名并发送给客户端

    def plan(self, image, width):
        random.seed(0)
        np.random.seed(0)
        im = image.copy()
        try_num = 5
        qs = []
        gs = []
        for _ in range(try_num):
            try:
                points_out, ang_out, width_out, depth_out = self.ggcnn.predict(im, 300)
                ggs = detect_grasps(points_out, ang_out, width_img=width_out, no_grasps=1)#这里得到了一系列候选抓取
                if len(ggs) == 0:
                    print("detect_grasps——error")
                    continue
                print("@@@中心和角度",ggs[0].center ,ggs[0].angle)
                g = Grasp2D.from_jaq(ggs[0].to_jacquard(scale=1))
                # g.width_px = width
                
                q = points_out[int(g.center[1]), int(g.center[0])]
            except Exception as e:
                print('--------------------出错了----------------------')
                print(e)
            else:
                qs.append(q)
                gs.append(g)
                # if q > 0.9:
                #     break
        if len(gs) == 0:
            return None
        g = gs[np.argmax(qs)]#取得是抓取质量最高的抓取
        q = qs[np.argmax(qs)]
        print("width",width)
        print("ggs[0]",ggs[0].width)
        print("width_px",g.width_px)
        
        p0, p1 = g.endpoints
        print("real_width",np.linalg.norm(p1-p0))
        print('-------------------------')
        print([p0, p1, g.depth, g.depth, q])
        plt.clf()
        plt.imshow(points_out)
        plt.colorbar()
        plt.savefig("predict_result.png")
        return [p0, p1, g.depth, g.depth, q]#[0, 0, 0,0, 0,points_out]#


def main(args):
    model_path = MODEL_PATH.joinpath(args.model_name)
    model_path = get_model(model_path)
    pp = Planer(model_path)
    Pyro4.config.SERIALIZERS_ACCEPTED.add('pickle')
    Pyro4.Daemon.serveSimple({pp: 'grasp'}, ns=False, host='', port=6665)


def test():
    model_path = get_model(MODEL_PATH.joinpath('gmd'))
    pp = Planer(model_path)
    ggcnn = GGCNNTorch(model_path)
    for p in TEST_PATH.iterdir():
        if p.joinpath('image.npy').exists():
            test_im = p.joinpath('image.npy')
            out_path = TEST_OUTPUT.joinpath(p.name)
            depth = np.load(test_im)
            points_out, ang_out, width_out, depth_out = ggcnn.predict(depth, 300)
            if out_path.exists():
                rmtree(out_path)
            out_path.mkdir(parents=True)
            np.save((out_path.joinpath('points_out.npy')), points_out)
            np.save((out_path.joinpath('ang_out.npy')), ang_out)
            np.save((out_path.joinpath('width_out.npy')), width_out)
            np.save((out_path.joinpath('depth_out.npy')), depth_out)
            np.save((out_path.joinpath('depth.npy')), depth)
            gs = plot_output(depth_out, points_out, ang_out, 1, width_out)
            plt.savefig(out_path.joinpath('result.png'))
            gj = [g.to_jacquard() for g in gs]
            np.save((out_path.joinpath('gj.npy')), gj)
            rr = pp.plan(depth, 90)
            if rr is not None:
                np.save((out_path.joinpath('pp.npy')), np.array(rr))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='dataset to npz')
    parser.add_argument('-m', '--model-name', metavar='gmd', type=str, default='gmd',
                        help='使用的模型的名字')
    parser.add_argument('-t', '--test', action='store_true')
    args = parser.parse_args()
    if args.test:
        test()
    else:
        main(args)
