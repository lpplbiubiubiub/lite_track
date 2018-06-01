# coding:utf8
import visdom
import time
import numpy as np
import cv2
import colorsys


def hsv2rgb(h,s,v):
    return tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h,s,v))


class Visualizer(object):
    '''
       封装了visdom的基本操作，但是你仍然可以通过`self.vis.function`
       或者`self.function`调用原生的visdom接口
       比如
       self.text('hello visdom')
       self.histogram(t.randn(1000))
       self.line(t.arange(0, 10),t.arange(1, 11))
    '''

    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)

        # 画的第几个数，相当于横坐标
        # 比如（’loss',23） 即loss的第23个点
        self.index = {}
        self.log_text = ''

    def reinit(self, env='default', **kwargs):
        '''
        修改visdom的配置
        '''
        self.vis = visdom.Visdom(env=env, **kwargs)
        return self

    def plot_many(self, d):
        '''
        一次plot多个
        @params d: dict (name, value) i.e. ('loss', 0.11)
        '''
        for k, v in d.iteritems():
            self.plot(k, v)

    def img_many(self, d):
        for k, v in d.iteritems():
            self.img(k, v)

    def plot(self, name, y, **kwargs):
        '''
        self.plot('loss', 1.00)
        '''
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=unicode(name),
                      opts=dict(title=name),
                      update=None if x == 0 else 'append',
                      **kwargs
                      )
        self.index[name] = x + 1

    def img(self, name, img_, **kwargs):
        '''
        self.img('input_img', t.Tensor(64, 64))
        self.img('input_imgs', t.Tensor(3, 64, 64))
        self.img('input_imgs', t.Tensor(100, 1, 64, 64))
        self.img('input_imgs', t.Tensor(100, 3, 64, 64), nrows=10)
        '''
        self.vis.images(img_.cpu().numpy(),
                        win=unicode(name),
                        opts=dict(title=name),
                        **kwargs
                        )

    def log(self, info, win='log_text'):
        '''
        self.log({'loss':1, 'lr':0.0001})
        '''

        self.log_text += ('[{time}] {info} <br>'.format(
            time=time.strftime('%m%d_%H%M%S'), \
            info=info))
        self.vis.text(self.log_text, win)

    def img_numpy(self, img, win_name):
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = img.transpose(2, 0, 1)
            self.vis.images(img, win=win_name)

    def image_track_detect(self, img, track_id, track_pos, det_id, detect_pos, win_name):
        if img.shape[2] == 3:
            [cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 3) for (x, y, w, h) in track_pos]
            [cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 3) for (x, y, w, h) in detect_pos]
            font = cv2.FONT_HERSHEY_COMPLEX
            [cv2.putText(img, str(idx), (x, y - 10), font, 1,
                         (255, 255, 255), 2) for idx, (x, y, _, _) in zip(track_id, track_pos)]
            [cv2.putText(img, str(idx), (x, y + 10), font, 1,
                         (255, 255, 0), 2) for idx, (x, y, _, _) in zip(det_id, track_pos)]
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = img.transpose(2, 0, 1)
            self.vis.images(img, win=win_name)

    def image_track(self, img, track_id, track_pos, track_color, track_conf, win_name):
        if img.shape[2] == 3:
            [cv2.rectangle(img, (int(x), int(y)), (int(x2), int(y2)), color, 2) for (x, y, x2, y2), color in zip(track_pos, track_color)]
            # [cv2.rectangle(img, (x, y), (x + w, y + h), hsv2rgb(conf, 0.7, 0.9), -1)
            # for (x, y, w, h), conf in zip(track_pos, track_conf)]
            font = cv2.FONT_HERSHEY_COMPLEX
            [cv2.putText(img, str(idx), (x, y - 10), font, 1,
                         color, 2) for idx, (x, y, _, _), color in zip(track_id, track_pos, track_color)]
            [cv2.putText(img, str(round(conf, 2)), (x, y + 10), font, 1,
                         color, 2) for conf, (x, y, _, _), color in zip(track_conf, track_pos, track_color)]
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = img.transpose(2, 0, 1)
            self.vis.images(img, win=win_name)

    def image_scatter(self, img, scatter_pos_list, win_name):
        if img.shape[2] == 3:
            [cv2.circle(img, (x, y), 2, (255, 255, 255)) for (x, y) in scatter_pos_list]
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = img.transpose(2, 0, 1)
            self.vis.images(img, win=win_name)

    def __getattr__(self, name):
        '''
        self.function 等价于self.vis.function
        自定义的plot,image,log,plot_many等除外
        '''
        return getattr(self.vis, name)
