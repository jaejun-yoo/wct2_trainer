import visdom
import nsml

class Logger:
    def __init__(self):
        self.vis = nsml.Visdom(visdom=visdom)
        self.last = None

    def scalar_summary(self, tag, value, step, scope=None):
        if self.last and self.last['step'] != step:
            nsml.report(**self.last, scope=scope)
            self.last = None
        if self.last is None:
            self.last = {'step':step, 'iter':step, 'epoch':1}
        self.last[tag] = value

    def image_summary(self, data, opts):
        self.vis.images(data,
            opts=opts
        )
