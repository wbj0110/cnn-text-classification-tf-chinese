#! /usr/bin/env python

from tornado import httpserver
from tornado import gen
from tornado.ioloop import  IOLoop
import  tornado.web
import json
import single_eval as sev

class IndexHandler(tornado.web.RequestHandler):
    def get(self):

            self.write("Hello,This is TextCNN")

class ClassifyHandler(tornado.web.RequestHandler):
    def get(self):
        data = self.get_argument('q', 'Hello')
        predict_result = sev.classify(data)
        self.write("this is Classfication for text,get method and result:{}".format(predict_result))
    def post(self):
        self.write("this is classfication for text ,post method")

class Application(tornado.web.Application):
    def __init__(self):
        handlers = [
            (r"/?",IndexHandler),
            (r"/classify/?",ClassifyHandler)
        ]
        tornado.web.Application.__init__(self,handlers=handlers)

def main():
    app = Application()
    app.listen(80)
    IOLoop.instance().start()


if __name__ == '__main__':
     main()
