from tornado import httpserver
from tornado import gen
from tornado.ioloop import  IOLoop
import  tornado.web
import json

class IndexHandler(tornado.web.RequestHandler):
    def get(self):
            data = self.get_argument('name', 'Hello')
            print(data)
            self.write("Hello,This is TextCNN")


class ClassifyHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("this is Classfication for text,get method")
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
