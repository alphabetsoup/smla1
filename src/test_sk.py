__author__ = 'laurence'
__project__ = 'smla1'


from sklearn import datasets

class TestSK:
    def helloworld(self):
        iris = datasets.load_iris()
        print(iris.data)

t = TestSK()
t.helloworld()