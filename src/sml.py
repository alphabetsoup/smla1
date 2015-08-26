__author__ = 'VAIO'

class sml:
    def it(self):
        f = open("test-public.txt")
        lines = f.readlines()
        line1 = lines[1]
        linesplit = line1.split()[2]
        print(linesplit)
        g = open("train.txt")
        for testline in lines:
            linesplit2 = testline.split()
            for word in linesplit2:
                linesplit2[0]

    @staticmethod
    def file_len(fname):
        with open(fname) as f:
            for i, l in enumerate(f):
                pass
        return i+1
t = sml()
t.it()
t.file_len("test-public.txt")