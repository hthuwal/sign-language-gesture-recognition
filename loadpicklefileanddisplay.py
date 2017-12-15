import sys
import pickle
x = pickle.load(open(sys.argv[1], 'rb'))
print "Length", len(x)
print x[0]
print "Press y to print all data and n to exit"
t = raw_input()
if t == 'y':
    for each in x:
        print each
