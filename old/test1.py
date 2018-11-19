
from utils import *
elapsed()
from data import fetch

print (len(fetch.cases))

for case in fetch.cases:
	d = fetch.load(case)

print ('Time elapsed: ', elapsed())
