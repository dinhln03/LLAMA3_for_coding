import numpy as np
'''
dccol : 1-8
dcpad : 1-10
mcecol: 0,1
mcerow: 0-32
'''

#w,h = 10,8

# def det2mce(detcol,detrow,detpol):
# 	dccol,dcpad = det2dc(detcol,detrow,detpol)
# 	if dccol<0 or dcpad<0:
# 			return -1,-1
# 	mcecol,mcerow = dc2mce(dccol,dcpad)
# 	return mcecol,mcerow

def mce2det(mcecol,mcerow):
	if mcecol<=17:
		detcol=mcecol
	else:
		detcol=mcecol-18
	if mcerow<=17:
		detrow=mcerow
		detpol='A'
	if mcerow>17:
		detrow=mcerow-18
		detpol='B'
	#detcol,detrow,detpol = dc2det(dccol,dcpad)
	im = 0 #not sure what this is
	return im,detcol,detrow,detpol
