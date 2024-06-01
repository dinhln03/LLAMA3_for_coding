from Tree import TreeNode
def zdir(root):
	if   root[0] == 0:
		return 0 #left
	elif root[1] ==0:
		return 1 #right
	else:
		return 2 #nothing


def add_nodes(root, nodes):
	for i in range(0,len(nodes)):
		if   nodes[i][0] == root.gfv():
			root.addChild(add_nodes(TreeNode(nodes[i], 0), nodes[0:i] + nodes[i+1:]))
		elif nodes[i][1] == root.gfv():
			root.addChild(add_nodes(TreeNode(nodes[i], 1), nodes[0:i] + nodes[i+1:]))

	return root
	

f  = open("Day24DB")
cm = [] 

for l in f.readlines():
	t = l.split('/')
	cm.append((int(t[0]), int(t[1]), 0))

cm   = sorted(cm, key=sum, reverse=True)
begs = filter(lambda x:     x[0] == 0 or x[1] == 0 , cm) 
cm   = filter(lambda x: not(x[0] == 0 or x[1] == 0), cm) 
rts  =    map(lambda x: TreeNode(x, zdir(x)), begs)
t    = TreeNode((0,0),2)

for root in rts:
	t.addChild(add_nodes(root, filter(lambda x: not(x[0] == root.l and x[1] == root.r), cm)))

print "Looking for max" 
print t.gwgh()
print t.glen()
print t.glnw()
