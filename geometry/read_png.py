import numpy as np
import png, itertools
import matplotlib.pyplot as plt
fig = "square_png.png"
Reader = png.Reader(fig)
row_count, column_count, pngdata, meta = Reader.asDirect()

# Make sure we're dealing with RGB files

bitdepth=meta['bitdepth']
plane_count=meta['planes']
assert plane_count == 3
# Returns the image data as a direct representation of an x * y * planes array.
# return (width, height, pixels, meta)
# to test asFloat(), asRGB()

image_2d = np.vstack(itertools.imap(np.uint16, pngdata))
image_3d = np.reshape(image_2d, (row_count,column_count,plane_count))

print type(image_3d), image_3d.shape
print type(image_2d), image_2d.shape

# http://pypng.googlecode.com/svn/trunk/code/exnumpy.py

mask_wall = [255,255,255]
background = 0
#walls = np.zeros((column_count,row_count))

exits = []
 # ``image_3d`` will contain the image as a three-dimensional numpy
 #     array, having dimensions ``(row_count,column_count,plane_count)``.
X = []
Y = []

def is_wall(v):
    return (v == mask_wall).all()


#for (i,j) in itertools.product(range(row_count), range(column_count)):
for i in range(row_count):
    for j in range(column_count):
        if is_wall(image_3d[i,j]) and i!=j:
            # walls[i,j] = np.Inf
            X.append(i)
            Y.append(j)
            plt.plot(i,j,".r")
minX = min(X)
maxX = max(X)
minY = min(Y)
maxY = max(Y)

w = maxX-minX
h = maxY-minY
space = np.zeros( (row_count, column_count ))

for (i,j) in itertools.product(range(minX,maxX), range(minY,maxY)):
        space[i,j] = np.Inf
        
    # elif (image_3d[i,j] != background).any():
    #     exits.append([i,j])



# plt.plot([X[0],X[0]],[Y[0], Y[-1]], [X[-1],X[-1]],[Y[0], Y[-1]], [X[0],X[-1]],[Y[0], Y[0]] , [X[0],X[-1]],[Y[-1], Y[-1]], "k" )
 
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.cla()
# cmap = plt.get_cmap()
# cmap.set_bad(color = 'k', alpha = 0.8)
# plt.imshow(space, cmap=cmap) #  lanczos nearest
plt.savefig("walls.png")


# for row in  image_3d:
#     if row.any():
#         l = row
#         print row
    
# np.savetxt("log.txt",image_2d, fmt="%d")
# Write the array to disk
# with file('test.txt', 'w') as outfile:
#     # X'm writing a header here just for the sake of readability
#     # Any line starting with "#" will be ignored by numpy.loadtxt
#     outfile.write('# Array shape: {0}\n'.format(image_2d.shape))

#     # Xterating through a ndimensional array produces slices along
#     # the last axis. This is equivalent to data[i,:,:] in this case
#     for data_slice in  image_3d: #range(plane_count):
#         # data_slice = image_3d[:,:,i]
#         # The formatting string indicates that X'm writing out
#         # the values in left-justified columns 7 characters in width
#         # with 2 decimal places.
#         np.savetxt(outfile, data_slice, fmt='%d')

#         # Writing out a break to indicate different slices...
#         outfile.write('# New slice\n')
