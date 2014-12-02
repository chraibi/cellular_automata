from sys import platform as _platform
import os

def make_movie(direct):
    files = os.path.join(direct, '*.png')
    cmd = '"mf://%s" -mf w=800:h=600:fps=2:type=png -ovc lavc -lavcopts vcodec=mpeg4 -oac copy -o output.avi'%files
    os.system("mencoder %s" % cmd)


if __name__ == "__main__":
    directory = 'pngs'
    if _platform == "linux" or _platform == "linux2":
        make_movie(directory)
    elif _platform == "darwin":
        print ('OSX not yet done!')
    elif _platform == "win32" or _platform == "cygwin":
        print ('Windows not yet done!')
    else:
        print('%s not known!' % _platform)