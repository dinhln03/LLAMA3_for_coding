"""
Script to show the wireframe of a given mesh (read from a file) in an interactive
Viewer.
"""


from viewer import *
from mesh.obj import OBJFile
import sys

if __name__ == "__main__":
    app = Viewer()
    if len(sys.argv) > 1:
        try:
            obj = OBJFile.read(sys.argv[1])
            app.scene.addObject(obj)
            app.title(sys.argv[1])
            app.scene.setTarget(obj.centroid)
        except Exception as e:
            raise e
    else:
        print("No input file given. Nothing to render.")
        print("Try 'python3 wireframe.py yourobj.obj'")

    app.show()
