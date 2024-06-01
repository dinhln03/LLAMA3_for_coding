#Author-HeNeos
#Description-Many triangles, I love triangles

import adsk.core, adsk.fusion, adsk.cam, traceback
import math


def get_points(n, angle, r):
    ans = [[0.0, 0.0]]*n
    for i in range(0, n):
        ans[i] = [r*math.cos(angle + 2*i*math.pi/n), r*math.sin(angle + 2*i*math.pi/n)]
    return ans

def run(context):
    try:
        app = adsk.core.Application.get()
        ui  = app.userInterface
        ui.messageBox('Are you ready')
        product = app.activeProduct
        design = adsk.fusion.Design.cast(product)
        rootComp = design.rootComponent
        sketches = rootComp.sketches
        xyPlane = rootComp.xYConstructionPlane
        # Create a new ObjectCollection.
        
        revolves = rootComp.features.revolveFeatures
    
        r = 4
        loftFeats = rootComp.features.loftFeatures
        loftInput = loftFeats.createInput(adsk.fusion.FeatureOperations.NewBodyFeatureOperation)
        loftSectionsObj = loftInput.loftSections
        n = 6
        for i in range(0, 100):
            angle = (math.pi)*abs(math.sin(i/10))
            
            ctorPlanes = rootComp.constructionPlanes
            plane = ctorPlanes.createInput()
            offset = adsk.core.ValueInput.createByString(str(i)+" cm")
            plane.setByOffset(xyPlane, offset)
            Plane = ctorPlanes.add(plane)
            sketch = sketches.add(Plane)
            lines = sketch.sketchCurves.sketchLines

            Points = []
            Lines = []
            p = get_points(n, angle, r)
            for j in range(0, n):
                point = adsk.core.Point3D.create(p[j][0], p[j][1], 0)
                Points.append(point)
            
            for j in range(0, n-1):
                line = lines.addByTwoPoints(Points[j], Points[j+1])
                Lines.append(line)
            Lines.append(lines.addByTwoPoints(Points[n-1], Points[0]))

            for i in range(0, n-1):
                sketch.sketchCurves.sketchArcs.addFillet(Lines[i], Lines[i].endSketchPoint.geometry, Lines[i+1], Lines[i+1].startSketchPoint.geometry, 0.5)
            sketch.sketchCurves.sketchArcs.addFillet(Lines[n-1], Lines[n-1].endSketchPoint.geometry, Lines[0], Lines[0].startSketchPoint.geometry, 0.5)

            profile = sketch.profiles.item(0)
            sketch.isVisible = False
            Plane.isLightBulbOn = False
            loftSectionsObj.add(profile)        
        
        loftInput.isSolid=True
        loftFeats.add(loftInput)
    except:
        if ui:
            ui.messageBox('Failed:\n{}'.format(traceback.format_exc()))

    #axis = lines.addByTwoPoints(adsk.core.Point3D.create(-1,-4,0), adsk.core.Point3D.create(1,-4,0))
    #circle1 = circles.addByCenterRadius(adsk.core.Point3D.create(0,0,0), 2)


def stop(context):
    try:
        app = adsk.core.Application.get()
        ui  = app.userInterface
        ui.messageBox('Finished')

    except:
        if ui:
            ui.messageBox('Failed:\n{}'.format(traceback.format_exc()))
