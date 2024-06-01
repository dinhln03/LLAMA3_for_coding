import os
import sys
import subprocess

CondylesFeaturesExtractor = "/Users/prisgdd/Documents/Projects/CNN/CondylesFeaturesExtractor-build/src/bin/condylesfeaturesextractor"

parser = argparse.ArgumentParser()
parser.add_argument('-meshDir', action='store', dest='meshDir', help='Input file to classify', 
                    default = "/Users/prisgdd/Desktop/TestPipeline/inputGroups/Mesh")
parser.add_argument('-outputDir', action='store', dest='outputDir', help='Directory for output files', 
					default="/Users/prisgdd/Desktop/TestPipeline/outputSurfRemesh")
parser.add_argument('-meanGroup', action='store', dest='meanGroup', help='Directory with all the mean shapes', 
					default="/Users/prisgdd/Documents/Projects/CNN/drive-download-20161123T180828Z")

args = parser.parse_args()
meshDir= args.meshDir
outputDir = args.outputDir
meanGroup = args.meanGroup

# Verify directory integrity
if not os.path.isdir(meshDir) or not os.path.isdir(outputDir):
	sys.exit("Error: At least one input is not a directory.")

listMesh = os.listdir(meshDir)

if listMesh.count(".DS_Store"):
	listMesh.remove(".DS_Store")

for i in range(0,len(listMesh)):
	command = list()

	command.append(CondylesFeaturesExtractor)
	command.append("--input")
	command.append(meshDir + "/" + listMesh[i])

	outputFile = outputDir + "/" + listMesh[i].split(".")[:-1][0] + "-Features.vtk"
	print outputFile
	file = open(outputFile, 'w')
	file.close()
	command.append("--output")
	command.append(outputFile)

	command.append("--meanGroup")
	command.append(meanGroup)	

	subprocess.call(command)







