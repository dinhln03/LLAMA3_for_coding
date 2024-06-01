from django.shortcuts import render, get_object_or_404
from django.http import HttpResponse
from gfx.models import Material, Mesh, Shader, Model
import subprocess
import os
import base64
import platform



def get_mesh( request, mesh_id):

	mesh = get_object_or_404(Mesh, pk=mesh_id)

	"""
		TODO:	Make it grab it from the cache instead of generating it
				every single time like some savage that can't handle living
				in the twenty-first century
	"""
	if( False ):#exportedMesh.findwith( mesh_id )):
		pass

	#
	# Can't find it, then grab the mesh, and export it
	#
	processName = None
	sys = platform.system()
	if( sys == "Darwin" ):
		processName = '/Applications/blender.app/Contents/MacOS/blender'
	else:
		return HttpResponse("Unknown operating system `{}`".format(sys))

	if( subprocess.call([processName,"--background", mesh.mesh.name, "--python","./gfx/export.py"]) == 1 ):
		return HttpResponse("There was an error")

	filename, fileExtension = os.path.splitext(mesh.mesh.name)
	newFileContents = open("{0}.js".format(filename)).read()

	return HttpResponse(newFileContents)





def get_texture( request, texture_id):
	return HttpResponse("You're looking at texture %s." % texture_id )




def get_material( request, material_id):
	material = get_object_or_404(Material, pk=material_id)

	vertSource = "{}".format( material.getVertex() )
	fragSource = "{}".format( material.getFragment() )

	return HttpResponse(
		"""{{
			"id":{0},
			"vertex":"{1}",
			"fragment":"{2}"
		}}
		""".format(
			material.id,
			base64.b64encode( vertSource ),
			base64.b64encode( fragSource ),
		)
	)



def get_shader( request, shader_id):
	shader = get_object_or_404(Shader, pk=shader_id)
	return HttpResponse(
		"""{{
			"id":{0},
			"tag":"{1}",
			"content":"{2}"
		}}""".format(
			shader.id,
			shader.tag,
			shader.content
		)
	)



def search_models( request ):
	model = get_object_or_404(Model, name=request.GET.get('tag', None))
	return HttpResponse(
		"""{{
			"id":{0},
			"tag":"{1}",
			"mesh_id":{2},
			"material_id":{3}
		}}""".format(
			model.id,
			model.name,
			model.mesh.id,
			model.material.id
		)
	)
