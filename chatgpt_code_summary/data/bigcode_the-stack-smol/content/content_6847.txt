import connexion
from openapi_server.annotator.phi_types import PhiType
from openapi_server.get_annotations import get_annotations
from openapi_server.models.error import Error  # noqa: E501
from openapi_server.models.text_id_annotation_request import TextIdAnnotationRequest  # noqa: E501
from openapi_server.models.text_id_annotation_response import TextIdAnnotationResponse  # noqa: E501


def create_text_id_annotations(text_id_annotation_request=None):  # noqa: E501
    """Annotate IDs in a clinical note

    Return the ID annotations found in a clinical note # noqa: E501

    :param text_id_annotation_request:
    :type text_id_annotation_request: dict | bytes

    :rtype: TextIdAnnotationResponse
    """
    if connexion.request.is_json:
        try:
            annotation_request = TextIdAnnotationRequest.from_dict(connexion.request.get_json())  # noqa: E501
            note = annotation_request.note
            annotations = get_annotations(note, phi_type=PhiType.ID)

            res = TextIdAnnotationResponse(annotations)
            status = 200
        except Exception as error:
            status = 500
            res = Error("Internal error", status, str(error))
    return res, status
