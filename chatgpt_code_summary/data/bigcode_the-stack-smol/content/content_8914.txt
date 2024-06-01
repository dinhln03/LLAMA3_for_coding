from rdflib import URIRef, Namespace
from definednamespace import DefinedNamespace


class RDF(DefinedNamespace):
    
	# http://www.w3.org/1999/02/22-rdf-syntax-ns#Property
	direction: URIRef               # The base direction component of a CompoundLiteral.
	first: URIRef                   # The first item in the subject RDF list.
	language: URIRef                # The language component of a CompoundLiteral.
	object: URIRef                  # The object of the subject RDF statement.
	predicate: URIRef               # The predicate of the subject RDF statement.
	rest: URIRef                    # The rest of the subject RDF list after the first item.
	subject: URIRef                 # The subject of the subject RDF statement.
	type: URIRef                    # The subject is an instance of a class.
	value: URIRef                   # Idiomatic property used for structured values.

	# http://www.w3.org/2000/01/rdf-schema#Class
	Alt: URIRef                     # The class of containers of alternatives.
	Bag: URIRef                     # The class of unordered containers.
	CompoundLiteral: URIRef         # A class representing a compound literal.
	List: URIRef                    # The class of RDF Lists.
	Property: URIRef                # The class of RDF properties.
	Seq: URIRef                     # The class of ordered containers.
	Statement: URIRef               # The class of RDF statements.

	# http://www.w3.org/2000/01/rdf-schema#Datatype
	HTML: URIRef                    # The datatype of RDF literals storing fragments of HTML content
	JSON: URIRef                    # The datatype of RDF literals storing JSON content.
	PlainLiteral: URIRef            # The class of plain (i.e. untyped) literal values, as used in RIF and OWL 2
	XMLLiteral: URIRef              # The datatype of XML literal values.
	langString: URIRef              # The datatype of language-tagged string values

	_NS = Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#")
