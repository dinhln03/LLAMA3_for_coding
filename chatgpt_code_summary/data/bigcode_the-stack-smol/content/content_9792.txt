

ROOT_SCOPE_METHOD( MD( 'Not', 'NOT_CLAUSE_FACTORY_single()' ) )


TEST( """ if [ Not @ ( Integer ) consume [ 1 2 3 ] ] then [ . 1 ] else [ . 2 ] == 2 """ )
TEST( """ Not @ ( Word ) consume [ 1 2 3 ] next value == 1 """ )


OBJECT( 'NOT_CLAUSE_FACTORY',
  methods = [
    MS( ARG( CW( '@' ), CG( 'CLAUSE', 'clause' ) ), """
      JUMP__return_ANY( CONTEXT, CONTEXT, $CA(NOT_CLAUSE_new( PARAM_clause )) ) ;
    """ ),
  ]
)

OBJECT( 'NOT_CLAUSE',
  inherit = [ 'CLAUSE' ],
  attributes = [
    A( 'ANY', 'clause' ),
  ],
  methods = [
    MS( ARG( CW( 'consume' ), CG( 'LIST', 'phrase' ) ), """
      JUMP__consume_LIST( $CA(FRAME__NOT_CLAUSE_1_new( CONTEXT, PARAM_phrase )), ACTION->clause, PARAM_phrase ) ;
    """ ),
  ],
  dump = D( '%s', '$DUMP( object->clause )' )
)

FRAME( 'NOT_CLAUSE_1',
  attributes = [
    A( 'ANY', 'phrase' ),
  ],
  methods = [
    MS( ARG( CW( 'return' ), CG( 'ANY', 'value' ) ), """
      nom_fail( ACTION->parent, "Input rejected by Not", $NONE ) ;
    """ ),
    MS( ARG( CW( 'fail' ), CG( 'ANY', 'error' ) ), """
      JUMP__return_ANY( ACTION->parent, ACTION->parent, $CA(ELEMENT_new( $NONE, ACTION->phrase )) ) ;
    """ ),
  ]
)


