# Project Quex (http://quex.sourceforge.net); License: MIT;
# (C) 2005-2020 Frank-Rene Schaefer; 
#_______________________________________________________________________________
from   quex.input.setup                               import NotificationDB
from   quex.input.regular_expression.pattern          import Pattern_Prep
import quex.input.regular_expression.core             as     regular_expression
from   quex.input.code.base                           import SourceRef, \
                                                             SourceRef_DEFAULT, \
                                                             SourceRefObject
from   quex.engine.state_machine.core                 import DFA  
import quex.engine.state_machine.construction.sequentialize as sequentialize
import quex.engine.state_machine.construction.repeat        as repeat
import quex.engine.state_machine.algebra.difference   as     difference    
import quex.engine.state_machine.algebra.intersection as     intersection    
import quex.engine.state_machine.algorithm.beautifier as     beautifier    
import quex.engine.state_machine.check.swallow        as     swallow
import quex.engine.state_machine.check.outrun         as     outrun
import quex.engine.state_machine.check.identity       as     identity
import quex.engine.state_machine.check.tail           as     tail
from   quex.engine.misc.tools                         import typed
from   quex.engine.misc.interval_handling             import NumberSet
from   quex.engine.counter                            import IndentationCount_Pre, \
                                                             cc_type_name_db, \
                                                             cc_type_db
from   quex.engine.counter_builder                    import CountActionMap_Builder
import quex.engine.misc.error                         as     error
import quex.engine.misc.error_check                   as     error_check
from   quex.engine.misc.file_in                       import check, \
                                                             check_or_die, \
                                                             skip_whitespace, \
                                                             read_identifier, \
                                                             read_integer
from   quex.constants  import E_CharacterCountType
from   quex.blackboard import setup as Setup

def parse_CountActionMap(fh):
    return _base_parse(fh, CountActionMapFromParser_Builder(fh))

def parse_IndentationSetup(fh):
    return _base_parse(fh, IndentationSetup_Builder(fh))

def _base_parse(fh, builder, IndentationSetupF=False):
    """Parses pattern definitions of the form:
   
          [ \t]                                       => grid 4;
          [:intersection([:alpha:], [\X064-\X066]):]  => space 1;

    In other words the right hand side *must* be a character set.

    ADAPTS: result to contain parsing information.
    """

    # NOTE: Catching of EOF happens in caller: parse_section(...)
    #
    while 1 + 1 == 2:
        skip_whitespace(fh)
        if check(fh, ">"): 
            break
        
        # A regular expression state machine
        pattern, identifier, sr = _parse_definition_head(fh, builder.identifier_list)
        if pattern is None and not builder.keyword_else_f:
            error.log("Keyword '\\else' cannot be used in indentation setup.", fh)

        # '_parse_definition_head()' ensures that only identifiers mentioned in 
        # 'result' are accepted. 
        if builder.requires_count():
            count = _read_value_specifier(fh, identifier, 1)
            builder.specify(identifier, pattern, count, sr)
        else:
            builder.specify(identifier, pattern, sr)

        if not check(fh, ";"):
            error.log("Missing ';' after '%s' specification." % identifier, fh)

    return builder.finalize() 

class CharacterSetVsAction_BuilderBase:
    def __init__(self, IdentifierList, KeywordElseAdmissibleF):
        self.identifier_list = IdentifierList
        self.keyword_else_f  = KeywordElseAdmissibleF

class CountActionMapFromParser_Builder(CharacterSetVsAction_BuilderBase):
    """Line/column number count specification.
    ___________________________________________________________________________
    The main result of the parsing the the Base's .count_command_map which is 
    an instance of CountActionMap_Builder.
    ____________________________________________________________________________
    """
    @typed(sr=SourceRef)
    def __init__(self, fh):
        self.sr              = SourceRef.from_FileHandle(fh)
        self.__fh            = fh
        self._ca_map_builder = CountActionMap_Builder()
        CharacterSetVsAction_BuilderBase.__init__(self, 
                                                  ("columns", "grid", "lines"), 
                                                  KeywordElseAdmissibleF=True) 

    def finalize(self):
        # Finalize / Produce 'LineColumnCount' object.
        # 
        ca_map = self._ca_map_builder.finalize(
                              Setup.buffer_encoding.source_set.minimum(), 
                              Setup.buffer_encoding.source_set.least_greater_bound(), 
                              self.sr)
        _check_grid_values_integer_multiples(ca_map)
        check_defined(ca_map, self.sr, E_CharacterCountType.LINE)
        return ca_map 

    def requires_count(self):
        return True

    @typed(sr=SourceRef, Identifier=(str,str))
    def specify(self, Identifier, Pattern, Count, sr):
        if Pattern is None:
            self._ca_map_builder.define_else(cc_type_db[Identifier], Count, sr)
        else:
            trigger_set = _extract_trigger_set(sr, Identifier, Pattern) 
            self._ca_map_builder.add(trigger_set, cc_type_db[Identifier], Count, sr)

class IndentationSetup_Builder(CharacterSetVsAction_BuilderBase):
    """Indentation counter specification.
    ____________________________________________________________________________
    The base's .count_command_map contains information about how to count the 
    space at the beginning of the line. The count until the first non-whitespace
    is the 'indentation'. 
    
    +bad:

    The spec contains information about what characters are not supposed to
    appear in indentation (bad characters). Depending on the philosophical
    basis, some might consider 'space' as evil, others consider 'tab' as evil.

    +newline:

    A detailed state machine can be defined for 'newline'. This might be 
    '\n|(\r\n)' or more complex things.

    +suppressor:

    A newline might be suppressed by '\' for example. For that, it might be
    specified as 'newline suppressor'.
    ____________________________________________________________________________
    """
    @typed(sr=SourceRef)
    def __init__(self, fh):
        self.__fh = fh
        self.sm_whitespace         = SourceRefObject("whitespace", None)
        self.sm_badspace           = SourceRefObject("bad", None)
        self.sm_newline            = SourceRefObject("newline", None)
        self.sm_newline_suppressor = SourceRefObject("suppressor", None)
        self.sm_suspend_list       = []

        if fh == -1: self.sr = SourceRef_DEFAULT
        else:        self.sr = SourceRef.from_FileHandle(self.__fh)
        CharacterSetVsAction_BuilderBase.__init__(self,
                                                  ("whitespace", "suspend", "newline", "suppressor", "bad"),
                                         KeywordElseAdmissibleF=False) 

    def finalize(self):
        # Finalize / Produce 'IndentationCount' object.
        # 
        if self.sm_whitespace.get() is None:
            self.sm_whitespace.set(self.__sm_whitespace_default(), SourceRef_DEFAULT)
        if self.sm_newline.get() is None:
            self.sm_newline.set(self.__sm_newline_default(), SourceRef_DEFAULT)

        # -- consistency
        self._consistency_check()

        # Transform 'SourceRefObject' into 'Pattern_Prep' objects
        # (TODO: Why not use it in the first place?)
        def get_pattern(SRO):
            if SRO is None or SRO.get() is None: return None
            return Pattern_Prep(SRO.get(), PatternString="<indentation %s>" % SRO.name, Sr=SRO.sr)

        pattern_suspend_list = [ get_pattern(sro) for sro in self.sm_suspend_list ]
        pattern_suspend_list = [ x for x in pattern_suspend_list if x is not None ]

        if self.sm_newline_suppressor.set_f():
            sm_suppressed_newline = sequentialize.do([self.sm_newline_suppressor.get(),
                                                      self.sm_newline.get()])
            sm_suppressed_newline = beautifier.do(sm_suppressed_newline)
            pattern_suppressed_newline = Pattern_Prep(sm_suppressed_newline, 
                                                      PatternString="<indentation suppressed newline>",
                                                      Sr=self.sm_newline_suppressor.sr)
        else:
            pattern_suppressed_newline = None

        return IndentationCount_Pre(self.sr, 
                                    get_pattern(self.sm_whitespace),
                                    get_pattern(self.sm_badspace),
                                    get_pattern(self.sm_newline),
                                    pattern_suppressed_newline,
                                    pattern_suspend_list)

    def requires_count(self):
        return False

    def specify(self, identifier, pattern, sr):
        sm = pattern.extract_sm()
        if   identifier == "whitespace": 
            self.__specify(self.sm_whitespace, sm, sr)
        elif identifier == "bad":        
            self.__specify(self.sm_badspace, sm, sr)
        elif identifier == "newline":    
            self.__specify(self.sm_newline, sm, sr)
        elif identifier == "suppressor": 
            self.__specify(self.sm_newline_suppressor, sm , sr)
        elif identifier == "suspend":    
            self.__specify_suspend(sm, sr)
        else:                            
            return False
        return True

    @typed(sr=SourceRef)
    def __specify(self, member_ref, Sm, sr):
        assert Sm is not None 
        _error_if_defined_before(member_ref, sr)

        if not Sm.is_DFA_compliant(): Sm = beautifier.do(Sm)

        member_ref.set(Sm, sr)

    @typed(sr=SourceRef)
    def __specify_suspend(self, Sm, sr):
        for before in self.sm_suspend_list:
            if not identity.do(before.get(), Sm): continue
            error.log("'suspend' has been defined before;", sr, DontExitF=True)
            error.log("at this place.", before.sr)

        sm_suspend = SourceRefObject("suspend", None)
        self.__specify(sm_suspend, Sm, sr)
        self.sm_suspend_list.append(sm_suspend)

    def __sm_newline_default(self):
        """Default newline: '(\n)|(\r\n)'
        """
        sm = DFA.from_character_set(NumberSet(ord('\n')))
        if Setup.dos_carriage_return_newline_f:
            sm.add_transition_sequence(sm.init_state_index, [ord('\r'), ord('\n')])
        return sm

    def __sm_whitespace_default(self):
        """Try to define default whitespace ' ' or '\t' if their positions
        are not yet occupied in the count_command_map.
        """
        sm_whitespace = DFA.from_character_set(NumberSet.from_integer_list([ord(' '), ord('\t')]))
        sm_whitespace = beautifier.do(repeat.do(sm_whitespace, 1))
        if self.sm_badspace.get() is not None:
            sm_whitespace = difference.do(sm_whitespace, self.sm_badspace.get())
            if    sm_whitespace.is_Empty() \
               or outrun.do(self.sm_badspace.get(), sm_whitespace):
                error.log("Cannot define default 'whitespace' in the frame of the given\n"
                          "definition of 'bad'.", self.sm_badspace.sr)
        return sm_whitespace

    def _consistency_check(self):
        """
        Required defintions:
           -- WHITESPACE (Default done automatically) => Assert.
           -- NEWLINE (Default done automatically)    => Assert.

        Inadmissible 'eat-into'.
           -- SUPPRESSOR shall not eat into [NEWLINE]
           -- NEWLINE    shall not eat into [WHITESPACE, BADSPACE, SUSPEND, SUPPRESSOR]
           -- WHITESPACE shall not eat into [SUPPRESSOR, NEWLINE, SUSPEND].
           -- BADSPACE   shall not eat into [SUPPRESSOR, NEWLINE, SUSPEND].

        No common lexemes:
           -- WHITESPACE and BADSPACE may not have common lexemes.

        Outrun:
           -- NEWLINE    may not start with SUSPEND and vice versa
           -- NEWLINE    may not start with SUPPRESSOR and vice versa
           -- SUPPRESSOR may not start with SUSPEND and vice versa
           -- WHITESPACE shall not outrun BADSPACE, but the contrary is ok.
              (BADSPACE  may outrun WHITESPACE (e.g: lexeme with 'tab' after whitespace')
        """
        # (1) Required definitions _____________________________________________
        assert self.sm_whitespace.set_f()
        assert self.sm_newline.set_f()

        whitespace   = self.sm_whitespace
        newline      = self.sm_newline
        badspace     = self.sm_badspace
        suppressor   = self.sm_newline_suppressor
        suspend_list = self.sm_suspend_list

        # (2) Inadmissible 'eat-into' __________________________________________
        #
        cmp_list = [
            (newline, badspace), (newline, whitespace), (newline, suppressor),
            (suppressor, newline),
            (whitespace, newline), (whitespace, suppressor),
            (badspace, newline),   (badspace, suppressor),
        ] \
        + [ (whitespace, x) for x in suspend_list ] \
        + [ (newline, x) for x in suspend_list ]    \
        + [ (badspace, x) for x in suspend_list ] 

        def _error(FormatStr, Sro0, Sro1):
            error.log(FormatStr % (Sro0.name, Sro1.name), Sro0.sr, DontExitF=True)
            error.log("'%s' defined here." % Sro1.name, Sro1.sr)

        def _iterate(SroPairList):
            for first_sro, second_sro in cmp_list:
                first, second = first_sro.get(), second_sro.get()
                if first is None or second is None: continue
                yield first_sro, first, second_sro, second

        for first_sro, first, second_sro, second in _iterate(cmp_list):
            if swallow.ending_A_beginning_B(first, second):
                _error("'%s' may eat into beginning of '%s'.", first_sro, second_sro)
            elif swallow.inside_A_match_B(first, second):
                _error("'%s' may swallow something matched by '%s'.", first_sro, second_sro)
            
        for sm_suspend in self.sm_suspend_list:
            only_common_f, \
            common_f       = tail.do(self.sm_newline.get(), sm_suspend.get())

            error_check.tail(only_common_f, common_f, 
                             "indentation handler's newline", self.sm_newline.sr, 
                             "suspend", sm_suspend.sr)

        # (3) Inadmissible common lexemes _____________________________________
        #
        if badspace.get() and not intersection.do([badspace.get(), whitespace.get()]).is_Empty():
            _error("'%s' and '%s' match on common lexemes.", whitespace, badspace)

        # (3) Inadmissible outruns ____________________________________________
        #
        cmp_list = [ (newline, suppressor), (suppressor, newline), (whitespace, badspace) ]
        for x in suspend_list:
            cmp_list.extend([
                (newline,    x), (x, newline), 
                (suppressor, x), (x, suppressor)
            ])

        for first_sro, first, second_sro, second in _iterate(cmp_list):
            if outrun.do(second, first):
                _error("'%s' may outrun '%s'.", first_sro, second_sro)


def _parse_definition_head(fh, IdentifierList):

    if check(fh, "\\default"): 
        error.log("'\\default' has been replaced by keyword '\\else' since quex 0.64.9!", fh)
    elif check(fh, "\\else"): 
        pattern = None
    else:                      
        pattern = regular_expression.parse(fh, AllowPreContextF=False, 
                                           AllowPostContextF=False)

    skip_whitespace(fh)
    check_or_die(fh, "=>", " after character set definition.")

    skip_whitespace(fh)
    identifier = read_identifier(fh, OnMissingStr="Missing identifier following '=>'.")
    error.verify_word_in_list(identifier, IdentifierList,
                              "Unrecognized specifier '%s'." % identifier, fh)
    skip_whitespace(fh)

    return pattern, identifier, SourceRef.from_FileHandle(fh)

def _read_value_specifier(fh, Keyword, Default=None):
    skip_whitespace(fh)
    value = read_integer(fh)
    if value is not None:     return value

    # not a number received, is it an identifier?
    variable = read_identifier(fh)
    if   variable:            return variable
    elif Default is not None: return Default

    error.log("Missing integer or variable name after keyword '%s'." % Keyword, fh) 

__CountActionMap_DEFAULT = None
def LineColumnCount_Default():
    global __CountActionMap_DEFAULT

    if __CountActionMap_DEFAULT is None:
        builder = CountActionMap_Builder()
        builder.add(NumberSet(ord('\n')), E_CharacterCountType.LINE, 1, SourceRef_DEFAULT)
        builder.add(NumberSet(ord('\t')), E_CharacterCountType.GRID, 4, SourceRef_DEFAULT)
        builder.define_else(E_CharacterCountType.COLUMN,   1, SourceRef_DEFAULT)     # Define: "\else"
        __CountActionMap_DEFAULT = builder.finalize(
                                      Setup.buffer_encoding.source_set.minimum(), 
                                      Setup.buffer_encoding.source_set.least_greater_bound(), # Apply:  "\else"
                                      SourceRef_DEFAULT) 
    return __CountActionMap_DEFAULT


def _error_if_defined_before(Before, sr):
    if not Before.set_f(): return

    error.log("'%s' has been defined before;" % Before.name, sr, 
              DontExitF=True)
    error.log("at this place.", Before.sr)

def _extract_trigger_set(sr, Keyword, Pattern):
    if Pattern is None:
        return None
    elif isinstance(Pattern, NumberSet):
        return Pattern

    def check_can_be_matched_by_single_character(SM):
        bad_f      = False
        init_state = SM.get_init_state()
        if SM.get_init_state().is_acceptance(): 
            bad_f = True
        elif len(SM.states) != 2:
            bad_f = True
        # Init state MUST transit to second state. Second state MUST not have any transitions
        elif len(init_state.target_map.get_target_state_index_list()) != 1:
            bad_f = True
        else:
            tmp = set(SM.states.keys())
            tmp.remove(SM.init_state_index)
            other_state_index = next(iter(tmp))
            if len(SM.states[other_state_index].target_map.get_target_state_index_list()) != 0:
                bad_f = True

        if bad_f:
            error.log("For '%s' only patterns are addmissible which\n" % Keyword + \
                      "can be matched by a single character, e.g. \" \" or [a-z].", sr)

    sm = Pattern.extract_sm()
    check_can_be_matched_by_single_character(sm)

    transition_map = sm.get_init_state().target_map.get_map()
    assert len(transition_map) == 1
    return list(transition_map.values())[0]

def _check_grid_values_integer_multiples(CaMap):
    """If there are no spaces and the grid is on a homogeneous scale,
       => then the grid can be transformed into 'easy-to-compute' spaces.
    """
    grid_value_list = []
    min_info        = None
    for character_set, info in CaMap:
        if info.cc_type == E_CharacterCountType.COLUMN: 
            return
        elif info.cc_type != E_CharacterCountType.GRID: 
            continue
        elif type(info.value) in (str, str): 
            # If there is one single 'variable' grid value, 
            # then no assumptions can be made.
            return
        grid_value_list.append(info.value)
        if min_info is None or info.value < min_info.value:
            min_info = info

    if min_info is None:
        return

    # Are all grid values a multiple of the minimum?
    if all(x % min_info.value == 0 for x in grid_value_list):
        error.warning("Setup does not contain spaces, only grids (tabulators). All grid\n" \
                      "widths are multiples of %i. The grid setup %s is equivalent to\n" \
                      % (min_info.value, repr(sorted(grid_value_list))[1:-1]) + \
                      "a setup with space counts %s. Space counts are faster to compute.\n" \
                      % repr([x / min_info.value for x in sorted(grid_value_list)])[1:-1],
                      min_info.sr)
    return

def check_defined(CaMap, SourceReference, CCT):
    """Checks whether the character counter type has been defined in the 
    map.
    
    THROWS: Error in case that is has not been defined.
    """
    for character_set, info in CaMap:
        if info.cc_type == CCT: 
            return

    error.warning("Setup does not define '%s'." % cc_type_name_db[CCT], SourceReference, 
                  SuppressCode=NotificationDB.warning_counter_setup_without_newline)



