"""State-Split Transformation
-----------------------------
(C) Frank-Rene Schaefer

The 'State-Split' is a procedure transforms a state machine that triggers on
some 'pure' values (e.g. Unicode Characters) into a state machine that triggers
on the code unit sequences (e.g. UTF8 Code Units) that correspond to the
original values. For example, a state transition on a Unicode Character
'0x1329D' as shown below,

        [ A ]--->( 0x1329D )---->[ B ]

is translated into a sequence of UTF16 transitions with a new intermediate
state 'i' as follows.

        [ A ]--( 0xD80C )-->[ i ]-->( 0xDE9E )-->[ B ]

This is so, since the character 0x1329D in Unicode is represented as the
sequence 0xD80C, 0xDE9E. The present algorithm exploits the fact that
translations of adjacent character result in sequences of adjacent intervals.

 .----------------------------------------------------------------------------.
 | This procedure is to be used for encodings of dynamic size, i.e. where the |
 | number of code units to represent a 'pure' value changes depending on the  |
 | value itself (e.g. UTF8, UTF16).                                           |
 '----------------------------------------------------------------------------'

PRINCIPLE:

A state transition is described by a 'trigger set' and a target state.  If an
input occurs that belongs to the 'trigger set' the state machine transits into
the specific target state. Trigger sets are composed of one ore more intervals
of adjacent values. If the encoding has some type of continuity, it can be
assumed that an interval in the pure values can be represented by a sequence of
intervals in the transformed state machine. This is, indeed true for the
encodings UTF8 and UTF16.

The algorithm below considers intervals of pure values and translates them
into interval sequences. All interval sequences of a triggger set that 
triggers to a target state are then combined into a set of state transitions.

A unicode transition from state A to state B:

         [ A ]-->(x0, x1)-->[ B ]

is translated into a chain of utf8-byte sequence transitions that might look
like this

     [ A ]-->(b0)-->[ 1 ]-->(c0,c1)-->[ B ] 
         \                             /
          `->(d1)-->[ 2 ]---(e0,e1)---' 

That means that intermediate states may be introduced to reflect the different
byte sequences that represent the original interval.

IDEAS:
    
In a simple approach one would translate each element of a interval into an
utf8-byte sequence and generate state transitions between A and B.  Such an
approach, however, produces a huge computational overhead and charges the later
Hopcroft Minimization with a huge state machine.

To avoid such an hughe computational effort, the Hopcroft Minimzation can be
prepared on the basis of transition intervals. 
    
(A) Backwards: In somewhat greater intervals, the following might occur:


                 .-->(d1)-->[ 1 ]---(A3,BF)---. 
                /                              \
               /  ,->(d1)-->[ 2 ]---(80,BF)--.  \
              /  /                            \  \
             [ A ]-->(b0)-->[ 3 ]-->(80,BF)-->[ B ] 
                 \                             /
                  `->(d1)-->[ 4 ]---(80,81)---' 

That means, that for states 2 and 3 the last transition is on [80, BF]
to state B. Thus, the intermediate states 2 and 3 are equivalent. Both
can be replaced by a single state. 

(B) Forwards: The first couple of bytes in the correspondent utf8 sequences
    might be the same. Then, no branch is required until the first differing
    byte.

PROCESS:

(1) The original interval translated into a list of interval sequence
    that represent the values in the target encoding.

(2) The interval sequences are plugged in between the state A and B
    of the state machine.
"""
from   quex.engine.state_machine.state.core          import DFA_State
import quex.engine.state_machine.transformation.base as     base
import quex.engine.state_machine.index               as     state_machine_index
from   quex.engine.misc.interval_handling            import NumberSet

from   quex.engine.misc.tools import flatten_list_of_lists
from   collections import defaultdict

class EncodingTrafoBySplit(base.EncodingTrafo):
    """Transformation that takes a lexatom and produces a lexatom sequence.
    """
    def __init__(self, Name, ErrorRangeByCodeUnitDb):
        base.EncodingTrafo.__init__(self, Name, 
                                    NumberSet.from_range(0, 0x110000),
                                    ErrorRangeByCodeUnitDb)

    def do_transition(self, from_target_map, FromSi, ToSi, BadLexatomSi):
        """Translates to transition 'FromSi' --> 'ToSi' inside the state
        machine according to the specific coding (see derived class, i.e.
        UTF8 or UTF16).

        'BadLexatomSi' is None => no bad lexatom detection.
                       else, transitions to 'bad lexatom state' are added
                       on invalid code units.

        RETURNS: [0] True if complete, False else.
                 [1] StateDb of newly generated states.
        """
        number_set = from_target_map[ToSi]

        # Check whether a modification is necessary
        if number_set.least_greater_bound() <= self.UnchangedRange: 
            # 'UnchangedRange' => No change to numerical values.
            return True, None

        if not self.cut_forbidden_range(number_set):
            # 'number_set' solely contains forbidden elements.
            del from_target_map[ToSi]
            return False, None

        transformed_interval_sequence_list = flatten_list_of_lists(
            self.get_interval_sequences(interval)
            for interval in number_set.get_intervals(PromiseToTreatWellF=True)
        )

        # Second, enter the new transitions.
        new_target_map, \
        new_state_db    = self.plug_interval_sequences(FromSi, ToSi, 
                                                       transformed_interval_sequence_list, 
                                                       BadLexatomSi)

        # Absorb new transitions into the target map of the 'from state'.
        del from_target_map[ToSi]
        from_target_map.update(new_target_map)

        return True, new_state_db

    def _do_single(self, Code): 
        number_set    = NumberSet.from_range(Code, Code+1)
        if number_set.is_empty():
            return -1
        interval_list = number_set.get_intervals(PromiseToTreatWellF=True)
        assert len(interval_list) == 1
        interval_sequence_list = self.get_interval_sequences(interval_list[0])
        # A single code element can only produce a single interval sequence!
        assert len(interval_sequence_list) == 1
        assert all(x.size() == 1 for x in interval_sequence_list[0])
        
        return [x.begin for x in interval_sequence_list[0]]

    def variable_character_sizes_f(self):
        return True

    def lexatom_n_per_character_in_state_machine(self, SM):
        lexatom_n = None
        for state in SM.states.itervalues():
            for number_set in state.target_map.get_map().itervalues():
                candidate_lexatom_n = self.lexatom_n_per_character(number_set)
                if   candidate_lexatom_n is None:      return None
                elif lexatom_n is None:                lexatom_n = candidate_lexatom_n
                elif lexatom_n != candidate_lexatom_n: return None
        return lexatom_n

    def hopcroft_minimization_always_makes_sense(self): 
        return True

    def plug_interval_sequences(self, FromSi, ToSi, IntervalSequenceList,
                                BadLexatomSi):
        """Transform the list of interval sequences into intermediate state
        transitions. 
        
        'BadLexatomSi' is None => no bad lexatom detection.
                       else, transitions to 'bad lexatom state' are added
                       on invalid code units.
        
        RETURN: [0] Target map update for the first state.
                [1] State Db update for intermediate states.

        """
        def simplify(tm_db, tm_end_inv, ToSi):
            """Those states which trigger on the same intervals to 'ToSi' are
            equivalent, i.e. can replaced by one state.
            """
            # Find the states that trigger on the same interval list to the 
            # terminal 'ToSi'.
            equivalence_db = {}
            replacement_db = {}
            for from_si, interval_list in tm_end_inv.iteritems():
                key           = tuple(sorted(interval_list))
                equivalent_si = equivalence_db.get(key)
                if equivalent_si is None: equivalence_db[key]     = from_si
                else:                     replacement_db[from_si] = equivalent_si

            # Replace target states which are equivalent
            result = {}
            for from_si, tm in tm_db.iteritems():
                new_tm = defaultdict(NumberSet)
                for target_si, interval in tm.iteritems():
                    replacement_si = replacement_db.get(target_si)
                    if replacement_si is not None: target_si = replacement_si
                    new_tm[target_si].quick_append_interval(interval)

                if any(number_set.is_empty() for si, number_set in new_tm.items()):
                    for si, number_set in new_tm.iteritems():
                        print "#sim", si, number_set

                if from_si in tm_end_inv:
                    for interval in tm_end_inv[from_si]:
                        new_tm[ToSi].quick_append_interval(interval)

                result[from_si] = new_tm

            return result

        tm_db,      \
        tm_end_inv, \
        position_db = _get_intermediate_transition_maps(FromSi, ToSi, 
                                                        IntervalSequenceList)

        result_tm_db = simplify(tm_db, tm_end_inv, ToSi)

        if BadLexatomSi is not None:
            for si, position in position_db.iteritems():
                # The 'positon 0' is done by 'do_state_machine'. It is concerned
                # with the first state's transition.
                assert position != 0
                self._add_transition_to_bad_lexatom_detector(result_tm_db[si],
                                                             BadLexatomSi,
                                                             position)

        for tm in result_tm_db.itervalues():
            assert not any(number_set.is_empty() for number_set in tm.itervalues())

        # Generate the target map to be inserted into state 'FromSi'.
        # Generate list of intermediate states that implement the sequence
        # of intervals.
        first_tm     = result_tm_db.pop(FromSi)
        new_state_db = dict(
            (si, DFA_State.from_TargetMap(tm)) for si, tm in result_tm_db.iteritems()
        )
        return first_tm, new_state_db

def __bunch_iterable(IntervalSequenceList, Index):
    """Iterate over sub-bunches of sequence in 'IntervalSequenceList' which are
    the same at the given 'Position'. The 'IntervalSequenceList' must be sorted!
    That is, same intervals must be adjacent. 

    EXAMPLE: 
               Index                = 1
               IntervalSequenceList = [
                  [ interval01, interval12, interval21, ], 
                  [ interval01, interval12, interval21, ], 
                  [ interval02, interval12, interval22, interval30 ], 
                  [ interval02, interval13, interval22, interval30 ], 
                  [ interval02, interval13, interval23, ] ]

    That is, the interval sequences are grouped according to groups where the
    second interval (Index=1) is equal, the yields are as follows:

         (1)    [ [ interval01, interval12, interval21, ], 
                  [ interval01, interval12, interval21, ] ]

         (2)    [ [ interval02, interval12, interval22, interval30 ] ]

         (3)    [ [ interval02, interval13, interval22, interval30 ], 
                  [ interval02, interval13, interval23, ] ]

    NOTE: Two sequences of different lengths are *never* grouped together
          -- by purpose.

    The index is provided in order to avoid the creation of shorted sub-
    sequences. Instead, the caller focusses on sub-sequences behind 'Index'.
    Obviously, this function only makes sense if the intervals before 'Index'
    are all the same.

    YIELDS: [0] Interval which is the same for group of sequenes at 'Index'.
            [1] Group of sequences.
            [2] 'LastF' -- telling whether the interval is the last in the 
                sequence.
            
    """
    prev_interval = None
    prev_i        = -1
    prev_last_f   = False
    for i, sequence in enumerate(IntervalSequenceList):
        interval = sequence[Index]
        if interval.is_empty(): print "#bu:", interval; assert False
        L        = len(sequence)
        last_f   = L == Index + 1
        if interval != prev_interval or last_f != prev_last_f:
            if prev_i != -1:
                yield prev_interval, IntervalSequenceList[prev_i:i], prev_last_f
            prev_i        = i 
            prev_interval = interval
            prev_last_f   = last_f

    yield prev_interval, IntervalSequenceList[prev_i:], prev_last_f

def _get_intermediate_transition_maps(FromSi, ToSi, interval_sequence_list):
    """Several transitions are to be inserted in between state 'FromSi' and 
    'ToSi'. The transitions result from the list of sequences in 
    'interval_sequence_list'. This function develops the transition maps
    of the states involved. Also, it notifies about the 'position' of each
    state in the code unit sequence. Thus, the caller may insert error-detectors
    on invalid code units.

    FORBIDDEN: There cannot be a sequence that starts with the exact intervals
               as a shorter sequences. Example:

           [ (0, 1), (0, 2), (0, 3) ]   # 
           [ (0, 1), (0, 2) ]           # Bad, very bad!

    This would mean that after (0, 1), (0, 2) the 'ToSi' is reached, but then
    after (0, 3) again. The result is an *iteration* on 'ToSi'

           --(0, 1)-->( A )--(0, 2)-->( ToSi )---->
                                    |           |
                                    '-<-(0, 3)--'

    Consequently, such a list of interval sequences cannot represent a linear
    transition.

    RETURNS: [0] Transition Map DB:  state_index --> 'TransitionMap' 

                 with TransitionMap: target_state_index --> Interval

                 That is 'TransitionMap[target_state_index]' tells through which
                 intervals the 'state_index' triggers to 'target_states'

                 The 'Transition Map DB' does not contain transitions to the
                 'ToSi'--the end state.
     
             [1] Inverse End Transition Map:

                 Transitions to the end state are stored inversely:

                        from_state_index --> list of Interval-s

                 The end state can be reached by more than one interval, so a
                 list of Interval-s is associated with the transition
                 'from_state_index' to 'ToSi'.

             [1] PositionDB:    state_index --> position in code unit sequence.
    """
    # Sort the list of sequences, so that adjacent intervals are listed one
    # after the other. This is necessary for '__bunch_iterable()' to function.
    interval_sequence_list.sort()

    worklist = [
        # The state at 'BeginStateIndex' is concerned with the intervals
        # at position '0' in the 'interval_sequence_list'. The list needs to
        # be grouped according to the first interval, and for each distinct
        # interval a transition to another state must be generated.
        (FromSi, interval_sequence_list, 0)
    ]
    tm_db       = defaultdict(dict)
    tm_end_inv  = defaultdict(list)
    position_db = {}
    while worklist:
        si, sequence_group, index = worklist.pop()
        # -- State 'si' triggers on intervals at 'index' in 'sequence_group'.
        tm              = tm_db[si]
        # -- State 'si' comes at position 'index' in a sequence of code units.
        # (position of 'FromSi' shall not appear in the 'position_db' since
        #  the error detection of the first state is done in the caller.)
        if si != FromSi: position_db[si] = index

        # Group the sequences according to the interval at position 'index'.
        for interval, sub_group, last_f in __bunch_iterable(sequence_group, index):
            # Transit to new state for the given sub-group of sequences.
            if not last_f:
                # For each 'interval' a deliberate target state is generated.
                # => each target state is only reached by a single Interval.
                new_si = state_machine_index.get()
                tm[new_si] = interval
                worklist.append((new_si, sub_group, index+1))
            else:
                # If the 'interval' is the last in the sequence, the 'ToSi' is 
                # reached. Obviously this may/should happen more than once. 
                tm_end_inv[si].append(interval)

    return tm_db, tm_end_inv, position_db

