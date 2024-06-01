def equivalent(left, right):
    if left.alphabet != right.alphabet:
        raise ValueError("Input alphabets must be equal!")

    transitions = []
    previous_states = []
    alphabet = left.alphabet
    states = [(left.initial_state(), right.initial_state())]

    while len(states) != 0:
        l, r = states.pop()
        previous_states.append((l.name, r.name))

        for value in alphabet:
            next_l, next_r = l.next_state(value), r.next_state(value)
            if (next_l is None and next_r is not None) \
                    or (next_r is None and next_l is not None):
                return False

            if (next_l[0], next_r[0]) not in previous_states:
                transitions.append((next_l[1], next_r[1]))
                states.append((left[next_l[0]], right[next_r[0]]))

    for (left, right) in transitions:
        if left != right:
            return False
    return True
