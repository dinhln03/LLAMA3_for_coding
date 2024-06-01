import numpy as np
from pddlgym.core import get_successor_states, InvalidAction
from pddlgym.inference import check_goal


def get_all_reachable(s, A, env, reach=None):
    reach = {} if not reach else reach

    reach[s] = {}
    for a in A:
        try:
            succ = get_successor_states(s,
                                        a,
                                        env.domain,
                                        raise_error_on_invalid_action=True,
                                        return_probs=True)
        except InvalidAction:
            succ = {s: 1.0}
        reach[s][a] = {s_: prob for s_, prob in succ.items()}
        for s_ in succ:
            if s_ not in reach:
                reach.update(get_all_reachable(s_, A, env, reach))
    return reach


def vi(S, succ_states, A, V_i, G_i, goal, env, gamma, epsilon):

    V = np.zeros(len(V_i))
    P = np.zeros(len(V_i))
    pi = np.full(len(V_i), None)
    print(len(S), len(V_i), len(G_i), len(P))
    print(G_i)
    P[G_i] = 1

    i = 0
    diff = np.inf
    while True:
        print('Iteration', i, diff)
        V_ = np.copy(V)
        P_ = np.copy(P)

        for s in S:
            if check_goal(s, goal):
                continue
            Q = np.zeros(len(A))
            Q_p = np.zeros(len(A))
            cost = 1
            for i_a, a in enumerate(A):
                succ = succ_states[s, a]

                probs = np.fromiter(iter(succ.values()), dtype=float)
                succ_i = [V_i[succ_s] for succ_s in succ_states[s, a]]
                Q[i_a] = cost + np.dot(probs, gamma * V_[succ_i])
                Q_p[i_a] = np.dot(probs, P_[succ_i])
            V[V_i[s]] = np.min(Q)
            P[V_i[s]] = np.max(Q_p)
            pi[V_i[s]] = A[np.argmin(Q)]

        diff = np.linalg.norm(V_ - V, np.inf)
        if diff < epsilon:
            break
        i += 1
    return V, pi
