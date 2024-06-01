from collections import namedtuple

Vote = namedtuple('Vote', 'user post vote')


def create_vote(vote_dict, cutoff):
    """
    changes the vote to the [-1, 1] range
    """
    modified_vote = 1 if float(vote_dict['vote']) > cutoff else -1
    return Vote(
        user=str(vote_dict['user']),
        post=str(vote_dict['post']),
        vote=modified_vote
    )
