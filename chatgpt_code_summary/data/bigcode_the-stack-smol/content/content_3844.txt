def average_rating(rating_list):

    if not rating_list:
        # if rating_list is empty return 0
        return 0

    return round(sum(rating_list) / len(rating_list))