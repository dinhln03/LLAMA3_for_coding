def belong(in_list1: list, in_list2: list) -> list:
    """
    Check wheter or not all the element in list in_list1 belong into in_list2
    :param in_list1: the source list
    :param in_list2: the target list where to find the element in in_list1
    :return: return True if the statement is verified otherwise return False
    """
    return all(element in in_list2 for element in in_list1)

if __name__ == "__main__":
    print(belong([1,2,3,4],[4,5,6,5,7,0,4,2,3]))