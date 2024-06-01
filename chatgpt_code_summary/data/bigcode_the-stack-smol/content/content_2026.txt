from time import sleep

from ec2mc import __main__

def test_user_commands():
    """test all user commands."""
    assert __main__.main([
        "user", "create", "ec2mc_test_user", "setup_users", "--default"
    ]) is not False
    sleep(5)
    assert __main__.main([
        "user", "list"
    ]) is not False
    assert __main__.main([
        "user", "set_group", "EC2MC_TEST_USER", "basic_users"
    ]) is not False
    assert __main__.main([
        "user", "be", "takingitcasual"
    ]) is not False
    assert __main__.main([
        "user", "rotate_key", "Ec2Mc_TeSt_UsEr"
    ]) is not False
    assert __main__.main([
        "user", "delete", "eC2mC_tEsT_uSeR"
    ]) is not False
