from brownie import AdvancedCollectible, network
import pytest
from scripts.advanced_collectible.deploy_and_create import deploy_and_create, get_contract
from scripts.utils.helpful_scripts import LOCAL_BLOCKCHAIN_ENVIRONMENTS, get_account


def test_can_create_advanced_collectible():
    if network.show_active() not in LOCAL_BLOCKCHAIN_ENVIRONMENTS:
        pytest.skip("Only for local testing")
    advanced_collectible, creation_transaction = deploy_and_create()
    # getting the requestId value from the requestedCollectible event
    requestId = creation_transaction.events["requestedCollectible"]["requestId"]
    randomNumber = 777
    get_contract("vrf_coordinator").callBackWithRandomness(
        requestId, randomNumber, advanced_collectible.address, {"from": get_account()})

    assert advanced_collectible.tokenCounter() == 1
    assert advanced_collectible.tokenIdToBreed(0) == randomNumber % 3
