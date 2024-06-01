# Copyright 2021 Canonical Ltd.
# See LICENSE file for licensing details.

from pathlib import Path
from subprocess import check_output
from time import sleep

import pytest
import yaml
from selenium import webdriver
from selenium.common.exceptions import JavascriptException, WebDriverException
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support.ui import WebDriverWait


METADATA = yaml.safe_load(Path("./metadata.yaml").read_text())


@pytest.mark.abort_on_fail
async def test_build_and_deploy(ops_test):
    my_charm = await ops_test.build_charm(".")
    image_path = METADATA["resources"]["oci-image"]["upstream-source"]

    await ops_test.model.deploy(my_charm, resources={"oci-image": image_path})

    charm_name = METADATA["name"]
    await ops_test.model.wait_for_idle(
        [charm_name],
        raise_on_blocked=True,
        raise_on_error=True,
        timeout=300,
    )
    assert ops_test.model.applications[charm_name].units[0].workload_status == "waiting"
    assert (
        ops_test.model.applications[charm_name].units[0].workload_status_message
        == "Waiting for kubeflow-profiles relation data"
    )


@pytest.mark.abort_on_fail
async def test_add_profile_relation(ops_test):
    charm_name = METADATA["name"]
    # TODO: Point kubeflow-profiles to latest/stable when Rev 54 or higher are promoted
    await ops_test.model.deploy("kubeflow-profiles", channel="latest/edge")
    await ops_test.model.add_relation("kubeflow-profiles", charm_name)
    await ops_test.model.wait_for_idle(
        ["kubeflow-profiles", charm_name],
        status="active",
        raise_on_blocked=True,
        raise_on_error=True,
        timeout=300,
    )


async def test_status(ops_test):
    charm_name = METADATA["name"]
    assert ops_test.model.applications[charm_name].units[0].workload_status == "active"


def fix_queryselector(elems):
    """Workaround for web components breaking querySelector.

    Because someone thought it was a good idea to just yeet the moral equivalent
    of iframes everywhere over a single page 🤦

    Shadow DOM was a terrible idea and everyone involved should feel professionally
    ashamed of themselves. Every problem it tried to solved could and should have
    been solved in better ways that don't break the DOM.
    """

    selectors = '").shadowRoot.querySelector("'.join(elems)
    return 'return document.querySelector("' + selectors + '")'


@pytest.fixture()
async def driver(request, ops_test):
    status = yaml.safe_load(
        check_output(
            ["juju", "status", "-m", ops_test.model_full_name, "--format=yaml"]
        )
    )
    endpoint = status["applications"]["kubeflow-dashboard"]["address"]
    application = ops_test.model.applications["kubeflow-dashboard"]
    config = await application.get_config()
    port = config["port"]["value"]
    url = f"http://{endpoint}.nip.io:{port}/"
    options = Options()
    options.headless = True

    with webdriver.Firefox(options=options) as driver:
        wait = WebDriverWait(driver, 180, 1, (JavascriptException, StopIteration))
        for _ in range(60):
            try:
                driver.get(url)
                break
            except WebDriverException:
                sleep(5)
        else:
            driver.get(url)

        yield driver, wait, url

        driver.get_screenshot_as_file(f"/tmp/selenium-{request.node.name}.png")


def test_links(driver):
    driver, wait, url = driver

    # Ensure that sidebar links are set up properly
    links = [
        "/jupyter/",
        "/katib/",
        "/pipeline/#/experiments",
        "/pipeline/#/pipelines",
        "/pipeline/#/runs",
        "/pipeline/#/recurringruns",
        # Removed temporarily until https://warthogs.atlassian.net/browse/KF-175 is fixed
        # "/pipeline/#/artifacts",
        # "/pipeline/#/executions",
        "/volumes/",
        "/tensorboards/",
    ]

    for link in links:
        print("Looking for link: %s" % link)
        script = fix_queryselector(["main-page", f"iframe-link[href='{link}']"])
        wait.until(lambda x: x.execute_script(script))

    # Ensure that quick links are set up properly
    links = [
        "/pipeline/",
        "/pipeline/#/runs",
        "/jupyter/new?namespace=kubeflow",
        "/katib/",
    ]

    for link in links:
        print("Looking for link: %s" % link)
        script = fix_queryselector(
            [
                "main-page",
                "dashboard-view",
                f"iframe-link[href='{link}']",
            ]
        )
        wait.until(lambda x: x.execute_script(script))

    # Ensure that doc links are set up properly
    links = [
        "https://charmed-kubeflow.io/docs/kubeflow-basics",
        "https://microk8s.io/docs/addon-kubeflow",
        "https://www.kubeflow.org/docs/started/requirements/",
    ]

    for link in links:
        print("Looking for link: %s" % link)
        script = fix_queryselector(
            [
                "main-page",
                "dashboard-view",
                f"a[href='{link}']",
            ]
        )
        wait.until(lambda x: x.execute_script(script))
