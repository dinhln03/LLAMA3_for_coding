import subprocess

import py
import pytest


@pytest.fixture(
    params=["tests/dataset-rdstmc", "tests/dataset-wiki", "tests/dataset-rntutor"]
)
def datasetdir(request):
    return py.path.local(request.param)


@pytest.fixture
def messages(datasetdir):
    msgdir = datasetdir.join("messages")
    return msgdir.listdir(fil="*.xml")


@pytest.fixture
def rncdir(datasetdir):
    return datasetdir.join("schemas")


@pytest.fixture
def rootrnc(rncdir):
    return rncdir.join("root.rnc")


@pytest.fixture
def rncschemas(rootrnc):
    return rootrnc.dirpath().listdir("*.rnc")


def test_validate_by_rnc_onemsg(rootrnc, messages):
    cmd = ["pyjing", "-c"]
    cmd.append(rootrnc.strpath)
    cmd.append(messages[0].strpath)
    subprocess.check_call(cmd)


def test_validate_by_rnc_allmsgs(rootrnc, messages):
    cmd = ["pyjing", "-c"]
    cmd.append(rootrnc.strpath)
    cmd.extend(map(str, messages))
    subprocess.check_call(cmd)


def test_rnc2rng(rootrnc, tmpdir, rncschemas):
    cmd = ["pytrang"]
    rngname = rootrnc.new(dirname=tmpdir, ext=".rng")
    cmd.append(rootrnc.strpath)
    cmd.append(rngname.strpath)
    subprocess.check_call(cmd)
    rngnames = tmpdir.listdir(fil="*.rng")
    assert len(rngnames) == len(rncschemas)
    for rnc, rng in zip(sorted(rngnames), sorted(rncschemas)):
        assert rnc.purebasename == rng.purebasename


"""RNG section ========================
"""


@pytest.fixture
def rngschemas(rootrnc, tmpdir, rncschemas):
    cmd = ["pytrang"]
    rngname = rootrnc.new(dirname=tmpdir, ext=".rng")
    cmd.append(rootrnc.strpath)
    cmd.append(rngname.strpath)
    subprocess.check_call(cmd)
    rngnames = tmpdir.listdir(fil="*.rng")
    assert len(rngnames) == len(rncschemas)
    for rnc, rng in zip(sorted(rngnames), sorted(rncschemas)):
        assert rnc.purebasename == rng.purebasename

    return rngnames


@pytest.fixture
def rootrng(rngschemas):
    rootschema = rngschemas[0].new(basename="root.rng")
    assert rootschema in rngschemas
    rootschema.ensure()
    return rootschema


def test_validate_by_rng_onemsg(rootrng, messages):
    cmd = ["pyjing"]
    cmd.append(rootrng.strpath)
    cmd.append(messages[0].strpath)
    subprocess.check_call(cmd)


def test_validate_by_rng_allmsgs(rootrng, messages):
    cmd = ["pyjing"]
    cmd.append(rootrng.strpath)
    cmd.extend(map(str, messages))
    subprocess.check_call(cmd)
