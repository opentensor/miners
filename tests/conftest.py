from pytest import fixture

def pytest_addoption(parser):
    parser.addoption(
        "--name",
        action="store"
    )

@fixture()
def name(request):
    return request.config.getoption("--name")