import nox


@nox.session
def format(session):
    session.install("-r", "requirements.dev.txt")

    session.run("black", ".")
    session.run("isort", ".")


@nox.session
def lint(session):
    session.install("msprime")
    session.install("-r", "requirements.dev.txt")
    session.install(".")

    session.run("pylint", "lshmm")
    session.run("pylint", "tests")


@nox.session
def test(session):
    session.install("tskit")
    session.install("msprime")
    session.install("-r", "requirements.dev.txt")
    session.install(".")

    session.run("pytest")


@nox.session
def build_docs(session):
    session.install("tskit")
    session.install("msprime")
    session.install("-r", "docs/requirements.docs.txt")
    session.install(".")

    session.cd("docs")
    session.run("rm", "-rf", "api_reference", external=True)
    session.run("python3", "generate_api_reference.py")

    session.run("rm", "-rf", "html", external=True)
    session.run("python3", "-m", "sphinx", "-W", "-b", "html", ".", "html")


@nox.session
def serve_docs(session):
    session.cd("docs/html")
    session.run("python3", "-m", "http.server")


@nox.session
def pip_compile(session):
    session.install("pip-tools")

    session.run("pip-compile", *session.posargs)


nox.options.sessions = ["format", "lint", "test", "build_docs"]
