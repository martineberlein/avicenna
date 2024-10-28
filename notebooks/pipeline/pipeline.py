def pipeline_decision(params):
    actions = []

    if params["branch"] == "main":
        if params["changed_files"] > 100:
            actions.append("skip build")  # Line A: Skip build for large changes on main
        else:
            actions.append("build")  # Line B: Build main branch

        if params["dependency_stability"] == "unstable":
            actions.append("run integration tests")  # Line C: Run integration tests if dependencies are unstable
        elif params["dependency_stability"] == "stable" and params["test_results"] == "pass":
            actions.append("deploy")  # Line D: Deploy to production if main and stable dependencies

    elif params["branch"].startswith("feature/"):
        actions.append("build")  # Line E: Build feature branch

        if params["changed_files"] < 50 and params["review_status"] == "approved":
            actions.append("run unit tests")  # Line F: Run tests on small, approved feature branches

    if params["critical_bug_found"]:
        actions.append("halt deployment")  # Line G: Halt if critical bug

    if params["last_deployment_status"] == "failure" and params["branch"] == "main":
        actions.append("notify team")  # Line H: Notify team on repeated failure in main branch

    return actions


def main(inp):
    return pipeline_decision(inp)

