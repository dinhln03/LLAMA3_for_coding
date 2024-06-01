import github
import pandas as pd


def get_issues(repo_addr):
    g = github.Github()
    repo = g.get_repo(repo_addr)
    return repo.get_issues()


def fetch_issue_activity(repo_addr):
    g = github.Github()
    issues = g.get_repo(repo_addr).get_issues(state="all")

    events = []
    for issue in issues:
        if issue.pull_request is not None:
            continue

        events.append((issue.created_at, 1))
        if issue.state == "closed":
            events.append((issue.closed_at, -1))

    df = pd.DataFrame(events, columns=["date", "action"])

    df.sort_values("date", inplace=True)
    df["open"] = df["action"].cumsum()
    df["total_events"] = abs(df["action"]).cumsum()
    df["closed"] = (df["total_events"] - df["open"]) // 2

    return df
