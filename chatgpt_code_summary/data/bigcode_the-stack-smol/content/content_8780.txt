import typer
from typing import Optional
from article_ripper import get_document, html_to_md

app = typer.Typer()


@app.command()
def fun(url: str, out: Optional[str] = None) -> None:
    doc = get_document(url)
    doc_summary = doc.summary()
    if out is None:
        print(doc_summary)
    else:
        with open(out, "w") as f:
            f.write(doc_summary)


def run() -> None:
    app()


if __name__ == "__main__":
    run()
