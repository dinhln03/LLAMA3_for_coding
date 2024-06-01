import click
from typing import Sequence, Tuple

from click.formatting import measure_table, iter_rows


class OrderedCommand(click.Command):
    def get_params(self, ctx):
        rv = super().get_params(ctx)
        rv.sort(key=lambda o: (not o.required, o.name))
        return rv

    def format_options(self, ctx, formatter) -> None:
        """Writes all the options into the formatter if they exist."""
        opts = []
        for param in self.get_params(ctx):
            rv = param.get_help_record(ctx)
            if rv is not None:
                opts.append(rv)

        if opts:
            with formatter.section("Options"):
                self.write_dl(formatter, opts)

    @staticmethod
    def write_dl(formatter, rows: Sequence[Tuple[str, str]], col_max: int = 30, col_spacing: int = 2) -> None:
        rows = list(rows)
        widths = measure_table(rows)
        if len(widths) != 2:
            raise TypeError("Expected two columns for definition list")

        first_col = min(widths[0], col_max) + col_spacing

        for first, second in iter_rows(rows, len(widths)):
            formatter.write(f"{'':>{formatter.current_indent}}{first}")
            if not second:
                formatter.write("\n")
                continue
            if len(first) <= first_col - col_spacing:
                formatter.write(" " * (first_col - len(first)))
            else:
                formatter.write("\n")
                formatter.write(" " * (first_col + formatter.current_indent))

            if "[" in second:
                text, meta = second.split("[")
                formatter.write(f"[{meta} {text}\n")
            else:
                formatter.write(f"{second}\n")


def add_options(options):
    def _add_options(func):
        for option in reversed(options):
            func = option(func)
        return func
    return _add_options


class Defaults:
    DOCKER_IMAGE = "docker.io/yellowdogco/virtual-screening-worker-public:3.3.0"
    PORTAL_API_URL = "https://portal.yellowdog.co/api"
    NAMESPACE = "virtual-screening"
    RETRIES = 10


shared_options = [
    click.option("--api_key_id", envvar="API_KEY_ID", required=True,
                 help="The application's API key ID for authenticating with the platform API. It is recommended to "
                      "supply this via the environment variable API_KEY_ID"),
    click.option("--api_key_secret", envvar="API_KEY_SECRET", required=True,
                 help="The application's API key secret for authenticating with the platform API. It is recommended to "
                      "supply this via the environment variable API_KEY_SECRET"),
    click.option("--template_id", envvar="TEMPLATE_ID", required=True,
                 help="The compute requirement template ID to use for provisioning compute"),
    click.option("--platform_api_url", envvar="PLATFORM_API_URL", default=Defaults.PORTAL_API_URL,
                 help="The URL of the platform API"),
    click.option("--namespace", envvar="NAMESPACE", default=Defaults.NAMESPACE,
                 help="The namespace within which all work and compute will be created"),
    click.option("--docker_image", envvar="DOCKER_IMAGE", default=Defaults.DOCKER_IMAGE,
                 help="The docker image that will be executed by the workers"),
    click.option("--retries", envvar="RETRIES", type=int, default=Defaults.RETRIES,
                 help="The number of times each failed task should be retried"),
]
