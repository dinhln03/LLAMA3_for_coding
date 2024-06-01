#!/usr/bin/env python3

import iterm2
# To install, update, or remove packages from PyPI, use Scripts > Manage > Manage Dependencies...
import subprocess

async def main(connection):
    component = iterm2.StatusBarComponent(
            short_description = 'k8s current context',
            detailed_description = 'Display k8s current context',
            knobs = [],
            exemplar = 'cluster-1',
            update_cadence = 3,
            identifier = 'com.github.bassaer.iterm2-k8s-context'
    )

    @iterm2.StatusBarRPC
    async def coro(knobs):
        result = subprocess.run(
                ['kubectl', 'config', 'current-context'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                encoding='utf-8')
        tmpl = '⎈  {}'
        if result.returncode != 0:
            return tmpl.format('Error');
        return tmpl.format(result.stdout.strip());

    await component.async_register(connection, coro)

# This instructs the script to run the "main" coroutine and to keep running even after it returns.
iterm2.run_forever(main)
