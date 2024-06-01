import libtmux


def ensure_server() -> libtmux.Server:
    '''
    Either create new or return existing server
    '''
    return libtmux.Server()


def spawn_session(name: str, kubeconfig_location: str, server: libtmux.Server):
    if server.has_session(name):
        return
    else:
        session = server.new_session(name)
        session.set_environment("KUBECONFIG", kubeconfig_location)
        # the new_session will create default window and pane which will not contain KUBECONFIG, add manually
        session.attached_window.attached_pane.send_keys("export KUBECONFIG={}".format(kubeconfig_location))
