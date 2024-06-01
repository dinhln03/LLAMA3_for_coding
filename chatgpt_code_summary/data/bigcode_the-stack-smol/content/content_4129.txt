import os
import layout
import callbacks  # layout needs to be defined before creating callbacks
import routes
import appserver

server = appserver.app.server
if __name__ == "__main__":
    debug_mode = True if os.getenv("DEBUG", "false") == "true" else False
    if debug_mode is True:
        print(f"Initiating server. Debug mode enabled.")
        # appserver.app.enable_dev_tools(debug=True)
    else:
        print(f"Initiating server.")

    appserver.app.run_server(
        debug=debug_mode,
        host="0.0.0.0",
        port=5000
    )