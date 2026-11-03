#!/usr/bin/env python3
"""Native GTK4 + WebKit desktop shell for Artificial Brain."""
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import gi
gi.require_version('Gtk', '4.0')
gi.require_version('WebKit', '6.0')
from gi.repository import Gtk, Gio, GLib, WebKit

ROOT = Path(__file__).resolve().parent
URL = 'http://127.0.0.1:8765/'
STATUS_URL = URL + 'api/status'


def server_ready():
    try:
        with urllib.request.urlopen(STATUS_URL, timeout=1.5) as response:
            return response.status == 200
    except Exception:
        return False


def start_server():
    if server_ready():
        return True
    log = open('/tmp/artificial-brain-gui.log', 'a')
    subprocess.Popen([str(ROOT / 'run-gui.sh')], cwd=ROOT,
                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                     start_new_session=True)
    for _ in range(40):
        if server_ready():
            return True
        time.sleep(0.5)
    return False


class ArtificialBrainApp(Gtk.Application):
    def __init__(self):
        super().__init__(application_id='com.mygame77.ArtificialBrain',
                         flags=Gio.ApplicationFlags.DEFAULT_FLAGS)

    def do_activate(self):
        window = Gtk.ApplicationWindow(application=self)
        window.set_title('Artificial Brain')
        window.set_default_size(1200, 800)
        window.set_size_request(720, 520)

        if not server_ready():
            label = Gtk.Label(label='Starting Artificial Brain…')
            label.set_margin_top(32)
            window.set_child(label)
            window.present()
            def load_when_ready():
                if server_ready():
                    view = self.make_view()
                    window.set_child(view)
                    return GLib.SOURCE_REMOVE
                return GLib.SOURCE_CONTINUE
            GLib.timeout_add(500, load_when_ready)
        else:
            window.set_child(self.make_view())
            window.present()

    def make_view(self):
        view = WebKit.WebView()
        settings = view.get_settings()
        settings.set_enable_developer_extras(False)
        view.load_uri(URL)
        return view


if __name__ == '__main__':
    if not start_server():
        print('Artificial Brain server did not start. Check /tmp/artificial-brain-gui.log.', file=sys.stderr)
        raise SystemExit(1)
    app = ArtificialBrainApp()
    raise SystemExit(app.run(sys.argv[:1]))