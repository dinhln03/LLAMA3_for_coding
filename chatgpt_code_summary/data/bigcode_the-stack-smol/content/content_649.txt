#!/usr/bin/env python
import sys
import subprocess

try:
    import gtk
except:
    print >> sys.stderr, "You need to install the python gtk bindings"
    sys.exit(1)
 
# import vte
try:
    import vte
except:
    error = gtk.MessageDialog (None, gtk.DIALOG_MODAL, gtk.MESSAGE_ERROR, gtk.BUTTONS_OK,
'You need to install python bindings for libvte')
    error.run()
    sys.exit (1)

def on_key_press_event(widget, event):
    keyname = gtk.gdk.keyval_name(event.keyval)
    '''print "Key %s (%d) was pressed" % (keyname, event.keyval)
    v.feed_child(keyname, len(keyname))
    v2.feed_child(keyname, len(keyname))'''
    for i in terms:
        i.emit("key-press-event", event)
    if (event.keyval == 65293):
           text.set_text("")

nbterm = 3
terms = []
if __name__ == '__main__':

    w = gtk.Window()
    hbox = gtk.HBox()

    x = 0
    y = 0
    for i in range(0, len(sys.argv)):


        v = vte.Terminal ()
        v.connect ("child-exited", lambda term: gtk.main_quit())
        v.fork_command()
        window = gtk.Window()

        if (i > 0):
            print sys.argv[i]
            r=subprocess.Popen(["/bin/bash", "-i", "-c", sys.argv[i]], shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
			#v.feed_child(sys.argv[i], len(sys.argv[i]))
            #line=r.stdout.readline()
            #print line
            v.feed_child(sys.argv[i], len(sys.argv[i]))
            e = gtk.gdk.Event(gtk.gdk.KEY_PRESS)
            e.keyval = 65293
            e.send_event = True
            window.set_title("Window %s" % (sys.argv[i]))
        else:
			window.set_title("Window %d" % (i+1))
        terms.append(v)
        window.add(v)
        window.connect('delete-event', lambda window, event: gtk.main_quit())
        window.move(x, y)
        window.set_default_size(200, 100)
        #window.set_title("Window %d" % (i+1))
        window.show_all()

        if (i > 0):
            e.window = window.get_window()
            v.emit("key-press-event", e)

        x += 780
        if (i-1 % 3 == 0):
            y += 450
            x = 0

    text = gtk.Entry()
    text.connect("key_press_event", on_key_press_event)
    w.set_default_size(200, 15)
    w.move(0, 0)
    hbox.pack_start(text, True, True, 0)
    w.add(hbox)
    w.connect('delete-event', lambda window, event: gtk.main_quit())
    w.show_all()

    text.set_can_focus(True)
    text.grab_focus()
    gtk.main()
